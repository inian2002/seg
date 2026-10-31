"""
train.py  —  Full-supervised 3D seismic segmentation
原始代码逻辑完整保留，仅新增两个功能：
  1. --restore  从指定 checkpoint 继续训练
  2. label=-1   稀疏标注支持（-1 不计损失，不计指标）

数据：原始数据已由外部完成归一化，代码内不做任何处理。

──────────────────────────────────────────────────────────────────
指标设计说明
──────────────────────────────────────────────────────────────────
【问题根因】原版 val() 是 patch 级别的平均：
  - 背景纯净 patch（无 salt）→ union=0 → IoU=1.0
  - 假设 300/336 patch 是背景 → 平均 IoU ≈ 0.97（虚高）
  - predict.py 大体推理用全局 IoU → 0.34（真实）
  两者差距巨大，训练时无法判断模型真实性能。

【修复方案】val() 改为 全局累积（global accumulation）：
  epoch 内累加所有 patch 的 inter / pred_pos / target_pos，
  最后统一计算 IoU 和 Dice。
  这与 predict.py 大体滑窗的计算逻辑完全一致，
  训练时 Val IoU 将与 predict.py 结果吻合。

【Train IoU 说明】
  稀疏标注（含-1）时，Train IoU 仅统计有标注体素（label≠-1），
  数值偏高属预期行为，仍以 Val IoU（全标注验证集）为真实指标。
  全标注（无-1）时，Train IoU 与原版完全一致。

──────────────────────────────────────────────────────────────────
用法
──────────────────────────────────────────────────────────────────
# 全标注从头训练（原版行为，完全不变）
python train.py --train_path /seam_patches --val_path /seam_val --save output/seam

# 从 SEAM 预训练权重微调 F3 稀疏标注
python train.py \\
    --restore    output/seam/checkpoints/epoch_020.pth \\
    --train_path /f3_patches_sparse \\
    --val_path   /f3_patches \\
    --save       output/f3_ft \\
    --lr         0.0001 --epochs 100
"""

import torch
import os
import numpy as np
import json
from datetime import datetime
from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from argues import config_2

args = config_2.args

from models.SaltHRNet3D import SaltHRNet3D
from utils.dataloader import salt_dataset
from utils.dataloader_val import salt_dataset_val
from utils import logger
from utils.checkpoint import load_model_state
from utils.reconstruction import (
    build_anisotropic_mask,
    reconstruction_loss_dict,
    sigmoid_rampup,
)

if args.da_type == 'pasa':
    from utils.pasa import Weak_DA, adjust_learning_rate, LossAverage
else:
    from utils.common import Weak_DA, adjust_learning_rate, LossAverage

from utils.loss import Weight_BCE_loss, dice_loss


# ═══════════════════════════════════════════════════════════════════
# 1. 损失函数
# ═══════════════════════════════════════════════════════════════════
def bce_dice_loss(output, target):
    """
    BCE + Dice 混合损失。
    loss.py 中的两个函数已修复 -1 支持：
      · 全标注（无-1）→ 与原版完全一致
      · 稀疏标注（含-1）→ -1 位置自动屏蔽，不计损失
    """
    loss_bce = Weight_BCE_loss(output, target)
    loss_dice = dice_loss(output, target)
    return 0.5 * loss_bce + 0.5 * loss_dice


def restore_checkpoint(model, optimizer, restore_path, device):
    start_epoch = 1
    if not os.path.isfile(restore_path):
        raise FileNotFoundError(f"[restore] 找不到权重文件: {restore_path}")

    ckpt = torch.load(restore_path, map_location=device)
    state = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
    load_model_state(model, state, verbose=True)

    if isinstance(ckpt, dict) and 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
            print("  [ckpt] optimizer state restored")
        except ValueError as exc:
            print(f"  [ckpt] optimizer state skipped: {exc}")

    if isinstance(ckpt, dict) and 'epoch' in ckpt:
        start_epoch = ckpt['epoch'] + 1

    return ckpt, start_epoch


# ═══════════════════════════════════════════════════════════════════
# 2. 训练函数
# ═══════════════════════════════════════════════════════════════════
def train(model, train_loader, optimizer, criterion, args, epoch, device):
    model.train()
    train_loss = LossAverage()
    seg_loss_meter = LossAverage()
    recon_loss_meter = LossAverage()

    # 全局累积指标（排除 -1 体素）
    g_inter = 0.0
    g_pred_pos = 0.0
    g_target_pos = 0.0

    step = len(train_loader)
    bar = tqdm(enumerate(train_loader), total=step,
               desc=f"Epoch {epoch}/{args.epochs}")

    for idx, (data, target) in bar:
        data, target = data.to(device), target.to(device)
        data, target = Weak_DA(data, target)

        optimizer.zero_grad()
        output = model(data)
        seg_loss = criterion(output, target)

        # Immediate backward to free graph 1
        seg_loss.backward()
        loss_val = seg_loss.item()
        recon_display = 0.0

        if bool(args.use_recon) and getattr(model, 'recon_head', None) is not None:
            masked_data, recon_mask = build_anisotropic_mask(
                data,
                horizontal_ratio=args.recon_horizontal_ratio,
                vertical_ratio=args.recon_vertical_ratio,
                block_ratio=args.recon_block_ratio,
            )
            aux_out = model(masked_data, return_aux=True)
            recon_terms = reconstruction_loss_dict(
                aux_out['recon'],
                data,
                recon_mask,
                edge_weight=args.recon_edge_weight,
                freq_weight=args.recon_freq_weight,
                low_freq_ratio=args.recon_low_freq_ratio,
            )
            ramp = sigmoid_rampup(epoch, args.recon_rampup)
            recon_loss = args.recon_weight * ramp * recon_terms['total']

            # Immediate backward to free graph 2
            recon_loss.backward()
            recon_display = recon_loss.item()
            recon_loss_meter.update(recon_display, data.size(0))
            loss_val += recon_display

        optimizer.step()

        # ── 指标累积（排除 -1）────────────────────────────────────
        # 全标注时 valid_mask 全为 True，与原版逻辑等价
        # 稀疏标注时仅统计有标注体素（label≠-1）
        with torch.no_grad():
            valid = (target != -1)
            pred = (output > 0.5).float()
            pv = pred[valid]
            tv = target[valid].float()

            inter = (pv * tv).sum().item()
            pred_pos = pv.sum().item()
            tgt_pos = tv.sum().item()

            g_inter += inter
            g_pred_pos += pred_pos
            g_target_pos += tgt_pos

            run_union = g_pred_pos + g_target_pos - g_inter
            run_iou = g_inter / run_union if run_union > 0 else 1.0

        iter_step = (epoch - 1) * step + idx
        if args.optim == 'SGD':
            adjust_learning_rate(optimizer, args.lr,
                                 args.epochs * step, iter_step)

        train_loss.update(loss_val, data.size(0))
        seg_loss_meter.update(seg_loss.item(), data.size(0))
        bar.set_postfix(
            loss=f"{loss_val:.4f}",
            seg=f"{seg_loss.item():.4f}",
            rec=f"{recon_display:.4f}",
            iou=f"{run_iou:.4f}"
        )

    # epoch 级全局指标
    union = g_pred_pos + g_target_pos - g_inter
    if union < 1.0:
        epoch_iou = 1.0
        epoch_dice = 1.0
    else:
        epoch_iou = g_inter / union
        epoch_dice = 2.0 * g_inter / (g_pred_pos + g_target_pos + 1e-6)

    log = OrderedDict({
        'Train_Loss': train_loss.avg,
        'Train_SegLoss': seg_loss_meter.avg,
        'Train_IoU': round(float(epoch_iou), 4),
        'Train_Dice': round(float(epoch_dice), 4),
    })
    if bool(args.use_recon):
        log['Train_ReconLoss'] = recon_loss_meter.avg
        log['Train_ReconRamp'] = round(sigmoid_rampup(epoch, args.recon_rampup), 4)
    return log


# ═══════════════════════════════════════════════════════════════════
# 3. 验证函数  —  全局累积，与 predict.py 大体推理指标一致
# ═══════════════════════════════════════════════════════════════════
def val(model, val_loader, device):
    """
    【关键修复】使用全局累积替代 patch 级别平均。

    原版（patch 平均）的问题：
      背景纯净 patch → union=0 → IoU 强制为 1.0
      大量此类 patch 导致 Val IoU 虚高（0.97），
      与 predict.py 大体推理的真实 IoU（0.34）严重不一致。

    修复后（全局累积）：
      epoch 内累加所有 patch 的 inter/pred_pos/target_pos，
      最终统一计算 IoU 和 Dice，
      结果与 predict.py 大体滑窗计算逻辑完全一致。

    注意：--val_path 必须指向全标注验证集（label 只有 0/1）。
    """
    model.eval()

    g_inter = 0.0
    g_pred_pos = 0.0
    g_target_pos = 0.0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            for b in range(data.size(0)):
                p = (output[b, 0] > 0.5).float()
                t = (target[b, 0] > 0.5).float()

                # 全标注验证集：valid 全为 True，等价于不过滤
                # 保留此行以防万一验证集混入 -1
                valid = (target[b, 0] != -1)
                pv = p[valid]
                tv = t[valid]

                inter = (pv * tv).sum().item()
                pred_pos = pv.sum().item()
                tgt_pos = tv.sum().item()

                g_inter += inter
                g_pred_pos += pred_pos
                g_target_pos += tgt_pos

    union = g_pred_pos + g_target_pos - g_inter
    if union < 1.0:
        iou = 1.0
        dice = 1.0
    else:
        iou = g_inter / union
        dice = 2.0 * g_inter / (g_pred_pos + g_target_pos + 1e-6)

    return OrderedDict({
        'IoU': round(float(iou), 4),
        'Dice': round(float(dice), 4),
    })


# ═══════════════════════════════════════════════════════════════════
# 4. 主程序入口
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    args = config_2.args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 输出目录 ──────────────────────────────────────────────────
    save_path = args.save
    ckpt_path = os.path.join(save_path, 'checkpoints')
    os.makedirs(ckpt_path, exist_ok=True)

    start_time = datetime.now()

    # ── 保存配置 ──────────────────────────────────────────────────
    args_dict = vars(args)
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)
    with open(os.path.join(save_path, 'config.txt'), 'w') as f:
        f.write(f"Experiment Start Time: "
                f"{start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 40 + "\n")
        for k, v in args_dict.items():
            f.write(f"{k}: {v}\n")
        f.write("-" * 40 + "\n")

    print(f"实验已启动 | 输出目录: {save_path}")
    print(f"⚠  --val_path 请指向【全标注】验证集文件夹（label 只有 0/1）")
    print(f"   Val IoU 已修复为全局累积，与 predict.py 大体推理指标一致")

    # ── 数据加载 ──────────────────────────────────────────────────
    train_set = salt_dataset(data_dir=args.train_path, train=True)
    val_set = salt_dataset_val(data_dir=args.val_path, train=False)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.n_threads, shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        num_workers=args.n_threads, shuffle=False, pin_memory=True)

    print(f"train: {len(train_set)} patches  |  val: {len(val_set)} patches")

    # ── 模型与优化器 ──────────────────────────────────────────────
    model = SaltHRNet3D(base=args.base_width,
                     enable_reconstruction=bool(args.use_recon)).to(device)
    optimizer = eval(f'torch.optim.{args.optim}')(
        model.parameters(), lr=args.lr)
    criterion = bce_dice_loss

    # ══════════════════════════════════════════════════════════════
    # ✅ 新增功能：从 checkpoint 继续训练
    #
    # --restore 不指定（默认 None）→ 从头训练，与原版完全一致
    # --restore 指定路径           → 加载权重继续训练
    #
    # 兼容三种格式：
    #   1. {'net':..., 'optimizer':..., 'epoch':...}  标准格式（全监督）
    #   2. {'net':..., 'teacher_net':...}              半监督格式（取 net）
    #   3. 纯 state_dict                               直接加载
    # ══════════════════════════════════════════════════════════════
    start_epoch = 1

    if args.restore is not None:
        ckpt, start_epoch = restore_checkpoint(
            model, optimizer, args.restore, device)
        print(f"✅ 已从 {args.restore} 恢复"
              f"（saved epoch={ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'}）"
              f"，从 epoch {start_epoch} 继续")
    else:
        print("ℹ  未指定 --restore，从头训练")

    # ── 日志 ──────────────────────────────────────────────────────
    log = logger.Train_Logger(save_path, 'train_log')
    best_score = -1.0
    val_name = "F3"
    val_iou_key = f'Val_{val_name}_IoU'
    val_dice_key = f'Val_{val_name}_Dice'

    # ══════════════════════════════════════════════════════════════
    # 训练循环（结构与原版完全一致）
    # ══════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, args.epochs + 1):

        train_log = train(model, train_loader, optimizer,
                          criterion, args, epoch, device)

        val_log = val(model, val_loader, device)
        val_iou = val_log['IoU']
        val_dice = val_log['Dice']

        # 合并日志
        log_info = {
            **train_log,
            val_iou_key: val_iou,
            val_dice_key: val_dice,
        }
        log.update(epoch, log_info)

        # 保存每轮模型
        epoch_model_path = os.path.join(
            ckpt_path, f'epoch_{epoch:03d}.pth')
        torch.save({
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'train_log': train_log,
            'val_log': val_log,
        }, epoch_model_path)

        # 更新 latest.pth
        torch.save({
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }, os.path.join(ckpt_path, 'latest.pth'))

        # 最优模型（以全局累积 Val IoU+Dice 为标准）
        current_score = val_iou + val_dice
        if current_score > best_score:
            best_score = current_score
            torch.save({
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'score': best_score,
            }, os.path.join(save_path, 'best_model.pth'))
            print(f"✨ Epoch {epoch}: 新最优 "
                  f"({val_iou_key}+{val_dice_key}={best_score:.4f})")

        print(f"Epoch {epoch}: {val_name} -> "
              f"IoU={val_iou:.4f}, Dice={val_dice:.4f}")

    # ── 结束 ──────────────────────────────────────────────────────
    end_time = datetime.now()
    duration = end_time - start_time
    with open(os.path.join(save_path, 'config.txt'), 'a') as f:
        f.write(f"Experiment End Time: "
                f"{end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {duration}\n")

    print(f"✅ 训练完成！结果存放于: {save_path}")
