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


def bce_dice_loss(output, target):

    loss_bce = Weight_BCE_loss(output, target)
    loss_dice = dice_loss(output, target)
    return 0.5 * loss_bce + 0.5 * loss_dice


def restore_checkpoint(model, optimizer, restore_path, device):
    start_epoch = 1
    if not os.path.isfile(restore_path):
        raise FileNotFoundError(f"[restore] Checkpoint not found: {restore_path}")

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


def train(model, train_loader, optimizer, criterion, args, epoch, device):
    model.train()
    train_loss = LossAverage()
    seg_loss_meter = LossAverage()
    recon_loss_meter = LossAverage()


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


            recon_loss.backward()
            recon_display = recon_loss.item()
            recon_loss_meter.update(recon_display, data.size(0))
            loss_val += recon_display

        optimizer.step()


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


def val(model, val_loader, device):

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


if __name__ == '__main__':
    args = config_2.args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    save_path = args.save
    ckpt_path = os.path.join(save_path, 'checkpoints')
    os.makedirs(ckpt_path, exist_ok=True)

    start_time = datetime.now()


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

    print(f"Experiment started | Output directory: {save_path}")
    print("--val_path must contain dense validation labels (0/1 only).")
    print("Validation IoU pools labeled voxels across the validation patches.")


    train_set = salt_dataset(data_dir=args.train_path, train=True)
    val_set = salt_dataset_val(data_dir=args.val_path, train=False)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.n_threads, shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        num_workers=args.n_threads, shuffle=False, pin_memory=True)

    print(f"train: {len(train_set)} patches  |  val: {len(val_set)} patches")


    model = SaltHRNet3D(base=args.base_width,
                     enable_reconstruction=bool(args.use_recon)).to(device)
    optimizer = eval(f'torch.optim.{args.optim}')(
        model.parameters(), lr=args.lr)
    criterion = bce_dice_loss


    start_epoch = 1

    if args.restore is not None:
        ckpt, start_epoch = restore_checkpoint(
            model, optimizer, args.restore, device)
        print(f"Restored from {args.restore}"
              f" (saved epoch={ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'})"
              f"; continuing from epoch {start_epoch}")
    else:
        print("No --restore checkpoint supplied; training from scratch.")


    log = logger.Train_Logger(save_path, 'train_log')
    best_score = -1.0
    val_name = "F3"
    val_iou_key = f'Val_{val_name}_IoU'
    val_dice_key = f'Val_{val_name}_Dice'


    for epoch in range(start_epoch, args.epochs + 1):

        train_log = train(model, train_loader, optimizer,
                          criterion, args, epoch, device)

        val_log = val(model, val_loader, device)
        val_iou = val_log['IoU']
        val_dice = val_log['Dice']


        log_info = {
            **train_log,
            val_iou_key: val_iou,
            val_dice_key: val_dice,
        }
        log.update(epoch, log_info)


        epoch_model_path = os.path.join(
            ckpt_path, f'epoch_{epoch:03d}.pth')
        torch.save({
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'train_log': train_log,
            'val_log': val_log,
        }, epoch_model_path)


        torch.save({
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }, os.path.join(ckpt_path, 'latest.pth'))


        current_score = val_iou + val_dice
        if current_score > best_score:
            best_score = current_score
            torch.save({
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'score': best_score,
            }, os.path.join(save_path, 'best_model.pth'))
            print(f"Epoch {epoch}: New best "
                  f"({val_iou_key}+{val_dice_key}={best_score:.4f})")

        print(f"Epoch {epoch}: {val_name} -> "
              f"IoU={val_iou:.4f}, Dice={val_dice:.4f}")


    end_time = datetime.now()
    duration = end_time - start_time
    with open(os.path.join(save_path, 'config.txt'), 'a') as f:
        f.write(f"Experiment End Time: "
                f"{end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {duration}\n")

    print(f"Training complete. Results saved to: {save_path}")
