"""
utils/pseudo_utils.py  ——  伪标签自训练工具模块

【核心模块】
  1. SpatialContinuityPrior  : 基于盐体空间连续性假设的置信度调权
  2. ConfidenceEMA           : 体素级指数滑动平均，稳定多轮伪标签
  3. PseudoLabelRefiner      : 综合置信度过滤 + 空间先验的伪标签精化器
  4. pseudo_label_stats      : 打印伪标签质量统计（如有真实标注可对比）

【原理说明】
────────────────────────────────────────────────────────────────
  间隔 40 稀疏标注的核心问题：
    某些空间区域（如图像角落）完全没有标注约束，
    模型在这些区域的预测受训练集分布驱动，可能漏检。

  空间连续性先验（盐体地质假设）：
    盐体是连续的地质体，已知标注为 1 的体素周围大概率也是盐体。
    因此，对 -1 位置的伪标签置信度，按照与已知盐体标注的距离加权：
      · 紧邻已知盐体 → 置信度阈值适当放松（更容易被标为 1）
      · 远离已知标注 → 阈值保持严格

  EMA 稳定（多轮自训练防错误累积）：
    每轮的新预测与上轮 EMA 融合，避免单次高方差预测污染标签。
────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation


# ═══════════════════════════════════════════════════════════════════════════
# 1.  空间连续性先验
# ═══════════════════════════════════════════════════════════════════════════

class SpatialContinuityPrior:
    """
    基于地质连续性的伪标签阈值调整。

    已知盐体（label=1）周围 r 体素内，
    将正样本置信度阈值从 conf_high 降低至 conf_high * (1 - alpha)，
    使得这些区域更容易被判定为盐体。

    对应论文：
      "SEAI: Seismic Structure-Aware Iterative Pseudo-Label Refinement"
      思想借鉴：利用地震体连续性先验约束半监督学习
    """

    def __init__(self, salt_radius=12, background_radius=6, alpha=0.15):
        """
        salt_radius      : 盐体标注周围的影响半径（体素）
        background_radius: 背景标注周围的影响半径
        alpha            : 最大阈值放松比例（0.15 = 放松15%）
        """
        self.salt_radius = salt_radius
        self.background_radius = background_radius
        self.alpha = alpha

    def compute_prior_maps(self, label_3d):
        """
        label_3d: (D, H, W)，值 {0, 1, -1}

        返回：
          pos_relax: (D, H, W) ∈ [0, alpha]  —  正样本阈值放松量
          neg_relax: (D, H, W) ∈ [0, alpha]  —  负样本阈值放松量
          （放松量越大 = 越容易填入伪标签）
        """
        salt_mask = (label_3d == 1).astype(np.float32)
        bg_mask = (label_3d == 0).astype(np.float32)
        unlabeled = (label_3d == -1)

        # 到已知盐体的距离
        if salt_mask.sum() > 0:
            dist_salt = distance_transform_edt(1 - salt_mask)
            pos_weight = np.clip(
                1.0 - dist_salt / (self.salt_radius + 1e-6), 0.0, 1.0)
        else:
            pos_weight = np.zeros_like(label_3d, dtype=np.float32)

        # 到已知背景的距离
        if bg_mask.sum() > 0:
            dist_bg = distance_transform_edt(1 - bg_mask)
            neg_weight = np.clip(
                1.0 - dist_bg / (self.background_radius + 1e-6), 0.0, 1.0)
        else:
            neg_weight = np.zeros_like(label_3d, dtype=np.float32)

        pos_relax = (pos_weight * self.alpha) * unlabeled.astype(np.float32)
        neg_relax = (neg_weight * self.alpha) * unlabeled.astype(np.float32)

        return pos_relax, neg_relax

    def adjusted_thresholds(self, label_3d, conf_high, conf_low):
        """
        返回与 label_3d 同 shape 的逐体素阈值图。
          adj_high: 正样本置信度阈值（盐体周围更低）
          adj_low : 负样本置信度阈值（背景周围更高）
        """
        pos_relax, neg_relax = self.compute_prior_maps(label_3d)
        adj_high = conf_high - pos_relax  # 盐体周围：降低阈值，更容易填
        adj_low = conf_low + neg_relax  # 背景周围：提高阈值，更容易填
        return adj_high.astype(np.float32), adj_low.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  置信度 EMA（防止多轮错误累积）
# ═══════════════════════════════════════════════════════════════════════════

class ConfidenceEMA:
    """
    体素级指数滑动平均（EMA），用于稳定多轮伪标签生成。

    原理：
      第 t 轮预测 prob_t，与历史 EMA prob_ema 融合：
        prob_ema ← momentum * prob_ema + (1 - momentum) * prob_t

      多轮后 prob_ema 比单次推理更稳定，减少噪声体素被错误标注的概率。

    存储格式：每个 npz 文件对应一个 .npy 文件存储其 EMA 概率图。
    """

    def __init__(self, ema_dir, momentum=0.7):
        """
        ema_dir : EMA 概率图的存储目录
        momentum: EMA 动量（高动量 = 历史权重大 = 更稳定但更新慢）
        """
        self.ema_dir = ema_dir
        self.momentum = momentum
        os.makedirs(ema_dir, exist_ok=True)

    def _ema_path(self, fname):
        return os.path.join(self.ema_dir, fname.replace('.npz', '_ema.npy'))

    def update(self, fname, new_prob):
        """
        new_prob: (D, H, W) numpy array ∈ [0, 1]
        返回融合后的 EMA 概率图。
        """
        ema_path = self._ema_path(fname)
        if os.path.exists(ema_path):
            old_ema = np.load(ema_path)
            new_ema = self.momentum * old_ema + (1 - self.momentum) * new_prob
        else:
            new_ema = new_prob  # 第一轮直接使用
        np.save(ema_path, new_ema.astype(np.float32))
        return new_ema


# ═══════════════════════════════════════════════════════════════════════════
# 3.  综合伪标签精化器
# ═══════════════════════════════════════════════════════════════════════════

class PseudoLabelRefiner:
    """
    综合以下三个机制的伪标签精化器：
      A. TTA 空间软票（主脚本调用，输入已是均值概率）
      B. 置信度 EMA（稳定多轮预测）
      C. 空间连续性先验（盐体地质假设）

    使用示例（在 pseudo_self_train.py 中替换 generate_pseudo_labels）：
      refiner = PseudoLabelRefiner(ema_dir='.../ema', ...)
      fill_rate = refiner.refine_dataset(model, src_dir, dst_dir, tta_fn)
    """

    def __init__(
            self,
            ema_dir,
            conf_high=0.85,
            conf_low=0.15,
            ema_momentum=0.7,
            salt_radius=12,
            bg_radius=6,
            alpha=0.15,
    ):
        self.conf_high = conf_high
        self.conf_low = conf_low
        self.ema = ConfidenceEMA(ema_dir, momentum=ema_momentum)
        self.prior = SpatialContinuityPrior(
            salt_radius=salt_radius,
            background_radius=bg_radius,
            alpha=alpha
        )

    def fill_one(self, fname, label_3d, prob_raw):
        """
        fname    : npz 文件名（用于 EMA 索引）
        label_3d : (D, H, W) 原始标注（含 -1）
        prob_raw : (D, H, W) 当前轮 TTA 概率图

        返回：filled_label (D, H, W)，-1 区域部分被填为 0/1
        """
        # B. EMA 融合
        prob_ema = self.ema.update(fname, prob_raw)

        # C. 空间连续性先验 → 逐体素阈值
        adj_high, adj_low = self.prior.adjusted_thresholds(
            label_3d, self.conf_high, self.conf_low)

        # 伪标签填充（仅作用于 -1 区域）
        pseudo = label_3d.copy()
        unlabeled = (label_3d == -1)

        pos_fill = unlabeled & (prob_ema > adj_high)
        neg_fill = unlabeled & (prob_ema < adj_low)

        pseudo[pos_fill] = 1.0
        pseudo[neg_fill] = 0.0

        n_filled = int(pos_fill.sum() + neg_fill.sum())
        n_total = int(unlabeled.sum())
        return pseudo, n_filled, n_total

    def refine_dataset(self, model, src_dir, dst_dir, tta_fn, device):
        """
        tta_fn: 函数签名 tta_fn(model, data_3d, device) -> prob_3d (D,H,W)
        """
        import torch
        from tqdm import tqdm

        os.makedirs(dst_dir, exist_ok=True)
        files = sorted([f for f in os.listdir(src_dir) if f.endswith('.npz')])
        total_filled = total_unlabeled = 0

        model.eval()
        for fname in tqdm(files, desc="  [Refiner] 精化伪标签"):
            npz = np.load(os.path.join(src_dir, fname))
            data_raw = npz['data'].astype(np.float32)
            label_raw = npz['label'].astype(np.float32)
            has_ch = (data_raw.ndim == 4)
            data_3d = data_raw[0] if has_ch else data_raw
            label_3d = label_raw[0] if has_ch else label_raw

            # TTA 推理（传入的 tta_fn 已处理 TTA）
            prob = tta_fn(model, data_3d, device)

            pseudo_3d, n_fill, n_total = self.fill_one(fname, label_3d, prob)
            total_filled += n_fill
            total_unlabeled += n_total

            out_label = pseudo_3d[None] if has_ch else pseudo_3d
            np.savez_compressed(
                os.path.join(dst_dir, fname),
                data=data_raw, label=out_label
            )

        fill_rate = total_filled / max(total_unlabeled, 1) * 100
        print(f"  Refiner 填充率: {total_filled:,}/{total_unlabeled:,} ({fill_rate:.1f}%)")
        return fill_rate


# ═══════════════════════════════════════════════════════════════════════════
# 4.  伪标签质量统计（可选，有真实标注时调用）
# ═══════════════════════════════════════════════════════════════════════════

def pseudo_label_stats(pseudo_dir, gt_dir):
    """
    对比伪标签目录与真实标注目录，统计填充体素的精度/召回率。
    仅用于有全标注数据时的实验分析。

    pseudo_dir: 含伪标签的 npz 目录
    gt_dir    : 真实全标注 npz 目录（label 只有 0/1）
    """
    files = sorted([f for f in os.listdir(pseudo_dir) if f.endswith('.npz')])
    gt_files = set(os.listdir(gt_dir))

    tp = fp = fn = pseudo_pos = pseudo_neg = 0

    for fname in files:
        if fname not in gt_files:
            continue
        p_npz = np.load(os.path.join(pseudo_dir, fname))
        g_npz = np.load(os.path.join(gt_dir, fname))

        p_label = p_npz['label'].astype(np.float32)
        g_label = g_npz['label'].astype(np.float32)
        if p_label.ndim == 4: p_label = p_label[0]
        if g_label.ndim == 4: g_label = g_label[0]

        # 只统计伪标签填入的体素（原始为 -1，现在为 0/1）
        # 注意：这里假设原始稀疏标注存在 gt_dir，无法直接识别哪些是伪标签
        # 用 g_label != -1 且 p_label != -1 来近似
        filled = (p_label != -1) & (g_label != -1)

        p_f = p_label[filled]
        g_f = g_label[filled]

        tp += int(((p_f == 1) & (g_f == 1)).sum())
        fp += int(((p_f == 1) & (g_f == 0)).sum())
        fn += int(((p_f == 0) & (g_f == 1)).sum())
        pseudo_pos += int((p_f == 1).sum())
        pseudo_neg += int((p_f == 0).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    print(f"\n  伪标签质量统计 (对比 {len(files)} 块)")
    print(f"  Pseudo Pos: {pseudo_pos:,}  Pseudo Neg: {pseudo_neg:,}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")

    return dict(precision=precision, recall=recall, f1=f1)


