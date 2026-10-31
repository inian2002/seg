import os
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation


class SpatialContinuityPrior:


    def __init__(self, salt_radius=12, background_radius=6, alpha=0.15):

        self.salt_radius = salt_radius
        self.background_radius = background_radius
        self.alpha = alpha

    def compute_prior_maps(self, label_3d):

        salt_mask = (label_3d == 1).astype(np.float32)
        bg_mask = (label_3d == 0).astype(np.float32)
        unlabeled = (label_3d == -1)


        if salt_mask.sum() > 0:
            dist_salt = distance_transform_edt(1 - salt_mask)
            pos_weight = np.clip(
                1.0 - dist_salt / (self.salt_radius + 1e-6), 0.0, 1.0)
        else:
            pos_weight = np.zeros_like(label_3d, dtype=np.float32)


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

        pos_relax, neg_relax = self.compute_prior_maps(label_3d)
        adj_high = conf_high - pos_relax
        adj_low = conf_low + neg_relax
        return adj_high.astype(np.float32), adj_low.astype(np.float32)


class ConfidenceEMA:


    def __init__(self, ema_dir, momentum=0.7):

        self.ema_dir = ema_dir
        self.momentum = momentum
        os.makedirs(ema_dir, exist_ok=True)

    def _ema_path(self, fname):
        return os.path.join(self.ema_dir, fname.replace('.npz', '_ema.npy'))

    def update(self, fname, new_prob):

        ema_path = self._ema_path(fname)
        if os.path.exists(ema_path):
            old_ema = np.load(ema_path)
            new_ema = self.momentum * old_ema + (1 - self.momentum) * new_prob
        else:
            new_ema = new_prob
        np.save(ema_path, new_ema.astype(np.float32))
        return new_ema


class PseudoLabelRefiner:


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


        prob_ema = self.ema.update(fname, prob_raw)


        adj_high, adj_low = self.prior.adjusted_thresholds(
            label_3d, self.conf_high, self.conf_low)


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

        import torch
        from tqdm import tqdm

        os.makedirs(dst_dir, exist_ok=True)
        files = sorted([f for f in os.listdir(src_dir) if f.endswith('.npz')])
        total_filled = total_unlabeled = 0

        model.eval()
        for fname in tqdm(files, desc="  [Refiner] Refining pseudo-labels"):
            npz = np.load(os.path.join(src_dir, fname))
            data_raw = npz['data'].astype(np.float32)
            label_raw = npz['label'].astype(np.float32)
            has_ch = (data_raw.ndim == 4)
            data_3d = data_raw[0] if has_ch else data_raw
            label_3d = label_raw[0] if has_ch else label_raw


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
        print(f"  Refiner fill rate: {total_filled:,}/{total_unlabeled:,} ({fill_rate:.1f}%)")
        return fill_rate


def pseudo_label_stats(pseudo_dir, gt_dir):

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

    print(f"\n  Pseudo-label quality statistics ({len(files)} patches)")
    print(f"  Pseudo Pos: {pseudo_pos:,}  Pseudo Neg: {pseudo_neg:,}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")

    return dict(precision=precision, recall=recall, f1=f1)
