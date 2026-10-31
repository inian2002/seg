import math
import random

import torch
import torch.nn.functional as F


def sigmoid_rampup(epoch: int, rampup_epochs: int) -> float:
    if rampup_epochs <= 0:
        return 1.0
    current = max(0.0, min(float(epoch - 1), float(rampup_epochs)))
    phase = 1.0 - current / float(rampup_epochs)
    return float(math.exp(-5.0 * phase * phase))


def _num_from_ratio(length: int, ratio: float) -> int:
    if ratio <= 0:
        return 0
    return max(1, min(length, int(round(length * ratio))))


def build_anisotropic_mask(data: torch.Tensor,
                           horizontal_ratio: float = 0.18,
                           vertical_ratio: float = 0.10,
                           block_ratio: float = 0.12):
    """
    Build a seismic-aware mask for 3D cubes.

    horizontal_ratio: mask H/W slices to emphasize crossline/inline continuity.
    vertical_ratio  : mask D slices more conservatively to preserve depth cues.
    block_ratio     : mask one random 3D block for local structure completion.
    """
    if data.dim() != 5:
        raise ValueError(f"Expected 5D input (B,C,D,H,W), got {tuple(data.shape)}")

    B, C, D, H, W = data.shape
    masked = data.clone()
    loss_mask = torch.zeros_like(data, dtype=torch.bool)
    fill_value = data.mean(dim=(-3, -2, -1), keepdim=True)

    for b in range(B):
        n_h = _num_from_ratio(H, horizontal_ratio * 0.5)
        n_w = _num_from_ratio(W, horizontal_ratio * 0.5)
        n_d = _num_from_ratio(D, vertical_ratio)

        if n_h > 0:
            idx_h = torch.randperm(H, device=data.device)[:n_h]
            masked[b, :, :, idx_h, :] = fill_value[b]
            loss_mask[b, :, :, idx_h, :] = True

        if n_w > 0:
            idx_w = torch.randperm(W, device=data.device)[:n_w]
            masked[b, :, :, :, idx_w] = fill_value[b]
            loss_mask[b, :, :, :, idx_w] = True

        if n_d > 0:
            idx_d = torch.randperm(D, device=data.device)[:n_d]
            masked[b, :, idx_d, :, :] = fill_value[b]
            loss_mask[b, :, idx_d, :, :] = True

        if block_ratio > 0:
            max_d = max(1, int(round(D * block_ratio)))
            max_h = max(1, int(round(H * block_ratio)))
            max_w = max(1, int(round(W * block_ratio)))
            bd = random.randint(1, max_d)
            bh = random.randint(1, max_h)
            bw = random.randint(1, max_w)
            sd = random.randint(0, max(0, D - bd))
            sh = random.randint(0, max(0, H - bh))
            sw = random.randint(0, max(0, W - bw))
            masked[b, :, sd:sd + bd, sh:sh + bh, sw:sw + bw] = fill_value[b]
            loss_mask[b, :, sd:sd + bd, sh:sh + bh, sw:sw + bw] = True

    return masked, loss_mask.float()


def masked_l1_loss(pred: torch.Tensor,
                   target: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    weight = mask.float()
    denom = weight.sum().clamp(min=1.0)
    return (torch.abs(pred - target) * weight).sum() / denom


def _diff_masks(mask: torch.Tensor):
    mask_d = torch.maximum(mask[:, :, 1:, :, :], mask[:, :, :-1, :, :])
    mask_h = torch.maximum(mask[:, :, :, 1:, :], mask[:, :, :, :-1, :])
    mask_w = torch.maximum(mask[:, :, :, :, 1:], mask[:, :, :, :, :-1])
    return mask_d, mask_h, mask_w


def edge_aware_gradient_loss(pred: torch.Tensor,
                             target: torch.Tensor,
                             mask: torch.Tensor,
                             edge_boost: float = 1.0) -> torch.Tensor:
    pd = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    ph = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    pw = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]

    td = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
    th = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
    tw = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]

    md, mh, mw = _diff_masks(mask.float())

    edge_d = torch.abs(td).detach()
    edge_h = torch.abs(th).detach()
    edge_w = torch.abs(tw).detach()

    wd = md * (1.0 + edge_boost * edge_d / (edge_d.mean() + 1e-6))
    wh = mh * (1.0 + edge_boost * edge_h / (edge_h.mean() + 1e-6))
    ww = mw * (1.0 + edge_boost * edge_w / (edge_w.mean() + 1e-6))

    loss_d = (torch.abs(pd - td) * wd).sum() / wd.sum().clamp(min=1.0)
    loss_h = (torch.abs(ph - th) * wh).sum() / wh.sum().clamp(min=1.0)
    loss_w = (torch.abs(pw - tw) * ww).sum() / ww.sum().clamp(min=1.0)
    return (loss_d + loss_h + loss_w) / 3.0


def _low_frequency_mask(shape, ratio: float, device):
    d, h, w = shape
    ratio = max(1e-3, min(float(ratio), 0.5))
    zz = torch.linspace(-1.0, 1.0, d, device=device)
    yy = torch.linspace(-1.0, 1.0, h, device=device)
    xx = torch.linspace(-1.0, 1.0, w, device=device)
    zz, yy, xx = torch.meshgrid(zz, yy, xx, indexing='ij')
    radius = torch.sqrt(zz * zz + yy * yy + xx * xx)
    return (radius <= ratio).float()


def frequency_amplitude_loss(pred: torch.Tensor,
                             target: torch.Tensor,
                             low_freq_ratio: float = 0.08) -> torch.Tensor:
    pred_fft = torch.fft.fftn(pred, dim=(-3, -2, -1), norm='ortho')
    tgt_fft = torch.fft.fftn(target, dim=(-3, -2, -1), norm='ortho')

    pred_amp = torch.abs(torch.fft.fftshift(pred_fft, dim=(-3, -2, -1)))
    tgt_amp = torch.abs(torch.fft.fftshift(tgt_fft, dim=(-3, -2, -1)))

    low = _low_frequency_mask(pred_amp.shape[-3:], low_freq_ratio, pred.device)
    low = low.view(1, 1, *low.shape).expand_as(pred_amp)
    high = 1.0 - low

    low_loss = (torch.abs(pred_amp - tgt_amp) * low).sum() / low.sum().clamp(min=1.0)
    high_loss = (torch.abs(pred_amp - tgt_amp) * high).sum() / high.sum().clamp(min=1.0)
    return 0.7 * low_loss + 0.3 * high_loss


def reconstruction_loss_dict(recon_pred: torch.Tensor,
                             target: torch.Tensor,
                             mask: torch.Tensor,
                             edge_weight: float = 0.25,
                             freq_weight: float = 0.10,
                             low_freq_ratio: float = 0.08):
    l1 = masked_l1_loss(recon_pred, target, mask)
    grad = edge_aware_gradient_loss(recon_pred, target, mask, edge_boost=1.0)
    freq = frequency_amplitude_loss(recon_pred, target, low_freq_ratio=low_freq_ratio)
    total = l1 + edge_weight * grad + freq_weight * freq
    return {
        'l1': l1,
        'grad': grad,
        'freq': freq,
        'total': total,
    }
