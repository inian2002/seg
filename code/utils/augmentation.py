import random
import torch


def apply_spatial_aug(data: torch.Tensor,
                      label: torch.Tensor):

    if random.random() < 0.5:
        data  = torch.flip(data,  dims=[-3])
        label = torch.flip(label, dims=[-3])
    if random.random() < 0.5:
        data  = torch.flip(data,  dims=[-2])
        label = torch.flip(label, dims=[-2])
    if random.random() < 0.5:
        data  = torch.flip(data,  dims=[-1])
        label = torch.flip(label, dims=[-1])
    return data, label


def apply_intensity_aug(data: torch.Tensor, args) -> torch.Tensor:

    data = data.clone()


    if random.random() < args.noise_prob:
        std  = random.uniform(args.noise_std_min, args.noise_std_max)
        dstd = data.std().item()
        data = data + torch.randn_like(data) * std * max(dstd, 1e-6)


    if random.random() < args.contrast_prob:
        scale = random.uniform(args.gamma_min, args.gamma_max)
        for b in range(data.shape[0]):
            m       = data[b].mean()
            data[b] = m + (data[b] - m) * scale


    if random.random() < args.cutout_prob:
        data = _cutout_3d(data, max_ratio=args.cutout_ratio)

    return data


def _cutout_3d(data: torch.Tensor, max_ratio: float = 0.25) -> torch.Tensor:
    B, C, D, H, W = data.shape
    data = data.clone()
    for b in range(B):
        fill = data[b].mean().item()
        cd   = random.randint(1, max(1, int(D * max_ratio)))
        ch   = random.randint(1, max(1, int(H * max_ratio)))
        cw   = random.randint(1, max(1, int(W * max_ratio)))
        sd   = random.randint(0, D - cd)
        sh   = random.randint(0, H - ch)
        sw   = random.randint(0, W - cw)
        data[b, :, sd:sd+cd, sh:sh+ch, sw:sw+cw] = fill
    return data
