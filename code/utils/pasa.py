import torch
import random


def Weak_DA(*args):

    assert len(args) > 0
    out = list(args)


    if random.random() < 0.5:
        out = [torch.flip(x, dims=[-1]) for x in out]
    if random.random() < 0.5:
        out = [torch.flip(x, dims=[-2]) for x in out]


    if random.random() < 0.7:
        scale = random.uniform(0.8, 1.2)
        out[0] = out[0] * scale


    if random.random() < 0.7:
        sigma = random.uniform(0.0, 0.05)
        noise = torch.randn_like(out[0]) * sigma
        out[0] = out[0] + noise

    return out


def adjust_learning_rate(optimizer, base_lr, max_iters,
                         cur_iters, power=0.9, nbb_mult=10):
    if optimizer.param_groups[0]['lr'] <= 1e-5:
        return
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult


class LossAverage(object):


    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = round(self.sum / self.count, 4)


if __name__ == '__main__':
    import torch


    data  = torch.randn(2, 1, 64, 64, 64)
    label = torch.randint(0, 2, (2, 1, 64, 64, 64)).float()

    label_orig = label.clone()
    data2, label2 = Weak_DA(data, label)


    unique_vals = label2.unique()
    assert set(unique_vals.tolist()).issubset({0.0, 1.0}), \
        f"Labels were corrupted by augmentation; unique values: {unique_vals}"

    print("Physics-Aware DA check passed:")
    print(f"   data  shape  : {data2.shape}")
    print(f"   Unique label values: {unique_vals.tolist()} (0/1 only)")
    print("   No vertical rotation; amplitude scaling and noise affect data only.")
