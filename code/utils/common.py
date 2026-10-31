import torch
import random
import numpy as np


def Weak_DA(*args):

    assert len(args) > 0
    out = list(args)

    if random.random() < 0.5:
        out = [torch.flip(x, dims=[-1]) for x in out]
    if random.random() < 0.5:
        out = [torch.flip(x, dims=[-2]) for x in out]
    if random.random() < 0.5:
        out = [torch.flip(x, dims=[-3]) for x in out]

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
    x = torch.arange(2 * 1 * 4 * 3 * 5).view(2, 1, 4, 3, 5)
    y = x.clone()
    x2, y2 = Weak_DA(x, y)
    assert torch.all(x2 == y2), "Weak_DA sync failed!"
    print("Weak_DA sync OK")
