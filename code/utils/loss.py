import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class lambda_BCE3d(nn.Module):

    def __init__(self, w=0.5):
        super(lambda_BCE3d, self).__init__()
        self.w = w

    def forward(self, y, target):
        w1 = torch.sum((target == 1).float())
        w2 = torch.sum((target == 0).float())
        w  = (w2 / (w1 + 1e-6)) * self.w
        w  = (target == 1).float() * (w - 1) + 1
        mask  = torch.zeros_like(target).to(device)
        mask2 = torch.zeros_like(target).float().to(device) + 0.5
        w      = torch.where(target == -1, mask,  w)
        target = torch.where(target == -1, mask2, target)
        loss   = F.binary_cross_entropy(
            input=y, target=target, weight=w, reduction='sum'
        ) / torch.count_nonzero(w)
        return loss


def BCE3D_loss(output, target):

    neg_count = torch.sum(target == 0)
    pos_count = torch.sum(target == 1)
    beta = neg_count / (pos_count + neg_count)
    w = torch.zeros_like(target, dtype=torch.float32)
    w[torch.where(target == -1)] = 0
    w[torch.where(target == 1)]  = beta
    w[torch.where(target == 0)]  = 1 - beta
    target = target.float()
    w.requires_grad = False
    return F.binary_cross_entropy(
        input=output, target=target, weight=w, reduction='mean')


class Tversky(nn.Module):

    def __init__(self, smooth=1, alpha=0.8, reduction='mean'):
        super(Tversky, self).__init__()
        self.smooth    = smooth
        self.reduction = reduction
        self.alpha     = alpha

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0]
        predict = predict.contiguous().view(predict.shape[0], -1)
        target  = target.contiguous().view(target.shape[0],  -1)
        zeros   = torch.zeros_like(predict).float()
        tp  = torch.sum(torch.where(target == -1., zeros, predict * target),   dim=1)
        fn  = torch.sum(torch.where(target == -1., zeros, target * (1-predict)), dim=1)
        fp  = torch.sum(torch.where(target == -1., zeros, (1-target) * predict), dim=1)
        tv  = (tp + self.smooth) / (
            tp + self.alpha * fn + (1 - self.alpha) * fp + self.smooth)
        return (1 - tv).mean()


def Weight_BCE_loss(output, target):
    neg_count   = torch.sum((target == 0).float())
    pos_count   = torch.sum((target == 1).float())
    valid_count = (pos_count + neg_count).clamp(min=1.0)
    beta        = neg_count / valid_count

    w = torch.zeros_like(target, dtype=torch.float32)
    w[target == 1] = beta
    w[target == 0] = 1.0 - beta


    target_clean = target.clone().float()
    target_clean[target == -1] = 0.0

    w.requires_grad = False


    return F.binary_cross_entropy(
        input=output, target=target_clean, weight=w, reduction='sum'
    ) / valid_count


def dice_loss(preds, targets, weight=None, smooth=1., reduction='mean'):
    assert reduction in [None, 'mean', 'sum']

    if weight is None:


        weight = (targets != -1).float().to(targets.device)
    else:
        weight = weight.to(targets.device)

    b       = preds.shape[0]
    preds   = preds.reshape(b, -1)
    targets = targets.reshape(b, -1)
    weight  = weight.reshape(b, -1)


    targets_clean = targets.clone().float()
    targets_clean[targets == -1] = 0.0

    inter    = (preds * targets_clean * weight).sum(1)
    dice_eff = (2 * inter + smooth) / (
        ((preds + targets_clean) * weight).sum(1) + smooth)
    loss = 1 - dice_eff

    if reduction == 'sum':   return loss.sum()
    elif reduction == 'mean': return loss.mean()
    else:                     return loss


if __name__ == '__main__':
    targets = torch.tensor(np.ones((3, 3, 3), dtype=np.float32))
    print(targets.shape, targets)
