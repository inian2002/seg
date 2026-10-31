import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class lambda_BCE3d(nn.Module):
    """Original lambda_BCE3d — unchanged."""
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
    """Original BCE3D_loss — unchanged."""
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
    """Original Tversky — unchanged."""
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


# ══════════════════════════════════════════════════════════════════
# Weight_BCE_loss  ─ 修复 -1 标签支持，向后兼容
#
# 原版 BUG（含 -1 时）：
#   neg_count = sum(1 - target)  → -1 位置贡献 2，严重高估负样本数
#   pos_count = sum(target)      → -1 位置贡献 -1，可能使 pos_count 为负
#
# label 只有 {0,1} 时（原版场景）：
#   sum(target==0) ≡ sum(1-target)，sum(target==1) ≡ sum(target)
#   valid_count = N_total → reduction='sum'/N_total ≡ 原版 'mean'
#   ✅ 行为与原版完全一致
#
# label 含 -1 时：
#   仅统计 0/1 位置计算 beta，-1 位置 weight=0，不贡献任何损失
# ══════════════════════════════════════════════════════════════════
def Weight_BCE_loss(output, target):
    neg_count   = torch.sum((target == 0).float())
    pos_count   = torch.sum((target == 1).float())
    valid_count = (pos_count + neg_count).clamp(min=1.0)
    beta        = neg_count / valid_count        # 正样本权重

    w = torch.zeros_like(target, dtype=torch.float32)
    w[target == 1] = beta
    w[target == 0] = 1.0 - beta
    # target==-1 处 w 保持 0，不计损失

    # 将 -1 替换为占位值 0（weight=0 不影响梯度）
    target_clean = target.clone().float()
    target_clean[target == -1] = 0.0

    w.requires_grad = False
    # reduction='sum'/valid_count ≡ 有效像素上的均值
    # 无 -1 时 valid_count==N_total，与原版 reduction='mean' 完全一致
    return F.binary_cross_entropy(
        input=output, target=target_clean, weight=w, reduction='sum'
    ) / valid_count


# ══════════════════════════════════════════════════════════════════
# dice_loss  ─ 修复 -1 标签支持，向后兼容
#
# 原版 BUG（含 -1 时，weight=None）：
#   weight = ones → -1 位置 weight=1
#   intersection = preds * targets * 1 → -1 位置 = preds*(-1) = 负值
#   分子/分母均被负值污染，结果完全错误
#
# label 只有 {0,1} 时（原版场景）：
#   (targets != -1) 全为 True → weight 全为 1 → 与原版完全一致
#
# label 含 -1 时：
#   -1 位置 weight=0，自动屏蔽
# ══════════════════════════════════════════════════════════════════
def dice_loss(preds, targets, weight=None, smooth=1., reduction='mean'):
    assert reduction in [None, 'mean', 'sum']

    if weight is None:
        # 无 -1：全为 1，与原版完全一致
        # 有 -1：-1 位置 weight=0，自动屏蔽
        weight = (targets != -1).float().to(targets.device)
    else:
        weight = weight.to(targets.device)

    b       = preds.shape[0]
    preds   = preds.reshape(b, -1)
    targets = targets.reshape(b, -1)
    weight  = weight.reshape(b, -1)

    # -1 位置替换为 0（weight=0，不影响结果）
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
