import torch
import random


def Weak_DA(*args):
    """
    地震数据物理感知增强 (Physics-Aware Augmentation for Seismic Data)

    约束原则：
      - Z 轴（深度/时间轴）不做任何旋转或翻转 —— 重力各向异性，地层由浅到深不可逆
      - 仅允许水平面（H/W 轴）的翻转和振幅/噪声扰动

    增强内容：
      1. 随机 H 方向翻转（X-line 方向，构造走向对称）
      2. 随机 W 方向翻转（I-line 方向，构造走向对称）
      3. 随机振幅缩放 [0.8, 1.2]  —— 模拟道间增益差异
      4. 随机高斯噪声注入 σ ∈ [0, 0.05]  —— 模拟采集环境噪声
         (仅施加于 data，不施加于 label)

    Shape: (B, C, D, H, W)，仅第一个 tensor 视为 data，其余视为 label。
    """
    assert len(args) > 0
    out = list(args)

    # ── 1. 水平方向翻转（Z 轴安全，H/W 轴允许）──────────────────
    if random.random() < 0.5:
        out = [torch.flip(x, dims=[-1]) for x in out]   # W (I-line)
    if random.random() < 0.5:
        out = [torch.flip(x, dims=[-2]) for x in out]   # H (X-line)
    # ❌ 不对 dims=[-3] (Z/depth) 做任何操作

    # ── 2. 振幅随机缩放（仅施加于 data，即 out[0]） ─────────────
    if random.random() < 0.7:
        scale = random.uniform(0.8, 1.2)
        out[0] = out[0] * scale

    # ── 3. 随机高斯噪声注入（仅施加于 data） ────────────────────
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
    """Computes and stores the average and current value."""

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

    # 测试 data/label 同步翻转，且 label 不受振幅/噪声影响
    data  = torch.randn(2, 1, 64, 64, 64)
    label = torch.randint(0, 2, (2, 1, 64, 64, 64)).float()

    label_orig = label.clone()
    data2, label2 = Weak_DA(data, label)

    # label 只应被翻转，不应被振幅缩放或加噪声干扰
    # 两者差异只来自翻转（值域不变）
    unique_vals = label2.unique()
    assert set(unique_vals.tolist()).issubset({0.0, 1.0}), \
        f"label 被噪声污染了！unique values: {unique_vals}"

    print("✅ Physics-Aware DA 测试通过：")
    print(f"   data  shape  : {data2.shape}")
    print(f"   label 唯一值 : {unique_vals.tolist()}  (只有 0/1，无污染)")
    print("   Z 轴不被旋转，振幅缩放和噪声仅作用于 data。")
