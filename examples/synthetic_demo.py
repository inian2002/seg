import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from AFVM import AdaptiveFourierVarianceMatching


def make_sample(size, rng, shift, field=False):
    d, h, w = np.meshgrid(*[np.linspace(-1, 1, size)] * 3, indexing='ij')
    salt = ((d - 0.2) / 0.8) ** 2 + ((h - shift) / 0.45) ** 2 + (w / 0.5) ** 2 < 1
    signal = np.sin(18 * d + 1.5 * h + w) * (1 - salt) + 0.15 * rng.normal(size=d.shape)
    if field:
        signal = 1.6 * signal + 0.3 * np.cos(3 * h - w)
    signal = (signal - signal.mean()) / max(signal.std(), 1e-8)
    return signal.astype(np.float32), salt.astype(np.float32)


def scores(prob, target):
    pred = prob > 0.5
    truth = target > 0.5
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'IoU': tp / max(tp + fp + fn, 1),
            'Dice': 2 * tp / max(2 * tp + fp + fn, 1),
            'PA': (tp + tn) / target.size}


def main():
    parser = argparse.ArgumentParser(description='Run a deterministic synthetic FAMT-Salt3D example.')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'synthetic_demo')
    parser.add_argument('--mode', choices=['afvm', 'full'], default='full')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.size < 32 or args.size % 16 or args.steps < 1:
        parser.error('--size must be a multiple of 16 and at least 32; --steps must be positive.')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('--output must be empty or new to protect previous results.')
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ['source', 'target', 'adapted', 'sparse', 'validation']:
        (args.output / name).mkdir()
    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    for i, shift in enumerate([-0.15, 0.15]):
        data, label = make_sample(args.size, rng, shift)
        np.savez_compressed(args.output / 'source' / f'{i:02}.npz', data=data, label=label)
        data, label = make_sample(args.size, rng, shift, field=True)
        np.savez_compressed(args.output / 'target' / f'{i:02}.npz', data=data)
        sparse = np.full_like(label, -1)
        keep = sorted(set(range(0, args.size, 8)) | {args.size - 1})
        sparse[:, keep, :] = label[:, keep, :]
        np.savez_compressed(args.output / 'sparse' / f'{i:02}.npz', data=data, label=sparse)
    val_data, val_label = make_sample(args.size, rng, 0.05, field=True)
    np.savez_compressed(args.output / 'validation' / '00.npz', data=val_data, label=val_label)
    afvm = AdaptiveFourierVarianceMatching(beta_min=0.10, beta_max=0.20)
    afvm.process_dataset(args.output / 'source', args.output / 'target', args.output / 'adapted')
    changes = []
    for path in (args.output / 'source').glob('*.npz'):
        with np.load(path) as before, np.load(args.output / 'adapted' / path.name) as after:
            if not np.array_equal(before['label'], after['label']) or not np.isfinite(after['data']).all():
                raise RuntimeError('AFVM did not preserve labels or produced non-finite data.')
            changes.append(float(np.mean(np.abs(before['data'] - after['data']))))
    if not all(change > 0 for change in changes):
        raise RuntimeError('Synthetic adaptation did not change source amplitudes.')
    result = {'purpose': 'Synthetic workflow check, not reproduction of the reported survey metrics.',
              'seed': args.seed, 'size': args.size, 'mode': args.mode,
              'afvm_labels_preserved': True, 'numpy_version': np.__version__,
              'demo_beta_range': [0.10, 0.20], 'adaptation_mean_absolute_changes': changes}
    if args.mode == 'full':
        import torch
        from models.SaltHRNet3D import SaltHRNet3D
        from utils.loss import Weight_BCE_loss, dice_loss
        from utils.pasa import Weak_DA
        from utils.reconstruction import build_anisotropic_mask, reconstruction_loss_dict, sigmoid_rampup

        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.use_deterministic_algorithms(True)
        if args.device == 'cuda' and not torch.cuda.is_available():
            parser.error('CUDA requested but unavailable.')
        device = torch.device(args.device)
        model = SaltHRNet3D(base=2, enable_reconstruction=True).to(device)
        history = []
        for stage, folder, lr in [('pretrain', 'adapted', 5e-4), ('finetune', 'sparse', 1e-4)]:
            arrays = [np.load(p) for p in sorted((args.output / folder).glob('*.npz'))]
            x = torch.from_numpy(np.stack([v['data'] for v in arrays])[:, None]).to(device)
            y = torch.from_numpy(np.stack([v['label'] for v in arrays])[:, None]).to(device)
            for v in arrays:
                v.close()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            model.train()
            for step in range(1, args.steps + 1):
                optimizer.zero_grad()
                augmented, labels = Weak_DA(x, y)
                prob = model(augmented)
                seg = 0.5 * Weight_BCE_loss(prob, labels) + 0.5 * dice_loss(prob, labels)
                seg.backward()
                masked, mask = build_anisotropic_mask(augmented)
                aux = model(masked, return_aux=True)
                terms = reconstruction_loss_dict(aux['recon'], augmented, mask)
                rec = 0.08 * sigmoid_rampup(step, 20) * terms['total']
                rec.backward()
                optimizer.step()
                if not torch.isfinite(seg + rec):
                    raise RuntimeError('Non-finite training loss.')
                history.append({'stage': stage, 'step': step, 'seg_loss': float(seg.detach()),
                                'weighted_recon_loss': float(rec.detach())})
            torch.save({'net': model.state_dict(), 'base_width': 2}, args.output / f'{stage}.pth')
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(val_data[None, None]).to(device))[0, 0].cpu().numpy()
        if pred.shape != val_label.shape or not np.isfinite(pred).all():
            raise RuntimeError('Invalid inference output.')
        np.save(args.output / 'probabilities.npy', pred)
        np.save(args.output / 'prediction.npy', (pred > 0.5).astype(np.uint8))
        result.update(torch_version=torch.__version__, device=str(device), history=history,
                      evaluation=scores(pred, val_label), base_width=2, steps_per_stage=args.steps)
    result['elapsed_seconds'] = round(time.perf_counter() - start, 3)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
