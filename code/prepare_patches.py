import argparse
import json
from pathlib import Path
import numpy as np


def starts(length, patch, stride):
    points = list(range(0, length - patch + 1, stride))
    if not points or points[-1] != length - patch:
        points.append(length - patch)
    return points


def main():
    parser = argparse.ArgumentParser(description='Extract NPZ patches from preprocessed NPY volumes.')
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--axes', type=int, nargs=3, required=True, metavar=('D', 'H', 'W'))
    parser.add_argument('--region', type=int, nargs=6, metavar=('D0', 'D1', 'H0', 'H1', 'W0', 'W1'))
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--inline-spacing', type=int, default=0)
    args = parser.parse_args()
    if sorted(args.axes) != [0, 1, 2] or args.patch_size < 16 or args.patch_size % 16:
        parser.error('--axes must permute 0,1,2; --patch-size must be a positive multiple of 16.')
    if not 1 <= args.stride <= args.patch_size or args.inline_spacing < 0:
        parser.error('Require 1 <= stride <= patch-size and inline-spacing >= 0.')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be new or empty.')
    data = np.load(args.data, mmap_mode='r', allow_pickle=False)
    if data.ndim != 3:
        parser.error('Data must be a 3D NPY array.')
    label = np.load(args.labels, mmap_mode='r', allow_pickle=False) if args.labels else None
    if label is not None and label.shape != data.shape:
        parser.error('Labels and data must have the same original shape and axis order.')
    if args.inline_spacing and label is None:
        parser.error('--inline-spacing requires --labels.')
    data = np.transpose(data, args.axes)
    if label is not None:
        label = np.transpose(label, args.axes)
    bounds = args.region or [0, data.shape[0], 0, data.shape[1], 0, data.shape[2]]
    if any(not (0 <= bounds[2*i] < bounds[2*i+1] <= data.shape[i]) for i in range(3)):
        parser.error('Region bounds must lie inside the transposed volume.')
    region = tuple(slice(bounds[2*i], bounds[2*i+1]) for i in range(3))
    data = data[region]
    if label is not None:
        label = label[region]
    if min(data.shape) < args.patch_size:
        parser.error('Region is smaller than a patch; do not pad labels implicitly.')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in starts(data.shape[0], args.patch_size, args.stride):
        for h in starts(data.shape[1], args.patch_size, args.stride):
            for w in starts(data.shape[2], args.patch_size, args.stride):
                patch = tuple(slice(i, i + args.patch_size) for i in (d, h, w))
                values = {'data': np.asarray(data[patch], dtype=np.float32)}
                if not np.isfinite(values['data']).all():
                    raise ValueError('Input patch contains NaN or infinity.')
                if label is not None:
                    y = np.asarray(label[patch], dtype=np.float32)
                    if not np.isin(y, [0, 1]).all():
                        raise ValueError('Input labels must be dense 0/1 labels.')
                    if args.inline_spacing:
                        sparse = np.full_like(y, -1)
                        keep = sorted(set(range(0, y.shape[1], args.inline_spacing)) | {y.shape[1]-1})
                        sparse[:, keep, :] = y[:, keep, :]
                        y = sparse
                    values['label'] = y
                name = f'd{d:04}_h{h:04}_w{w:04}.npz'
                np.savez_compressed(args.output / name, **values)
                rows.append({'file': name, 'start_in_transposed_volume': [d+bounds[0], h+bounds[2], w+bounds[4]]})
    manifest = {'data': args.data.name, 'labels': args.labels.name if args.labels else None,
                'axes': args.axes, 'region': bounds, 'patch_size': args.patch_size,
                'stride': args.stride, 'inline_spacing': args.inline_spacing, 'patches': rows}
    args.output.with_name(args.output.name + '_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Saved {len(rows)} patches to {args.output}')


if __name__ == '__main__':
    main()
