from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import label as connected_component_label


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.SaltHRNet3D import SaltHRNet3D
from utils.checkpoint import load_model_state
from utils.pasa import Weak_DA
from utils.reconstruction import build_anisotropic_mask


DEFAULT_MODEL_PATH = None


def seed_everything(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _as_4d(array: np.ndarray, name: str) -> Tuple[np.ndarray, bool]:

    array = np.asarray(array)
    if array.ndim == 3:
        return array[np.newaxis, ...], False
    if array.ndim == 4 and array.shape[0] == 1:
        return array, True
    raise ValueError(
        f"{name} must have shape (D,H,W) or (1,D,H,W), got {array.shape}"
    )


def _restore_shape(array_4d: np.ndarray, had_channel: bool) -> np.ndarray:
    return array_4d if had_channel else array_4d[0]


def _load_npz(input_path: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray,
                                             Optional[np.ndarray], bool]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input NPZ does not exist: {input_path}")

    with np.load(input_path, allow_pickle=False) as archive:
        if "data" not in archive.files:
            raise KeyError(f"{input_path} does not contain the required key 'data'")
        arrays = {key: archive[key] for key in archive.files}

    data_4d, had_channel = _as_4d(arrays["data"], "data")
    data_4d = data_4d.astype(np.float32, copy=False)

    label_4d: Optional[np.ndarray] = None
    if "label" in arrays:
        label_4d, label_had_channel = _as_4d(arrays["label"], "label")
        if label_4d.shape != data_4d.shape:
            raise ValueError(
                f"data shape {arrays['data'].shape} and label shape "
                f"{arrays['label'].shape} do not match"
            )
        if label_had_channel != had_channel:
            raise ValueError(
                "data and label must both be 3-D or both be 4-D with one channel"
            )
        label_4d = label_4d.astype(np.float32, copy=False)

    return arrays, data_4d, label_4d, had_channel


def apply_pasa(data_4d: np.ndarray,
               label_4d: Optional[np.ndarray]) -> Tuple[np.ndarray,
                                                         Optional[np.ndarray]]:


    data_tensor = torch.from_numpy(data_4d[np.newaxis, ...]).float()
    tensors: List[torch.Tensor] = [data_tensor]
    if label_4d is not None:
        tensors.append(torch.from_numpy(label_4d[np.newaxis, ...]).float())

    augmented = Weak_DA(*tensors)
    data_aug = augmented[0][0].cpu().numpy().astype(np.float32, copy=False)
    label_aug = None
    if label_4d is not None:
        label_aug = augmented[1][0].cpu().numpy().astype(np.float32, copy=False)
    return data_aug, label_aug


def make_anisotropic_mask(data_4d: np.ndarray,
                          horizontal_ratio: float,
                          vertical_ratio: float,
                          block_ratio: float) -> Tuple[np.ndarray, np.ndarray]:

    data_tensor = torch.from_numpy(data_4d[np.newaxis, ...]).float()
    masked, mask = build_anisotropic_mask(
        data_tensor,
        horizontal_ratio=horizontal_ratio,
        vertical_ratio=vertical_ratio,
        block_ratio=block_ratio,
    )
    return (
        masked[0].cpu().numpy().astype(np.float32, copy=False),
        mask[0].cpu().numpy().astype(np.float32, copy=False),
    )


def filter_small_components(binary_volume: np.ndarray,
                            min_size: int) -> Tuple[np.ndarray, int, int]:

    if min_size <= 0:
        return binary_volume.astype(np.float32, copy=True), 0, 0

    structure = np.ones((3, 3, 3), dtype=np.int8)
    components, component_count = connected_component_label(
        binary_volume.astype(np.uint8, copy=False), structure=structure
    )
    if component_count == 0:
        return binary_volume.astype(np.float32, copy=True), 0, 0

    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    keep = sizes >= int(min_size)
    kept_count = int(keep.sum())
    return keep[components].astype(np.float32), component_count, kept_count


def _extract_state_dict(checkpoint: object, use_student: bool) -> object:

    if not isinstance(checkpoint, dict):
        return checkpoint

    if use_student and "student_net" in checkpoint:
        state = checkpoint["student_net"]
    elif not use_student and "teacher_net" in checkpoint:
        state = checkpoint["teacher_net"]
    elif "net" in checkpoint:
        state = checkpoint["net"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state = checkpoint["model"]
    else:
        state = checkpoint


    if isinstance(state, dict) and state:
        keys = list(state.keys())
        if all(isinstance(key, str) and key.startswith("module.") for key in keys):
            return {key[len("module."):]: value for key, value in state.items()}
    return state


def load_model(model_path: Path,
               base_width: int,
               device: torch.device,
               use_student: bool) -> SaltHRNet3D:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")

    model = SaltHRNet3D(base=base_width, enable_reconstruction=True).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state = _extract_state_dict(checkpoint, use_student=use_student)
    load_model_state(model, state, verbose=True)
    model.eval()
    selected = "student" if use_student else "teacher/net"
    print(f"[model] loaded {model_path} ({selected}, base={base_width})")
    return model


def _parse_patch_size(value: Union[int, Sequence[int]]) -> Tuple[int, int, int]:

    if isinstance(value, (int, np.integer)):
        sizes = (int(value),) * 3
    else:
        sizes = tuple(int(item) for item in value)
        if len(sizes) == 1:
            sizes = sizes * 3
        elif len(sizes) != 3:
            raise ValueError("patch_size must contain one value or three values")
    if any(size <= 0 or size % 4 != 0 for size in sizes):
        raise ValueError("every patch_size value must be a positive multiple of 4")
    return sizes


def _axis_grid(length: int, patch: int, overlap: int) -> Tuple[List[int], int]:

    stride = patch - overlap


    count = max(1, int(np.round((length + overlap) / stride + 0.5)))
    starts = [index * stride for index in range(count)]
    return starts, stride * count + overlap


def _blend_window(shape: Tuple[int, int, int], overlap: int) -> np.ndarray:

    if overlap == 0:
        return np.ones(shape, dtype=np.float32)
    d, h, w = shape
    window = np.ones(shape, dtype=np.float32)
    sigma = 0.5 / ((overlap / 4.0) ** 2)
    edge = np.asarray(
        [np.exp(-((k - overlap + 1) ** 2) * sigma) for k in range(overlap)],
        dtype=np.float32,
    )
    for k in range(overlap):
        window[k, :, :] = window[d - k - 1, :, :] = edge[k]
        window[:, k, :] = window[:, h - k - 1, :] = edge[k]
        window[:, :, k] = window[:, :, w - k - 1] = edge[k]
    return window


@torch.inference_mode()
def predict_volume(model: SaltHRNet3D,
                   data_3d: np.ndarray,
                   device: torch.device,
                   patch_size: Union[int, Sequence[int]] = (196, 196, 256),
                   overlap: int = 8,
                   progress: bool = True) -> np.ndarray:

    data_3d = np.asarray(data_3d, dtype=np.float32)
    if data_3d.ndim != 3:
        raise ValueError(f"Model input must be 3-D (D,H,W), got {data_3d.shape}")

    patch = _parse_patch_size(patch_size)
    overlap = int(overlap)
    if overlap < 0 or any(overlap >= axis for axis in patch):
        raise ValueError("overlap must satisfy 0 <= overlap < patch_size")

    original_shape = tuple(int(v) for v in data_3d.shape)
    grids = [_axis_grid(length, p, overlap)
             for length, p in zip(original_shape, patch)]
    starts = [grid[0] for grid in grids]
    padded_shape = tuple(grid[1] for grid in grids)
    pad_width = tuple((0, padded - size)
                      for size, padded in zip(original_shape, padded_shape))
    padded = np.pad(data_3d, pad_width, mode="constant", constant_values=0.0)

    blend = _blend_window(patch, overlap)
    prediction = np.zeros(padded_shape, dtype=np.float32)
    weights = np.zeros(padded_shape, dtype=np.float32)
    total = len(starts[0]) * len(starts[1]) * len(starts[2])
    processed = 0
    started = time.time()

    for sd in starts[0]:
        for sh in starts[1]:
            for sw in starts[2]:
                patch_data = padded[
                    sd:sd + patch[0],
                    sh:sh + patch[1],
                    sw:sw + patch[2],
                ]
                tensor = torch.from_numpy(patch_data[None, None]).to(device)
                output = model(tensor)
                if not isinstance(output, torch.Tensor) or output.ndim != 5:
                    raise RuntimeError("SaltHRNet3D returned an unexpected output")
                if tuple(output.shape[-3:]) != patch:
                    raise RuntimeError(
                        f"Model output shape {tuple(output.shape[-3:])} does not "
                        f"match input window {patch}; use a patch size divisible by 4"
                    )
                predicted_patch = output[0, 0].float().cpu().numpy()
                prediction[
                    sd:sd + patch[0],
                    sh:sh + patch[1],
                    sw:sw + patch[2],
                ] += predicted_patch * blend
                weights[
                    sd:sd + patch[0],
                    sh:sh + patch[1],
                    sw:sw + patch[2],
                ] += blend

                processed += 1
                if progress:
                    elapsed = time.time() - started
                    print(
                        f"\r[predict] {processed}/{total} windows "
                        f"({elapsed:.1f}s)",
                        end="",
                        flush=True,
                    )

    if progress:
        print()
    return (prediction / np.maximum(weights, 1e-6))[
        :original_shape[0], :original_shape[1], :original_shape[2]
    ].astype(np.float32, copy=False)


def _output_path(explicit: Optional[str], output_dir: Optional[Path],
                 input_path: Path, suffix: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(path.suffix + ".npz" if path.suffix else ".npz")
        return path.resolve()
    parent = output_dir.resolve() if output_dir else input_path.parent
    return parent / f"{input_path.stem}_{suffix}.npz"


def _save_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"[save] {path}")


def _select_source(source: str,
                   original: np.ndarray,
                   pasa: np.ndarray,
                   masked: np.ndarray) -> np.ndarray:
    if source == "original":
        return original
    if source == "pasa":
        return pasa
    if source == "masked":
        return masked
    raise ValueError(f"Unknown source: {source}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create PASA, anisotropic-mask, and model-label NPZ files"
    )
    parser.add_argument("--input_npz", required=True,
                        help="Original NPZ containing data (and optionally label)")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH,
                        help="SaltHRNet3D checkpoint used for prediction")
    parser.add_argument("--output_dir", default=None,
                        help="Directory for default output names")
    parser.add_argument("--pasa_output", "--pasa_output_npz", "--output_pasa_npz",
                        dest="pasa_output", default=None,
                        help="PASA output NPZ path")
    parser.add_argument("--mask_output", "--mask_output_npz", "--output_mask_npz",
                        dest="mask_output", default=None,
                        help="Anisotropic-mask output NPZ path")
    parser.add_argument("--label_output", "--label_output_npz", "--output_label_npz",
                        dest="label_output", default=None,
                        help="Predicted-label output NPZ path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for PASA and anisotropic mask")
    parser.add_argument("--mask_source", choices=["original", "pasa"], default="pasa",
                        help="Data used to build the anisotropic mask")
    parser.add_argument("--predict_source", choices=["original", "pasa", "masked"],
                        default="original", help="Data sent to the segmentation model")
    parser.add_argument("--horizontal_ratio", type=float, default=0.18,
                        help="H/W slice ratio for anisotropic masking")
    parser.add_argument("--vertical_ratio", type=float, default=0.10,
                        help="D slice ratio for anisotropic masking")
    parser.add_argument("--block_ratio", type=float, default=0.12,
                        help="Maximum 3-D block ratio for anisotropic masking")
    parser.add_argument("--base_width", type=int, default=8,
                        help="SaltHRNet3D base width used during training")
    parser.add_argument("--patch_size", type=int, nargs="+", default=[196, 196, 256],
                        help="One cubic size or D H W window sizes; project default: 196 196 256")
    parser.add_argument("--overlap", type=int, default=8,
                        help="Inference-window overlap")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Probability threshold for the output label")
    parser.add_argument("--min_component_size", type=int, default=50000,
                        help="Remove smaller 26-connected regions; 0 disables filtering")
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, or a CUDA device such as cuda:0")
    parser.add_argument("--use_student", action="store_true",
                        help="Use student_net when the checkpoint has teacher/student nets")
    parser.add_argument("--skip_predict", action="store_true",
                        help="Only generate PASA and mask outputs; do not load the model")
    parser.add_argument("--quiet_progress", action="store_true",
                        help="Do not print per-window inference progress")
    return parser


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.threshold < 0.0 or args.threshold > 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if args.min_component_size < 0:
        raise ValueError("min_component_size must be non-negative")
    for name, value in (("horizontal_ratio", args.horizontal_ratio),
                        ("vertical_ratio", args.vertical_ratio),
                        ("block_ratio", args.block_ratio)):
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    input_path = Path(args.input_npz).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    pasa_path = _output_path(args.pasa_output, output_dir, input_path, "pasa")
    mask_path = _output_path(args.mask_output, output_dir, input_path,
                             "anisotropic_mask")
    label_path = _output_path(args.label_output, output_dir, input_path,
                              "pred_label")
    all_paths = [pasa_path, mask_path] + ([] if args.skip_predict else [label_path])
    if len({str(path) for path in all_paths}) != len(all_paths):
        raise ValueError("PASA, mask, and label outputs must use different paths")
    if any(path == input_path for path in all_paths):
        raise ValueError("An output path must not overwrite the input NPZ")

    seed_everything(args.seed)
    original_arrays, original_4d, original_label_4d, had_channel = _load_npz(input_path)
    pasa_4d, pasa_label_4d = apply_pasa(original_4d, original_label_4d)

    pasa_arrays = dict(original_arrays)
    pasa_arrays["data"] = _restore_shape(pasa_4d, had_channel)
    if pasa_label_4d is not None:
        pasa_arrays["label"] = _restore_shape(pasa_label_4d, had_channel)
    _save_npz(pasa_path, pasa_arrays)

    mask_source_4d = pasa_4d if args.mask_source == "pasa" else original_4d
    masked_4d, mask_4d = make_anisotropic_mask(
        mask_source_4d,
        horizontal_ratio=args.horizontal_ratio,
        vertical_ratio=args.vertical_ratio,
        block_ratio=args.block_ratio,
    )
    mask_arrays: Dict[str, np.ndarray] = dict(original_arrays)
    mask_arrays.update({
        "data": _restore_shape(masked_4d, had_channel),
        "mask": _restore_shape(mask_4d, had_channel),
    })
    selected_label = pasa_label_4d if args.mask_source == "pasa" else original_label_4d
    if selected_label is not None:
        mask_arrays["label"] = _restore_shape(selected_label, had_channel)
    _save_npz(mask_path, mask_arrays)

    if args.skip_predict:
        print("[done] PASA and anisotropic-mask outputs generated; prediction skipped")
        return

    if args.model_path is None:
        raise ValueError("--model_path is required unless --skip_predict is set")
    device = _resolve_device(args.device)
    model = load_model(Path(args.model_path).expanduser().resolve(),
                       base_width=args.base_width,
                       device=device,
                       use_student=args.use_student)
    predict_source_4d = _select_source(
        args.predict_source, original_4d, pasa_4d, masked_4d
    )
    probability_3d = predict_volume(
        model,
        predict_source_4d[0],
        device,
        patch_size=args.patch_size,
        overlap=args.overlap,
        progress=not args.quiet_progress,
    )
    raw_label_3d = (probability_3d >= args.threshold).astype(np.float32)
    predicted_label_3d, component_count, kept_count = filter_small_components(
        raw_label_3d, args.min_component_size
    )
    label_arrays: Dict[str, np.ndarray] = {
        key: value for key, value in original_arrays.items()
        if key not in {"data", "label", "prob", "ground_truth"}
    }
    label_arrays.update({
        "data": _restore_shape(predict_source_4d, had_channel),
        "label": _restore_shape(predicted_label_3d[np.newaxis, ...], had_channel),
        "label_raw": _restore_shape(raw_label_3d[np.newaxis, ...], had_channel),
        "prob": _restore_shape(probability_3d[np.newaxis, ...], had_channel),
    })
    if original_label_4d is not None:
        label_arrays["ground_truth"] = _restore_shape(original_label_4d, had_channel)
    _save_npz(label_path, label_arrays)
    if args.min_component_size > 0:
        print(
            f"[filter] kept {kept_count}/{component_count} components "
            f"with at least {args.min_component_size:,} voxels"
        )
    print(
        f"[done] label positives: {int(predicted_label_3d.sum()):,} / "
        f"{predicted_label_3d.size:,} ({predicted_label_3d.mean() * 100:.2f}%)"
    )


if __name__ == "__main__":
    main()
