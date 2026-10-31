import argparse
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


class AdaptiveFourierVarianceMatching:
    """Adaptive Fourier Variance Matching for 3D seismic NPZ patches.

    Paper equations:
      v_i = Var(x_i)
      v_i_norm = clip((v_i - P5) / (P95 - P5), 0, 1)
      beta_i = beta_min + (1 - v_i_norm) * (beta_max - beta_min)
      j(i) = argmin_j |v_i^s - v_j^t|

    AFVM keeps the source phase and replaces only low-frequency amplitudes
    inside the adaptive beta_i window.
    """

    def __init__(self, beta_min=0.005, beta_max=0.03):
        if beta_min <= 0 or beta_max <= 0:
            raise ValueError("beta_min and beta_max must be positive.")
        if beta_min > beta_max:
            raise ValueError("beta_min must be <= beta_max.")
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.p5 = None
        self.p95 = None

    @staticmethod
    def _spatial_shape(data):
        if data.ndim == 3:
            return data.shape
        if data.ndim == 4:
            return data.shape[-3:]
        raise ValueError(
            f"AFVM expects data with shape (D,H,W) or (C,D,H,W), got {data.shape}"
        )

    @staticmethod
    def _fft_axes(data):
        return tuple(range(data.ndim - 3, data.ndim))

    @staticmethod
    def _match_data_shape(source_data, target_data):
        source_shape = AdaptiveFourierVarianceMatching._spatial_shape(source_data)
        target_shape = AdaptiveFourierVarianceMatching._spatial_shape(target_data)
        if source_shape != target_shape:
            raise ValueError(
                f"Spatial shape mismatch: source {source_shape} vs target {target_shape}"
            )

        if source_data.ndim == target_data.ndim:
            if source_data.shape[:-3] != target_data.shape[:-3]:
                raise ValueError(
                    f"Channel shape mismatch: source {source_data.shape} vs "
                    f"target {target_data.shape}"
                )
            return source_data, target_data

        if source_data.ndim == 4 and target_data.ndim == 3:
            target_data = np.broadcast_to(target_data, source_data.shape)
            return source_data, target_data

        if source_data.ndim == 3 and target_data.ndim == 4:
            if target_data.shape[0] != 1:
                raise ValueError(
                    "Cannot match 3D source data with multi-channel target data."
                )
            return source_data, target_data[0]

        raise ValueError(
            f"Unsupported source/target shapes: {source_data.shape}, {target_data.shape}"
        )

    @staticmethod
    def _compute_variance(npz_path):
        with np.load(npz_path) as npz:
            if "data" not in npz.files:
                raise KeyError(f"{npz_path} does not contain key 'data'.")
            data = npz["data"].astype(np.float32, copy=False)
            return float(np.var(data))

    @staticmethod
    def _load_data(npz_path):
        with np.load(npz_path) as npz:
            if "data" not in npz.files:
                raise KeyError(f"{npz_path} does not contain key 'data'.")
            return npz["data"]

    def fit_source_variance_range(self, source_variances):
        source_variances = np.asarray(source_variances, dtype=np.float64)
        if source_variances.size == 0:
            raise ValueError("source_variances is empty.")
        self.p5 = float(np.percentile(source_variances, 5))
        self.p95 = float(np.percentile(source_variances, 95))
        return self.p5, self.p95

    def beta_from_variance(self, variance):
        if self.p5 is None or self.p95 is None:
            raise RuntimeError("Call fit_source_variance_range before beta_from_variance.")

        if self.p95 <= self.p5:
            variance_norm = 0.5
        else:
            variance_norm = (float(variance) - self.p5) / (self.p95 - self.p5)
            variance_norm = float(np.clip(variance_norm, 0.0, 1.0))

        beta = self.beta_min + (1.0 - variance_norm) * (
            self.beta_max - self.beta_min
        )
        return float(beta)

    @staticmethod
    def create_window_function(spatial_shape, beta):
        d_size, h_size, w_size = spatial_shape
        window = np.zeros(spatial_shape, dtype=bool)

        extents = [
            max(1, int(round(d_size * beta))),
            max(1, int(round(h_size * beta))),
            max(1, int(round(w_size * beta))),
        ]
        centers = [d_size // 2, h_size // 2, w_size // 2]
        limits = [d_size, h_size, w_size]

        slices = []
        for center, extent, limit in zip(centers, extents, limits):
            start = center - extent // 2
            end = start + extent
            if start < 0:
                start = 0
                end = min(limit, extent)
            if end > limit:
                end = limit
                start = max(0, end - extent)
            slices.append(slice(start, end))

        window[tuple(slices)] = True
        return window

    def apply_afvm(self, source_data, target_data, beta):
        source_data, target_data = self._match_data_shape(source_data, target_data)
        source_float = source_data.astype(np.float32, copy=False)
        target_float = target_data.astype(np.float32, copy=False)

        axes = self._fft_axes(source_float)
        fft_source = np.fft.fftn(source_float, axes=axes)
        fft_target = np.fft.fftn(target_float, axes=axes)
        fft_source_shifted = np.fft.fftshift(fft_source, axes=axes)
        fft_target_shifted = np.fft.fftshift(fft_target, axes=axes)

        amp_source = np.abs(fft_source_shifted)
        amp_target = np.abs(fft_target_shifted)
        phase_source = np.angle(fft_source_shifted)

        window = self.create_window_function(self._spatial_shape(source_float), beta)
        while window.ndim < amp_source.ndim:
            window = window[None, ...]

        amp_afvm = np.where(window, amp_target, amp_source)
        fft_afvm_shifted = amp_afvm * np.exp(1j * phase_source)
        fft_afvm = np.fft.ifftshift(fft_afvm_shifted, axes=axes)
        adapted_data = np.fft.ifftn(fft_afvm, axes=axes).real
        return adapted_data.astype(source_data.dtype, copy=False)

    def process_npz_file(self, source_npz_path, target_npz_path, output_path, beta):
        with np.load(source_npz_path) as source_npz:
            if "data" not in source_npz.files:
                raise KeyError(f"{source_npz_path} does not contain key 'data'.")
            arrays = {key: source_npz[key] for key in source_npz.files}

        target_data = self._load_data(target_npz_path)
        arrays["data"] = self.apply_afvm(arrays["data"], target_data, beta)
        np.savez_compressed(output_path, **arrays)
        return arrays["data"]

    def process_dataset(self, source_dir, target_dir, output_dir, num_target_samples=0):
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        source_files = sorted(source_dir.glob("*.npz"))
        target_files = sorted(target_dir.glob("*.npz"))
        if not source_files:
            raise ValueError(f"No NPZ files found in source directory: {source_dir}")
        if not target_files:
            raise ValueError(f"No NPZ files found in target directory: {target_dir}")

        if num_target_samples and num_target_samples > 0:
            sample_count = min(num_target_samples, len(target_files))
            sample_indices = np.random.choice(
                len(target_files), sample_count, replace=False
            )
            target_files = [target_files[i] for i in sorted(sample_indices)]

        print(f"Found {len(source_files)} synthetic files")
        print(f"Found {len(target_files)} real reference files")
        print(
            f"Using AFVM beta range: beta_min={self.beta_min}, "
            f"beta_max={self.beta_max}"
        )

        print("\nScanning source variances...")
        source_variances = np.array(
            [self._compute_variance(path) for path in tqdm(source_files, desc="Source var")],
            dtype=np.float64,
        )
        p5, p95 = self.fit_source_variance_range(source_variances)
        print(f"Source variance percentiles: P5={p5:.6g}, P95={p95:.6g}")

        print("\nScanning target variances...")
        target_variances = np.array(
            [self._compute_variance(path) for path in tqdm(target_files, desc="Target var")],
            dtype=np.float64,
        )

        print(f"\nProcessing {len(source_files)} synthetic files with AFVM...")
        processed = 0
        for source_file, source_var in tqdm(
            list(zip(source_files, source_variances)), desc="Applying AFVM"
        ):
            beta = self.beta_from_variance(source_var)
            target_idx = int(np.argmin(np.abs(target_variances - source_var)))
            target_file = target_files[target_idx]
            output_file = output_dir / source_file.name
            try:
                self.process_npz_file(source_file, target_file, output_file, beta)
                processed += 1
            except Exception as exc:
                print(f"\nError processing {source_file.name}: {exc}")

        print(f"\nCompleted! AFVM files saved to: {output_dir}")
        print(f"Total files processed: {processed}")


AFVM = AdaptiveFourierVarianceMatching


def main():
    parser = argparse.ArgumentParser(
        description="Apply Adaptive Fourier Variance Matching to seismic NPZ data"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Directory containing synthetic seismic NPZ files",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Directory containing real seismic NPZ files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save AFVM-aligned files",
    )
    parser.add_argument(
        "--beta_min",
        type=float,
        default=0.005,
        help="Minimum adaptive low-frequency replacement ratio",
    )
    parser.add_argument(
        "--beta_max",
        type=float,
        default=0.03,
        help="Maximum adaptive low-frequency replacement ratio",
    )
    parser.add_argument(
        "--num_target_samples",
        type=int,
        default=0,
        help="Number of target samples to match against; 0 means use all targets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when target samples are sub-sampled",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    afvm = AdaptiveFourierVarianceMatching(
        beta_min=args.beta_min,
        beta_max=args.beta_max,
    )
    afvm.process_dataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        output_dir=args.output_dir,
        num_target_samples=args.num_target_samples,
    )


if __name__ == "__main__":
    main()
