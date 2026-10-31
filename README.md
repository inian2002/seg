# FAMT-Salt3D

Source code and supplied seismic volumes for **Sparsely supervised 3D seismic salt interpretation using frequency-domain adaptation and multi-task learning**.

Repository: <https://github.com/inian2002/seg>

## Contents

- `code/AFVM.py`: Adaptive Fourier Variance Matching for seismic NPZ patches.
- `code/models/SaltHRNet3D.py`: the 3D segmentation network and auxiliary reconstruction branch.
- `code/train.py`: supervised/sparse-label training with AFVM-prepared data, PASA and anisotropic reconstruction.
- `code/argues/config_2.py`: training options.
- `code/utils/`: data loaders, augmentation, losses, reconstruction, logging and checkpoint utilities.
- `prepare_transfer.py`: prepares a pretrained checkpoint for transfer by removing optimizer and epoch state.
- `result/f3.csv` and `result/seam.csv`: supplied comparison metric summaries.
- `data/`: data documentation, download helper and SHA-256 checksums.

## Installation and computational requirements

Create a Python environment and install the imported dependencies:

```bash
python -m venv .venv
# Activate the environment using the command appropriate for your operating system.
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with your GPU/driver. The manuscript reports Python 3.13, PyTorch 2.11 and CUDA 13, with a 32 GB NVIDIA GPU for training. The dependency file lists required packages without claiming that every available version has been validated. Large 3D volumes and 256-cubed patches require substantial memory; memory-map NPY files when only inspecting them.

## Download the supplied volumes

Large files are available as individual assets in [the data release](https://github.com/inian2002/seg/releases/tag/famt-salt3d-data-v1), rather than in Git history.

```bash
python data/download_data.py
```

This downloads the F3 and SEAM volumes and verifies their SHA-256 hashes. Use `--dataset f3` or `--dataset seam` to download one volume, and `--output-dir PATH` to choose another directory. Shapes, sizes and hashes are in [`data/manifest.json`](data/manifest.json).

## Data expected by training

The supplied NPY files are float32 seismic amplitude volumes. They are **not** labeled training-patch archives. The loaders expect one NPZ archive per patch with:

- `data`: an externally preprocessed amplitude array with shape `(D, H, W)`;
- `label`: an equally shaped array, with `0` for background, `1` for salt and `-1` for unknown training voxels.

Validation labels must be dense (`0`/`1`). The loaders add a channel dimension and convert arrays to float32; they do not normalize amplitudes internally. Confirm volume axis order before extracting patches; the download helper preserves stored arrays without transposition. Labels, patch-generation scripts, trained checkpoints and the full-volume inference script are not included in the supplied folder. The metric CSVs are supplied summaries, not results recomputed by a script in this release. These limits are stated explicitly so the repository is not mistaken for a complete reproduction package.

## Example workflow

The paths below refer to labeled NPZ patch folders prepared separately.

```bash
python code/AFVM.py --source_dir path/to/seam_patches --target_dir path/to/f3_references --output_dir path/to/adapted_seam

cd code
python train.py --train_path path/to/adapted_seam --val_path path/to/seam_validation --save ../output/seam --epochs 200 --lr 0.0005 --da_type pasa --use_recon 1
```

Training writes per-epoch checkpoints, `latest.pth`, a best checkpoint and logs beneath `--save`. For transfer learning, remove the saved optimizer/epoch state before fine-tuning:

```bash
cd ..
python prepare_transfer.py output/seam/best_model.pth output/seam/transfer.pth
cd code
python train.py --restore ../output/seam/transfer.pth --train_path path/to/f3_sparse_patches --val_path path/to/f3_dense_validation --save ../output/f3 --epochs 100 --lr 0.0001 --da_type pasa --use_recon 1
```

Check available arguments with `python code/AFVM.py --help` or, from `code/`, `python train.py --help`. The checkpoint preparation utility should be used with trusted checkpoint files. Experimental results have not been rerun as part of this repository upload.

## License and data provenance

Source code is released under the [MIT License](LICENSE), as authorized by the authors. Original third-party notices remain applicable. The MIT software license does not override rights or access conditions in the original seismic datasets; consult their providers for applicable data terms.

The uploaded volume names identify the Dutch F3 survey and SEAM synthetic data. The authors supplied the preprocessed arrays in this release; see the manuscript's experimental-data section for survey references and processing context.
