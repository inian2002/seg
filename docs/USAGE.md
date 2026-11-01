# User guide

Run commands from the repository root. Code comments and docstrings are omitted; usage is documented here.

## Installation and resources

Use a dedicated Python environment (Python 3.11 or newer):

```text
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS, then install:

```text
python -m pip install -r requirements.txt
```

PyTorch must match the machine's CPU/CUDA setup. The local validation environment used Python 3.13.13 and PyTorch 2.11.0+cu130; the example was run on CPU. The manuscript reports full experiments on a 32 GB NVIDIA GeForce RTX 5090 with CUDA 13.0. Full 256-cubed training is substantially more demanding than the 32-cubed example; smaller test inputs do not reproduce the reported accuracy. Keep at least several GB of free RAM for volume loading and enough disk space for intermediate NPZ files and checkpoints. The two public amplitude files total about 1.02 GB (decimal).

## Runnable synthetic example

```text
python examples/synthetic_demo.py --device cpu --output outputs/demo
```

The script generates two toy labeled source cubes, two target-reference cubes, sparse target labels, and a held-out toy cube. It runs AFVM, two dense-label optimization steps, two sparse-label steps with PASA and reconstruction, and inference. No downloaded surveys or pretrained weights are required.

Expected outputs include `source/`, `target/`, `adapted/`, `sparse/`, `validation/`, `pretrain.pth`, `finetune.pth`, `probabilities.npy`, `prediction.npy`, and `summary.json`. The summary records losses, metrics, seed and versions. It checks that adaptation changes amplitudes, preserves labels, and produces finite training/inference values. IoU and Dice may be zero after this deliberately short run; successful execution does not establish scientific performance.

The example uses base width 2, 32-cubed inputs and AFVM beta 0.10–0.20 so the small frequency window is visible. These are demonstration settings. The research defaults are base width 8, 256-cubed inputs and AFVM beta 0.005–0.03. Run `--help` for all example options. A NumPy-only check is available with `--mode afvm`. Use a new output folder for each run.

## Data and array conventions

```text
python data/download_data.py
```

The downloader verifies published sizes and SHA-256 hashes. It preserves the released arrays exactly. See [data documentation](../data/README.md). The files contain preprocessed F3 and SEAM seismic amplitude volumes.

Training NPZ files must contain `data` and `label`, both shaped `(D, H, W)`, where `D` is Timeline/Depth, `H` is Inline and `W` is Crossline. Data must be finite, preprocessed `float32` amplitudes. Dense labels are 0 (background) and 1 (salt); sparse training additionally uses -1 (unlabeled). Validation labels must be dense. Do not classify unlabeled voxels as background. Keep only NPZ files in training/validation directories because the supplied loaders enumerate all files.

Confirm the original array-axis order from survey metadata before preprocessing. Shape or filename alone does not establish axis meaning. The network does not standardize inputs; survey-level normalization must be done before patch extraction. Public release volumes are already supplied as z-score arrays.

## Prepare patches and sparse labels

`code/prepare_patches.py` converts preprocessed NPY data and optional dense NPY labels into training-compatible NPZ patches. `--axes` explicitly specifies the original axes to place in `(D,H,W)` order. If the input is already `(D,H,W)`, use `--axes 0 1 2`; if confirmed to be `(Inline,Crossline,Depth)`, use `--axes 2 0 1`.

```text
python code/prepare_patches.py --data /path/seam.npy --labels /path/seam_labels.npy --axes 0 1 2 --output outputs/seam_train --patch-size 256 --stride 256
python code/prepare_patches.py --data /path/f3.npy --labels /path/f3_labels.npy --axes 0 1 2 --output outputs/f3_sparse --patch-size 256 --stride 256 --inline-spacing 35
```

Replace paths and axes with the actual dataset configuration. Optional `--region D0 D1 H0 H1 W0 W1` uses half-open bounds after axis permutation. Define separate spatial train/validation regions before patch extraction, accounting for overlap; the example commands do not choose a scientifically valid split automatically. The preprocessing manifest is saved beside the patch folder so the loader does not treat it as a sample.

`--inline-spacing 35` retains local Inline indices 0, 35, ..., 245 and 255 in a 256-cubed patch: 9/256 = 3.515625% labeled voxels. Unknown voxels receive -1. Always prepare validation separately without this option. The script does not crop, interpolate, normalize, or change labels silently beyond the region and sparsity options provided.

## AFVM adaptation

```text
python code/AFVM.py --source_dir outputs/seam_train --target_dir /path/f3_reference_npz --output_dir outputs/seam_adapted --beta_min 0.005 --beta_max 0.03 --seed 42
```

Source/target folders contain compatible NPZ cubes with `data`; source files also carry dense `label`. All reference files are used unless `--num_target_samples` is set. Source labels and other NPZ arrays are preserved; only source amplitudes are adapted. Output filenames match the source filenames. Target references must follow the intended training/evaluation protocol.

## Pretraining, transfer and fine-tuning

```text
python code/train.py --train_path outputs/seam_adapted --val_path /path/seam_validation --save outputs/seam --base_width 8 --batch_size 2 --epochs 200 --lr 0.0005 --seed 42 --da_type pasa --use_recon 1
python prepare_transfer.py outputs/seam/best_model.pth outputs/seam/transfer.pth
python code/train.py --restore outputs/seam/transfer.pth --train_path outputs/f3_sparse --val_path /path/f3_validation --save outputs/f3 --base_width 8 --batch_size 2 --epochs 100 --lr 0.0001 --seed 42 --da_type pasa --use_recon 1
```

`--restore` normally restores the optimizer and resumes from the saved epoch. `prepare_transfer.py` removes optimizer and epoch metadata for a new fine-tuning stage. Choose a pretraining checkpoint according to the stated validation protocol rather than the test set. CUDA is used automatically when available; otherwise training runs on CPU. On Windows, `--n_threads 0` avoids worker-process complications.

Outputs: `config.json`, `config.txt`, `train_log.csv`, TensorBoard events, `best_model.pth`, `checkpoints/epoch_NNN.pth` and `checkpoints/latest.pth`. The existing logger names validation columns `Val_F3_*` even for a SEAM validation folder; the data supplied by `--val_path` determines the actual survey. Validation metrics pool labeled voxels across patches; they are not automatically equivalent to a fused full-volume evaluation if patches overlap or the evaluation region differs.

| Option | Default | Meaning |
|---|---:|---|
| `--base_width` | 8 | Network width; match it when loading weights |
| `--da_type` | common | Choose `pasa` for the full framework |
| `--use_recon` | 1 | Enable auxiliary reconstruction |
| `--recon_weight` | 0.08 | Reconstruction loss weight |
| `--recon_rampup` | 20 | Ramp-up duration in epochs |
| `--recon_horizontal_ratio` | 0.18 | Combined H/W masking ratio |
| `--recon_vertical_ratio` | 0.10 | D masking ratio |
| `--recon_block_ratio` | 0.12 | Maximum block side ratio |
| `--recon_edge_weight` | 0.25 | Gradient-loss weight |
| `--recon_freq_weight` | 0.10 | Frequency-loss weight |
| `--recon_low_freq_ratio` | 0.08 | Frequency-loss low-band radius |

Use `python code/train.py --help` for remaining options. The original training loop seeds NumPy/PyTorch but does not fully seed Python's random module; exact repeated training is therefore not promised. The synthetic example explicitly seeds all three.

## Inference and artifact generation

The repository includes the existing project helper `generate_npz_artifacts.py`. It generates synchronized PASA data, a masked reconstruction input, and a probability/binary segmentation volume.

```text
python code/generate_npz_artifacts.py --input_npz /path/evaluation_volume.npz --model_path outputs/f3/best_model.pth --output_dir outputs/inference --base_width 8 --patch_size 256 --overlap 8 --threshold 0.05 --min_component_size 50000 --predict_source original
```

Specify `--patch_size 256` to follow the manuscript; the helper's historical default is 196 196 256. It accepts one cubic dimension or three dimensions. Segmentation uses weighted sliding-window fusion, a probability threshold, and optional removal of small 26-connected components. Set `--min_component_size 0` to disable filtering. `--device cpu` selects CPU; `--skip_predict` produces only augmentation and mask outputs without loading a model.

The prediction NPZ contains `data`, `prob`, `label_raw`, filtered `label`, and `ground_truth` when input labels were supplied. No labels are required for inference. All labels remain in the input's array coordinates.

A small tested inference command after the synthetic demo is:

```text
python code/generate_npz_artifacts.py --input_npz outputs/demo/validation/00.npz --model_path outputs/demo/finetune.pth --output_dir outputs/demo_inference --base_width 2 --patch_size 32 --overlap 8 --threshold 0.5 --min_component_size 0 --device cpu
```

## Common issues

- Shape mismatch: check axis order and matching data/label shapes. Training expects 3D arrays before the loader adds a channel axis.
- Out of memory: use the small CPU example first; full settings need substantially more GPU memory. Reduced settings change the experiment.
- Restore immediately finishes: use a transfer checkpoint with epoch metadata removed for a new fine-tuning run.
- Unexpected metrics: confirm threshold, label mask, evaluation region, overlap fusion and connected-component settings; do not derive accuracy or recall from IoU/Dice alone.
  
