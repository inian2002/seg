# Reproduction notes

## What is executable now

The repository provides AFVM, SaltHRNet3D, PASA, sparse-label losses, reconstruction, training/checkpoint utilities, explicit patch preparation, and the project's sliding-window inference helper. `examples/synthetic_demo.py` creates an independent synthetic test case and runs adaptation, short pretraining, sparse fine-tuning, and inference. All example data are generated locally from a fixed seed.

The workflow was checked on CPU with Python 3.13.13, NumPy 2.5.0 and PyTorch 2.11.0+cu130. See [the recorded example run](../examples/expected_cpu_summary.json). Wall time and floating-point results can differ by hardware/library version. The tiny model is deliberately undertrained; zero IoU after four optimization steps is not a failure of the execution check and is not a published research result.

## Materials and limits

The public release contains F3 and SEAM preprocessed amplitude volumes plus checksums. The repository also contains the authors' supplied `result/f3.csv` and `result/seam.csv`. These files are retained as supplied and are not outputs of the new example.

Exact reproduction of the manuscript's quantitative tables additionally requires the aligned dense/sparse salt labels, exact spatial splits/crops and resampling records, selected trained checkpoints, and the complete comparison-method configurations. These materials are not included in the current public release. The L3 volume is also not included. This document does not claim that data cannot legally be shared; it records what is currently supplied. The synthetic example is a runnable test case, not a substitute for missing experiment-specific materials or evidence of the reported IoU/Dice scores.

The manuscript reports F3 IoU 0.9367 and Dice 0.9673 at 3.52% annotation. This documentation update did not rerun those scientific experiments. A trained checkpoint and aligned evaluation labels are required to independently check them. In particular, running short toy training cannot verify the reported survey performance.

## Parameters to preserve for a survey replication

- Confirm `(D,H,W)=(Timeline/Depth,Inline,Crossline)` after any permutation; record the permutation and region.
- Match survey-level standardization, crop/resampling and train/validation/test separation.
- Use base width 8, 256-cubed inputs, batch size 2, AFVM beta 0.005–0.03, PASA and the documented reconstruction weights.
- Pretrain for 200 epochs at 5e-4 and fine-tune for 100 epochs at 1e-4, following the paper's checkpoint-selection policy.
- Sparse Inline annotations include both boundaries; spacing 35 in 256 samples retains 9 sections.
- For full-volume inference, match patch size 256, overlap 8, threshold 0.05 and minimum connected-component size 50,000. Compare predictions over the same evaluation region for every method.
- Record software versions and all random seeds. The original training script's partial seeding and GPU nondeterminism can prevent bitwise repeats.

## Licensing

The MIT license applies to repository software, including the newly added example and preprocessing utility. It does not change the original seismic-data providers' terms. The local demonstration arrays are generated from analytic functions and random noise and contain no third-party survey data.
