# Seismic amplitude volumes

Download the individual assets from <https://github.com/inian2002/seg/releases/tag/famt-salt3d-data-v1>, or run `python data/download_data.py` from the repository root.

| File | Shape | Type | Bytes |
|---|---|---|---|
| `F3_zscore_651_951_256.npy` | 651 x 951 x 256 | float32 | 633959552 |
| `SEAM_zscore_570_480_355.npy` | 570 x 480 x 355 | float32 | 388512128 |

SHA-256 hashes and permanent release URLs are in `manifest.json`. Files are retained exactly as supplied; the downloader does not resample or reorder axes. These are amplitude volumes; dense/sparse salt labels and trained weights are not included.

The repository's MIT license applies to the software. Original seismic-data rights and provider conditions remain applicable.
