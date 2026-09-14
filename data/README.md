# Data directory

This directory is intended to hold the exported FMCW radar dataset and processed feature files for training and visualization.

The repository is organized so that local, generated data artifacts stay out of version control. For a portfolio repository, inspect the `data/` directory structure and keep the original raw acquisition folders private or replace them with a small public sample.

Expected files:

- `alldata.npz` produced by the training pipeline or a public sample export.
- `preprocessed.npz` files produced by the DSP extractor under per-gesture folders.
- Raw `radar.npy` and `config.json` recordings captured by the acquisition setup.

Create the data folder locally before running the extract and train scripts.
