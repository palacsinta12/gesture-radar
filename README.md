# FMCW Radar Gesture Recognition

A portfolio-ready machine-learning and signal-processing project for recognizing human hand gestures from Frequency-Modulated Continuous Wave (FMCW) radar tensors.

The project combines:

- a custom radar DSP chain with clutter suppression, Range-Doppler transformation, and azimuth estimation
- feature extraction into RTM, DTM, and ATM images
- a TensorFlow/Keras 2D CNN for gesture classification
- visualizations and generated training artifacts for research and demo use

<div align="center">
  <img src="assets/range_doppler_animation.gif" alt="Range-Doppler animation" width="60%">
  <br>
  <em>Range-Doppler sample from the push-pull gesture.</em>
</div>

## Project Goals

This repository was built as an applied DSP and deep-learning experiment for human gesture recognition using a 60 GHz Infineon radar sensor. The final goal is to show a reproducible data pipeline from raw analog radar data into a trained classifying model.

## Hardware and Data

The signals were recorded with the Infineon CY8CKIT-062S2-AI evaluation kit using the BGT60TR13C 60 GHz radar sensor. The acquisition setup exposes a 1 Tx × 3 Rx antenna arrangement and produces FMCW frame batches that are converted into spatial-temporal gesture feature maps.

## Repository Structure

```text
gesture_radar/
├── data/                   # local dataset and sample exports
├── src/
│   ├── data/               # DSP data preparation and feature extraction
│   ├── helpers/            # radar DSP primitives
│   ├── models/             # model definitions
│   ├── animate.py          # animation rendering for range-doppler maps
│   ├── config.py          # global constants and hyperparameters
│   ├── train.py            # training pipeline
│   ├── visualize.py       # feature visualization
│   └── visualize_dsp.py    # DSP diagnostic plots
├── assets/                 # generated figures and demos
├── requirements.txt        # Python dependency pin suggestions
├── pyproject.toml          # packaging metadata and pytest config
├── Makefile                # convenience commands
└── README.md               # project documentation
```

## DSP Pipeline

The preprocessing path makes the gesture data learnable for a standard 2D CNN:

1. Static clutter attenuation via an MTI filter.
2. Range-Doppler extraction through the Doppler algorithm.
3. Beamforming across the horizontal antenna pair for azimuth tracking.
4. Conversion of the feature volumes into RTM, DTM, and ATM maps.
5. Decibel scaling for stable neural-network training.

The repository also includes the plotting scripts that generate comparative visual diagnostics such as:

- static clutter comparison before and after filtering
- linear-versus-dB magnitude scaling comparison
- gesture sample maps for RTM, DTM, and ATM
- range-Doppler animation and final training artifacts

## Model

The CNN is constructed in [src/models/cnn.py](src/models/cnn.py) with a small convolutional stack, batch normalization, pooling, dropout, and a dense classifier. The model keeps the time and spatial bins visible by preserving the feature map structure instead of collapsing the map too early with global pooling.

## Getting Started

Create a virtual environment and install the repository requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell use:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the main training pipeline:

```bash
python -m src.train
```

Generate visualization examples:

```bash
python -m src.visualize
python -m src.visualize_dsp
python -m src.animate
```

## Data Notes

The repository is designed to ingest a local dataset under the `data/` directory and includes a `data/README.md` note to remind contributors that raw acquisitions, NPY files, and preprocessing exports should remain local to avoid bloating the repository.

## Reproducibility

You can set up a clean experiment with package metadata via `pyproject.toml` and install with `pip`. The current training configuration is described in [src/config.py](src/config.py). If you want to reproduce the results on a different machine, align the TensorFlow version carefully because the TensorFlow/Keras interface is tightly version-sensitive in this repo.

## Licenses

This project is distributed under the MIT license. See [LICENSE](LICENSE).

## Portfolio Guidance

This project is structured for a GitHub portfolio with:

- a clean top-level repository README
- dependency metadata
- a Python package structure
- a small automated import smoke test
- generated-image and data artifacts ignored by default

For a public portfolio repository, consider replacing the included dataset files with a small, anonymized sample dataset or publishing the script-only pipeline while keeping the raw measurement files private.
