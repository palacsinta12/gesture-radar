"""Create DSP diagnostic figures for the FMCW signal-processing stack.

These scripts are not the main training pipeline. They show how the project
explains the signal transformation from raw sensor samples to the feature maps
that later feed the neural network.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from src.config import DATA_DIR, EPSILON
from src.helpers.DopplerAlgo import DopplerAlgo

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def find_first_raw_radar_file():
    """Return the first raw radar recording under the private data tree."""
    for dirpath, _, filenames in os.walk(DATA_DIR):
        if "radar.npy" in filenames:
            return os.path.join(dirpath, "radar.npy")
    return None


def find_best_frame(raw_data, doppler_algo, sensor_idx=0):
    """Pick the frame with the strongest motion energy as a plot anchor."""
    n_frames = raw_data.shape[0]
    max_energy = 0
    best_frame = 0
    rd_maps = []

    for f in range(n_frames):
        rd = doppler_algo.compute_doppler_map(raw_data[f, sensor_idx], sensor_idx)
        rd_maps.append(rd)

        center_bin = rd.shape[1] // 2
        motion_energy = np.sum(np.abs(rd[:, :center_bin - 2])) + np.sum(np.abs(rd[:, center_bin + 3:]))

        if f > 5 and motion_energy > max_energy:
            max_energy = motion_energy
            best_frame = f

    return best_frame, rd_maps


def plot_mti_effect():
    """Compare the same frame with and without the MTI clutter filter."""
    file_path = find_first_raw_radar_file()
    if not file_path:
        return

    raw_data = np.load(file_path) / 4095.0
    _, batch_nsensors, batch_nchirps, batch_nsamples = raw_data.shape
    sensor_idx = 0

    doppler_no_mti = DopplerAlgo(batch_nsamples, batch_nchirps, batch_nsensors, mti_alpha=0.0)
    doppler_with_mti = DopplerAlgo(batch_nsamples, batch_nchirps, batch_nsensors, mti_alpha=0.5)

    best_frame, _ = find_best_frame(raw_data, doppler_with_mti, sensor_idx)
    print(f"MTI Plot: Auto-detected maximum motion at frame {best_frame}")

    for f in range(best_frame + 1):
        rd_no_mti = doppler_no_mti.compute_doppler_map(raw_data[f, sensor_idx], sensor_idx)
        rd_with_mti = doppler_with_mti.compute_doppler_map(raw_data[f, sensor_idx], sensor_idx)

    rd_no_mti_db = 10 * np.log10(np.abs(rd_no_mti) + EPSILON)
    rd_with_mti_db = 10 * np.log10(np.abs(rd_with_mti) + EPSILON)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    im1 = ax1.imshow(rd_no_mti_db, aspect="auto", origin="lower", cmap="jet")
    ax1.set_title("Without MTI Filter\n(Static clutter present at 0 m/s)")
    ax1.set_xlabel("Doppler Velocity Bins (Center = 0 m/s)")
    ax1.set_ylabel("Range Bins (Distance)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Magnitude (dB)")

    im2 = ax2.imshow(rd_with_mti_db, aspect="auto", origin="lower", cmap="jet")
    ax2.set_title("With MTI Filter (alpha=0.5)\n(Static clutter attenuated by ~15dB)")
    ax2.set_xlabel("Doppler Velocity Bins (Center = 0 m/s)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Magnitude (dB)")

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "mti_comparison.png"), dpi=300)
    plt.close()


def plot_linear_vs_db():
    """Create the linear-versus-logarithmic diagnostic figure."""
    file_path = find_first_raw_radar_file()
    if not file_path:
        return

    raw_data = np.load(file_path) / 4095.0
    _, batch_nsensors, batch_nchirps, batch_nsamples = raw_data.shape
    doppler = DopplerAlgo(batch_nsamples, batch_nchirps, batch_nsensors, mti_alpha=0.5)

    best_frame, maps = find_best_frame(raw_data, doppler)
    rd_map = maps[best_frame]

    rd_linear = np.abs(rd_map)
    rd_db = 10 * np.log10(rd_linear + EPSILON)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    im1 = ax1.imshow(rd_linear, aspect="auto", origin="lower", cmap="viridis")
    ax1.set_title("Linear Scale\n(Only the peak reflection is prominent)")
    ax1.set_xlabel("Doppler Velocity Bins")
    ax1.set_ylabel("Range Bins")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Amplitude (Linear)")

    im2 = ax2.imshow(rd_db, aspect="auto", origin="lower", cmap="viridis")
    ax2.set_title("Decibel (dB) Scale\n(Full structural footprint and noise floor visible)")
    ax2.set_xlabel("Doppler Velocity Bins")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Magnitude (dB)")

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "linear_vs_db.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_mti_effect()
    plot_linear_vs_db()