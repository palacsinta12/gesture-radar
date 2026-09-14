"""Create a Range-Doppler animation from the private FMCW raw recording.

This script is a demo generator for the project README and portfolio visuals.
It searches the local data tree for a raw radar sample and writes the final
GIF into the repository assets folder.
"""

import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from src.config import DATA_DIR, EPSILON
from src.helpers.DopplerAlgo import DopplerAlgo

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def find_push_pull_file():
    """Return the first push-pull radar recording, falling back to any recording."""
    for dirpath, _, filenames in os.walk(DATA_DIR):
        if "radar.npy" in filenames and "push_pull" in dirpath.lower():
            return os.path.join(dirpath, "radar.npy")

    for dirpath, _, filenames in os.walk(DATA_DIR):
        if "radar.npy" in filenames:
            return os.path.join(dirpath, "radar.npy")

    return None


def generate_gif():
    """Render a 2-D range-doppler animation as a local GIF asset."""
    file_path = find_push_pull_file()
    if not file_path:
        print("No raw radar data found.")
        return

    print(f"Generating animation from: {file_path}")
    raw_data = np.load(file_path) / 4095.0
    _, n_sensors, n_chirps, n_samples = raw_data.shape

    start_frame = 32
    end_frame = min(64, raw_data.shape[0])

    doppler_algo = DopplerAlgo(n_samples, n_chirps, n_sensors, mti_alpha=0.5)

    rd_maps = []
    for f in range(end_frame):
        rd = doppler_algo.compute_doppler_map(raw_data[f, 0], 0)
        if f >= start_frame:
            rd_db = 10 * np.log10(np.abs(rd) + EPSILON)
            rd_maps.append(rd_db)

    fig, ax = plt.subplots(figsize=(6, 5))

    vmin = np.percentile(rd_maps, 5)
    vmax = np.percentile(rd_maps, 99.5)

    cax = ax.imshow(rd_maps[0].T, aspect="auto", origin="lower", cmap="jet", vmin=vmin, vmax=vmax)

    ax.set_xlabel("Doppler Velocity Bins")
    ax.set_ylabel("Range Bins")
    fig.colorbar(cax, ax=ax, label="Magnitude (dB)")

    def update(frame_idx):
        cax.set_data(rd_maps[frame_idx].T)
        ax.set_title(f"FMCW Range-Doppler Map (Push-Pull)\nFrame: {frame_idx + 1}/{len(rd_maps)}")
        return [cax]

    ani = animation.FuncAnimation(fig, update, frames=len(rd_maps), interval=100, blit=True)

    save_path = os.path.join(ASSETS_DIR, "range_doppler_animation.gif")
    print(f"Saving animation to {save_path} (this might take a few seconds)...")
    ani.save(save_path, writer="pillow", fps=10)
    plt.close()
    print("Animation saved successfully!")


if __name__ == "__main__":
    generate_gif()