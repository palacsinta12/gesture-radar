"""Create a figure-grid showing a representative RTM, DTM, and ATM sample.

The script reads the preprocessed NumPy dataset in the repository data folder
and writes a 3-column visualization grid into the public assets directory.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from src.config import DATA_DIR

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def plot_gesture_samples():
    """Generate a 3-column sample figure for each gesture class in the dataset."""
    full_path = os.path.join(DATA_DIR, "alldata.npz")
    if not os.path.exists(full_path):
        print(f"Dataset not found at {full_path}. Run the data extraction step first.")
        return

    dataset = np.load(full_path)
    data, target, labels = dataset["data"], dataset["target"], dataset["labels"]

    fig, axs = plt.subplots(nrows=len(labels), ncols=3, figsize=(10, 8))

    for row, cls_name in enumerate(labels):
        cls_idx = list(labels).index(cls_name)
        all_indices = np.where(target == cls_idx)[0]
        rand_idx = np.random.choice(all_indices)
        gesture_data = data[rand_idx]

        for col in range(3):
            img = gesture_data[..., col]
            axs[row, col].imshow(
                img.T,
                aspect="auto",
                origin="lower",
                cmap="gray_r",
                vmin=np.percentile(img, 5),
                vmax=np.percentile(img, 95),
            )

            if row == 0:
                axs[row, col].set_title(["RTM (Range-Time)", "DTM (Doppler-Time)", "ATM (Azimuth-Time)"][col])
            if col == 0:
                axs[row, col].set_ylabel(f"{cls_name}\nRange bins")
            if row == len(labels) - 1:
                axs[row, col].set_xlabel("Frame")

    plt.tight_layout()
    save_path = os.path.join(ASSETS_DIR, "gesture_samples.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved gesture samples plot to {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_gesture_samples()