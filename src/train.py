import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_STATE,
    TEST_SPLIT,
    DATA_DIR,
)
from src.models.cnn import build_cnn_model

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def plot_history(history, save_path=None):
    """Render and optionally save the training accuracy/loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history.history["accuracy"], label="Train Acc", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Test Acc", linewidth=2)
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Test Loss", linewidth=2)
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Train the FMCW gesture recognition CNN.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--test-split", type=float, default=TEST_SPLIT)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--data-file", type=str, default="alldata.npz")
    return parser.parse_args()


def main():
    args = parse_args()

    full_path = DATA_DIR / args.data_file
    if not full_path.exists():
        raise FileNotFoundError(f"Dataset not found at {full_path}. Run preprocessing first.")

    dataset = np.load(full_path)
    X, y, labels = dataset["data"], dataset["target"], dataset["labels"]
    num_classes = len(labels)

    y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_split,
        random_state=args.random_state,
        stratify=y,
    )

    n_channels = X_train.shape[-1]
    for i in range(n_channels):
        mean_val = np.mean(X_train[..., i])
        std_val = np.std(X_train[..., i])
        X_train[..., i] = (X_train[..., i] - mean_val) / (std_val + 1e-10)
        X_test[..., i] = (X_test[..., i] - mean_val) / (std_val + 1e-10)

    model = build_cnn_model(input_shape=X_train.shape[1:], num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n🎯 Final Test Accuracy: {test_acc * 100:.2f}% | Final Test Loss: {test_loss:.4f}")

    plot_history(history, save_path=ASSETS_DIR / "training_history.png")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "confusion_matrix.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()