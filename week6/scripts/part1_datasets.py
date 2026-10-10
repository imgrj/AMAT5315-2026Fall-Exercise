import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.datasets import load_mnist, load_cifar10, class_frequencies, CIFAR10_CLASSES

def main():
    root = Path(__file__).resolve().parent.parent
    evidence_dir = root / "evidence"
    artifacts_dir = root / "artifacts"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=== Part 1: Understand the datasets ===")
    
    # 1. MNIST
    print("Loading MNIST...")
    mnist_train_x, mnist_train_y = load_mnist(split="train")
    mnist_test_x, mnist_test_y = load_mnist(split="test")
    
    print(f"MNIST train: {mnist_train_x.shape}, pixel range: [{mnist_train_x.min()}, {mnist_train_x.max()}]")
    print(f"MNIST test:  {mnist_test_x.shape}, pixel range: [{mnist_test_x.min()}, {mnist_test_x.max()}]")
    assert mnist_train_x.shape == (60000, 28, 28)
    assert mnist_test_x.shape == (10000, 28, 28)

    mnist_train_freq = class_frequencies(mnist_train_y)
    mnist_test_freq = class_frequencies(mnist_test_y)

    # Verify digit 1 most common, digit 5 least common
    most_common_train = int(np.argmax(mnist_train_freq))
    least_common_train = int(np.argmin(mnist_train_freq))
    most_common_test = int(np.argmax(mnist_test_freq))
    least_common_test = int(np.argmin(mnist_test_freq))
    print(f"MNIST train: most common digit={most_common_train} ({mnist_train_freq[most_common_train]:.4f}), least common digit={least_common_train} ({mnist_train_freq[least_common_train]:.4f})")
    print(f"MNIST test:  most common digit={most_common_test} ({mnist_test_freq[most_common_test]:.4f}), least common digit={least_common_test} ({mnist_test_freq[least_common_test]:.4f})")

    # Plot MNIST: First two training images per class + Class fractions
    fig = plt.figure(figsize=(10, 6), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.35)
    
    # Subgrid for images (2 rows x 10 cols)
    gs_img = gs[0].subgridspec(2, 10, wspace=0.08, hspace=0.08)
    for c in range(10):
        c_idxs = np.where(mnist_train_y == c)[0][:2]
        for row_idx, idx in enumerate(c_idxs):
            ax = fig.add_subplot(gs_img[row_idx, c])
            ax.imshow(mnist_train_x[idx], cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(str(c), fontsize=10, pad=3)
    fig.text(0.5, 0.96, "MNIST: first two training images per class\n60,000 training · 10,000 test · 28 × 28 grayscale",
             ha="center", va="top", fontsize=11, fontweight="bold")

    # Subgrid for class fractions
    ax_bar = fig.add_subplot(gs[1])
    x_indices = np.arange(10)
    width = 0.35
    ax_bar.bar(x_indices - width/2, mnist_train_freq, width, label="Training", color="#1f77b4")
    ax_bar.bar(x_indices + width/2, mnist_test_freq, width, label="Test", color="#ff7f0e")
    ax_bar.set_xlabel("Digit class", fontsize=10)
    ax_bar.set_ylabel("Class fraction", fontsize=10)
    ax_bar.set_xticks(x_indices)
    ax_bar.set_xticklabels(x_indices)
    ax_bar.set_ylim(0.0, 0.15)
    ax_bar.legend(loc="upper right", frameon=True)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)

    mnist_fig_path = evidence_dir / "mnist-dataset.png"
    plt.savefig(mnist_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved MNIST evidence to {mnist_fig_path}")

    # 2. CIFAR-10
    print("\nLoading CIFAR-10...")
    cifar_train_x, cifar_train_y = load_cifar10(split="train")
    cifar_test_x, cifar_test_y = load_cifar10(split="test")
    print(f"CIFAR train: {cifar_train_x.shape}, pixel range: [{cifar_train_x.min()}, {cifar_train_x.max()}]")
    print(f"CIFAR test:  {cifar_test_x.shape}, pixel range: [{cifar_test_x.min()}, {cifar_test_x.max()}]")
    assert cifar_train_x.shape == (50000, 32, 32, 3)
    assert cifar_test_x.shape == (10000, 32, 32, 3)

    cifar_train_freq = class_frequencies(cifar_train_y)
    cifar_test_freq = class_frequencies(cifar_test_y)
    print("CIFAR-10 train freq:", cifar_train_freq)
    print("CIFAR-10 test freq:", cifar_test_freq)
    assert np.allclose(cifar_train_freq, 0.1), "CIFAR-10 train classes must each be exactly 0.10"
    assert np.allclose(cifar_test_freq, 0.1), "CIFAR-10 test classes must each be exactly 0.10"

    # Plot CIFAR-10: First two training images per class + Class fractions
    fig = plt.figure(figsize=(11, 6.5), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.35)

    gs_img = gs[0].subgridspec(2, 10, wspace=0.08, hspace=0.08)
    for c in range(10):
        c_idxs = np.where(cifar_train_y == c)[0][:2]
        for row_idx, idx in enumerate(c_idxs):
            ax = fig.add_subplot(gs_img[row_idx, c])
            ax.imshow(cifar_train_x[idx], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(CIFAR10_CLASSES[c], fontsize=8, pad=3)
    fig.text(0.5, 0.96, "CIFAR-10: first two training images per class\n50,000 training · 10,000 test · 32 × 32 RGB",
             ha="center", va="top", fontsize=11, fontweight="bold")

    ax_bar = fig.add_subplot(gs[1])
    ax_bar.bar(x_indices - width/2, cifar_train_freq, width, label="Training", color="#1f77b4")
    ax_bar.bar(x_indices + width/2, cifar_test_freq, width, label="Test", color="#ff7f0e")
    ax_bar.set_xlabel("Object class", fontsize=10)
    ax_bar.set_ylabel("Class fraction", fontsize=10)
    ax_bar.set_xticks(x_indices)
    ax_bar.set_xticklabels(CIFAR10_CLASSES, rotation=20, ha="right", fontsize=9)
    ax_bar.set_ylim(0.0, 0.15)
    ax_bar.legend(loc="upper right", frameon=True)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)

    cifar_fig_path = evidence_dir / "cifar-dataset.png"
    plt.savefig(cifar_fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved CIFAR-10 evidence to {cifar_fig_path}")

    # Summary JSON in artifacts
    summary = {
        "mnist": {
            "train_samples": int(mnist_train_x.shape[0]),
            "test_samples": int(mnist_test_x.shape[0]),
            "resolution": [28, 28],
            "channels": 1,
            "most_common_train_digit": most_common_train,
            "least_common_train_digit": least_common_train,
            "train_class_frequencies": mnist_train_freq.tolist(),
            "test_class_frequencies": mnist_test_freq.tolist(),
        },
        "cifar10": {
            "train_samples": int(cifar_train_x.shape[0]),
            "test_samples": int(cifar_test_x.shape[0]),
            "resolution": [32, 32],
            "channels": 3,
            "classes": CIFAR10_CLASSES,
            "balanced_fraction": 0.1,
            "train_class_frequencies": cifar_train_freq.tolist(),
            "test_class_frequencies": cifar_test_freq.tolist(),
        }
    }
    with open(artifacts_dir / "part1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Part 1 verification complete!")

if __name__ == "__main__":
    main()
