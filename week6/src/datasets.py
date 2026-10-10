import gzip
import os
import struct
import tarfile
from pathlib import Path
import numpy as np

def load_mnist(data_dir=None, split="train"):
    """
    Load MNIST images and labels preserving native pixels (28x28, 0-255 uint8).
    split: 'train' (60,000) or 't10k'/'test' (10,000)
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "inputs" / "mnist"
    else:
        data_dir = Path(data_dir)

    split_prefix = "train" if split == "train" else "t10k"
    img_path = data_dir / f"{split_prefix}-images-idx3-ubyte.gz"
    lbl_path = data_dir / f"{split_prefix}-labels-idx1-ubyte.gz"

    with gzip.open(img_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Invalid MNIST image magic: {magic}"
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

    with gzip.open(lbl_path, "rb") as f:
        magic, num_lbl = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Invalid MNIST label magic: {magic}"
        assert num_lbl == num, f"Image count {num} != label count {num_lbl}"
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels

def load_cifar10(data_dir=None, split="train"):
    """
    Load CIFAR-10 images and labels preserving native pixels (32x32x3 RGB, 0-255 uint8).
    split: 'train' (50,000) or 'test' (10,000)
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "inputs" / "cifar10"
    else:
        data_dir = Path(data_dir)

    archive_path = data_dir / "cifar-10-binary.tar.gz"
    all_images = []
    all_labels = []

    with tarfile.open(archive_path, "r:gz") as tar:
        if split == "train":
            batch_names = [f"cifar-10-batches-bin/data_batch_{i}.bin" for i in range(1, 6)]
        else:
            batch_names = ["cifar-10-batches-bin/test_batch.bin"]

        for name in batch_names:
            member = tar.getmember(name)
            f = tar.extractfile(member)
            raw = f.read()
            record_len = 1 + 3072 # 1 byte label + 3072 bytes
            num_records = len(raw) // record_len
            records = np.frombuffer(raw, dtype=np.uint8).reshape(num_records, record_len)
            labels = records[:, 0]
            # Reshape 3072 into (3, 32, 32) then transpose to (32, 32, 3)
            images = records[:, 1:].reshape(num_records, 3, 32, 32).transpose(0, 2, 3, 1)
            all_images.append(images)
            all_labels.append(labels)

    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return images, labels

def binarize(images, threshold=128):
    """Binarize grayscale images at threshold (default 128) into {0, 1}."""
    return (images >= threshold).astype(np.int32)

def to_spins(binary_images):
    """Convert {0, 1} bits into spins {-1, +1} via s_i = 2*v_i - 1."""
    return (2 * binary_images - 1).astype(np.int32)

def class_frequencies(labels, num_classes=10):
    """Compute empirical distribution fraction p_hat(y = c) for each class."""
    counts = np.bincount(labels, minlength=num_classes)
    return counts / float(len(labels))

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
