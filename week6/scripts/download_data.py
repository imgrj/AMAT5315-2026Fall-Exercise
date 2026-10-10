import os
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUTS_DIR = BASE_DIR / "inputs"
MNIST_DIR = INPUTS_DIR / "mnist"
CIFAR_DIR = INPUTS_DIR / "cifar10"

MNIST_DIR.mkdir(parents=True, exist_ok=True)
CIFAR_DIR.mkdir(parents=True, exist_ok=True)

mnist_base = "https://storage.googleapis.com/cvdf-datasets/mnist"
mnist_files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

print("Downloading MNIST datasets...")
for fname in mnist_files:
    dest = MNIST_DIR / fname
    if not dest.exists():
        url = f"{mnist_base}/{fname}"
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"Already exists: {dest}")

print("\nDownloading CIFAR-10 binary dataset...")
cifar_url = "https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz"
cifar_dest = CIFAR_DIR / "cifar-10-binary.tar.gz"
if not cifar_dest.exists():
    try:
        print(f"Downloading {cifar_url} -> {cifar_dest}")
        urllib.request.urlretrieve(cifar_url, cifar_dest)
    except Exception as e:
        alt_url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
        print(f"Retrying with {alt_url} -> {cifar_dest}")
        urllib.request.urlretrieve(alt_url, cifar_dest)
else:
    print(f"Already exists: {cifar_dest}")

print("Dataset download completed successfully!")
