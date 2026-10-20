"""Export reference noise predictions from the pinned diffusers model.

Run this with the pinned PyTorch reference environment:

    uv run --python 3.12 --with torch --with diffusers==0.40.0 \
        python scripts/export_reference_unet.py

The JAX port in ``src/unet.py`` is then compared against
``artifacts/part3/reference_unet.npz`` inside ``scripts/part3_cifar.py`` and by
the acceptance test for Part 3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import UNet2DModel

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "inputs" / "cifar-model"
OUT = ROOT / "artifacts" / "part3" / "reference_unet.npz"
LEVELS = [0, 1, 10, 100, 500, 999]


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = UNet2DModel.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float32)
    model.eval()
    rng = np.random.default_rng(20260707)
    x = rng.standard_normal((len(LEVELS), 3, 32, 32)).astype(np.float32)
    t = np.asarray(LEVELS, dtype=np.int64)
    with torch.no_grad():
        eps = model(torch.from_numpy(x), torch.from_numpy(t)).sample.numpy()
    np.savez(OUT, x=x, t=t, eps=eps)
    count = int(sum(p.numel() for p in model.parameters()))
    print(json.dumps({"config": dict(model.config), "parameters": count, "levels": LEVELS}, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
