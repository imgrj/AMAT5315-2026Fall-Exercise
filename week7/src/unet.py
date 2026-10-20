"""JAX port of the diffusers v0.40.0 ``UNet2DModel`` for google/ddpm-cifar10-32.

The downloaded configuration fixes the architecture: a U-Net with side lengths
32 -> 16 -> 8 -> 4 -> 8 -> 16 -> 32, channel counts 128, 256, 256, 256,
two residual blocks per down stage, three per up stage, attention at side
length 16 on both paths and in the 4x4 middle, and a 512-wide time embedding
derived from 128 sinusoidal features.

Weights are read from the pinned ``safetensors`` file.  Convolution kernels are
transposed from PyTorch's OIHW to JAX's HWIO and linear weights from (out, in)
to (in, out) once, at load time.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

NUM_CLASSES = 10
GROUPS = 32
NORM_EPS = 1e-6
TIME_FEATURES = 128
TIME_EMBED = 512


# ------------------------------------------------------------------ loading ---


def load_safetensors(path) -> dict:
    """Read a safetensors file into ``{name: np.ndarray}``."""
    path = Path(path)
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        data = handle.read()
    arrays = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        dtype = np.dtype("float32" if info["dtype"] == "F32" else info["dtype"].lower())
        arrays[name] = np.frombuffer(data[begin:end], dtype=dtype).reshape(info["shape"]).copy()
    return arrays


def load_unet(model_dir) -> dict:
    """Load and transpose the published weights for the JAX forward pass."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    raw = load_safetensors(model_dir / "diffusion_pytorch_model.safetensors")
    params = {}
    for name, value in raw.items():
        if name.endswith(".weight") and value.ndim == 4:  # conv: OIHW -> HWIO
            params[name] = jnp.asarray(np.transpose(value, (2, 3, 1, 0)))
        elif name.endswith(".weight") and value.ndim == 2:  # linear: (out, in) -> (in, out)
            params[name] = jnp.asarray(value.T)
        else:
            params[name] = jnp.asarray(value)
    return {"params": params, "config": config}


def parameter_count(params) -> int:
    return int(sum(int(np.prod(v.shape)) for v in params.values()))


# ------------------------------------------------------------------ helpers ---


def _silu(x):
    return x * jax.nn.sigmoid(x)


def _linear(x, params, prefix):
    return x @ params[prefix + ".weight"] + params[prefix + ".bias"]


def _conv(x, params, prefix, stride: int = 1, padding: str = "SAME"):
    weight = params[prefix + ".weight"]
    out = jax.lax.conv_general_dilated(
        x,
        weight,
        window_strides=(stride, stride),
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return out + params[prefix + ".bias"]


def _group_norm(x, params, prefix, groups: int = GROUPS, eps: float = NORM_EPS):
    n, h, w, c = x.shape
    grouped = x.reshape(n, h, w, groups, c // groups)
    mean = jnp.mean(grouped, axis=(1, 2, 4), keepdims=True)
    var = jnp.var(grouped, axis=(1, 2, 4), keepdims=True)
    normalised = (grouped - mean) / jnp.sqrt(var + eps)
    normalised = normalised.reshape(n, h, w, c)
    return normalised * params[prefix + ".weight"] + params[prefix + ".bias"]


def time_embedding(levels) -> jnp.ndarray:
    """Sinusoidal features (direction sin, cos) of the model's time indices."""
    levels = jnp.asarray(levels, dtype=jnp.float32).reshape(-1)
    half = TIME_FEATURES // 2
    exponent = -np.log(10000.0) * np.arange(half, dtype=np.float64) / (half - 1.0)
    frequencies = jnp.asarray(np.exp(exponent), dtype=jnp.float32)
    args = levels[:, None] * frequencies[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


def _time_mlp(params, levels):
    emb = time_embedding(levels)
    emb = _linear(emb, params, "time_embedding.linear_1")
    emb = _silu(emb)
    return _linear(emb, params, "time_embedding.linear_2")


def _attention(x, params, prefix, heads: int, dim_head: int):
    n, h, w, c = x.shape
    residual = x
    normed = _group_norm(x, params, prefix + ".group_norm")
    flat = normed.reshape(n, h * w, c)

    def project(name):
        value = _linear(flat, params, prefix + "." + name)
        return value.reshape(n, h * w, heads, dim_head).transpose(0, 2, 1, 3)

    query, key, value = project("query"), project("key"), project("value")
    logits = jnp.einsum("nhqd,nhkd->nhqk", query, key) * (dim_head**-0.5)
    weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    attended = jnp.einsum("nhqk,nhkd->nhqd", weights, value)
    attended = attended.transpose(0, 2, 1, 3).reshape(n, h * w, c)
    attended = _linear(attended, params, prefix + ".proj_attn").reshape(n, h, w, c)
    return residual + attended


def _resnet(x, params, prefix, temb):
    hidden = _group_norm(x, params, prefix + ".norm1")
    hidden = _silu(hidden)
    hidden = _conv(hidden, params, prefix + ".conv1", padding="SAME")
    hidden = hidden + _linear(_silu(temb), params, prefix + ".time_emb_proj")[:, None, None, :]
    hidden = _group_norm(hidden, params, prefix + ".norm2")
    hidden = _silu(hidden)
    hidden = _conv(hidden, params, prefix + ".conv2", padding="SAME")
    if prefix + ".conv_shortcut.weight" in params:
        x = _conv(x, params, prefix + ".conv_shortcut", padding="VALID")
    return x + hidden


def _downsample(x, params, prefix):
    x = jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
    return _conv(x, params, prefix + ".conv", stride=2, padding="VALID")


def _upsample(x, params, prefix):
    x = jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)
    return _conv(x, params, prefix + ".conv", padding="SAME")


# -------------------------------------------------------------------- model ---


def predict_eps(loaded, x, model_index):
    """One noise prediction for a batch of ``32 x 32 x 3`` images in NHWC.

    ``model_index`` is the model's own time index (the sampler passes ``t - 1``).
    """
    params = loaded["params"]
    config = loaded["config"]
    channels = list(config["block_out_channels"])
    down_types = list(config["down_block_types"])
    up_types = list(config["up_block_types"])
    layers = int(config["layers_per_block"])
    attention_head_dim = config.get("attention_head_dim")
    x = jnp.asarray(x, dtype=jnp.float32)
    temb = _time_mlp(params, model_index)

    output = _conv(x, params, "conv_in", padding="SAME")
    skips = [output]

    for i, block_type in enumerate(down_types):
        out_channels = channels[i]
        heads = 1 if attention_head_dim is None else out_channels // attention_head_dim
        dim_head = out_channels if attention_head_dim is None else attention_head_dim
        for j in range(layers):
            output = _resnet(output, params, f"down_blocks.{i}.resnets.{j}", temb)
            if block_type.startswith("Attn"):
                output = _attention(output, params, f"down_blocks.{i}.attentions.{j}", heads, dim_head)
            skips.append(output)
        if i < len(down_types) - 1:
            output = _downsample(output, params, f"down_blocks.{i}.downsamplers.0")
            skips.append(output)

    output = _resnet(output, params, "mid_block.resnets.0", temb)
    output = _attention(output, params, "mid_block.attentions.0", 1, channels[-1])
    output = _resnet(output, params, "mid_block.resnets.1", temb)

    reversed_channels = list(reversed(channels))
    for i, block_type in enumerate(up_types):
        out_channels = reversed_channels[i]
        in_channels = reversed_channels[min(i + 1, len(channels) - 1)]
        heads = 1 if attention_head_dim is None else out_channels // attention_head_dim
        dim_head = out_channels if attention_head_dim is None else attention_head_dim
        for j in range(layers + 1):
            skip = skips.pop()
            output = jnp.concatenate([output, skip], axis=-1)
            output = _resnet(output, params, f"up_blocks.{i}.resnets.{j}", temb)
            if block_type.startswith("Attn"):
                output = _attention(output, params, f"up_blocks.{i}.attentions.{j}", heads, dim_head)
        if i < len(up_types) - 1:
            output = _upsample(output, params, f"up_blocks.{i}.upsamplers.0")

    output = _group_norm(output, params, "conv_norm_out")
    output = _silu(output)
    output = _conv(output, params, "conv_out", padding="SAME")
    return output
