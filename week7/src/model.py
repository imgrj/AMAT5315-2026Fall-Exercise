"""Time-conditioned fully connected predictor (learning sheet Eq. 8-11).

The time encoding is fixed by ``diffusion.toml`` and ``flow.toml``: for
``j = 0, ..., 15`` the features are sine and cosine of

    value * time_frequency_base ** (-j / 16),

with ``value = t - 1`` for the noise predictor (levels ``t = 1, ..., 1000``) and
``value = 1000 * s`` for the velocity predictor (``s in [0, 1]``).  The 32
features are appended to the point's coordinates; three hidden layers of 128
SiLU units follow, then a linear output with two coordinates.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

TIME_DIM = 32
HIDDEN = 128
DEPTH = 3
FREQUENCY_BASE = 10_000.0
FLOW_TIME_SCALE = 1000.0


def frequencies() -> np.ndarray:
    """``omega_j = 10000 ** (-j / 16)``, ``j = 0, ..., 15`` (Eq. 8)."""
    j = np.arange(TIME_DIM // 2, dtype=np.float64)
    return FREQUENCY_BASE ** (-j / (TIME_DIM // 2))


def time_features(value) -> jnp.ndarray:
    """Eq. (9): all 16 sines followed by all 16 cosines of ``value * omega_j``."""
    value = jnp.asarray(value, dtype=jnp.float32).reshape(-1)
    omega = jnp.asarray(frequencies(), dtype=jnp.float32)
    args = value[:, None] * omega[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


def init_predictor(key, in_dim: int = 2, out_dim: int = 2, hidden: int = HIDDEN, depth: int = DEPTH):
    """Zero biases and normal weights with standard deviation ``1/sqrt(input width)``."""
    sizes = [in_dim + TIME_DIM] + [hidden] * depth + [out_dim]
    keys = jax.random.split(key, len(sizes) - 1)
    weights, biases = [], []
    for k, fan_in, fan_out in zip(keys, sizes[:-1], sizes[1:]):
        weights.append(jax.random.normal(k, (fan_in, fan_out), dtype=jnp.float32) / np.sqrt(fan_in))
        biases.append(jnp.zeros((fan_out,), dtype=jnp.float32))
    return {"weights": weights, "biases": biases}


def apply(params, x, encoded_time) -> jnp.ndarray:
    """Evaluate the network for a batch; ``encoded_time`` is ``t - 1`` or ``1000 s``."""
    x = jnp.asarray(x, dtype=jnp.float32)
    encoded = jnp.asarray(encoded_time, dtype=jnp.float32).reshape(-1)
    if encoded.shape[0] == 1 and x.shape[0] != 1:
        encoded = jnp.broadcast_to(encoded, (x.shape[0],))
    h = jnp.concatenate([x, time_features(encoded)], axis=-1)
    n_layers = len(params["weights"])
    for i in range(n_layers - 1):
        h = jax.nn.silu(h @ params["weights"][i] + params["biases"][i])
    return h @ params["weights"][-1] + params["biases"][-1]


def noise_predictor(params):
    """Wrap the MLP as ``eps_theta(x_t, t)``; the network sees ``t - 1``."""

    def predict(x_t, t):
        return apply(params, x_t, jnp.asarray(t, dtype=jnp.float32) - 1.0)

    return predict


def velocity_predictor(params, time_scale: float = FLOW_TIME_SCALE):
    """Wrap the MLP as ``v_theta(x_s, s)``; the network sees ``1000 * s``."""

    def predict(x_s, s):
        return apply(params, x_s, time_scale * jnp.asarray(s, dtype=jnp.float32))

    return predict


def parameter_count(params) -> int:
    return int(sum(np.size(np.asarray(w)) for w in params["weights"]) + sum(np.size(np.asarray(b)) for b in params["biases"]))


def save_params(path, params):
    np.savez(
        path,
        **{f"w{i}": np.asarray(w) for i, w in enumerate(params["weights"])},
        **{f"b{i}": np.asarray(b) for i, b in enumerate(params["biases"])},
    )


def load_params(path):
    data = np.load(path)
    weights = [jnp.asarray(data[k]) for k in sorted((k for k in data.files if k.startswith("w")), key=lambda s: int(s[1:]))]
    biases = [jnp.asarray(data[k]) for k in sorted((k for k in data.files if k.startswith("b")), key=lambda s: int(s[1:]))]
    return {"weights": weights, "biases": biases}
