"""Imitation learning of Strong Branching for spin-glass Branch-and-Bound.

Trains the small shared per-spin network of `train_branch.design.toml`:

    f_theta(phi) = w2 . tanh(W1 phi + b1) + b2          (12 -> 48 tanh -> 1)

one score per candidate spin, compared with a masked softmax over the
candidates of the same node, and fits it to Strong Branching's choice with a
cross-entropy loss plus a weight penalty.

Run from `week12/`:

    uv run --with jax==0.12.1 python train_branch.py \
        --dataset artifacts/bnb_strong/dataset.json --out artifacts/model.json
"""

from __future__ import annotations

import argparse
import json
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import random  # noqa: E402

FEATURES = 12
HIDDEN = 48
LEARNING_RATE = 4e-3
BATCH = 64
EPOCHS = 120
WEIGHT_DECAY = 2e-4
SEED = 2026
CHECKPOINT_EVERY = 5
N_PROBES = 6


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def split_nodes(data):
    """Split by instance, and refuse a split that shares a spin glass."""
    train_instances = set(int(i) for i in data["train_instances"])
    val_instances = set(int(i) for i in data["validation_instances"])
    overlap = train_instances & val_instances
    if overlap:
        raise SystemExit(f"training and validation share instances {sorted(overlap)}")

    train, val = [], []
    for node in data["nodes"]:
        inst = int(node["instance"])
        if inst in train_instances:
            train.append(node)
        elif inst in val_instances:
            val.append(node)
        else:
            raise SystemExit(f"instance {inst} belongs to neither split")
    return train, val, sorted(train_instances), sorted(val_instances)


def to_arrays(nodes):
    """Pad candidate lists with a 0/1 mask and build the imitation target."""
    k = max(len(node["candidates"]) for node in nodes)
    x = np.zeros((len(nodes), k, FEATURES), dtype=np.float64)
    mask = np.zeros((len(nodes), k), dtype=np.float64)
    y = np.zeros((len(nodes), k), dtype=np.float64)
    for n, node in enumerate(nodes):
        cands = node["candidates"]
        best = set(int(s) for s in node["best"])
        for c, cand in enumerate(cands):
            x[n, c, :] = np.asarray(cand["features"], dtype=np.float64)
            mask[n, c] = 1.0
            if int(cand["spin"]) in best:
                y[n, c] = 1.0 / len(best)
    return jnp.asarray(x), jnp.asarray(mask), jnp.asarray(y)


def init_params(key):
    """Glorot-scale normal weights; biases start at zero."""
    k1, k2 = random.split(key)
    w1 = random.normal(k1, (HIDDEN, FEATURES)) * np.sqrt(1.0 / FEATURES)
    w2 = random.normal(k2, (HIDDEN,)) * np.sqrt(1.0 / HIDDEN)
    return {
        "w1": w1,
        "b1": jnp.zeros((HIDDEN,)),
        "w2": w2,
        "b2": jnp.zeros(()),
    }


def logits(params, x):
    """Scores of every candidate at every node: `x` has shape (N, K, 12)."""
    h = jnp.tanh(jnp.einsum("nkf,hf->nkh", x, params["w1"]) + params["b1"])
    return h @ params["w2"] + params["b2"]


def masked_log_softmax(scores, mask):
    neg = jnp.where(mask > 0.0, scores, -1e30)
    m = jnp.max(neg, axis=1, keepdims=True)
    shifted = neg - m
    denom = jnp.exp(shifted) * (mask > 0.0)
    log_z = jnp.log(jnp.sum(denom, axis=1, keepdims=True))
    return shifted - log_z


def cross_entropy(params, x, mask, y):
    logp = masked_log_softmax(logits(params, x), mask)
    return -jnp.sum(y * logp) / jnp.sum(mask[:, 0])


def penalised(params, x, mask, y):
    ce = cross_entropy(params, x, mask, y)
    reg = WEIGHT_DECAY * (jnp.sum(params["w1"] ** 2) + jnp.sum(params["w2"] ** 2))
    return ce + reg


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="artifacts/bnb_strong/dataset.json")
    ap.add_argument("--out", default="artifacts/model.json")
    args = ap.parse_args(argv)

    data = load_dataset(args.dataset)
    train_nodes, val_nodes, train_inst, val_inst = split_nodes(data)
    if not train_nodes or not val_nodes:
        raise SystemExit("both splits must contain nodes")

    xt, mt, yt = to_arrays(train_nodes)
    xv, mv, yv = to_arrays(val_nodes)

    # Training statistics standardize every feature; the floor keeps a constant
    # feature from blowing up.
    flat = np.asarray(xt).reshape(-1, FEATURES)
    keep = np.asarray(mt).reshape(-1) > 0
    mean = flat[keep].mean(axis=0)
    std = np.maximum(flat[keep].std(axis=0), 1e-6)

    def standardize(x):
        return (x - jnp.asarray(mean)) / jnp.asarray(std)

    xt, xv = standardize(xt), standardize(xv)

    params = init_params(random.PRNGKey(SEED))
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    grad_fn = jax.jit(jax.grad(penalised))
    ce_fn = jax.jit(cross_entropy)

    key = random.PRNGKey(SEED + 1)
    n_train = int(xt.shape[0])
    history = {"epoch": [], "train_ce": [], "val_ce": []}
    best = {"val_ce": float("inf"), "epoch": 0, "params": params}

    for epoch in range(1, EPOCHS + 1):
        key, sub = random.split(key)
        order = np.asarray(random.permutation(sub, n_train))
        step = 0
        for start in range(0, n_train, BATCH):
            idx = jnp.asarray(order[start : start + BATCH])
            grads = grad_fn(params, xt[idx], mt[idx], yt[idx])
            step += 1
            m = jax.tree_util.tree_map(lambda a, g: beta1 * a + (1 - beta1) * g, m, grads)
            v = jax.tree_util.tree_map(lambda a, g: beta2 * a + (1 - beta2) * g * g, v, grads)
            mhat = jax.tree_util.tree_map(lambda a: a / (1 - beta1**step), m)
            vhat = jax.tree_util.tree_map(lambda a: a / (1 - beta2**step), v)
            params = jax.tree_util.tree_map(
                lambda p, mm, vv: p - LEARNING_RATE * mm / (jnp.sqrt(vv) + eps),
                params,
                mhat,
                vhat,
            )

        train_ce = float(ce_fn(params, xt, mt, yt))
        val_ce = float(ce_fn(params, xv, mv, yv))
        history["epoch"].append(epoch)
        history["train_ce"].append(train_ce)
        history["val_ce"].append(val_ce)
        if epoch == 1 or epoch % CHECKPOINT_EVERY == 0:
            if val_ce < best["val_ce"] - 1e-12:
                best = {"val_ce": val_ce, "epoch": epoch, "params": params}
        if epoch % 20 == 0 or epoch == 1 or epoch == EPOCHS:
            print(
                f"epoch {epoch:3d}  train {train_ce:.4f}  val {val_ce:.4f}",
                flush=True,
            )

    params = best["params"]
    logits_fn = jax.jit(logits)

    # Six validation probes carry features, logits and probabilities so the Rust
    # solver can prove it computes the same network.
    probes = []
    for node in val_nodes[:N_PROBES]:
        cands = np.asarray([c["features"] for c in node["candidates"]], dtype=np.float64)
        cands_j = jnp.asarray((cands - mean) / std)
        scores = logits_fn(params, cands_j[None, :, :])[0]
        mask = jnp.ones((1, cands.shape[0]))
        probs = jnp.exp(masked_log_softmax(scores[None, :], mask)[0])
        probes.append(
            {
                "instance": int(node["instance"]),
                "node": int(node["node"]),
                "features": cands.tolist(),
                "logits": np.asarray(scores).tolist(),
                "probs": np.asarray(probs).tolist(),
            }
        )

    model = {
        "schema": "branchopt-model-v1",
        "features": FEATURES,
        "hidden": HIDDEN,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "w1": np.asarray(params["w1"]).reshape(-1).tolist(),
        "b1": np.asarray(params["b1"]).reshape(-1).tolist(),
        "w2": np.asarray(params["w2"]).reshape(-1).tolist(),
        "b2": float(np.asarray(params["b2"])),
        "best_epoch": int(best["epoch"]),
        "best_val_ce": float(best["val_ce"]),
        "history": history,
        "train": {
            "nodes": len(train_nodes),
            "instances": train_inst,
        },
        "validation": {
            "nodes": len(val_nodes),
            "instances": val_inst,
        },
        "overlap": len(set(train_inst) & set(val_inst)),
        "optimizer": {
            "name": "Adam",
            "learning_rate": LEARNING_RATE,
            "batch": BATCH,
            "epochs": EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
        },
        "probes": probes,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=1)

    print(
        f"train nodes {len(train_nodes)} from {len(train_inst)} spin glasses; "
        f"validation nodes {len(val_nodes)} from {len(val_inst)} spin glasses; "
        f"overlap {model['overlap']}",
        flush=True,
    )
    print(
        f"best checkpoint epoch {best['epoch']} validation cross-entropy "
        f"{best['val_ce']:.4f}; final train {history['train_ce'][-1]:.4f} "
        f"val {history['val_ce'][-1]:.4f}",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
