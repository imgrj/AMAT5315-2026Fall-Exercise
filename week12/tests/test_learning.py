"""Parts 3 and 4: the recorded labels, the trained network, the JAX/Rust parity
and the explored nodes on unseen spin glasses.

The specification is `train_branch.design.toml` plus `benchmark.eval` in
`branchopt.design.toml`.
"""
from __future__ import annotations

import numpy as np
import pytest

FEATURES = 12
HIDDEN = 48


def forward(model: dict, phi: np.ndarray) -> np.ndarray:
    """`w2 . tanh(W1 phi + b1) + b2` with the saved standardization."""
    mean = np.asarray(model["mean"], dtype=float)
    std = np.maximum(np.asarray(model["std"], dtype=float), 1e-6)
    w1 = np.asarray(model["w1"], dtype=float).reshape(HIDDEN, FEATURES)
    b1 = np.asarray(model["b1"], dtype=float)
    w2 = np.asarray(model["w2"], dtype=float)
    z = (np.asarray(phi, dtype=float) - mean) / std
    return np.tanh(z @ w1.T + b1) @ w2 + model["b2"]


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=float) - np.max(values)
    exps = np.exp(shifted)
    return exps / exps.sum()


def test_dataset_splits_by_instance(dataset):
    """36 training graphs, 12 validation graphs, no instance in both."""
    train = set(dataset["train_instances"])
    validation = set(dataset["validation_instances"])
    assert train & validation == set()
    assert len(train) == 36
    assert len(validation) == 12
    for node in dataset["nodes"]:
        assert (node["instance"] in train) != (node["instance"] in validation)
        assert len(node["candidates"]) >= 2
        for cand in node["candidates"]:
            assert len(cand["features"]) == FEATURES


def test_no_instance_contributes_more_than_eighty_nodes(dataset):
    counts: dict[int, int] = {}
    for node in dataset["nodes"]:
        counts[node["instance"]] = counts.get(node["instance"], 0) + 1
    assert max(counts.values()) <= 80


def test_labels_record_the_strong_branching_scores(dataset):
    """The product score of every candidate is the score of Eq. 7."""
    for node in dataset["nodes"]:
        scores = []
        for cand in node["candidates"]:
            expected = max(cand["gain0"], 1e-6) * max(cand["gain1"], 1e-6)
            assert cand["score"] == pytest.approx(expected, rel=1e-9)
            scores.append(cand["score"])
        best = max(scores)
        assert set(node["best"]) == {
            cand["spin"] for cand, s in zip(node["candidates"], scores) if s >= best - 1e-9 * max(1.0, abs(best))
        }
        # The strongest gain is a genuine lower-bound rise or the pruned marker.
        assert all(
            cand["gain0"] >= 0.0 and cand["gain1"] >= 0.0 for cand in node["candidates"]
        )


def test_training_history_and_checkpoint(model):
    history = model["history"]
    assert len(history["epoch"]) == 120
    assert history["epoch"][0] == 1 and history["epoch"][-1] == 120
    assert np.all(np.isfinite(history["train_ce"]))
    assert np.all(np.isfinite(history["val_ce"]))
    assert history["train_ce"][-1] < history["train_ce"][0]
    assert history["val_ce"][-1] < history["val_ce"][0]

    epochs = np.asarray(history["epoch"])
    val = np.asarray(history["val_ce"])
    checkpoints = (epochs == 1) | (epochs % 5 == 0)
    best = int(np.argmin(np.where(checkpoints, val, np.inf)))
    assert model["best_epoch"] == int(epochs[best])
    assert model["best_val_ce"] == pytest.approx(float(val[best]), abs=1e-9)


def test_numpy_forward_reproduces_the_saved_probes(model):
    """The exported weights and the stored probabilities agree to machine precision."""
    assert len(model["probes"]) == 6
    for probe in model["probes"]:
        phi = np.asarray(probe["features"], dtype=float)
        logits = forward(model, phi)
        assert np.allclose(logits, probe["logits"], atol=1e-9, rtol=1e-9)
        probs = softmax(logits)
        assert np.allclose(probs, probe["probs"], atol=1e-9, rtol=1e-9)
        assert probs.sum() == pytest.approx(1.0, abs=1e-12)


def test_rust_matches_jax_on_the_probes(eval_test):
    """Exactness by separation needs the two implementations to agree."""
    assert eval_test["model"]["probe_probability_max_diff"] < 1e-4
    assert eval_test["model"]["probe_logit_max_diff"] < 1e-6
    assert len(eval_test["model"]) > 0


def test_every_rule_stays_exact(eval_test, eval_transfer):
    for run in (eval_test, eval_transfer):
        assert run["summary"]["largest_highs_error"] < 1e-6
        for inst in run["instances"]:
            for name, record in inst["policies"].items():
                assert record["error"] < 1e-6, (run["parameters"]["spins"], inst["id"], name)


def test_non_strong_rules_solve_one_lp_per_node(eval_test, eval_transfer):
    """No hidden lookahead: the network only chooses a spin."""
    for run in (eval_test, eval_transfer):
        for name, record in run["summary"]["policies"].items():
            if name == "strong":
                continue
            for inst in run["instances"]:
                assert inst["policies"][name]["lp_solves"] == inst["policies"][name]["nodes"]


def test_learned_rule_explores_fewer_nodes(eval_test, eval_transfer):
    """Fewer nodes than most-frac on 22-spin and on the larger 30-spin graphs."""
    for run in (eval_test, eval_transfer):
        policies = {k: v["mean_nodes"] for k, v in run["summary"]["policies"].items()}
        assert policies["neural"] < policies["most-frac"]
        assert policies["root-neural"] < policies["most-frac"]
        assert policies["strong"] < policies["most-frac"]


def test_size_transfer_keeps_the_tree_small(eval_transfer):
    """The 22-spin network still cuts most of the 30-spin tree."""
    policies = {k: v["mean_nodes"] for k, v in eval_transfer["summary"]["policies"].items()}
    assert policies["most-frac"] / policies["neural"] > 2.0
    assert policies["most-frac"] - policies["neural"] > 50.0


def test_first_splits_are_recorded(eval_test):
    """The instance where the network saves most keeps both opening splits."""
    inst = eval_test["summary"]["largest_neural_saving_instance"]
    record = eval_test["instances"][inst]
    assert len(record["policies"]["most-frac"]["first_splits"]) >= 2
    assert len(record["policies"]["neural"]["first_splits"]) >= 2
    assert record["policies"]["neural"]["nodes"] < record["policies"]["most-frac"]["nodes"]
