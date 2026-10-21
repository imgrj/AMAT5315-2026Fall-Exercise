"""Verification tests for Week 7 (diffusion and flow matching).

Correctness criteria come from ``week7/SPEC.md`` (restated from the revised
learning sheet), never from the implementation. Two families:

* unit tests check the mathematical contracts (schedule, closed-form noising,
  exact conditional means, DDPM/DDIM updates, flow matching, energy distance)
  and the ``diffusion.toml`` / ``flow.toml`` values;
* acceptance tests read the committed ``evidence/*_metrics.json`` and enforce
  the sheet's tolerances.
"""

from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from src import diffusion, flow, metrics, mixture, model, schedule

STAGES = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000]


# ---------------------------------------------------------------- schedule ---


def test_schedule_is_the_sheet_schedule():
    s = schedule.linear_schedule()
    assert s.betas.shape == (1000,)
    assert np.isclose(float(s.betas[0]), 1e-4, rtol=1e-6)
    assert np.isclose(float(s.betas[-1]), 0.02, rtol=1e-6)
    assert np.all(np.diff(np.asarray(s.betas)) > 0)
    assert 3e-5 < float(s.abars[-1]) < 5e-5  # sheet: about 4e-5


def test_schedule_alphas_are_complementary():
    s = schedule.linear_schedule()
    assert np.allclose(np.asarray(s.alphas) + np.asarray(s.betas), 1.0)
    assert float(s.abars[0]) == 1.0
    assert np.isclose(float(s.abars[10]), float(np.prod(s.alphas[:10])), rtol=1e-6)


# ------------------------------------------------------------ configuration ---


def test_diffusion_toml_matches_the_sheet():
    config = tomllib.loads((ROOT / "diffusion.toml").read_text(encoding="utf-8"))
    assert config["model"]["hidden_widths"] == [128, 128, 128]
    assert config["model"]["activation"] == "SiLU"
    assert config["model"]["time_features"] == 32
    assert config["model"]["time_frequency_base"] == 10_000.0
    assert config["model"]["time_offset"] == -1
    assert config["training"]["updates"] == 25_000
    assert config["training"]["batch_size"] == 512
    assert config["training"]["optimizer"] == "Adam"
    assert config["training"]["learning_rate"] == 0.001
    assert config["training"]["final_learning_rate"] == 0.0
    assert config["training"]["seed"] == 2026
    assert config["training"]["checkpoint_updates"] == STAGES


def test_flow_toml_matches_the_sheet():
    config = tomllib.loads((ROOT / "flow.toml").read_text(encoding="utf-8"))
    assert config["model"]["hidden_widths"] == [128, 128, 128]
    assert config["model"]["time_scale"] == 1000.0
    assert config["model"]["time_frequency_base"] == 10_000.0
    assert config["training"]["updates"] == 25_000
    assert config["training"]["batch_size"] == 512
    assert config["training"]["seed"] == 2029


def test_time_features_follow_equations_8_and_9():
    value = jnp.asarray([0.0, 17.0, 999.0])
    features = np.asarray(model.time_features(value))
    omega = 10_000.0 ** (-np.arange(16) / 16.0)
    expected = np.concatenate(
        [np.sin(np.asarray(value)[:, None] * omega[None, :]), np.cos(np.asarray(value)[:, None] * omega[None, :])],
        axis=-1,
    )
    assert features.shape == (3, 32)
    assert np.allclose(features, expected, atol=1e-5)


def test_noise_and_velocity_predictors_use_the_specified_time_argument():
    params = model.init_predictor(jax.random.PRNGKey(0))
    x = jnp.asarray([[0.3, -0.4]])
    t = 137
    assert np.allclose(
        np.asarray(model.noise_predictor(params)(x, t)),
        np.asarray(model.apply(params, x, jnp.asarray([t - 1.0]))),
        atol=1e-6,
    )
    s = 0.42
    assert np.allclose(
        np.asarray(model.velocity_predictor(params)(x, s)),
        np.asarray(model.apply(params, x, jnp.asarray([1000.0 * s]))),
        atol=1e-6,
    )


def test_initialization_uses_the_sheet_standard_deviation():
    params = model.init_predictor(jax.random.PRNGKey(1))
    for weight in params["weights"]:
        fan_in = int(weight.shape[0])
        assert np.isclose(float(jnp.std(weight)), 1.0 / math.sqrt(fan_in), rtol=0.05)
    for bias in params["biases"]:
        assert np.allclose(np.asarray(bias), 0.0)


# ----------------------------------------------------------------- mixture ---


def test_mixture_exact_samples_have_the_target_moments():
    x = np.asarray(mixture.sample(jax.random.PRNGKey(0), 40000))
    assert x.shape == (40000, 2)
    assert np.allclose(x.mean(axis=0), 0.0, atol=0.05)
    assert np.allclose(x.var(axis=0), 4.59, atol=0.10)  # 3^2/2 + 0.09 per coordinate
    labels = np.asarray(mixture.nearest_component(x))
    populations = np.bincount(labels, minlength=4) / len(labels)
    assert np.all(np.abs(populations - 0.25) < 0.02)


def test_exact_posterior_mean_reduces_to_the_observation_at_low_noise():
    x0 = mixture.sample(jax.random.PRNGKey(1), 4000)
    eps = jax.random.normal(jax.random.PRNGKey(2), x0.shape)
    abar = 0.9999
    x_t = math.sqrt(abar) * x0 + math.sqrt(1 - abar) * eps
    x0_hat = np.asarray(mixture.exact_posterior_mean(x_t, abar))
    err = np.linalg.norm(x0_hat - x_t, axis=-1)
    assert np.median(err) < 0.01 * np.median(np.linalg.norm(x_t, axis=-1))


def test_exact_posterior_mean_contracts_toward_zero_at_high_noise():
    x_t = jax.random.normal(jax.random.PRNGKey(3), (4000, 2))
    x0_hat = np.asarray(mixture.exact_posterior_mean(x_t, 4e-5))
    assert np.mean(np.linalg.norm(x0_hat, axis=-1)) < 0.2


def test_exact_noise_identity_holds():
    x_t = jax.random.normal(jax.random.PRNGKey(4), (2000, 2)) * 2.0
    for abar in (0.999, 0.5, 0.01):
        a = math.sqrt(abar)
        b = math.sqrt(1 - abar)
        lhs = np.asarray(mixture.exact_conditional_noise(x_t, abar))
        rhs = (x_t - a * np.asarray(mixture.exact_posterior_mean(x_t, abar))) / b
        assert np.allclose(lhs, rhs, atol=1e-5)


def test_exact_noise_beats_the_zero_predictor():
    x0 = mixture.sample(jax.random.PRNGKey(5), 8000)
    eps = jax.random.normal(jax.random.PRNGKey(6), x0.shape)
    abar = 0.36
    x_t = math.sqrt(abar) * x0 + math.sqrt(1 - abar) * eps
    pred = mixture.exact_conditional_noise(x_t, abar)
    mse_pred = float(np.mean(np.sum((pred - eps) ** 2, axis=-1)))
    mse_zero = float(np.mean(np.sum(eps**2, axis=-1)))
    assert mse_pred < 0.7 * mse_zero


# --------------------------------------------------------------- diffusion ---


def test_direct_noising_matches_the_closed_form():
    x0 = mixture.sample(jax.random.PRNGKey(7), 20000)
    eps = jax.random.normal(jax.random.PRNGKey(8), x0.shape)
    abar = 0.01
    x_t = np.asarray(diffusion.direct_noise(x0, abar, eps))
    assert np.allclose(x_t.mean(axis=0), 0.0, atol=0.03)
    expected_var = abar * 4.59 + (1 - abar)
    assert np.allclose(np.asarray(x_t.var(axis=0)), expected_var, atol=0.04)


def test_ddpm_last_step_adds_no_noise():
    s = schedule.linear_schedule()
    x = jnp.asarray([[0.5, -0.2]])

    def eps_fn(x_t, t):
        return jnp.zeros_like(x_t)

    out_a = diffusion.ddpm_step(eps_fn, x, s, t=1, key=jax.random.PRNGKey(0))
    out_b = diffusion.ddpm_step(eps_fn, x, s, t=1, key=jax.random.PRNGKey(1))
    assert np.array_equal(np.asarray(out_a), np.asarray(out_b))


def test_ddpm_step_matches_algorithm_2():
    s = schedule.linear_schedule()
    t = 137
    x = jnp.asarray([[1.25, -0.75]])
    eps_hat = jnp.asarray([[0.3, -0.1]])
    key = jax.random.PRNGKey(9)
    out = diffusion.ddpm_step(lambda x_t, tt: eps_hat, x, s, t=t, key=key)
    alpha_t = float(s.alphas[t - 1])
    beta_t = float(s.betas[t - 1])
    abar_t = float(s.abars[t])
    mean = (x - beta_t / math.sqrt(1 - abar_t) * eps_hat) / math.sqrt(alpha_t)
    z = jax.random.normal(key, x.shape)
    assert np.allclose(np.asarray(out), np.asarray(mean + math.sqrt(beta_t) * z), atol=1e-5)


def test_ddpm_skipped_mean_matches_equation_17():
    s = schedule.linear_schedule()
    t, t_next = 42, 40
    x = jnp.asarray([[0.4, 1.1]])
    eps_hat = jnp.asarray([[-0.2, 0.5]])
    mean = diffusion.ddpm_skipped_mean(lambda x_t, tt: eps_hat, x, s, t=t, t_next=t_next)
    a = float(s.abars[t]) / float(s.abars[t_next])
    expected = (x - (1 - a) / math.sqrt(1 - float(s.abars[t])) * eps_hat) / math.sqrt(a)
    assert np.allclose(np.asarray(mean), np.asarray(expected), atol=1e-5)


def test_adjacent_ddpm_jump_reduces_to_part_1():
    s = schedule.linear_schedule()
    t = 137
    x = jnp.asarray([[0.4, 1.1]])
    eps_hat = jnp.asarray([[-0.2, 0.5]])
    mean = diffusion.ddpm_skipped_mean(lambda x_t, tt: eps_hat, x, s, t=t, t_next=t - 1)
    expected = (x - float(s.betas[t - 1]) / math.sqrt(1 - float(s.abars[t])) * eps_hat) / math.sqrt(
        float(s.alphas[t - 1])
    )
    assert np.allclose(np.asarray(mean), np.asarray(expected), atol=1e-5)


def test_ddpm_skipped_noise_has_variance_one_minus_a():
    s = schedule.linear_schedule()
    t, t_next = 300, 260
    x = jnp.zeros((8000, 2))
    out = diffusion.ddpm_skipped_step(lambda x_t, tt: jnp.zeros_like(x_t), x, s, t=t, t_next=t_next,
                                      key=jax.random.PRNGKey(21))
    a = float(s.abars[t]) / float(s.abars[t_next])
    assert np.allclose(np.asarray(out).var(axis=0), 1 - a, atol=0.05)


def test_ddim_is_deterministic_and_matches_tweedie_with_one_step():
    s = schedule.linear_schedule()
    x_t = jnp.asarray([[2.0, 0.5], [-1.0, 2.5]])
    eps_hat = jnp.asarray([[0.1, -0.2], [0.3, 0.4]])
    eps_fn = lambda x, t: eps_hat
    a = diffusion.ddim_step(eps_fn, x_t, s, t=500, t_next=0)
    b = diffusion.ddim_step(eps_fn, x_t, s, t=500, t_next=0)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    abar_t = float(s.abars[500])
    x0_hat = (x_t - math.sqrt(1 - abar_t) * eps_hat) / math.sqrt(abar_t)
    assert np.allclose(np.asarray(a), np.asarray(x0_hat), atol=1e-5)


def test_sampler_levels_are_evenly_spaced_and_end_at_zero():
    for steps in (1, 2, 5, 10, 20, 50, 100, 1000):
        levels = diffusion.even_levels(steps, T=1000)
        assert levels[0] == 1000
        assert levels[-1] == 0
        assert len(levels) == steps + 1
        assert np.all(np.diff(levels) < 0)


# -------------------------------------------------------------------- flow ---


def test_exact_flow_velocity_limits():
    x = jax.random.normal(jax.random.PRNGKey(10), (2000, 2)) * 1.5
    v1 = np.asarray(flow.exact_marginal_velocity(x, s=1.0))
    assert np.allclose(v1, x, atol=1e-4)


def test_flow_euler_is_deterministic():
    def velocity_fn(x, s):
        return -x

    x1 = jnp.asarray([[0.3, 0.4]])
    steps = 10
    a = flow.euler_sample(velocity_fn, x1, steps=steps)
    b = flow.euler_sample(velocity_fn, x1, steps=steps)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert np.allclose(np.asarray(a), (1 + 1 / steps) ** steps * np.asarray(x1), atol=1e-6)


# ----------------------------------------------------------------- metrics ---


def test_energy_distance_is_zero_for_identical_sets_and_positive_for_shifted():
    x = jax.random.normal(jax.random.PRNGKey(11), (600, 2))
    assert abs(metrics.energy_distance(x, x)) < 1e-12
    assert metrics.energy_distance(x, x + 1.0) > 0.2


def test_null_scale_is_reported_as_a_95_quantile():
    q = metrics.null_scale(jax.random.PRNGKey(12), size=800, pairs=8)
    assert q["q95"] > 0.0
    assert q["q95"] < 0.05


def test_cluster_stats_recover_exact_samples():
    x = np.asarray(mixture.sample(jax.random.PRNGKey(13), 20000))
    stats = metrics.cluster_stats(x)
    assert np.all(np.abs(stats["populations"] - 0.25) < 0.02)
    assert np.all(np.abs(np.asarray(stats["centres"]) - mixture.CENTERS) < 0.03)
    assert np.all(np.abs(np.asarray(stats["spreads"]) - 0.30) < 0.02)


# ------------------------------------------------------- acceptance: part 1 ---


def _load(name: str):
    path = EVIDENCE / name
    if not path.exists():
        pytest.fail(f"missing evidence {path}; run the part script first")
    return json.loads(path.read_text())


def test_forward_diffusion_end_state_is_standard_gaussian():
    data = _load("part1_metrics.json")["forward"]
    assert data["n"] >= 2000
    assert np.all(np.abs(data["mean"]) < 0.1)
    assert np.all(np.abs(np.asarray(data["variance"]) - 1.0) < 0.1)


def test_generated_mixture_samples_recover_populations_centres_and_widths():
    data = _load("part1_metrics.json")["generation"]
    assert data["n"] == 2000 and data["steps"] == 1000
    assert np.all(np.abs(np.asarray(data["populations"]) - 0.25) < 0.04)
    assert np.all(np.abs(np.asarray(data["spreads"]) - 0.30) < 0.03)
    assert np.all(np.abs(np.asarray(data["centres"]) - mixture.CENTERS) < 0.05)


def test_training_uses_the_sheet_checkpoints_and_levels_the_loss_near_a_quarter():
    data = _load("part1_metrics.json")["training"]
    assert data["stage_updates"] == STAGES
    assert data["updates"] == 25000
    assert data["batch_size"] == 512
    assert abs(data["final_loss"] - 0.25) < 0.05


def test_optional_denoiser_matches_the_exact_posterior_mean():
    data = _load("part1_metrics.json")["denoiser"]
    rel = dict(zip(data["t"], data["relative_rms"]))
    for t in (100, 300, 600):
        assert rel[t] < 0.1, (t, rel[t])


# ------------------------------------------------------- acceptance: part 2 ---


def test_ddim_is_deterministic_while_ddpm_is_stochastic():
    data = _load("part2_metrics.json")["determinism"]
    assert data["ddim_max_abs_diff"] == 0.0
    assert data["ddpm_same_cluster_fraction"] < 0.45


def test_both_samplers_reproduce_the_target():
    data = _load("part2_metrics.json")["determinism"]
    assert np.all(np.abs(np.asarray(data["ddim_populations"]) - 0.25) < 0.015)
    for width in np.asarray(data["ddim_spreads"]).ravel():
        assert abs(width - 0.30) < 0.03
    for width in np.asarray(data["ddpm_spreads"]).ravel():
        assert abs(width - 0.30) < 0.03
    assert data["constants"]["exact_learned_width_gap"] < 0.02


# ------------------------------------------------------- acceptance: part 3 ---


def test_learned_velocity_matches_the_exact_marginal_velocity():
    data = _load("part3_metrics.json")["velocity"]
    assert data["s"] == [0.1, 0.5, 0.9]
    assert np.all(np.asarray(data["relative_rms"]) < 0.1)


def test_few_step_samplers_reach_the_target_and_ddpm_needs_more_steps():
    sweep = _load("part3_metrics.json")["sweep"]
    index = sweep["steps"].index(20)
    assert float(sweep["flow"]["energy"][index]) < 0.005
    assert float(sweep["flow"]["width_error"][index]) < 0.03
    assert np.asarray(sweep["null_quantile_95"]).item() > 0.0


# ------------------------------------------------------- acceptance: part 4/5 ---


def test_pretrained_sampler_generates_cifar_images():
    data = _load("part4_metrics.json")
    assert data["param_count"] > 30_000_000
    assert data["images"] == 4
    assert data["levels"] == 1000
    assert data["revision"] == "267b167dc01f0e4e61923ea244e8b988f84deb80"
    stats = np.asarray(data["pixel_stats"])
    assert stats.shape == (4, 4)
    assert np.all(np.isfinite(stats))


def test_repaint_keeps_observed_pixels_and_varies_inside_the_hole():
    data = _load("part5_metrics.json")
    assert data["seeds"] == [7201, 7202, 7203, 7204]
    assert data["jumps"] == 50
    assert data["attempts_per_jump"] == 2
    assert data["network_evaluations"] == 99
    assert data["observed_max_change"] < 1e-5
    mse = np.asarray(data["hidden_mse"])
    assert mse.shape == (4, 4)
    assert np.all(mse > 0.0)
    assert np.all(np.asarray(data["hidden_variance_mean"]) > 0.0)
