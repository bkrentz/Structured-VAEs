from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from free_energy import bayesian_negative_free_energy, point_negative_free_energy
from inference import GaussianPosterior, SufficientStatistics, gaussian_row_covariance, sample_gaussian_trajectories, sufficient_statistics
from model import BayesianModelParams, BayesianNeuralParams, DecoderTailParams, DiagonalGaussianPosterior, GaussianRowPosterior, ModelConfig, NoiseParams, PointDynamicsParams, PointModelParams, PointNeuralParams, decode_log_rates_bayesian, forecast_origin_site_mask, initialise_bayesian_model, initialise_point_model, poisson_log_prob_from_log_rate
from utils import Array, AdamState, _actual_variance, _assert_finite_tree, _chol_solve, _clip_gradient_tree, _log_variance_parameter, _symmetrise, adam_init, adam_update, spectral_radius


## Training settings

@dataclass(frozen=True)
class PointTrainingConfig:
    """Optimisation settings for the point-estimate structured VAE."""

    num_steps: int = 2000
    batch_size: int = 12
    learning_rate: float = 2e-3
    dynamics_step_size: float = 0.08
    mask_horizon: int = 30
    num_mc_samples: int = 4
    gradient_clip: float = 10.0
    context_only_dynamics_stats: bool = True
    log_every: int = 20
    seed: int = 0


@dataclass(frozen=True)
class BayesianTrainingConfig:
    """Optimisation settings for the Bayesian/ARD structured VAE."""

    num_steps: int = 2000
    batch_size: int = 12
    learning_rate: float = 1.5e-3
    mask_horizon: int = 30
    num_mc_samples: int = 4
    gradient_clip: float = 10.0
    q_a_step_size: float = 0.08
    noise_step_size: float = 0.06
    vogn_step_size: float = 0.04
    ard_step_size: float = 0.04
    ard_warmup_steps: int = 60
    ard_update_every: int = 5
    kappa_a: float = 1.0
    kappa_w: float = 1.0
    alpha_min: float = 1e-3
    alpha_max: float = 1e4
    context_only_dynamics_stats: bool = True
    log_every: int = 20
    seed: int = 0


## Training containers

class PointTrainState(NamedTuple):
    dynamics: PointDynamicsParams
    neural: PointNeuralParams
    optimizer: AdamState


class BayesianTrainState(NamedTuple):
    qA: GaussianRowPosterior
    noise: NoiseParams
    qW1: DiagonalGaussianPosterior
    log_alpha: Array
    neural: BayesianNeuralParams
    optimizer: AdamState


## Point-estimate dynamics updates

def point_dynamics_m_step(
    stats: SufficientStatistics,
    config: ModelConfig,
) -> PointDynamicsParams:
    """Closed-form target for A, diagonal Q, m0, and diagonal S0."""

    latent_dim = stats.sum_ztzt.shape[0]
    regularised_sxx = (
        _symmetrise(stats.sum_ztzt)
        + config.information_jitter * jnp.eye(latent_dim)
    )
    A = _chol_solve(regularised_sxx, stats.sum_ztp1zt.T).T
    residual = (
        stats.sum_ztp1ztp1
        - A @ stats.sum_ztp1zt.T
        - stats.sum_ztp1zt @ A.T
        + A @ stats.sum_ztzt @ A.T
    )
    q_diag = jnp.maximum(
        jnp.diagonal(_symmetrise(residual))
        / jnp.maximum(stats.num_transitions, 1.0),
        config.variance_floor,
    )
    m0 = stats.sum_z0 / jnp.maximum(stats.num_sequences, 1.0)
    initial_second = stats.sum_z0z0 / jnp.maximum(stats.num_sequences, 1.0)
    initial_cov = _symmetrise(initial_second - jnp.outer(m0, m0))
    s0_diag = jnp.maximum(jnp.diagonal(initial_cov), config.variance_floor)
    noise = NoiseParams(
        _log_variance_parameter(q_diag, config.variance_floor),
        m0,
        _log_variance_parameter(s0_diag, config.variance_floor),
    )
    return PointDynamicsParams(A, noise)


def damp_point_dynamics(
    old: PointDynamicsParams,
    target: PointDynamicsParams,
    step_size: float,
    config: ModelConfig,
) -> PointDynamicsParams:
    rho = jnp.asarray(step_size)
    A = (1.0 - rho) * old.A + rho * target.A
    q_old = _actual_variance(old.noise.log_q_diag, config.variance_floor)
    q_new = _actual_variance(target.noise.log_q_diag, config.variance_floor)
    s_old = _actual_variance(old.noise.log_s0_diag, config.variance_floor)
    s_new = _actual_variance(target.noise.log_s0_diag, config.variance_floor)
    noise = NoiseParams(
        _log_variance_parameter((1.0 - rho) * q_old + rho * q_new, config.variance_floor),
        (1.0 - rho) * old.noise.m0 + rho * target.noise.m0,
        _log_variance_parameter((1.0 - rho) * s_old + rho * s_new, config.variance_floor),
    )
    return PointDynamicsParams(A, noise)


def _transition_mask_from_sites(site_mask: Array) -> Array:
    return site_mask[:, :-1] * site_mask[:, 1:]


## Bayesian dynamics updates


def _scale_statistics(stats: SufficientStatistics, factor: float) -> SufficientStatistics:
    factor = jnp.asarray(factor, dtype=stats.sum_z0.dtype)
    return SufficientStatistics(
        stats.sum_z0 * factor,
        stats.sum_z0z0 * factor,
        stats.sum_ztzt * factor,
        stats.sum_ztp1zt * factor,
        stats.sum_ztp1ztp1 * factor,
        stats.num_sequences * factor,
        stats.num_transitions * factor,
    )


def update_qA_conjugate(
    old: GaussianRowPosterior,
    stats: SufficientStatistics,
    noise: NoiseParams,
    alpha: Array,
    config: ModelConfig,
    train_config: BayesianTrainingConfig,
) -> GaussianRowPosterior:
    """Natural-parameter move toward the exact conjugate q(A) optimum."""

    q_diag = _actual_variance(noise.log_q_diag, config.variance_floor)
    tau = 1.0 / q_diag
    prior_precision = jnp.diag(train_config.kappa_a * alpha)
    target_precision = prior_precision[None] + tau[:, None, None] * stats.sum_ztzt[None]
    target_precision = jax.vmap(_symmetrise)(target_precision)
    target_natural = tau[:, None] * stats.sum_ztp1zt
    rho = jnp.asarray(train_config.q_a_step_size)
    precision = (1.0 - rho) * old.precision + rho * target_precision
    natural = (1.0 - rho) * old.natural_mean + rho * target_natural
    mean = jax.vmap(_chol_solve)(precision, natural)
    return GaussianRowPosterior(mean, precision, natural)


def update_bayesian_noise(
    old: NoiseParams,
    stats: SufficientStatistics,
    qA: GaussianRowPosterior,
    config: ModelConfig,
    train_config: BayesianTrainingConfig,
) -> NoiseParams:
    """Damped point updates for diagonal Q, m0, and diagonal S0."""

    A_cov = gaussian_row_covariance(qA, config.information_jitter)
    A_second = A_cov + qA.mean[:, :, None] * qA.mean[:, None, :]
    syy = jnp.diagonal(stats.sum_ztp1ztp1)
    linear = jnp.einsum("jk,jk->j", qA.mean, stats.sum_ztp1zt)
    quadratic = jnp.einsum("jkl,lk->j", A_second, stats.sum_ztzt)
    residual = syy - 2.0 * linear + quadratic
    q_target = jnp.maximum(
        residual / jnp.maximum(stats.num_transitions, 1.0),
        config.variance_floor,
    )
    m0_target = stats.sum_z0 / jnp.maximum(stats.num_sequences, 1.0)
    second0 = stats.sum_z0z0 / jnp.maximum(stats.num_sequences, 1.0)
    cov0 = _symmetrise(second0 - jnp.outer(m0_target, m0_target))
    s0_target = jnp.maximum(jnp.diagonal(cov0), config.variance_floor)

    rho = jnp.asarray(train_config.noise_step_size)
    q_old = _actual_variance(old.log_q_diag, config.variance_floor)
    s_old = _actual_variance(old.log_s0_diag, config.variance_floor)
    return NoiseParams(
        _log_variance_parameter((1.0 - rho) * q_old + rho * q_target, config.variance_floor),
        (1.0 - rho) * old.m0 + rho * m0_target,
        _log_variance_parameter((1.0 - rho) * s_old + rho * s0_target, config.variance_floor),
    )


def _full_data_negative_log_likelihood_w1(
    W1: Array,
    tail: DecoderTailParams,
    z_samples: Array,
    counts: Array,
    scale: float,
    config: ModelConfig,
) -> Array:
    log_rates = jax.vmap(
        lambda z: decode_log_rates_bayesian(tail, W1, z, config)
    )(z_samples)
    log_likelihood = jnp.sum(
        poisson_log_prob_from_log_rate(log_rates, counts[None], config.dt),
        axis=(-3, -2, -1),
    )
    return -scale * jnp.mean(log_likelihood)


def poisson_ggn_diagonal_w1(
    W1: Array,
    tail: DecoderTailParams,
    z_samples: Array,
    scale: float,
    config: ModelConfig,
) -> Array:
    """Positive diagonal GGN of the full-data negative Poisson log likelihood."""

    preactivation = jnp.einsum("sbtk,hk->sbth", z_samples, W1) + tail.b1
    hidden = jnp.tanh(preactivation)
    log_rate = jnp.einsum("sbth,dh->sbtd", hidden, tail.W2) + tail.b2
    log_rate = jnp.clip(log_rate, -config.log_rate_clip, config.log_rate_clip)
    mean = config.dt * jnp.exp(log_rate)
    weighted_output = jnp.einsum(
        "sbtd,dh->sbth", mean, jnp.square(tail.W2)
    )
    hidden_derivative_sq = jnp.square(1.0 - jnp.square(hidden))
    factor = weighted_output * hidden_derivative_sq
    curvature = jnp.einsum(
        "sbth,sbtk->hk", factor, jnp.square(z_samples)
    )
    return scale * curvature / float(z_samples.shape[0])


def update_vogn_first_layer(
    key: Array,
    old: DiagonalGaussianPosterior,
    tail: DecoderTailParams,
    posterior: GaussianPosterior,
    counts: Array,
    alpha: Array,
    full_data_scale: float,
    config: ModelConfig,
    train_config: BayesianTrainingConfig,
) -> tuple[DiagonalGaussianPosterior, Mapping[str, Array]]:
    """One diagonal VOGN natural update for the first decoder layer."""

    z_samples = jax.lax.stop_gradient(
        sample_gaussian_trajectories(
            key, posterior, train_config.num_mc_samples
        )
    )
    loss_function = lambda W1: _full_data_negative_log_likelihood_w1(
        W1,
        tail,
        z_samples,
        counts,
        full_data_scale,
        config,
    )
    gradient = jax.grad(loss_function)(old.mean)
    curvature = poisson_ggn_diagonal_w1(
        old.mean, tail, z_samples, full_data_scale, config
    )
    prior_precision = train_config.kappa_w * alpha[None, :]
    target_precision = prior_precision + curvature
    target_natural = curvature * old.mean - gradient
    rho = jnp.asarray(train_config.vogn_step_size)
    precision = (1.0 - rho) * old.precision + rho * target_precision
    precision = jnp.clip(precision, 1e-5, 1e8)
    natural = (1.0 - rho) * old.natural_mean + rho * target_natural
    mean = natural / precision
    return (
        DiagonalGaussianPosterior(mean, precision, natural),
        {
            "vogn_gradient_norm": jnp.linalg.norm(gradient),
            "mean_vogn_curvature": jnp.mean(curvature),
        },
    )


def ard_target(
    qA: GaussianRowPosterior,
    qW1: DiagonalGaussianPosterior,
    config: ModelConfig,
    train_config: BayesianTrainingConfig,
) -> Array:
    """Empirical-Bayes shared column-wise ARD target including variances."""

    A_cov = gaussian_row_covariance(qA, config.information_jitter)
    A_energy = jnp.sum(
        jnp.square(qA.mean) + jnp.diagonal(A_cov, axis1=-2, axis2=-1),
        axis=0,
    )
    W_variance = 1.0 / qW1.precision
    W_energy = jnp.sum(jnp.square(qW1.mean) + W_variance, axis=0)
    numerator = qA.mean.shape[0] + qW1.mean.shape[0]
    denominator = (
        train_config.kappa_a * A_energy
        + train_config.kappa_w * W_energy
    )
    return numerator / jnp.maximum(denominator, 1e-12)


def update_ard(
    log_alpha: Array,
    qA: GaussianRowPosterior,
    qW1: DiagonalGaussianPosterior,
    config: ModelConfig,
    train_config: BayesianTrainingConfig,
) -> Array:
    target = jnp.clip(
        ard_target(qA, qW1, config, train_config),
        train_config.alpha_min,
        train_config.alpha_max,
    )
    rho = train_config.ard_step_size
    updated = (1.0 - rho) * log_alpha + rho * jnp.log(target)
    return jnp.clip(
        updated,
        jnp.log(train_config.alpha_min),
        jnp.log(train_config.alpha_max),
    )

## Training loops


def _point_train_step(
    state: PointTrainState,
    key: Array,
    counts: Array,
    site_mask: Array,
    config: ModelConfig,
    train_config: PointTrainingConfig,
) -> tuple[PointTrainState, Mapping[str, Array]]:
    loss_function = lambda neural: point_negative_free_energy(
        neural,
        key,
        state.dynamics,
        counts,
        site_mask,
        config,
        train_config.num_mc_samples,
    )
    (loss, aux), grads = jax.value_and_grad(loss_function, has_aux=True)(state.neural)
    grads, raw_grad_norm = _clip_gradient_tree(grads, train_config.gradient_clip)
    neural, optimizer = adam_update(
        state.neural,
        grads,
        state.optimizer,
        train_config.learning_rate,
    )

    posterior = jax.tree_util.tree_map(jax.lax.stop_gradient, aux["posterior"])
    transition_mask = (
        _transition_mask_from_sites(site_mask)
        if train_config.context_only_dynamics_stats
        else jnp.ones_like(site_mask[:, :-1])
    )
    stats = sufficient_statistics(posterior, transition_mask)
    dynamics_target = point_dynamics_m_step(stats, config)
    dynamics = damp_point_dynamics(
        state.dynamics,
        dynamics_target,
        train_config.dynamics_step_size,
        config,
    )
    metrics = {
        "loss": loss,
        "free_energy": -loss,
        "expected_log_likelihood": jnp.mean(aux["expected_log_likelihood"]),
        "latent_kl": -jnp.mean(aux["expected_log_prior"] + aux["entropy"]),
        "gradient_norm": raw_grad_norm,
        "spectral_radius": spectral_radius(dynamics.A),
    }
    return PointTrainState(dynamics, neural, optimizer), metrics


def _bayesian_train_step(
    state: BayesianTrainState,
    key: Array,
    counts: Array,
    site_mask: Array,
    config: ModelConfig,
    train_config: BayesianTrainingConfig,
    num_train_sequences: int,
) -> tuple[BayesianTrainState, Mapping[str, Array]]:
    key_objective, key_vogn = jax.random.split(key)
    loss_function = lambda neural: bayesian_negative_free_energy(
        neural,
        key_objective,
        state.qA,
        state.noise,
        state.qW1,
        state.log_alpha,
        counts,
        site_mask,
        config,
        train_config,
        num_train_sequences,
    )
    (loss, aux), grads = jax.value_and_grad(loss_function, has_aux=True)(state.neural)
    grads, raw_grad_norm = _clip_gradient_tree(grads, train_config.gradient_clip)
    neural, optimizer = adam_update(
        state.neural,
        grads,
        state.optimizer,
        train_config.learning_rate,
    )

    posterior = jax.tree_util.tree_map(jax.lax.stop_gradient, aux["posterior"])
    transition_mask = (
        _transition_mask_from_sites(site_mask)
        if train_config.context_only_dynamics_stats
        else jnp.ones_like(site_mask[:, :-1])
    )
    batch_stats = sufficient_statistics(posterior, transition_mask)
    full_scale = float(num_train_sequences) / float(counts.shape[0])
    stats = _scale_statistics(batch_stats, full_scale)
    alpha = jnp.exp(state.log_alpha)
    qA = update_qA_conjugate(
        state.qA, stats, state.noise, alpha, config, train_config
    )
    noise = update_bayesian_noise(
        state.noise, stats, qA, config, train_config
    )
    qW1, vogn_metrics = update_vogn_first_layer(
        key_vogn,
        state.qW1,
        neural.decoder_tail,
        posterior,
        counts,
        alpha,
        full_scale,
        config,
        train_config,
    )
    metrics = {
        "loss": loss,
        "free_energy": -loss,
        "expected_log_likelihood": jnp.mean(aux["expected_log_likelihood"]),
        "latent_kl": -jnp.mean(aux["expected_log_prior"] + aux["entropy"]),
        "kl_a": aux["kl_a"],
        "kl_w": aux["kl_w"],
        "gradient_norm": raw_grad_norm,
        "spectral_radius_mean_A": spectral_radius(qA.mean),
        **vogn_metrics,
    }
    return BayesianTrainState(
        qA,
        noise,
        qW1,
        state.log_alpha,
        neural,
        optimizer,
    ), metrics


def _history_entry(step: int, metrics: Mapping[str, Any]) -> dict[str, float]:
    entry = {name: float(np.asarray(value)) for name, value in metrics.items()}
    entry["step"] = int(step)
    return entry


def evaluate_point_objective(
    key: Array,
    params: PointModelParams,
    counts_prefix: Array,
    config: ModelConfig,
    mask_horizon: int,
    num_samples: int = 2,
) -> dict[str, float]:
    site_mask = forecast_origin_site_mask(
        counts_prefix.shape[0], counts_prefix.shape[1], mask_horizon
    )
    loss, aux = point_negative_free_energy(
        PointNeuralParams(params.recognition, params.decoder),
        key,
        params.dynamics,
        counts_prefix,
        site_mask,
        config,
        num_samples,
    )
    return {
        "free_energy": float(-loss),
        "expected_log_likelihood": float(jnp.mean(aux["expected_log_likelihood"])),
        "latent_kl": float(-jnp.mean(aux["expected_log_prior"] + aux["entropy"])),
    }


def evaluate_bayesian_objective(
    key: Array,
    params: BayesianModelParams,
    counts_prefix: Array,
    config: ModelConfig,
    train_config: BayesianTrainingConfig,
    num_train_sequences: int,
) -> dict[str, float]:
    site_mask = forecast_origin_site_mask(
        counts_prefix.shape[0],
        counts_prefix.shape[1],
        train_config.mask_horizon,
    )
    loss, aux = bayesian_negative_free_energy(
        BayesianNeuralParams(params.recognition, params.decoder_tail),
        key,
        params.qA,
        params.noise,
        params.qW1,
        params.log_alpha,
        counts_prefix,
        site_mask,
        config,
        train_config,
        num_train_sequences,
    )
    return {
        "free_energy": float(-loss),
        "expected_log_likelihood": float(jnp.mean(aux["expected_log_likelihood"])),
        "latent_kl": float(-jnp.mean(aux["expected_log_prior"] + aux["entropy"])),
        "kl_a": float(aux["kl_a"]),
        "kl_w": float(aux["kl_w"]),
    }


def fit_point_model(
    train_counts_prefix: Array,
    validation_counts_prefix: Array,
    config: ModelConfig = ModelConfig(),
    train_config: PointTrainingConfig = PointTrainingConfig(),
    initial_params: PointModelParams | None = None,
    verbose: bool = True,
) -> tuple[PointModelParams, list[dict[str, float]]]:
    """Fit the point model using masked structured inference and generalized EM."""

    key = jax.random.PRNGKey(train_config.seed)
    if initial_params is None:
        key, init_key = jax.random.split(key)
        initial_params = initialise_point_model(init_key, train_counts_prefix, config)
    neural = PointNeuralParams(initial_params.recognition, initial_params.decoder)
    state = PointTrainState(initial_params.dynamics, neural, adam_init(neural))
    site_mask = forecast_origin_site_mask(
        train_config.batch_size,
        train_counts_prefix.shape[1],
        train_config.mask_horizon,
    )
    step_function = jax.jit(
        lambda state, key, counts, mask: _point_train_step(
            state, key, counts, mask, config, train_config
        )
    )
    validation_function = jax.jit(
        lambda key, dynamics, neural, counts, mask: point_negative_free_energy(
            neural,
            key,
            dynamics,
            counts,
            mask,
            config,
            max(1, train_config.num_mc_samples),
        )[0]
    )
    rng = np.random.default_rng(train_config.seed)
    history: list[dict[str, float]] = []
    num_train = train_counts_prefix.shape[0]
    validation_mask = forecast_origin_site_mask(
        validation_counts_prefix.shape[0],
        validation_counts_prefix.shape[1],
        train_config.mask_horizon,
    )

    for step in range(train_config.num_steps):
        indices = rng.choice(
            num_train,
            size=train_config.batch_size,
            replace=train_config.batch_size > num_train,
        )
        batch = train_counts_prefix[jnp.asarray(indices)]
        key, step_key = jax.random.split(key)
        state, metrics = step_function(state, step_key, batch, site_mask)

        if step % train_config.log_every == 0 or step == train_config.num_steps - 1:
            key, validation_key = jax.random.split(key)
            validation_loss = validation_function(
                validation_key,
                state.dynamics,
                state.neural,
                validation_counts_prefix,
                validation_mask,
            )
            entry = _history_entry(step, metrics)
            entry["validation_free_energy"] = float(-validation_loss)
            history.append(entry)
            _assert_finite_tree(state, "point-model training state")
            if verbose:
                print(
                    f"point step {step:04d} | free energy {entry['free_energy']:.2f} | "
                    f"val {entry['validation_free_energy']:.2f} | "
                    f"rho(A) {entry['spectral_radius']:.3f}",
                    flush=True,
                )

    params = PointModelParams(
        state.dynamics,
        state.neural.recognition,
        state.neural.decoder,
    )
    return params, history


def fit_bayesian_model(
    train_counts_prefix: Array,
    validation_counts_prefix: Array,
    config: ModelConfig = ModelConfig(),
    train_config: BayesianTrainingConfig = BayesianTrainingConfig(),
    initial_params: BayesianModelParams | None = None,
    verbose: bool = True,
) -> tuple[BayesianModelParams, list[dict[str, float]]]:
    """Fit q(A), a VOGN first layer, point noise, and shared ARD."""

    key = jax.random.PRNGKey(train_config.seed)
    if initial_params is None:
        key, init_key = jax.random.split(key)
        initial_params = initialise_bayesian_model(
            init_key, train_counts_prefix, config, train_config
        )
    neural = BayesianNeuralParams(
        initial_params.recognition, initial_params.decoder_tail
    )
    state = BayesianTrainState(
        initial_params.qA,
        initial_params.noise,
        initial_params.qW1,
        initial_params.log_alpha,
        neural,
        adam_init(neural),
    )
    site_mask = forecast_origin_site_mask(
        train_config.batch_size,
        train_counts_prefix.shape[1],
        train_config.mask_horizon,
    )
    num_train = train_counts_prefix.shape[0]
    step_function = jax.jit(
        lambda state, key, counts, mask: _bayesian_train_step(
            state,
            key,
            counts,
            mask,
            config,
            train_config,
            num_train,
        )
    )
    validation_mask = forecast_origin_site_mask(
        validation_counts_prefix.shape[0],
        validation_counts_prefix.shape[1],
        train_config.mask_horizon,
    )
    validation_function = jax.jit(
        lambda key, state, counts, mask: bayesian_negative_free_energy(
            state.neural,
            key,
            state.qA,
            state.noise,
            state.qW1,
            state.log_alpha,
            counts,
            mask,
            config,
            train_config,
            num_train,
        )[0]
    )
    rng = np.random.default_rng(train_config.seed)
    history: list[dict[str, float]] = []

    for step in range(train_config.num_steps):
        indices = rng.choice(
            num_train,
            size=train_config.batch_size,
            replace=train_config.batch_size > num_train,
        )
        batch = train_counts_prefix[jnp.asarray(indices)]
        key, step_key = jax.random.split(key)
        state, metrics = step_function(state, step_key, batch, site_mask)

        if (
            step >= train_config.ard_warmup_steps
            and (step - train_config.ard_warmup_steps)
            % train_config.ard_update_every
            == 0
        ):
            state = state._replace(
                log_alpha=update_ard(
                    state.log_alpha,
                    state.qA,
                    state.qW1,
                    config,
                    train_config,
                )
            )

        if step % train_config.log_every == 0 or step == train_config.num_steps - 1:
            key, validation_key = jax.random.split(key)
            validation_loss = validation_function(
                validation_key, state, validation_counts_prefix, validation_mask
            )
            entry = _history_entry(step, metrics)
            entry["validation_free_energy"] = float(-validation_loss)
            alpha = np.exp(np.asarray(state.log_alpha))
            entry["alpha_min"] = float(alpha.min())
            entry["alpha_max"] = float(alpha.max())
            history.append(entry)
            _assert_finite_tree(state, "Bayesian-model training state")
            if verbose:
                print(
                    f"bayes step {step:04d} | free energy {entry['free_energy']:.2f} | "
                    f"val {entry['validation_free_energy']:.2f} | "
                    f"rho(E[A]) {entry['spectral_radius_mean_A']:.3f} | "
                    f"alpha [{entry['alpha_min']:.2f}, {entry['alpha_max']:.2f}]",
                    flush=True,
                )

    params = BayesianModelParams(
        state.qA,
        state.noise,
        state.qW1,
        state.log_alpha,
        state.neural.recognition,
        state.neural.decoder_tail,
    )
    return params, history
