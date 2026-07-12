from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp

from inference import GaussianPosterior, gaussian_row_covariance, infer_bayesian_posterior, infer_point_posterior, posterior_second_moments, sample_gaussian_trajectories
from model import BayesianNeuralParams, DecoderTailParams, DiagonalGaussianPosterior, GaussianRowPosterior, ModelConfig, NoiseParams, PointDynamicsParams, PointNeuralParams, decode_log_rates_bayesian, decode_log_rates_point, poisson_log_prob_from_log_rate, recognise
from utils import Array, LOG_2PI, _actual_variance, _logdet_spd, _symmetrise


## Free energy terms

def gaussian_trajectory_entropy(posterior: GaussianPosterior) -> Array:
    """Entropy of each full trajectory Gaussian."""

    time_steps = posterior.mean.shape[1]
    latent_dim = posterior.mean.shape[2]
    dimension = time_steps * latent_dim
    return 0.5 * (
        dimension * (1.0 + LOG_2PI) - posterior.logdet_precision
    )


def expected_log_point_prior(
    posterior: GaussianPosterior,
    dynamics: PointDynamicsParams,
    config: ModelConfig,
) -> Array:
    """E_q[log p(z_0:T | A,Q,m0,S0)] for each sequence."""

    second, cross_second = posterior_second_moments(posterior)
    mean0 = posterior.mean[:, 0]
    cov0 = posterior.cov[:, 0]
    q_diag = _actual_variance(dynamics.noise.log_q_diag, config.variance_floor)
    s0_diag = _actual_variance(dynamics.noise.log_s0_diag, config.variance_floor)
    latent_dim = posterior.mean.shape[-1]
    time_steps = posterior.mean.shape[1]

    centred0 = mean0 - dynamics.noise.m0
    initial_residual = cov0 + centred0[:, :, None] * centred0[:, None, :]
    initial_quadratic = jnp.sum(
        jnp.diagonal(initial_residual, axis1=-2, axis2=-1) / s0_diag,
        axis=-1,
    )
    initial = -0.5 * (
        latent_dim * LOG_2PI + jnp.sum(jnp.log(s0_diag)) + initial_quadratic
    )

    if time_steps == 1:
        return initial

    A = dynamics.A
    previous = second[:, :-1]
    following = second[:, 1:]
    cross_right = jnp.matmul(cross_second, A.T)
    cross_left = jnp.matmul(A, jnp.swapaxes(cross_second, -1, -2))
    propagated = jnp.matmul(jnp.matmul(A, previous), A.T)
    residual = _symmetrise(following - cross_right - cross_left + propagated)
    transition_quadratic = jnp.sum(
        jnp.diagonal(residual, axis1=-2, axis2=-1) / q_diag,
        axis=(-2, -1),
    )
    transition = -0.5 * (
        (time_steps - 1)
        * (latent_dim * LOG_2PI + jnp.sum(jnp.log(q_diag)))
        + transition_quadratic
    )
    return initial + transition


def expected_log_bayesian_prior(
    posterior: GaussianPosterior,
    qA: GaussianRowPosterior,
    noise: NoiseParams,
    config: ModelConfig,
) -> Array:
    """E_{q(z)q(A)}[log p(z | A,Q,m0,S0)] for each sequence."""

    second, cross_second = posterior_second_moments(posterior)
    mean0 = posterior.mean[:, 0]
    cov0 = posterior.cov[:, 0]
    q_diag = _actual_variance(noise.log_q_diag, config.variance_floor)
    s0_diag = _actual_variance(noise.log_s0_diag, config.variance_floor)
    latent_dim = posterior.mean.shape[-1]
    time_steps = posterior.mean.shape[1]

    centred0 = mean0 - noise.m0
    initial_residual = cov0 + centred0[:, :, None] * centred0[:, None, :]
    initial_quadratic = jnp.sum(
        jnp.diagonal(initial_residual, axis1=-2, axis2=-1) / s0_diag,
        axis=-1,
    )
    initial = -0.5 * (
        latent_dim * LOG_2PI + jnp.sum(jnp.log(s0_diag)) + initial_quadratic
    )
    if time_steps == 1:
        return initial

    A_cov = gaussian_row_covariance(qA, config.information_jitter)
    A_second = A_cov + qA.mean[:, :, None] * qA.mean[:, None, :]
    Sxx = jnp.sum(second[:, :-1], axis=1)
    S10 = jnp.sum(cross_second, axis=1)
    S11 = jnp.sum(second[:, 1:], axis=1)
    syy = jnp.diagonal(S11, axis1=-2, axis2=-1)
    linear = jnp.einsum("jk,bjk->bj", qA.mean, S10)
    quadratic = jnp.einsum("jkl,blk->bj", A_second, Sxx)
    residual = syy - 2.0 * linear + quadratic
    transition_quadratic = jnp.sum(residual / q_diag, axis=-1)
    transition = -0.5 * (
        (time_steps - 1)
        * (latent_dim * LOG_2PI + jnp.sum(jnp.log(q_diag)))
        + transition_quadratic
    )
    return initial + transition


def kl_gaussian_rows_to_ard_prior(
    qA: GaussianRowPosterior,
    alpha: Array,
    kappa_a: float,
    jitter: float = 0.0,
) -> Array:
    """KL(q(A) || product Gaussian ARD prior)."""

    cov = gaussian_row_covariance(qA, jitter)
    prior_precision = kappa_a * alpha
    trace_term = jnp.einsum("k,rkk->r", prior_precision, cov)
    mean_term = jnp.sum(jnp.square(qA.mean) * prior_precision[None], axis=-1)
    logdet_prior_cov = -jnp.sum(jnp.log(prior_precision))
    logdet_cov = jax.vmap(_logdet_spd)(cov)
    latent_dim = qA.mean.shape[-1]
    return 0.5 * jnp.sum(
        trace_term + mean_term - latent_dim + logdet_prior_cov - logdet_cov
    )


def kl_diagonal_gaussian_to_ard_prior(
    qW1: DiagonalGaussianPosterior,
    alpha: Array,
    kappa_w: float,
) -> Array:
    """KL(q(W1) || product Gaussian ARD prior)."""

    variance = 1.0 / qW1.precision
    prior_precision = kappa_w * alpha[None, :]
    return 0.5 * jnp.sum(
        prior_precision * (variance + jnp.square(qW1.mean))
        - 1.0
        - jnp.log(prior_precision * variance)
    )


def sample_diagonal_gaussian(
    key: Array,
    posterior: DiagonalGaussianPosterior,
    num_samples: int,
) -> Array:
    noise = jax.random.normal(key, (num_samples,) + posterior.mean.shape)
    return posterior.mean[None] + noise / jnp.sqrt(posterior.precision)[None]


def point_negative_free_energy(
    neural: PointNeuralParams,
    key: Array,
    dynamics: PointDynamicsParams,
    counts: Array,
    site_mask: Array,
    config: ModelConfig,
    num_samples: int,
) -> tuple[Array, Mapping[str, Any]]:
    """Negative free energy evaluated at the Gaussian-site structured posterior."""

    sites = recognise(neural.recognition, counts, site_mask, config)
    posterior = infer_point_posterior(dynamics, sites, config)
    z_samples = sample_gaussian_trajectories(key, posterior, num_samples)
    log_rates = jax.vmap(
        lambda z: decode_log_rates_point(neural.decoder, z, config)
    )(z_samples)
    log_likelihood_samples = jnp.sum(
        poisson_log_prob_from_log_rate(log_rates, counts[None], config.dt),
        axis=(-2, -1),
    )
    expected_log_likelihood = jnp.mean(log_likelihood_samples, axis=0)
    expected_log_prior = expected_log_point_prior(posterior, dynamics, config)
    entropy = gaussian_trajectory_entropy(posterior)
    free_energy_per_sequence = expected_log_likelihood + expected_log_prior + entropy
    negative_free_energy = -jnp.mean(free_energy_per_sequence)
    aux = {
        "posterior": posterior,
        "expected_log_likelihood": expected_log_likelihood,
        "expected_log_prior": expected_log_prior,
        "entropy": entropy,
        "free_energy_per_sequence": free_energy_per_sequence,
    }
    return negative_free_energy, aux


def bayesian_negative_free_energy(
    neural: BayesianNeuralParams,
    key: Array,
    qA: GaussianRowPosterior,
    noise: NoiseParams,
    qW1: DiagonalGaussianPosterior,
    log_alpha: Array,
    counts: Array,
    site_mask: Array,
    config: ModelConfig,
    train_config: BayesianTrainingConfig,
    num_train_sequences: int,
) -> tuple[Array, Mapping[str, Any]]:
    """Negative free energy for the Bayesian model."""

    key_z, key_w = jax.random.split(key)
    sites = recognise(neural.recognition, counts, site_mask, config)
    posterior = infer_bayesian_posterior(qA, noise, sites, config)
    z_samples = sample_gaussian_trajectories(
        key_z, posterior, train_config.num_mc_samples
    )
    W1_samples = sample_diagonal_gaussian(
        key_w, qW1, train_config.num_mc_samples
    )
    log_rates = jax.vmap(
        lambda W1, z: decode_log_rates_bayesian(
            neural.decoder_tail, W1, z, config
        )
    )(W1_samples, z_samples)
    log_likelihood_samples = jnp.sum(
        poisson_log_prob_from_log_rate(log_rates, counts[None], config.dt),
        axis=(-2, -1),
    )
    expected_log_likelihood = jnp.mean(log_likelihood_samples, axis=0)
    expected_log_prior = expected_log_bayesian_prior(
        posterior, qA, noise, config
    )
    entropy = gaussian_trajectory_entropy(posterior)
    local_free_energy = expected_log_likelihood + expected_log_prior + entropy
    alpha = jnp.exp(log_alpha)
    kl_a = kl_gaussian_rows_to_ard_prior(
        qA, alpha, train_config.kappa_a, config.information_jitter
    )
    kl_w = kl_diagonal_gaussian_to_ard_prior(
        qW1, alpha, train_config.kappa_w
    )
    global_kl_per_sequence = (kl_a + kl_w) / float(num_train_sequences)
    objective = jnp.mean(local_free_energy) - global_kl_per_sequence
    aux = {
        "posterior": posterior,
        "expected_log_likelihood": expected_log_likelihood,
        "expected_log_prior": expected_log_prior,
        "entropy": entropy,
        "local_free_energy": local_free_energy,
        "kl_a": kl_a,
        "kl_w": kl_w,
        "objective": objective,
    }
    return -objective, aux
