from typing import NamedTuple

import jax
import jax.numpy as jnp

from model import GaussianRowPosterior, GaussianSites, ModelConfig, NoiseParams, PointDynamicsParams
from utils import Array, _actual_variance, _chol_solve, _logdet_spd, _spd_inverse, _symmetrise


## Inference containers

class GaussianPosterior(NamedTuple):
    """Moments and backward-conditionals of a block-tridiagonal Gaussian."""

    mean: Array
    cov: Array
    cross_cov: Array  # Cov(z[t+1], z[t])
    conditional_offset: Array
    conditional_cov: Array
    backward_gain: Array
    logdet_precision: Array


class SufficientStatistics(NamedTuple):
    sum_z0: Array
    sum_z0z0: Array
    sum_ztzt: Array
    sum_ztp1zt: Array
    sum_ztp1ztp1: Array
    num_sequences: Array
    num_transitions: Array


## Gaussian information construction and smoothing

def gaussian_row_covariance(q: GaussianRowPosterior, jitter: float = 0.0) -> Array:
    return jax.vmap(lambda precision: _spd_inverse(precision, jitter))(q.precision)


def point_information_blocks(
    dynamics: PointDynamicsParams,
    sites: GaussianSites,
    config: ModelConfig,
) -> tuple[Array, Array, Array]:
    """Build diagonal/lower blocks and information vectors for point dynamics."""

    A = dynamics.A
    q_diag = _actual_variance(
        dynamics.noise.log_q_diag, config.variance_floor
    )
    s0_diag = _actual_variance(
        dynamics.noise.log_s0_diag, config.variance_floor
    )
    q_precision = jnp.diag(1.0 / q_diag)
    s0_precision = jnp.diag(1.0 / s0_diag)
    outgoing = A.T @ q_precision @ A
    lower = -q_precision @ A

    batch_size, time_steps, latent_dim = sites.h.shape
    diag = jnp.zeros((batch_size, time_steps, latent_dim, latent_dim))
    diag = diag + jax.vmap(jax.vmap(jnp.diag))(sites.precision_diag)
    diag = diag.at[:, 0].add(s0_precision)
    if time_steps > 1:
        diag = diag.at[:, :-1].add(outgoing)
        diag = diag.at[:, 1:].add(q_precision)
    eta = sites.h.at[:, 0].add(s0_precision @ dynamics.noise.m0)
    lower_blocks = jnp.broadcast_to(
        lower[None, None], (batch_size, max(time_steps - 1, 0), latent_dim, latent_dim)
    )
    return diag, lower_blocks, eta


def bayesian_information_blocks(
    qA: GaussianRowPosterior,
    noise: NoiseParams,
    sites: GaussianSites,
    config: ModelConfig,
) -> tuple[Array, Array, Array]:
    """Build the expected Gaussian transition potential for random A."""

    q_diag = _actual_variance(noise.log_q_diag, config.variance_floor)
    s0_diag = _actual_variance(noise.log_s0_diag, config.variance_floor)
    tau = 1.0 / q_diag
    q_precision = jnp.diag(tau)
    s0_precision = jnp.diag(1.0 / s0_diag)
    A_cov = gaussian_row_covariance(qA, config.information_jitter)
    A_second = A_cov + qA.mean[:, :, None] * qA.mean[:, None, :]
    outgoing = jnp.einsum("j,jkl->kl", tau, A_second)
    lower = -q_precision @ qA.mean

    batch_size, time_steps, latent_dim = sites.h.shape
    diag = jnp.zeros((batch_size, time_steps, latent_dim, latent_dim))
    diag = diag + jax.vmap(jax.vmap(jnp.diag))(sites.precision_diag)
    diag = diag.at[:, 0].add(s0_precision)
    if time_steps > 1:
        diag = diag.at[:, :-1].add(outgoing)
        diag = diag.at[:, 1:].add(q_precision)
    eta = sites.h.at[:, 0].add(s0_precision @ noise.m0)
    lower_blocks = jnp.broadcast_to(
        lower[None, None], (batch_size, max(time_steps - 1, 0), latent_dim, latent_dim)
    )
    return diag, lower_blocks, eta


def _smooth_information_one(
    diagonal_blocks: Array,
    lower_blocks: Array,
    eta: Array,
    jitter: float,
) -> GaussianPosterior:
    """Solve one symmetric block-tridiagonal Gaussian in information form."""

    time_steps, latent_dim = eta.shape
    eye = jnp.eye(latent_dim, dtype=eta.dtype)

    first_precision = _symmetrise(diagonal_blocks[0]) + jitter * eye
    first_info = eta[0]

    if time_steps == 1:
        cov = _spd_inverse(first_precision)
        mean = cov @ first_info
        return GaussianPosterior(
            mean=mean[None],
            cov=cov[None],
            cross_cov=jnp.zeros((0, latent_dim, latent_dim), dtype=eta.dtype),
            conditional_offset=mean[None],
            conditional_cov=cov[None],
            backward_gain=jnp.zeros((0, latent_dim, latent_dim), dtype=eta.dtype),
            logdet_precision=_logdet_spd(first_precision),
        )

    def eliminate_step(
        carry: tuple[Array, Array],
        inputs: tuple[Array, Array, Array],
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        previous_precision, previous_info = carry
        lower, diagonal, info = inputs
        solved_upper = _chol_solve(previous_precision, lower.T)
        solved_info = _chol_solve(previous_precision, previous_info)
        precision = _symmetrise(diagonal - lower @ solved_upper) + jitter * eye
        updated_info = info - lower @ solved_info
        return (precision, updated_info), (precision, updated_info)

    _, forward_outputs = jax.lax.scan(
        eliminate_step,
        (first_precision, first_info),
        (lower_blocks, diagonal_blocks[1:], eta[1:]),
    )
    rest_precision, rest_info = forward_outputs
    schur_precision = jnp.concatenate([first_precision[None], rest_precision], axis=0)
    schur_info = jnp.concatenate([first_info[None], rest_info], axis=0)

    last_cov = _spd_inverse(schur_precision[-1])
    last_offset = last_cov @ schur_info[-1]
    last_mean = last_offset

    def backward_step(
        carry: tuple[Array, Array],
        inputs: tuple[Array, Array, Array],
    ) -> tuple[tuple[Array, Array], tuple[Array, Array, Array, Array, Array]]:
        next_mean, next_cov = carry
        precision, info, lower = inputs
        conditional_cov = _spd_inverse(precision)
        upper = lower.T
        gain = -conditional_cov @ upper
        offset = conditional_cov @ info
        mean = offset + gain @ next_mean
        cov = _symmetrise(conditional_cov + gain @ next_cov @ gain.T)
        cross_lower = next_cov @ gain.T
        return (mean, cov), (mean, cov, cross_lower, offset, conditional_cov)

    _, backward_outputs = jax.lax.scan(
        backward_step,
        (last_mean, last_cov),
        (
            schur_precision[:-1][::-1],
            schur_info[:-1][::-1],
            lower_blocks[::-1],
        ),
    )
    mean_rev, cov_rev, cross_rev, offset_rev, conditional_cov_rev = backward_outputs
    mean = jnp.concatenate([mean_rev[::-1], last_mean[None]], axis=0)
    cov = jnp.concatenate([cov_rev[::-1], last_cov[None]], axis=0)
    cross_cov = cross_rev[::-1]
    offset = jnp.concatenate([offset_rev[::-1], last_offset[None]], axis=0)
    conditional_cov = jnp.concatenate(
        [conditional_cov_rev[::-1], last_cov[None]], axis=0
    )

    def gain_from_precision(precision: Array, lower: Array) -> Array:
        return -_chol_solve(precision, lower.T)

    backward_gain = jax.vmap(gain_from_precision)(
        schur_precision[:-1], lower_blocks
    )
    logdet_precision = jnp.sum(jax.vmap(_logdet_spd)(schur_precision))
    return GaussianPosterior(
        mean,
        cov,
        cross_cov,
        offset,
        conditional_cov,
        backward_gain,
        logdet_precision,
    )


def smooth_information(
    diagonal_blocks: Array,
    lower_blocks: Array,
    eta: Array,
    jitter: float,
) -> GaussianPosterior:
    """Batch the information-form smoother over independent sequences."""

    return jax.vmap(_smooth_information_one, in_axes=(0, 0, 0, None))(
        diagonal_blocks, lower_blocks, eta, jitter
    )


def infer_point_posterior(
    dynamics: PointDynamicsParams,
    sites: GaussianSites,
    config: ModelConfig,
) -> GaussianPosterior:
    blocks = point_information_blocks(dynamics, sites, config)
    return smooth_information(*blocks, config.information_jitter)


def infer_bayesian_posterior(
    qA: GaussianRowPosterior,
    noise: NoiseParams,
    sites: GaussianSites,
    config: ModelConfig,
) -> GaussianPosterior:
    blocks = bayesian_information_blocks(qA, noise, sites, config)
    return smooth_information(*blocks, config.information_jitter)


def posterior_second_moments(
    posterior: GaussianPosterior,
) -> tuple[Array, Array]:
    second = posterior.cov + posterior.mean[..., :, None] * posterior.mean[..., None, :]
    cross_second = (
        posterior.cross_cov
        + posterior.mean[:, 1:, :, None] * posterior.mean[:, :-1, None, :]
    )
    return second, cross_second


def sufficient_statistics(
    posterior: GaussianPosterior,
    transition_mask: Array | None = None,
) -> SufficientStatistics:
    """Aggregate expected LDS sufficient statistics with Cov(next, previous) orientation."""

    second, cross_second = posterior_second_moments(posterior)
    batch_size, time_steps = posterior.mean.shape[:2]
    if transition_mask is None:
        transition_mask = jnp.ones((batch_size, max(time_steps - 1, 0)))
    weights = transition_mask[..., None, None]
    return SufficientStatistics(
        sum_z0=jnp.sum(posterior.mean[:, 0], axis=0),
        sum_z0z0=jnp.sum(second[:, 0], axis=0),
        sum_ztzt=jnp.sum(second[:, :-1] * weights, axis=(0, 1)),
        sum_ztp1zt=jnp.sum(cross_second * weights, axis=(0, 1)),
        sum_ztp1ztp1=jnp.sum(second[:, 1:] * weights, axis=(0, 1)),
        num_sequences=jnp.asarray(batch_size, dtype=posterior.mean.dtype),
        num_transitions=jnp.sum(transition_mask),
    )


def sample_gaussian_trajectories(
    key: Array,
    posterior: GaussianPosterior,
    num_samples: int,
) -> Array:
    """Reparameterise full trajectories using Gaussian backward conditionals."""

    batch_size, time_steps, latent_dim = posterior.mean.shape
    noise = jax.random.normal(
        key, (num_samples, batch_size, time_steps, latent_dim)
    )

    def sample_one(
        eps: Array,
        offset: Array,
        conditional_cov: Array,
        gain: Array,
    ) -> Array:
        chol_last = jnp.linalg.cholesky(_symmetrise(conditional_cov[-1]))
        last = offset[-1] + chol_last @ eps[-1]
        if time_steps == 1:
            return last[None]

        def step(z_next: Array, inputs: tuple[Array, Array, Array, Array]):
            off, cov, G, eps_t = inputs
            chol = jnp.linalg.cholesky(_symmetrise(cov))
            z = off + G @ z_next + chol @ eps_t
            return z, z

        _, reversed_samples = jax.lax.scan(
            step,
            last,
            (
                offset[:-1][::-1],
                conditional_cov[:-1][::-1],
                gain[::-1],
                eps[:-1][::-1],
            ),
        )
        return jnp.concatenate([reversed_samples[::-1], last[None]], axis=0)

    sample_batch = jax.vmap(sample_one, in_axes=(0, 0, 0, 0))
    sample_mc = jax.vmap(sample_batch, in_axes=(0, None, None, None))
    return sample_mc(
        noise,
        posterior.conditional_offset,
        posterior.conditional_cov,
        posterior.backward_gain,
    )
