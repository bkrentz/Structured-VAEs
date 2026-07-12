from dataclasses import dataclass
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from inference import GaussianPosterior, infer_bayesian_posterior, infer_point_posterior
from model import BayesianModelParams, DecoderParams, DecoderTailParams, ModelConfig, PointModelParams, RecognitionParams, decode_log_rates_bayesian, decode_log_rates_point, recognise
from utils import Array


## Forecast settings

@dataclass(frozen=True)
class ForecastConfig:
    """Monte Carlo settings for suffix forecasting."""

    num_samples: int = 300
    seed: int = 10


## Forecast containers

class ForecastSamples(NamedTuple):
    latents: Array
    rates: Array
    counts: Array


## Prefix inference

def infer_point_prefix(
    params: PointModelParams,
    counts_prefix: Array,
    config: ModelConfig,
) -> GaussianPosterior:
    site_mask = jnp.ones(counts_prefix.shape[:2], dtype=counts_prefix.dtype)
    sites = recognise(params.recognition, counts_prefix, site_mask, config)
    return infer_point_posterior(params.dynamics, sites, config)


def infer_bayesian_prefix(
    params: BayesianModelParams,
    counts_prefix: Array,
    config: ModelConfig,
) -> GaussianPosterior:
    site_mask = jnp.ones(counts_prefix.shape[:2], dtype=counts_prefix.dtype)
    sites = recognise(params.recognition, counts_prefix, site_mask, config)
    return infer_bayesian_posterior(params.qA, params.noise, sites, config)


## NumPy forecast backend

# Training needs JAX for gradients. Forecasting only needs array operations,
# so this keeps long notebook runs from compiling another large graph.


def _numpy_softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _numpy_chol_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve an SPD system through its Cholesky factor."""

    matrix = 0.5 * (matrix + matrix.T)
    chol = np.linalg.cholesky(matrix)
    return np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))


def _recognise_numpy(
    params: RecognitionParams,
    counts: Array,
    site_mask: np.ndarray,
    config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    counts_np = np.asarray(counts)
    W1 = np.asarray(params.W1)
    b1 = np.asarray(params.b1)
    Wm = np.asarray(params.W_mean)
    bm = np.asarray(params.b_mean)
    Wp = np.asarray(params.W_precision)
    bp = np.asarray(params.b_precision)
    x = np.log1p(counts_np) * site_mask[..., None]
    hidden = np.tanh(x @ W1.T + b1)
    mean = hidden @ Wm.T + bm
    precision = (_numpy_softplus(hidden @ Wp.T + bp) + config.min_site_precision)
    precision *= site_mask[..., None]
    return precision, precision * mean


def _numpy_information_smoother(
    diagonal: np.ndarray,
    lower: np.ndarray,
    eta: np.ndarray,
    jitter: float,
) -> GaussianPosterior:
    batch_size, time_steps, latent_dim = eta.shape
    eye = np.eye(latent_dim)
    means = []
    covariances = []
    crosses = []
    offsets = []
    conditional_covariances = []
    gains = []
    logdets = []

    for n in range(batch_size):
        schur = np.zeros((time_steps, latent_dim, latent_dim))
        info = np.zeros((time_steps, latent_dim))
        schur[0] = 0.5 * (diagonal[n, 0] + diagonal[n, 0].T) + jitter * eye
        info[0] = eta[n, 0]
        for t in range(1, time_steps):
            solved_upper = _numpy_chol_solve(schur[t - 1], lower[n, t - 1].T)
            solved_info = _numpy_chol_solve(schur[t - 1], info[t - 1])
            candidate = diagonal[n, t] - lower[n, t - 1] @ solved_upper
            schur[t] = 0.5 * (candidate + candidate.T) + jitter * eye
            info[t] = eta[n, t] - lower[n, t - 1] @ solved_info

        cond_cov = np.stack([_numpy_chol_solve(block, eye) for block in schur])
        offset = np.einsum("tij,tj->ti", cond_cov, info)
        gain = np.empty((max(time_steps - 1, 0), latent_dim, latent_dim))
        for t in range(time_steps - 1):
            gain[t] = -np.linalg.solve(schur[t], lower[n, t].T)

        mean = np.zeros((time_steps, latent_dim))
        cov = np.zeros((time_steps, latent_dim, latent_dim))
        cross = np.zeros((max(time_steps - 1, 0), latent_dim, latent_dim))
        mean[-1] = offset[-1]
        cov[-1] = cond_cov[-1]
        for t in range(time_steps - 2, -1, -1):
            mean[t] = offset[t] + gain[t] @ mean[t + 1]
            cov[t] = cond_cov[t] + gain[t] @ cov[t + 1] @ gain[t].T
            cov[t] = 0.5 * (cov[t] + cov[t].T)
            cross[t] = cov[t + 1] @ gain[t].T
        logdet = sum(np.linalg.slogdet(block)[1] for block in schur)
        means.append(mean)
        covariances.append(cov)
        crosses.append(cross)
        offsets.append(offset)
        conditional_covariances.append(cond_cov)
        gains.append(gain)
        logdets.append(logdet)

    return GaussianPosterior(
        np.stack(means),
        np.stack(covariances),
        np.stack(crosses),
        np.stack(offsets),
        np.stack(conditional_covariances),
        np.stack(gains),
        np.asarray(logdets),
    )


def _infer_point_prefix_numpy(
    params: PointModelParams,
    counts_prefix: Array,
    config: ModelConfig,
) -> GaussianPosterior:
    counts = np.asarray(counts_prefix)
    batch_size, time_steps, _ = counts.shape
    latent_dim = config.latent_dim
    mask = np.ones((batch_size, time_steps))
    site_precision, site_h = _recognise_numpy(params.recognition, counts, mask, config)
    A = np.asarray(params.dynamics.A)
    q_diag = np.exp(np.asarray(params.dynamics.noise.log_q_diag)) + config.variance_floor
    s0_diag = np.exp(np.asarray(params.dynamics.noise.log_s0_diag)) + config.variance_floor
    q_precision = np.diag(1.0 / q_diag)
    s0_precision = np.diag(1.0 / s0_diag)
    outgoing = A.T @ q_precision @ A
    lower_block = -q_precision @ A
    diagonal = np.zeros((batch_size, time_steps, latent_dim, latent_dim))
    for n in range(batch_size):
        for t in range(time_steps):
            diagonal[n, t] = np.diag(site_precision[n, t])
    diagonal[:, 0] += s0_precision
    if time_steps > 1:
        diagonal[:, :-1] += outgoing
        diagonal[:, 1:] += q_precision
    eta = site_h.copy()
    eta[:, 0] += s0_precision @ np.asarray(params.dynamics.noise.m0)
    lower = np.broadcast_to(
        lower_block, (batch_size, max(time_steps - 1, 0), latent_dim, latent_dim)
    ).copy()
    return _numpy_information_smoother(
        diagonal, lower, eta, config.information_jitter
    )


def _infer_bayesian_prefix_numpy(
    params: BayesianModelParams,
    counts_prefix: Array,
    config: ModelConfig,
) -> GaussianPosterior:
    counts = np.asarray(counts_prefix)
    batch_size, time_steps, _ = counts.shape
    latent_dim = config.latent_dim
    mask = np.ones((batch_size, time_steps))
    site_precision, site_h = _recognise_numpy(params.recognition, counts, mask, config)
    A_mean = np.asarray(params.qA.mean)
    A_cov = np.stack([_numpy_chol_solve(np.asarray(P), np.eye(P.shape[0])) for P in params.qA.precision])
    A_second = A_cov + A_mean[:, :, None] * A_mean[:, None, :]
    q_diag = np.exp(np.asarray(params.noise.log_q_diag)) + config.variance_floor
    s0_diag = np.exp(np.asarray(params.noise.log_s0_diag)) + config.variance_floor
    tau = 1.0 / q_diag
    q_precision = np.diag(tau)
    s0_precision = np.diag(1.0 / s0_diag)
    outgoing = np.einsum("j,jkl->kl", tau, A_second)
    lower_block = -q_precision @ A_mean
    diagonal = np.zeros((batch_size, time_steps, latent_dim, latent_dim))
    for n in range(batch_size):
        for t in range(time_steps):
            diagonal[n, t] = np.diag(site_precision[n, t])
    diagonal[:, 0] += s0_precision
    if time_steps > 1:
        diagonal[:, :-1] += outgoing
        diagonal[:, 1:] += q_precision
    eta = site_h.copy()
    eta[:, 0] += s0_precision @ np.asarray(params.noise.m0)
    lower = np.broadcast_to(
        lower_block, (batch_size, max(time_steps - 1, 0), latent_dim, latent_dim)
    ).copy()
    return _numpy_information_smoother(
        diagonal, lower, eta, config.information_jitter
    )


def _seed_from_key(key: Array) -> int:
    words = np.asarray(key, dtype=np.uint32).reshape(-1)
    seed = 0
    for index, word in enumerate(words):
        seed ^= int(word) << (index % 2 * 16)
    return seed % (2**32 - 1)


def _decode_point_numpy(
    params: DecoderParams,
    z: np.ndarray,
    config: ModelConfig,
) -> np.ndarray:
    hidden = np.tanh(z @ np.asarray(params.W1).T + np.asarray(params.b1))
    log_rate = hidden @ np.asarray(params.W2).T + np.asarray(params.b2)
    return np.clip(log_rate, -config.log_rate_clip, config.log_rate_clip)


def _decode_bayesian_numpy(
    tail: DecoderTailParams,
    W1: np.ndarray,
    z: np.ndarray,
    config: ModelConfig,
) -> np.ndarray:
    hidden = np.tanh(z @ W1.T + np.asarray(tail.b1))
    log_rate = hidden @ np.asarray(tail.W2).T + np.asarray(tail.b2)
    return np.clip(log_rate, -config.log_rate_clip, config.log_rate_clip)


def forecast_point_model(
    key: Array,
    params: PointModelParams,
    counts_prefix: Array,
    horizon: int,
    num_samples: int,
    config: ModelConfig,
) -> ForecastSamples:
    rng = np.random.default_rng(_seed_from_key(key))
    posterior = _infer_point_prefix_numpy(params, counts_prefix, config)
    terminal_mean = np.asarray(posterior.mean)[:, -1]
    terminal_cov = np.asarray(posterior.cov)[:, -1]
    terminal_chol = np.stack([np.linalg.cholesky(c) for c in terminal_cov])
    eps = rng.normal(size=(num_samples,) + terminal_mean.shape)
    z = terminal_mean[None] + np.einsum("bij,sbj->sbi", terminal_chol, eps)
    A = np.asarray(params.dynamics.A)
    q_diag = np.exp(np.asarray(params.dynamics.noise.log_q_diag)) + config.variance_floor
    latent_samples = []
    for _ in range(horizon):
        z = np.einsum("ij,sbj->sbi", A, z) + rng.normal(size=z.shape) * np.sqrt(q_diag)
        latent_samples.append(z.copy())
    latents = np.stack(latent_samples, axis=2)
    log_rates = np.stack(
        [_decode_point_numpy(params.decoder, z_s, config) for z_s in latents],
        axis=0,
    )
    rates = np.exp(log_rates)
    counts = rng.poisson(rates * config.dt).astype(np.float32)
    return ForecastSamples(latents, rates, counts)


def forecast_bayesian_model(
    key: Array,
    params: BayesianModelParams,
    counts_prefix: Array,
    horizon: int,
    num_samples: int,
    config: ModelConfig,
) -> ForecastSamples:
    rng = np.random.default_rng(_seed_from_key(key))
    posterior = _infer_bayesian_prefix_numpy(params, counts_prefix, config)
    terminal_mean = np.asarray(posterior.mean)[:, -1]
    terminal_cov = np.asarray(posterior.cov)[:, -1]
    terminal_chol = np.stack([np.linalg.cholesky(c) for c in terminal_cov])
    z = terminal_mean[None] + np.einsum(
        "bij,sbj->sbi",
        terminal_chol,
        rng.normal(size=(num_samples,) + terminal_mean.shape),
    )
    A_mean = np.asarray(params.qA.mean)
    A_cov = np.stack([_numpy_chol_solve(np.asarray(P), np.eye(P.shape[0])) for P in params.qA.precision])
    A_chol = np.stack([np.linalg.cholesky(c) for c in A_cov])
    A_samples = A_mean[None] + np.einsum(
        "rij,srj->sri",
        A_chol,
        rng.normal(size=(num_samples,) + A_mean.shape),
    )
    W_mean = np.asarray(params.qW1.mean)
    W_samples = W_mean[None] + rng.normal(
        size=(num_samples,) + W_mean.shape
    ) / np.sqrt(np.asarray(params.qW1.precision))[None]
    q_diag = np.exp(np.asarray(params.noise.log_q_diag)) + config.variance_floor
    latent_samples = []
    for _ in range(horizon):
        z = np.einsum("sij,sbj->sbi", A_samples, z) + rng.normal(size=z.shape) * np.sqrt(q_diag)
        latent_samples.append(z.copy())
    latents = np.stack(latent_samples, axis=2)
    log_rates = np.stack(
        [
            _decode_bayesian_numpy(params.decoder_tail, W_samples[s], latents[s], config)
            for s in range(num_samples)
        ],
        axis=0,
    )
    rates = np.exp(log_rates)
    counts = rng.poisson(rates * config.dt).astype(np.float32)
    return ForecastSamples(latents, rates, counts)


def bayesian_uncertainty_decomposition(
    key: Array,
    params: BayesianModelParams,
    counts_prefix: Array,
    horizon: int,
    num_global_samples: int,
    num_aleatoric_samples: int,
    config: ModelConfig,
) -> Mapping[str, Array]:
    rng = np.random.default_rng(_seed_from_key(key))
    posterior = _infer_bayesian_prefix_numpy(params, counts_prefix, config)
    terminal_mean = np.asarray(posterior.mean)[:, -1]
    terminal_cov = np.asarray(posterior.cov)[:, -1]
    terminal_chol = np.stack([np.linalg.cholesky(c) for c in terminal_cov])
    eps = rng.normal(
        size=(num_global_samples, num_aleatoric_samples) + terminal_mean.shape
    )
    z = terminal_mean[None, None] + np.einsum("bij,grbj->grbi", terminal_chol, eps)
    A_mean = np.asarray(params.qA.mean)
    A_cov = np.stack([_numpy_chol_solve(np.asarray(P), np.eye(P.shape[0])) for P in params.qA.precision])
    A_chol = np.stack([np.linalg.cholesky(c) for c in A_cov])
    A_samples = A_mean[None] + np.einsum(
        "rij,grj->gri",
        A_chol,
        rng.normal(size=(num_global_samples,) + A_mean.shape),
    )
    W_mean = np.asarray(params.qW1.mean)
    W_samples = W_mean[None] + rng.normal(
        size=(num_global_samples,) + W_mean.shape
    ) / np.sqrt(np.asarray(params.qW1.precision))[None]
    q_diag = np.exp(np.asarray(params.noise.log_q_diag)) + config.variance_floor
    all_counts = []
    for _ in range(horizon):
        z = np.einsum("gij,grbj->grbi", A_samples, z) + rng.normal(size=z.shape) * np.sqrt(q_diag)
        log_rates = np.stack(
            [
                np.stack(
                    [
                        _decode_bayesian_numpy(
                            params.decoder_tail, W_samples[g], z[g, r], config
                        )
                        for r in range(num_aleatoric_samples)
                    ],
                    axis=0,
                )
                for g in range(num_global_samples)
            ],
            axis=0,
        )
        all_counts.append(rng.poisson(np.exp(log_rates) * config.dt))
    count_samples = np.stack(all_counts, axis=3)  # [G,R,B,H,D]
    conditional_mean = count_samples.mean(axis=1)
    aleatoric = count_samples.var(axis=1).mean(axis=0)
    epistemic = conditional_mean.var(axis=0)
    return {
        "aleatoric": aleatoric.mean(axis=(0, 2)),
        "epistemic": epistemic.mean(axis=(0, 2)),
        "total": (aleatoric + epistemic).mean(axis=(0, 2)),
    }
