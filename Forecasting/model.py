from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from data import make_damped_rotation
from utils import Array, _glorot, _log_variance_parameter, _softplus_inverse

if TYPE_CHECKING:
    from train import BayesianTrainingConfig


## Model settings

@dataclass(frozen=True)
class ModelConfig:
    """Shared architecture and variance settings for both fitted models."""

    latent_dim: int = 20
    recognition_hidden_dim: int = 24
    decoder_hidden_dim: int = 16
    dt: float = 0.20
    min_site_precision: float = 1e-4
    variance_floor: float = 1e-4
    initial_process_std: float = 0.20
    initial_site_precision: float = 0.35
    log_rate_clip: float = 7.0
    information_jitter: float = 1e-5


## Parameter containers

class RecognitionParams(NamedTuple):
    W1: Array
    b1: Array
    W_mean: Array
    b_mean: Array
    W_precision: Array
    b_precision: Array


class DecoderParams(NamedTuple):
    W1: Array
    b1: Array
    W2: Array
    b2: Array


class DecoderTailParams(NamedTuple):
    b1: Array
    W2: Array
    b2: Array


class NoiseParams(NamedTuple):
    log_q_diag: Array
    m0: Array
    log_s0_diag: Array


class PointDynamicsParams(NamedTuple):
    A: Array
    noise: NoiseParams


class PointNeuralParams(NamedTuple):
    recognition: RecognitionParams
    decoder: DecoderParams


class PointModelParams(NamedTuple):
    dynamics: PointDynamicsParams
    recognition: RecognitionParams
    decoder: DecoderParams


class GaussianRowPosterior(NamedTuple):
    """Row-factorised Gaussian posterior represented in natural form."""

    mean: Array  # [row, column]
    precision: Array  # [row, column, column]
    natural_mean: Array  # [row, column]


class DiagonalGaussianPosterior(NamedTuple):
    """Diagonal Gaussian posterior represented in natural form."""

    mean: Array
    precision: Array
    natural_mean: Array


class BayesianNeuralParams(NamedTuple):
    recognition: RecognitionParams
    decoder_tail: DecoderTailParams


class BayesianModelParams(NamedTuple):
    qA: GaussianRowPosterior
    noise: NoiseParams
    qW1: DiagonalGaussianPosterior
    log_alpha: Array
    recognition: RecognitionParams
    decoder_tail: DecoderTailParams


class GaussianSites(NamedTuple):
    mean: Array
    precision_diag: Array
    h: Array


## Model initialisation and neural mappings

def _initial_transition(latent_dim: int) -> Array:
    A = 0.65 * jnp.eye(latent_dim)
    if latent_dim >= 2:
        A = A.at[:2, :2].set(make_damped_rotation(0.90, 0.10))
    return A


def _initial_recognition(
    key: Array,
    num_observed_dims: int,
    config: ModelConfig,
) -> RecognitionParams:
    k1, k2, k3 = jax.random.split(key, 3)
    coordinate_scale = jnp.where(
        jnp.arange(config.latent_dim) < min(2, config.latent_dim), 1.0, 0.12
    )
    W_mean = 0.08 * _glorot(
        k2, (config.latent_dim, config.recognition_hidden_dim)
    )
    W_mean = W_mean * coordinate_scale[:, None]
    return RecognitionParams(
        W1=_glorot(k1, (config.recognition_hidden_dim, num_observed_dims)),
        b1=jnp.zeros((config.recognition_hidden_dim,)),
        W_mean=W_mean,
        b_mean=jnp.zeros((config.latent_dim,)),
        W_precision=0.02
        * _glorot(k3, (config.latent_dim, config.recognition_hidden_dim)),
        b_precision=_softplus_inverse(config.initial_site_precision)
        * jnp.ones((config.latent_dim,)),
    )


def _initial_decoder(
    key: Array,
    counts_prefix: Array,
    config: ModelConfig,
) -> DecoderParams:
    k1, k2 = jax.random.split(key)
    num_observed_dims = counts_prefix.shape[-1]
    mean_counts = jnp.mean(counts_prefix, axis=(0, 1))
    output_bias = jnp.log(jnp.maximum(mean_counts / config.dt, 0.25))
    coordinate_scale = jnp.where(
        jnp.arange(config.latent_dim) < min(2, config.latent_dim), 1.0, 0.12
    )
    W1 = _glorot(k1, (config.decoder_hidden_dim, config.latent_dim))
    W1 = W1 * coordinate_scale[None, :]
    return DecoderParams(
        W1=W1,
        b1=jnp.zeros((config.decoder_hidden_dim,)),
        W2=0.08 * _glorot(k2, (num_observed_dims, config.decoder_hidden_dim)),
        b2=output_bias,
    )


def initialise_point_model(
    key: Array,
    counts_prefix: Array,
    config: ModelConfig,
) -> PointModelParams:
    key_rec, key_dec = jax.random.split(key)
    recognition = _initial_recognition(key_rec, counts_prefix.shape[-1], config)
    decoder = _initial_decoder(key_dec, counts_prefix, config)
    noise = NoiseParams(
        log_q_diag=_log_variance_parameter(
            jnp.full((config.latent_dim,), config.initial_process_std**2),
            config.variance_floor,
        ),
        m0=jnp.zeros((config.latent_dim,)),
        log_s0_diag=_log_variance_parameter(
            jnp.ones((config.latent_dim,)), config.variance_floor
        ),
    )
    dynamics = PointDynamicsParams(_initial_transition(config.latent_dim), noise)
    return PointModelParams(dynamics, recognition, decoder)


def initialise_bayesian_model(
    key: Array,
    counts_prefix: Array,
    config: ModelConfig,
    train_config: BayesianTrainingConfig,
) -> BayesianModelParams:
    key_rec, key_dec = jax.random.split(key)
    recognition = _initial_recognition(key_rec, counts_prefix.shape[-1], config)
    decoder = _initial_decoder(key_dec, counts_prefix, config)
    A_mean = _initial_transition(config.latent_dim)
    alpha = jnp.ones((config.latent_dim,))
    prior_precision_a = train_config.kappa_a * alpha
    A_precision = jnp.tile(
        jnp.diag(prior_precision_a + 200.0)[None], (config.latent_dim, 1, 1)
    )
    A_natural = jnp.einsum("rij,rj->ri", A_precision, A_mean)
    qA = GaussianRowPosterior(A_mean, A_precision, A_natural)

    W1_mean = decoder.W1
    W1_precision = train_config.kappa_w * alpha[None, :] + 3.0
    W1_precision = jnp.broadcast_to(W1_precision, W1_mean.shape)
    qW1 = DiagonalGaussianPosterior(
        W1_mean,
        W1_precision,
        W1_precision * W1_mean,
    )
    noise = NoiseParams(
        log_q_diag=_log_variance_parameter(
            jnp.full((config.latent_dim,), config.initial_process_std**2),
            config.variance_floor,
        ),
        m0=jnp.zeros((config.latent_dim,)),
        log_s0_diag=_log_variance_parameter(
            jnp.ones((config.latent_dim,)), config.variance_floor
        ),
    )
    tail = DecoderTailParams(decoder.b1, decoder.W2, decoder.b2)
    return BayesianModelParams(qA, noise, qW1, jnp.log(alpha), recognition, tail)



def initialise_bayesian_from_point(
    point_params: PointModelParams,
    train_config: BayesianTrainingConfig,
    initial_a_variance: float = 2e-3,
    initial_w_variance: float = 2e-2,
) -> BayesianModelParams:
    """Convert a point checkpoint into the Bayesian variational family.

    This chooses an initial value; it does not change the Bayesian objective.
    """

    latent_dim = point_params.dynamics.A.shape[0]
    alpha = jnp.ones((latent_dim,), dtype=point_params.dynamics.A.dtype)
    A_precision = jnp.broadcast_to(
        (jnp.eye(latent_dim) / initial_a_variance)[None],
        (latent_dim, latent_dim, latent_dim),
    )
    A_natural = jnp.einsum("rij,rj->ri", A_precision, point_params.dynamics.A)
    qA = GaussianRowPosterior(point_params.dynamics.A, A_precision, A_natural)

    W1_precision = jnp.full_like(point_params.decoder.W1, 1.0 / initial_w_variance)
    qW1 = DiagonalGaussianPosterior(
        point_params.decoder.W1,
        W1_precision,
        W1_precision * point_params.decoder.W1,
    )
    tail = DecoderTailParams(
        point_params.decoder.b1,
        point_params.decoder.W2,
        point_params.decoder.b2,
    )
    return BayesianModelParams(
        qA=qA,
        noise=point_params.dynamics.noise,
        qW1=qW1,
        log_alpha=jnp.log(alpha),
        recognition=point_params.recognition,
        decoder_tail=tail,
    )

def forecast_origin_site_mask(
    batch_size: int,
    time_steps: int,
    mask_horizon: int,
    dtype: Any = jnp.float32,
) -> Array:
    """Return a mask whose final block is absent from recognition sites."""

    if mask_horizon < 0 or mask_horizon >= time_steps:
        raise ValueError("mask_horizon must satisfy 0 <= mask_horizon < time_steps.")
    if mask_horizon == 0:
        row = jnp.ones((time_steps,), dtype=dtype)
    else:
        row = jnp.concatenate(
            [
                jnp.ones((time_steps - mask_horizon,), dtype=dtype),
                jnp.zeros((mask_horizon,), dtype=dtype),
            ]
        )
    return jnp.broadcast_to(row[None], (batch_size, time_steps))


def recognise(
    params: RecognitionParams,
    counts: Array,
    site_mask: Array,
    config: ModelConfig,
) -> GaussianSites:
    """Map local counts to positive diagonal Gaussian-site precision."""

    x = jnp.log1p(counts) * site_mask[..., None]
    hidden = jnp.tanh(x @ params.W1.T + params.b1)
    mean = hidden @ params.W_mean.T + params.b_mean
    raw_precision = hidden @ params.W_precision.T + params.b_precision
    precision = jax.nn.softplus(raw_precision) + config.min_site_precision
    precision = precision * site_mask[..., None]
    h = precision * mean
    return GaussianSites(mean, precision, h)


def decode_log_rates_point(
    params: DecoderParams,
    z: Array,
    config: ModelConfig,
) -> Array:
    hidden = jnp.tanh(z @ params.W1.T + params.b1)
    log_rate = hidden @ params.W2.T + params.b2
    return jnp.clip(log_rate, -config.log_rate_clip, config.log_rate_clip)


def decode_log_rates_bayesian(
    tail: DecoderTailParams,
    W1: Array,
    z: Array,
    config: ModelConfig,
) -> Array:
    hidden = jnp.tanh(z @ W1.T + tail.b1)
    log_rate = hidden @ tail.W2.T + tail.b2
    return jnp.clip(log_rate, -config.log_rate_clip, config.log_rate_clip)


def poisson_log_prob_from_log_rate(
    log_rate: Array,
    counts: Array,
    dt: float,
) -> Array:
    log_mean = log_rate + jnp.log(dt)
    mean = jnp.exp(log_mean)
    return counts * log_mean - mean - gammaln(counts + 1.0)
