from typing import Mapping

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln, logsumexp

from forecast import ForecastSamples, _infer_bayesian_prefix_numpy, _infer_point_prefix_numpy
from model import BayesianModelParams, ModelConfig, PointModelParams
from utils import Array, _chol_solve


## Forecast metrics and latent diagnostics

def forecast_metrics(
    forecast: ForecastSamples,
    truth_counts: Array,
) -> Mapping[str, Array]:
    """Horizon-wise RMSE and interval coverage."""

    predictive_mean = jnp.mean(forecast.counts, axis=0)
    predictive_std = jnp.std(forecast.counts, axis=0)
    error = predictive_mean - truth_counts
    rmse = jnp.sqrt(jnp.mean(jnp.square(error), axis=(0, 2)))

    lower68 = jnp.quantile(forecast.counts, 0.16, axis=0)
    upper68 = jnp.quantile(forecast.counts, 0.84, axis=0)
    lower95 = jnp.quantile(forecast.counts, 0.025, axis=0)
    upper95 = jnp.quantile(forecast.counts, 0.975, axis=0)
    coverage68 = jnp.mean(
        (truth_counts >= lower68) & (truth_counts <= upper68), axis=(0, 2)
    )
    coverage95 = jnp.mean(
        (truth_counts >= lower95) & (truth_counts <= upper95), axis=(0, 2)
    )
    return {
        "predictive_mean": predictive_mean,
        "predictive_std": predictive_std,
        "rmse": rmse,
        "coverage68": coverage68,
        "coverage95": coverage95,
    }


def probabilistic_forecast_metrics(
    forecast: ForecastSamples,
    truth_counts: Array,
    dt: float,
) -> Mapping[str, Array]:
    """Complete horizon metrics using the Poisson-rate mixture."""

    base = dict(forecast_metrics(forecast, truth_counts))
    poisson_mean = jnp.maximum(forecast.rates * dt, 1e-12)
    log_prob = (
        truth_counts[None] * jnp.log(poisson_mean)
        - poisson_mean
        - gammaln(truth_counts[None] + 1.0)
    )
    mixture_log_prob = logsumexp(log_prob, axis=0) - jnp.log(forecast.rates.shape[0])
    base["nlpd"] = -jnp.mean(mixture_log_prob, axis=(0, 2))

    predictive_mean = jnp.maximum(base["predictive_mean"], 1e-8)
    y = truth_counts
    y_log_ratio = jnp.where(y > 0, y * jnp.log(y / predictive_mean), 0.0)
    deviance = 2.0 * (y_log_ratio - (y - predictive_mean))
    base["poisson_deviance"] = jnp.mean(deviance, axis=(0, 2))
    return base


def fit_linear_alignment(
    source: Array,
    target: Array,
    ridge: float = 1e-5,
) -> Mapping[str, Array]:
    """Least-squares map from fitted latent coordinates to true coordinates."""

    source_flat = source.reshape((-1, source.shape[-1]))
    target_flat = target.reshape((-1, target.shape[-1]))
    source_mean = jnp.mean(source_flat, axis=0)
    target_mean = jnp.mean(target_flat, axis=0)
    X = source_flat - source_mean
    Y = target_flat - target_mean
    gram = X.T @ X + ridge * jnp.eye(X.shape[-1])
    matrix = _chol_solve(gram, X.T @ Y)
    return {
        "matrix": matrix,
        "source_mean": source_mean,
        "target_mean": target_mean,
    }


def apply_linear_alignment(source: Array, alignment: Mapping[str, Array]) -> Array:
    return (
        (source - alignment["source_mean"]) @ alignment["matrix"]
        + alignment["target_mean"]
    )


def latent_r2(true_latents: Array, fitted_latents: Array) -> Array:
    alignment = fit_linear_alignment(fitted_latents, true_latents)
    aligned = apply_linear_alignment(fitted_latents, alignment)
    residual = jnp.sum(jnp.square(true_latents - aligned))
    centred = true_latents - jnp.mean(
        true_latents.reshape((-1, true_latents.shape[-1])), axis=0
    )
    total = jnp.sum(jnp.square(centred))
    return 1.0 - residual / jnp.maximum(total, 1e-12)


def ard_diagnostics(
    params: BayesianModelParams,
    config: ModelConfig,
) -> Mapping[str, Array]:
    return {"alpha": jnp.exp(params.log_alpha)}


## Numpy latent diagnostics

def point_latent_diagnostics(
    params: PointModelParams,
    counts_prefix: Array,
    true_latents_prefix: Array,
    config: ModelConfig,
) -> Mapping[str, Array]:
    posterior = _infer_point_prefix_numpy(params, counts_prefix, config)
    return {"latent_r2": latent_r2(jnp.asarray(true_latents_prefix), jnp.asarray(posterior.mean))}


def bayesian_latent_diagnostics(
    params: BayesianModelParams,
    counts_prefix: Array,
    true_latents_prefix: Array,
    config: ModelConfig,
) -> Mapping[str, Array]:
    posterior = _infer_bayesian_prefix_numpy(params, counts_prefix, config)
    return {"latent_r2": latent_r2(jnp.asarray(true_latents_prefix), jnp.asarray(posterior.mean))}
