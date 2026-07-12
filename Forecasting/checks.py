from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from data import SimulationConfig, simulate_dataset
from inference import gaussian_row_covariance, point_information_blocks, smooth_information
from model import ModelConfig, forecast_origin_site_mask, initialise_bayesian_model, initialise_point_model, recognise
from train import BayesianTrainingConfig, poisson_ggn_diagonal_w1


## Self checks

def run_self_checks(seed: int = 0, raise_on_failure: bool = True) -> Mapping[str, float]:
    """Run compact algebraic checks against dense Gaussian calculations."""

    simulation = SimulationConfig(
        num_train=2,
        num_validation=1,
        num_test=1,
        time_steps=8,
        observed_steps=6,
        decoder_hidden_dim=5,
        num_observed_dims=4,
        seed=seed,
    )
    model_config = ModelConfig(
        latent_dim=3,
        recognition_hidden_dim=6,
        decoder_hidden_dim=5,
        dt=simulation.dt,
    )
    dataset = simulate_dataset(simulation)
    counts = dataset.arrays("train")[0][:, : simulation.observed_steps]
    params = initialise_point_model(jax.random.PRNGKey(seed + 1), counts, model_config)
    site_mask = forecast_origin_site_mask(2, simulation.observed_steps, 2)
    sites = recognise(params.recognition, counts, site_mask, model_config)
    diagonal, lower, eta = point_information_blocks(
        params.dynamics, sites, model_config
    )
    posterior = smooth_information(
        diagonal, lower, eta, model_config.information_jitter
    )

    n = 0
    T = simulation.observed_steps
    K = model_config.latent_dim
    dense_precision = np.zeros((T * K, T * K))
    dense_eta = np.asarray(eta[n]).reshape(-1)
    for t in range(T):
        sl = slice(t * K, (t + 1) * K)
        dense_precision[sl, sl] = np.asarray(diagonal[n, t])
        dense_precision[sl, sl] += model_config.information_jitter * np.eye(K)
    for t in range(T - 1):
        current = slice(t * K, (t + 1) * K)
        following = slice((t + 1) * K, (t + 2) * K)
        dense_precision[following, current] = np.asarray(lower[n, t])
        dense_precision[current, following] = np.asarray(lower[n, t]).T
    dense_cov = np.linalg.inv(dense_precision)
    dense_mean = dense_cov @ dense_eta
    mean_error = float(
        np.max(np.abs(np.asarray(posterior.mean[n]).reshape(-1) - dense_mean))
    )
    covariance_error = 0.0
    cross_error = 0.0
    for t in range(T):
        sl = slice(t * K, (t + 1) * K)
        covariance_error = max(
            covariance_error,
            float(np.max(np.abs(np.asarray(posterior.cov[n, t]) - dense_cov[sl, sl]))),
        )
        if t < T - 1:
            following = slice((t + 1) * K, (t + 2) * K)
            cross_error = max(
                cross_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(posterior.cross_cov[n, t])
                            - dense_cov[following, sl]
                        )
                    )
                ),
            )

    masked_precision_max = float(
        np.max(np.abs(np.asarray(sites.precision_diag)[:, -2:]))
    )
    bayes_config = BayesianTrainingConfig(num_steps=1, batch_size=2)
    bayes = initialise_bayesian_model(
        jax.random.PRNGKey(seed + 2), counts, model_config, bayes_config
    )
    A_cov = np.asarray(
        gaussian_row_covariance(bayes.qA, model_config.information_jitter)
    )
    q_diag = np.exp(np.asarray(bayes.noise.log_q_diag)) + model_config.variance_floor
    covariance_precision_contribution = sum(
        A_cov[j] / q_diag[j] for j in range(K)
    )
    minimum_expected_precision_eigenvalue = float(
        np.min(np.linalg.eigvalsh(covariance_precision_contribution))
    )

    z_sample = np.asarray(posterior.mean[:1])[None]
    curvature = poisson_ggn_diagonal_w1(
        bayes.qW1.mean,
        bayes.decoder_tail,
        jnp.asarray(z_sample),
        1.0,
        model_config,
    )
    minimum_ggn = float(np.min(np.asarray(curvature)))
    results = {
        "dense_mean_max_abs_error": mean_error,
        "dense_covariance_max_abs_error": covariance_error,
        "cross_covariance_max_abs_error": cross_error,
        "masked_site_precision_max_abs": masked_precision_max,
        "minimum_A_covariance_precision_contribution_eigenvalue": minimum_expected_precision_eigenvalue,
        "minimum_poisson_ggn_diagonal": minimum_ggn,
    }
    failures = []
    if mean_error > 2e-4:
        failures.append("block mean")
    if covariance_error > 2e-4:
        failures.append("block covariance")
    if cross_error > 2e-4:
        failures.append("cross-covariance orientation")
    if masked_precision_max != 0.0:
        failures.append("masked recognition sites")
    if minimum_expected_precision_eigenvalue < -1e-7:
        failures.append("E[A^T Q^-1 A] covariance term")
    if minimum_ggn < -1e-7:
        failures.append("Poisson GGN positivity")
    if failures and raise_on_failure:
        raise AssertionError("Self-checks failed: " + ", ".join(failures))
    return results
