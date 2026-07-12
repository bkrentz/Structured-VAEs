from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from data import SyntheticDataset
from forecast import ForecastSamples
from utils import Array


## Plotting helpers

def _save_figure(fig: Any, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_dgp_example(
    dataset: SyntheticDataset,
    split: str = "train",
    sequence_index: int = 0,
    observed_dimensions: Sequence[int] = (0, 1, 2),
    output_path: str | Path | None = None,
):
    """Show one latent path, rate heatmap, and the forecast boundary."""

    import matplotlib.pyplot as plt

    _, rates, latents = dataset.arrays(split)
    rates = np.asarray(rates[sequence_index])
    latents = np.asarray(latents[sequence_index])
    boundary = dataset.config.observed_steps
    time = np.arange(dataset.config.time_steps)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].plot(latents[:, 0], latents[:, 1], linewidth=2)
    axes[0].scatter(latents[0, 0], latents[0, 1], marker="o", label="start")
    axes[0].scatter(latents[-1, 0], latents[-1, 1], marker="x", label="end")
    axes[0].set_title("True latent dynamics: damped rotation")
    axes[0].set_xlabel("latent coordinate 1")
    axes[0].set_ylabel("latent coordinate 2")
    axes[0].legend()

    image = axes[1].imshow(rates.T, aspect="auto", interpolation="nearest")
    axes[1].axvline(boundary - 0.5, linestyle="--", linewidth=1.5)
    axes[1].set_title("Mean Poisson rates and forecast origin")
    axes[1].set_xlabel("time bin")
    axes[1].set_ylabel("observation dimension")
    fig.colorbar(image, ax=axes[1], label="rate per unit time")

    for dimension in observed_dimensions:
        axes[2].plot(time, rates[:, dimension], label=f"dimension {dimension}")
    axes[2].axvline(boundary - 0.5, linestyle="--", linewidth=1.5)
    axes[2].set_title("Selected rate trajectories")
    axes[2].set_xlabel("time bin")
    axes[2].set_ylabel("rate per unit time")
    axes[2].legend(fontsize=8)
    _save_figure(fig, output_path)
    return fig, axes


def plot_training_histories(
    histories: Mapping[str, Sequence[Mapping[str, float]]],
    output_path: str | Path | None = None,
):
    """Plot training and internal masked-validation objectives."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for label, history in histories.items():
        steps = [entry["step"] for entry in history]
        axes[0].plot(steps, [entry["free_energy"] for entry in history], label=label)
        axes[1].plot(
            steps,
            [entry["validation_free_energy"] for entry in history],
            label=label,
        )
    axes[0].set_title("Training restricted free energy")
    axes[1].set_title("Validation restricted free energy")
    for ax in axes:
        ax.set_xlabel("training step")
        ax.set_ylabel("free energy per sequence")
        ax.legend()
    _save_figure(fig, output_path)
    return fig, axes


def plot_forecast_sequence(
    full_rates: Array,
    forecast: ForecastSamples,
    observed_steps: int,
    sequence_index: int = 0,
    observed_dimensions: Sequence[int] = (0, 1, 2),
    title: str = "Suffix forecast",
    output_path: str | Path | None = None,
):
    """Plot true rates, predictive mean rate, and one/two-sigma bands."""

    import matplotlib.pyplot as plt

    full_rates_np = np.asarray(full_rates[sequence_index])
    samples = np.asarray(forecast.rates[:, sequence_index])
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    horizon = samples.shape[1]
    full_time = np.arange(full_rates_np.shape[0])
    forecast_time = np.arange(observed_steps, observed_steps + horizon)

    fig, axes = plt.subplots(
        len(observed_dimensions),
        1,
        figsize=(11, 2.7 * len(observed_dimensions)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, dimension in zip(axes, observed_dimensions):
        ax.plot(
            full_time[:observed_steps],
            full_rates_np[:observed_steps, dimension],
            label="observed prefix rate",
        )
        ax.plot(
            full_time[observed_steps:],
            full_rates_np[observed_steps:, dimension],
            label="withheld true rate",
        )
        forecast_line = ax.plot(
            forecast_time,
            mean[:, dimension],
            linewidth=2,
            label="predictive mean rate",
        )[0]
        colour = forecast_line.get_color()
        ax.fill_between(
            forecast_time,
            np.maximum(mean[:, dimension] - 2.0 * std[:, dimension], 0.0),
            mean[:, dimension] + 2.0 * std[:, dimension],
            alpha=0.15,
            color=colour,
            label="mean +/- 2 sd",
        )
        ax.fill_between(
            forecast_time,
            np.maximum(mean[:, dimension] - std[:, dimension], 0.0),
            mean[:, dimension] + std[:, dimension],
            alpha=0.28,
            color=colour,
            label="mean +/- 1 sd",
        )
        ax.axvline(observed_steps - 0.5, linestyle="--", linewidth=1.5)
        ax.set_ylabel(f"rate {dimension}")
        ax.legend(fontsize=8, ncol=3)
    axes[0].set_title(title)
    axes[-1].set_xlabel("time bin")
    _save_figure(fig, output_path)
    return fig, axes


def plot_horizon_metrics(
    metrics_by_model: Mapping[str, Mapping[str, Array]],
    output_path: str | Path | None = None,
):
    """Compare count error and probabilistic score over forecast horizon."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for label, metrics in metrics_by_model.items():
        horizon = np.arange(1, len(np.asarray(metrics["rmse"])) + 1)
        axes[0].plot(horizon, np.asarray(metrics["rmse"]), marker="o", label=label)
        axes[1].plot(horizon, np.asarray(metrics["nlpd"]), marker="o", label=label)
    axes[0].set_title("Forecast RMSE grows with horizon")
    axes[1].set_title("Held-out Poisson negative log predictive density")
    axes[0].set_ylabel("RMSE")
    axes[1].set_ylabel("NLPD")
    for ax in axes:
        ax.set_xlabel("forecast horizon")
        ax.legend()
    _save_figure(fig, output_path)
    return fig, axes


def plot_interval_coverage(
    metrics_by_model: Mapping[str, Mapping[str, Array]],
    output_path: str | Path | None = None,
):
    """Show empirical coverage of 68% and 95% predictive intervals."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for label, metrics in metrics_by_model.items():
        horizon = np.arange(1, len(np.asarray(metrics["coverage68"])) + 1)
        axes[0].plot(horizon, np.asarray(metrics["coverage68"]), marker="o", label=label)
        axes[1].plot(horizon, np.asarray(metrics["coverage95"]), marker="o", label=label)
    axes[0].axhline(0.68, linestyle="--", linewidth=1.5)
    axes[1].axhline(0.95, linestyle="--", linewidth=1.5)
    axes[0].set_title("Coverage of nominal 68% intervals")
    axes[1].set_title("Coverage of nominal 95% intervals")
    for ax in axes:
        ax.set_xlabel("forecast horizon")
        ax.set_ylabel("empirical coverage")
        ax.set_ylim(0.0, 1.05)
        ax.legend()
    _save_figure(fig, output_path)
    return fig, axes


def plot_transition_eigenvalues(
    true_A: Array,
    learned: Mapping[str, Array],
    output_path: str | Path | None = None,
):
    """Compare basis-invariant transition eigenvalues in the complex plane."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), linestyle="--", linewidth=1, label="unit circle")
    true_eigenvalues = np.linalg.eigvals(np.asarray(true_A))
    ax.scatter(
        true_eigenvalues.real,
        true_eigenvalues.imag,
        marker="x",
        s=90,
        label="true",
    )
    for label, matrix in learned.items():
        eigenvalues = np.linalg.eigvals(np.asarray(matrix))
        ax.scatter(eigenvalues.real, eigenvalues.imag, s=55, label=label)
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("real part")
    ax.set_ylabel("imaginary part")
    ax.set_title("Learned transition eigenvalues")
    ax.legend(loc="upper right")
    _save_figure(fig, output_path)
    return fig, ax


def plot_ard(
    diagnostics: Mapping[str, Array],
    output_path: str | Path | None = None,
):
    """Show shared ARD precision by fitted latent coordinate."""

    import matplotlib.pyplot as plt

    alpha = np.asarray(diagnostics["alpha"])
    coordinate = np.arange(1, len(alpha) + 1)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.bar(coordinate, alpha)
    ax.set_yscale("log")
    ax.set_title("Large ARD precision means stronger shrinkage")
    ax.set_xlabel("fitted latent coordinate")
    ax.set_ylabel("ARD precision alpha")
    _save_figure(fig, output_path)
    return fig, ax


def plot_masking_intervention(
    masked_metrics: Mapping[str, Array],
    unmasked_metrics: Mapping[str, Array],
    output_path: str | Path | None = None,
):
    """One compact ablation: forecast score with versus without suffix-site masking."""

    import matplotlib.pyplot as plt

    horizon = np.arange(1, len(np.asarray(masked_metrics["nlpd"])) + 1)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(horizon, np.asarray(masked_metrics["nlpd"]), marker="o", label="forecast-origin masking")
    ax.plot(horizon, np.asarray(unmasked_metrics["nlpd"]), marker="o", label="ordinary unmasked training")
    ax.set_title("Forecast-origin masking tests whether the latent dynamics are usable")
    ax.set_xlabel("forecast horizon")
    ax.set_ylabel("held-out Poisson NLPD")
    ax.legend()
    _save_figure(fig, output_path)
    return fig, ax


def plot_uncertainty_decomposition(
    decomposition: Mapping[str, Array],
    output_path: str | Path | None = None,
):
    """Plot average aleatoric and epistemic predictive count variance."""

    import matplotlib.pyplot as plt

    horizon = np.arange(1, len(np.asarray(decomposition["total"])) + 1)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(horizon, np.asarray(decomposition["aleatoric"]), marker="o", label="aleatoric")
    ax.plot(horizon, np.asarray(decomposition["epistemic"]), marker="o", label="global-parameter epistemic")
    ax.plot(horizon, np.asarray(decomposition["total"]), marker="o", label="total")
    ax.set_title("Bayesian forecast uncertainty accumulates over the horizon")
    ax.set_xlabel("forecast horizon")
    ax.set_ylabel("average predictive count variance")
    ax.legend()
    _save_figure(fig, output_path)
    return fig, ax


__all__ = [
    "SimulationConfig",
    "ModelConfig",
    "PointTrainingConfig",
    "BayesianTrainingConfig",
    "ForecastConfig",
    "SyntheticDataset",
    "PointModelParams",
    "BayesianModelParams",
    "ForecastSamples",
    "simulate_dataset",
    "forecast_origin_site_mask",
    "initialise_point_model",
    "initialise_bayesian_model",
    "initialise_bayesian_from_point",
    "fit_point_model",
    "fit_bayesian_model",
    "infer_point_prefix",
    "infer_bayesian_prefix",
    "forecast_point_model",
    "forecast_bayesian_model",
    "bayesian_uncertainty_decomposition",
    "probabilistic_forecast_metrics",
    "point_latent_diagnostics",
    "bayesian_latent_diagnostics",
    "ard_diagnostics",
    "plot_dgp_example",
    "plot_training_histories",
    "plot_forecast_sequence",
    "plot_horizon_metrics",
    "plot_interval_coverage",
    "plot_transition_eigenvalues",
    "plot_ard",
    "plot_masking_intervention",
    "plot_uncertainty_decomposition",
    "run_self_checks",
]
