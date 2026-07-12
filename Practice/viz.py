from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from data import sample_dataset
from inference import kalman_smoother


def plot_trial_diagnostics(data, trial_idx: int = 0, output_path: str | None = None):
    z = data["z"][trial_idx]
    rates = data["rates"][trial_idx]
    counts = data["counts"][trial_idx]

    num_timesteps = counts.shape[0]
    num_neurons = counts.shape[1]
    time = range(num_timesteps)
    neurons_to_show = list(range(min(5, num_neurons)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(z[:, 0], z[:, 1], linewidth=2)
    axes[0, 0].scatter(z[0, 0], z[0, 1], color="green", label="start")
    axes[0, 0].scatter(z[-1, 0], z[-1, 1], color="red", label="end")
    axes[0, 0].set_title("latent trajectory")
    axes[0, 0].set_xlabel("z[0]")
    axes[0, 0].set_ylabel("z[1]")
    axes[0, 0].legend()

    count_image = axes[0, 1].imshow(counts.T, aspect="auto", interpolation="nearest")
    axes[0, 1].set_title("spike counts")
    axes[0, 1].set_xlabel("time bin")
    axes[0, 1].set_ylabel("neuron")
    fig.colorbar(count_image, ax=axes[0, 1], label="count")

    rate_image = axes[1, 0].imshow(rates.T, aspect="auto", interpolation="nearest")
    axes[1, 0].set_title("true firing rates")
    axes[1, 0].set_xlabel("time bin")
    axes[1, 0].set_ylabel("neuron")
    fig.colorbar(rate_image, ax=axes[1, 0], label="rate")

    for neuron_idx in neurons_to_show:
        axes[1, 1].plot(time, rates[:, neuron_idx], label=f"neuron {neuron_idx}")
    axes[1, 1].set_title("selected true firing rates")
    axes[1, 1].set_xlabel("time bin")
    axes[1, 1].set_ylabel("spikes per unit time")
    axes[1, 1].legend(fontsize=8)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    return fig, axes


def _save_if_requested(fig, output_path):
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)


## Training diagnostics

def history_to_arrays(history):
    """
    Convert training history into arrays
    """
    if len(history) == 0:
        return {}

    return {
        key: jnp.asarray([entry[key] for entry in history])
        for key in history[0]
    }


def plot_training_history(history, output_path: str | None = None):
    """
    Plot optimisation traces
    """
    arrays = history_to_arrays(history)
    if len(arrays) == 0:
        raise ValueError("history is empty")

    steps = np.asarray(arrays.get("step", jnp.arange(len(next(iter(arrays.values()))))))
    series = [
        ("free_energy", "free energy"),
        ("mean_expected_log_likelihood", "expected log likelihood"),
        ("mean_kl", "KL"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    axes = axes.ravel()

    for ax, (key, title) in zip(axes, series):
        if key not in arrays:
            ax.axis("off")
            continue

        ax.plot(steps, np.asarray(arrays[key]), linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("step")

    _save_if_requested(fig, output_path)

    return fig, axes


## Posterior diagnostics

def infer_posterior(model, params, counts, mask = None, jitter = 1e-6):
    """
    Run recognition and Kalman smoothing
    """
    sites = model.recognise(params.recognition, counts, mask)

    return kalman_smoother(
        params.dynamics,
        sites,
        transition_mask = model.transition_mask,
        variance_floor = model.variance_floor,
        jitter = jitter,
    )


def _flatten_latents(z):
    return jnp.reshape(z, (-1, z.shape[-1]))


def fit_linear_alignment(source, target, ridge = 1e-6):
    """
    Fit linear map from source latents to target latents
    """
    source_flat = _flatten_latents(source)
    target_flat = _flatten_latents(target)

    source_mean = jnp.mean(source_flat, axis=0)
    target_mean = jnp.mean(target_flat, axis=0)

    source_centred = source_flat - source_mean
    target_centred = target_flat - target_mean

    latent_dim = source_centred.shape[-1]
    eye = jnp.eye(latent_dim)

    matrix = jnp.linalg.solve(
        source_centred.T @ source_centred + ridge * eye,
        source_centred.T @ target_centred,
    )

    return {
        "matrix": matrix,
        "source_mean": source_mean,
        "target_mean": target_mean,
    }


def apply_linear_alignment(source, alignment):
    return (
        (source - alignment["source_mean"]) @ alignment["matrix"]
        + alignment["target_mean"]
    )


def align_covariances(cov, alignment):
    W = alignment["matrix"]
    return jnp.einsum("ki,...kl,lj->...ij", W, cov, W)


def align_latents(source, target, ridge = 1e-6):
    alignment = fit_linear_alignment(source, target, ridge)
    aligned = apply_linear_alignment(source, alignment)
    return aligned, alignment


def latent_r2(target, estimate, eps = 1e-8):
    target_flat = _flatten_latents(target)
    estimate_flat = _flatten_latents(estimate)

    residual = jnp.sum((target_flat - estimate_flat) ** 2)
    total = jnp.sum((target_flat - jnp.mean(target_flat, axis=0)) ** 2)

    return 1.0 - residual / jnp.maximum(total, eps)


def plot_latent_recovery(
    true_z,
    posterior_mean,
    posterior_cov = None,
    trial_idx = 0,
    align = True,
    credible_interval = 0.95,
    ellipse_stride = 1,
    output_path: str | None = None,
):
    """
    Plot true and inferred latent trajectories
    """
    if align:
        plotted_mean, alignment = align_latents(posterior_mean, true_z)
        plotted_cov = None if posterior_cov is None else align_covariances(posterior_cov, alignment)
    else:
        plotted_mean = posterior_mean
        plotted_cov = posterior_cov
        alignment = None

    true_trial = np.asarray(true_z[trial_idx])
    mean_trial = np.asarray(plotted_mean[trial_idx])
    time = np.arange(true_trial.shape[0])
    interval_pct = int(round(credible_interval * 100))
    z_score = 1.6448536269514722 if abs(credible_interval - 0.90) < 1e-6 else 1.959963984540054
    ellipse_scale = np.sqrt(-2.0 * np.log(max(1.0 - credible_interval, 1e-12)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    axes[0].plot(true_trial[:, 0], true_trial[:, 1], label="true", linewidth=2)
    inferred_line = axes[0].plot(mean_trial[:, 0], mean_trial[:, 1], label="inferred", linewidth=2)[0]

    if plotted_cov is not None:
        from matplotlib.patches import Ellipse

        cov_trial = np.asarray(plotted_cov[trial_idx])
        ellipse_color = inferred_line.get_color()
        stride = max(int(ellipse_stride), 1)

        for idx in range(0, mean_trial.shape[0], stride):
            cov_2d = 0.5 * (cov_trial[idx] + cov_trial[idx].T)
            eigvals, eigvecs = np.linalg.eigh(cov_2d)
            eigvals = np.maximum(eigvals, 1e-12)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            width, height = 2.0 * ellipse_scale * np.sqrt(eigvals)
            ellipse = Ellipse(
                xy=mean_trial[idx],
                width=width,
                height=height,
                angle=angle,
                facecolor="none",
                edgecolor=ellipse_color,
                linewidth=0.7,
                alpha=0.35,
                label=f"{interval_pct}% marginal posterior ellipses" if idx == 0 else None,
            )
            axes[0].add_patch(ellipse)

    axes[0].set_title(
        f"latent trajectory with {interval_pct}% ellipses"
        if plotted_cov is not None
        else "latent trajectory"
    )
    axes[0].set_xlabel("z[0]")
    axes[0].set_ylabel("z[1]")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend()

    if plotted_cov is not None:
        std_trial = np.sqrt(
            np.maximum(
                np.diagonal(np.asarray(plotted_cov[trial_idx]), axis1=-2, axis2=-1),
                1e-10,
            )
        )
    else:
        std_trial = None

    for dim, ax in enumerate(axes[1:]):
        if std_trial is not None:
            lower = mean_trial[:, dim] - z_score * std_trial[:, dim]
            upper = mean_trial[:, dim] + z_score * std_trial[:, dim]
            ax.fill_between(time, lower, upper, alpha=0.25, label=f"{interval_pct}% marginal posterior band")
        ax.plot(time, true_trial[:, dim], label="true", linewidth=2)
        ax.plot(time, mean_trial[:, dim], label="inferred", linewidth=2)
        ax.set_title(f"latent dim {dim}")
        ax.set_xlabel("time bin")
        ax.legend(fontsize=8)

    _save_if_requested(fig, output_path)

    return fig, axes, alignment


## Dynamics diagnostics

def transform_dynamics_to_target_coords(A, alignment):
    """
    Transform learned A into aligned latent coordinates
    """
    W = alignment["matrix"]
    transformed_T = jnp.linalg.pinv(W) @ A.T @ W
    return transformed_T.T


def plot_transition_matrices(
    true_A,
    learned_A,
    alignment = None,
    output_path: str | None = None,
):
    """
    Plot true and learned transition matrices
    """
    matrices = [
        ("true A", true_A),
        ("learned A", learned_A),
    ]

    if alignment is not None:
        matrices.append(
            ("learned A aligned", transform_dynamics_to_target_coords(learned_A, alignment))
        )

    values = jnp.concatenate([jnp.ravel(matrix) for _, matrix in matrices])
    vmax = float(jnp.max(jnp.abs(values)))
    vmin = -vmax

    fig, axes = plt.subplots(
        1,
        len(matrices),
        figsize=(4 * len(matrices), 4),
        constrained_layout=True,
    )

    if len(matrices) == 1:
        axes = [axes]

    for ax, (title, matrix) in zip(axes, matrices):
        image = ax.imshow(
            np.asarray(matrix),
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
        )
        ax.set_title(title)
        fig.colorbar(image, ax=ax)

    _save_if_requested(fig, output_path)

    return fig, axes


def fit_diagnostic_metrics(
    true_z,
    posterior_mean,
    true_A = None,
    learned_A = None,
    ridge = 1e-6,
):
    """
    Compute scalar recovery diagnostics
    """
    aligned_mean, alignment = align_latents(posterior_mean, true_z, ridge)

    metrics = {
        "latent_r2": latent_r2(true_z, aligned_mean),
    }

    if true_A is not None and learned_A is not None:
        learned_A_aligned = transform_dynamics_to_target_coords(learned_A, alignment)
        metrics["transition_relative_error"] = (
            jnp.linalg.norm(learned_A_aligned - true_A)
            / jnp.maximum(jnp.linalg.norm(true_A), ridge)
        )

    return metrics, alignment


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    data = sample_dataset(key)
    plot_trial_diagnostics(data, trial_idx=0, output_path="outputs/dgp_diagnostics.png")
    plt.show()
