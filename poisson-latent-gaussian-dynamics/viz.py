from pathlib import Path

import jax
import matplotlib.pyplot as plt

from data import sample_dataset


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


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    data = sample_dataset(key)
    plot_trial_diagnostics(data, trial_idx=0, output_path="outputs/dgp_diagnostics.png")
    plt.show()
