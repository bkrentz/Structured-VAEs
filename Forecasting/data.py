from dataclasses import dataclass
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp

from utils import Array


## Data settings

@dataclass(frozen=True)
class SimulationConfig:
    """Settings for the synthetic Poisson LDS experiment."""

    num_train: int = 12
    num_validation: int = 2
    num_test: int = 3
    time_steps: int = 100
    observed_steps: int = 50
    true_latent_dim: int = 2
    decoder_hidden_dim: int = 12
    num_observed_dims: int = 10
    dt: float = 0.20
    rotation_radius: float = 0.975
    rotation_angle: float = 0.20
    process_std: tuple[float, ...] = (0.075, 0.055)
    initial_std: tuple[float, ...] = (0.90, 0.70)
    mean_rate: float = 6.0
    seed: int = 7

    @property
    def forecast_horizon(self) -> int:
        return self.time_steps - self.observed_steps


## Data containers

class TrueGenerativeParams(NamedTuple):
    A: Array
    q_diag: Array
    m0: Array
    s0_diag: Array
    W1: Array
    b1: Array
    W2: Array
    b2: Array


@dataclass(frozen=True)
class SyntheticDataset:
    """Full synthetic data and its fixed train/validation/test partition."""

    counts: Array
    rates: Array
    latents: Array
    split: Mapping[str, slice]
    true_params: TrueGenerativeParams
    config: SimulationConfig

    def arrays(self, name: str) -> tuple[Array, Array, Array]:
        sl = self.split[name]
        return self.counts[sl], self.rates[sl], self.latents[sl]


## Synthetic data

def make_damped_rotation(radius: float, angle: float) -> Array:
    return radius * jnp.array(
        [[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]],
        dtype=jnp.float32,
    )


def _simulate_latents(
    key: Array,
    A: Array,
    q_diag: Array,
    m0: Array,
    s0_diag: Array,
    num_sequences: int,
    time_steps: int,
) -> Array:
    key0, key_noise = jax.random.split(key)
    latent_dim = A.shape[0]
    z0 = m0 + jax.random.normal(key0, (num_sequences, latent_dim)) * jnp.sqrt(s0_diag)
    noise = jax.random.normal(
        key_noise, (time_steps - 1, num_sequences, latent_dim)
    ) * jnp.sqrt(q_diag)

    def step(z_prev: Array, eps: Array) -> tuple[Array, Array]:
        z_next = z_prev @ A.T + eps
        return z_next, z_next

    _, rest = jax.lax.scan(step, z0, noise)
    return jnp.swapaxes(jnp.concatenate([z0[None], rest], axis=0), 0, 1)


def _decode_true_rates(params: TrueGenerativeParams, z: Array) -> Array:
    hidden = jnp.tanh(z @ params.W1.T + params.b1)
    log_rate = hidden @ params.W2.T + params.b2
    return jnp.exp(jnp.clip(log_rate, -5.0, 5.0))


def simulate_dataset(config: SimulationConfig = SimulationConfig()) -> SyntheticDataset:
    """Simulate train, validation, and test sequences from one fixed Poisson LDS."""

    if config.true_latent_dim != 2:
        raise ValueError("The visual damped-rotation DGP is defined for true_latent_dim=2.")
    total = config.num_train + config.num_validation + config.num_test
    key = jax.random.PRNGKey(config.seed)
    key_w1, key_w2, key_latent, key_count = jax.random.split(key, 4)

    A = make_damped_rotation(config.rotation_radius, config.rotation_angle)
    q_diag = jnp.square(jnp.asarray(config.process_std, dtype=jnp.float32))
    s0_diag = jnp.square(jnp.asarray(config.initial_std, dtype=jnp.float32))
    m0 = jnp.zeros((config.true_latent_dim,), dtype=jnp.float32)

    W1 = 0.85 * jax.random.normal(
        key_w1, (config.decoder_hidden_dim, config.true_latent_dim)
    )
    W2 = 0.38 * jax.random.normal(
        key_w2, (config.num_observed_dims, config.decoder_hidden_dim)
    )
    b1 = jnp.linspace(-0.35, 0.35, config.decoder_hidden_dim)
    neuron_offsets = jnp.linspace(-0.35, 0.35, config.num_observed_dims)
    b2 = jnp.log(config.mean_rate) + neuron_offsets

    true_params = TrueGenerativeParams(A, q_diag, m0, s0_diag, W1, b1, W2, b2)
    latents = _simulate_latents(
        key_latent, A, q_diag, m0, s0_diag, total, config.time_steps
    )
    rates = _decode_true_rates(true_params, latents)
    counts = jax.random.poisson(key_count, rates * config.dt).astype(jnp.float32)

    train_end = config.num_train
    validation_end = train_end + config.num_validation
    split = {
        "train": slice(0, train_end),
        "validation": slice(train_end, validation_end),
        "test": slice(validation_end, total),
    }
    return SyntheticDataset(counts, rates, latents, split, true_params, config)
