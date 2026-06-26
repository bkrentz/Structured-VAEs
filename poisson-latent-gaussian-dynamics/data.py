from dataclasses import dataclass

import jax
import jax.numpy as jnp 


@dataclass
class DGPParams:
    """
    A       = dynamics matrix
    q_scale = noise for the dynamics
    W1, b1  = first layer of nonlinear rate function
    W2, b2  = second layer of nonlinear rate function
    dt      = width of the time bins
    """
    A: jnp.ndarray
    q_scale: float
    W1: jnp.ndarray
    b1: jnp.ndarray
    W2: jnp.ndarray
    b2: jnp.ndarray
    dt: float

# Construct the parameters
def make_transition_matrix(latent_dim: int = 2, radius: float = 0.95) -> jnp.ndarray:
    """
    Make the transition matrix A for the latent dynamics
    """
    # angle
    theta = -0.1

    # 1D case
    if latent_dim ==1:
        return jnp.array([[radius]])

    # rotation matrix
    rotation = radius * jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta), jnp.cos(theta)],
        ]
    )

    # Construct A - generalisable to higher dimensions
    A = jnp.eye(latent_dim) * 0.8
    A = A.at[:2, :2].set(rotation)

    return A

def init_dgp_params(
    key: jnp.ndarray,
    latent_dim: int = 2,
    hidden_dim: int = 32,
    num_neurons: int = 30,
    dt: float = 0.05,
    q_scale: float = 0.05,
) -> DGPParams:
    """
    Function that creates a full set of simulation settings for the DGP
    """
    key_w1, key_w2 = jax.random.split(key)

    # transition
    A = make_transition_matrix(latent_dim)

    # Constructs weights and biases
    W1 = 0.8 * jax.random.normal(key_w1, (hidden_dim, latent_dim))
    b1 = jnp.zeros((hidden_dim,))
    W2 = 0.6 * jax.random.normal(key_w2, (num_neurons, hidden_dim))
    b2 = 1 * jnp.ones((num_neurons,))

    return DGPParams(A=A, q_scale=q_scale, W1=W1, b1=b1, W2=W2, b2=b2, dt=dt)

# Compute poisson rate from latent dynamics
def latent_to_rates(z: jnp.ndarray, params: DGPParams) -> jnp.ndarray:
    hidden = jnp.tanh(z @ params.W1.T + params.b1)
    log_rates = hidden @ params.W2.T + params.b2
    log_rates = jnp.clip(log_rates, -5.0, 5.0)
    return jnp.exp(log_rates)

# Sample the latent trajectories
def sample_latents(
    key: jnp.ndarray,
    params: DGPParams,
    num_trials: int, 
    num_timesteps: int,
) -> jnp.ndarray:
    """
    Sample the latent dynamics for a given number of trials and timesteps
    """
    # Shapes, repro 
    latent_dim = params.A.shape[0]
    key_z0, key_noise = jax.random.split(key)

    # Initial state
    z0 = jax.random.normal(key_z0, (num_trials, latent_dim))

    # Sample all the noise at once. 
    noise = params.q_scale * jax.random.normal(key_noise, (num_timesteps-1, num_trials, latent_dim),
    )

    def step(z_prev, eps_t):
        z_next = z_prev @ params.A.T + eps_t
        return z_next, z_next

    # Evolve the latent trajectories
    _, z_rest = jax.lax.scan(step, z0, noise)

    # Stitch together the initial state and the rest of the trajectories
    z_time_first = jnp.concatenate([z0[None, ...], z_rest], axis=0)

    # Swao time and trial axes
    # Thus, we have (trials, timesteps, latent_dim)
    return jnp.swapaxes(z_time_first, 0, 1)


# Give the whole shebang
def sample_dataset(
    key: jnp.ndarray,
    num_trials: int = 256,
    num_timesteps: int = 100,
    latent_dim: int = 2,
    hidden_dim: int = 32,
    num_neurons: int = 30,
    dt: float = 0.05,
    ):
    """ 
    Sample the entire dataset for the DGP

    Returns:
    dict: 
    - z: latent trajectories
    - rates: poisson rates
    - counts: poisson counts
    - params: parameters
    """
    # Repro 
    key_params, key_latents, key_counts = jax.random.split(key, num=3)

    # Params
    params = init_dgp_params(key_params, latent_dim, hidden_dim, num_neurons, dt)

    # Latent trajectories 
    # Shape (trials, timesteps, latent_dim
    z = sample_latents(key_latents, params, num_trials, num_timesteps)

    # Poisson rates
    rates = latent_to_rates(z, params)

    # Sample the counts
    counts = jax.random.poisson(key_counts, rates * params.dt)

    return {
        "z": z,
        "rates": rates, 
        "counts": counts, 
        "params": params,
    }

if __name__ == "__main__":

    # Repro
    key = jax.random.PRNGKey(1)
    
    # Get the data
    data = sample_dataset(key)

    # Print the sampled data, and its dimensions
    print("z", data["z"].shape)
    print("rates:", data["rates"].shape)
    print("counts:", data["counts"].shape)
    print("mean count/ bin", jnp.mean(data["counts"]))
    print("mean rate:", jnp.mean(data["rates"]))









    






