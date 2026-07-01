from typing import Any, NamedTuple

import jax
import jax.numpy as jnp 
from jax.example_libraries import optimizers

from model import (LDSParams, RecognitionParams, DecoderParams, ModelParams)
from inference import (
    LDSSmoothingPosterior,
    LDSSufficientStatistics,
    kalman_smoother,
    expected_sufficient_stats,
    kl_to_prior,
)


## Containers

# Container for the neural network parameters
class NeuralParams(NamedTuple): 
    recognition: RecognitionParams
    decoder: DecoderParams

# Container for the training state
class TrainState(NamedTuple):
    dynamics: LDSParams
    opt_state: Any
    key: jnp.ndarray
    step: jnp.ndarray


## Packing helpers

# Extract from a model parameters container
def split_neural_params(params: ModelParams) -> NeuralParams:
    return NeuralParams(
        recognition=params.recognition,
        decoder=params.decoder,
    )

split_nerual_params = split_neural_params

# Merge back into a model parameters container
def merge_params(dynamics: LDSParams, neural: NeuralParams) -> ModelParams:
    return ModelParams(
        dynamics=dynamics,
        recognition=neural.recognition,
        decoder=neural.decoder,
    )


## Variance helpers

# log variance, with subtraction of variance floor
def _log_variance_param(variance, variance_floor, eps = 1e-6):
    return jnp.log(jnp.maximum(variance-variance_floor, eps))


def _actual_variance(log_variance_param, variance_floor):
    return jnp.exp(log_variance_param) + variance_floor


## Dynamics helpers

def _least_squares_A(S00, S10, transition_mask = None, jitter = 1e-6):
    """
    Analytical M-step for A gives 
    A = (S10 @ S00^{-1})
    where 
    S00 = sum_n E[z_{n} z_{n}^T | all sites]
    S10 = sum_n E[z_{n+1} z_{n}^T | all sites]
    """
    latent_dim = S00.shape[0]
    eye = jnp.eye(latent_dim)

    if transition_mask is None: 
        return jnp.linalg.solve(S00 + jitter * eye, S10.T).T

    # this is the case where we want to mask some of the latent dynamics
    transition_mask = jnp.asarray(transition_mask)

    # Basically, we solve row by row of the system of equations for each row of the transition matrix
    def solve_row(mask_row, target_row):
        M = jnp.diag(mask_row)
        system = M @ S00 @ M + jitter * eye
        rhs = mask_row * target_row
        row = jnp.linalg.solve(system, rhs)
        return row * mask_row

    return jax.vmap(solve_row)(transition_mask, S10)



def dynamics_m_step(
    stats: LDSSufficientStatistics,
    transition_mask = None,
    variance_floor = 1e-6,
    jitter = 1e-6,
):
    """
    M-step update for the dynamics parameters
    """
    # Extract the sufficient statistics
    S00 = stats.sum_ztzt
    S10 = stats.sum_ztp1zt
    S11 = stats.sum_ztp1ztp1

    ## Compute the updates
    # A = (S10 @ S00^{-1})
    A = _least_squares_A(S00, S10, transition_mask, jitter)

    # Now for the variance
    # note that we can literally throw away non diagnoals as the update separates for constant transition A 
    residual_second_moment = S11 - A @ S10.T - S10 @ A.T + A @ S00 @ A.T
    num_transitions = jnp.asarray(stats.num_transitions, dtype = S00.dtype)
    Q = residual_second_moment / num_transitions
    q_diag = jnp.maximum(jnp.diag(Q), variance_floor)

    # Now compute the parameters of the initial distribution 
    num_sequences = jnp.asarray(stats.num_sequences, dtype = S00.dtype)

    # Initial mean
    m0 = stats.sum_z0 / num_sequences

    # Initial covariance
    Ez0z0 = stats.sum_z0z0 / num_sequences
    S0 = Ez0z0 - jnp.outer(m0, m0)
    S0 = 0.5 * (S0 + S0.T)
    s0_diag = jnp.maximum(jnp.diag(S0), variance_floor)

    return LDSParams(
        A = A, 
        log_q_diag = _log_variance_param(q_diag, variance_floor),
        m0 = m0,
        log_s0_diag = _log_variance_param(s0_diag, variance_floor),
    )


# Damped updates as we are both minibatching and doing variational EM. 
def damped_dynamics_update(
    old: LDSParams,
    new: LDSParams,
    step_size = 1.0,
    transition_mask = None,
    variance_floor = 1e-6,
    jitter = 1e-6,
):

    rho = jnp.asarray(step_size)

    # For A
    A = (1.0 - rho) * old.A + rho * new.A
    if transition_mask is not None:
        A = A * transition_mask

    # for q
    q_old = _actual_variance(old.log_q_diag, variance_floor)
    q_new = _actual_variance(new.log_q_diag, variance_floor)
    q_diag = (1.0 - rho) * q_old + rho * q_new

    # for s0
    s0_old = _actual_variance(old.log_s0_diag, variance_floor)
    s0_new = _actual_variance(new.log_s0_diag, variance_floor)
    s0_diag = (1.0 - rho) * s0_old + rho * s0_new

    # for m0 
    m0 = (1.0 - rho) * old.m0 + rho * new.m0

    return LDSParams(
        A = A,
        log_q_diag = _log_variance_param(q_diag, variance_floor),
        m0 = m0,
        log_s0_diag = _log_variance_param(s0_diag, variance_floor),
    )

# Convert posterior moments into a dynamics update
def dynamics_update_from_posterior(
    model,
    old_dynamics: LDSParams,
    posterior: LDSSmoothingPosterior,
    dynamics_step_size = 0.05,
    jitter = 1e-6,
):
    # We want to fix the posterior, and update the dynamics
    # Hence, we want to stop gradients flowing through the posterior
    posterior = jax.tree_util.tree_map(jax.lax.stop_gradient, posterior)
    stats = expected_sufficient_stats(posterior)

    new_dynamics = dynamics_m_step(
        stats, 
        transition_mask = model.transition_mask,
        variance_floor = model.variance_floor,
        jitter = jitter,
    )

    return damped_dynamics_update(
        old = old_dynamics,
        new = new_dynamics,
        step_size = dynamics_step_size,
        transition_mask = model.transition_mask,
        variance_floor = model.variance_floor,
        jitter = jitter,
    )


## Posterior sampling

# Now, to update neural params, we follow gradients. We thus need MC estimates of expectations
# Hence, we need to sample from the variational posterior. 
def sample_posterior_marginals(
    key, 
    posterior: LDSSmoothingPosterior,
    num_samples = 1,
    jitter = 1e-6,
):
    mean = posterior.mean
    cov = posterior.cov

    # Jax expects batched inputs, so we need to make fake for vmap
    unbatched = mean.ndim == 2
    if unbatched:
        mean = mean[None, ...]
        cov = cov[None, ...]

    latent_dim = mean.shape[-1]
    eye = jnp.eye(latent_dim)

    # We know that the marginals are gaussian, we sample with z_{s,t} = m_t + L_t @ eps_{s,t}, where eps_{s,t} ~ N(0, I)
    chol = jnp.linalg.cholesky(cov + jitter * eye)
    noise = jax.random.normal(key, (num_samples,) + mean.shape)

    samples = mean[None, ...] + jnp.einsum("...ij,s...j->s...i", chol, noise)

    if unbatched:
        samples = samples[:, 0]

    return samples


## Free energy Computation

# Compute an estimate of the Free Energy for a minibatch
def batch_free_energy(
    key, 
    model, 
    params: ModelParams, 
    counts, 
    mask = None, 
    num_samples = 1,
    jitter = 1e-6,
): 
    sites = model.recognise(params.recognition, counts, mask)
    posterior = kalman_smoother(
        params.dynamics, 
        sites, 
        transition_mask = model.transition_mask,
        variance_floor = model.variance_floor,
        jitter = jitter,
    )

    # Note that we just need marginals on individual latenst as the likelihood decouples over time 
    z_samples = sample_posterior_marginals(
        key,
        posterior, 
        num_samples, 
        jitter = jitter,
    )

    # Compute the expected log likelihood component
    sample_log_likelihood = jax.vmap(
        lambda z: model.poisson_log_likelihood(params.decoder, z, counts, mask)
    )(z_samples)
    expected_log_likelihood = jnp.mean(sample_log_likelihood, axis = 0)

    # compute the kl 
    kl = kl_to_prior(posterior, sites)

    # Put this together to get the free enbergy estimate
    free_energy_per_sequence = expected_log_likelihood - kl
    free_energy = jnp.mean(free_energy_per_sequence)

    # Bundle up all the other bits. 
    aux = {
        "posterior": posterior,
        "sites": sites,
        "expected_log_likelihood": expected_log_likelihood,
        "kl": kl,
        "free_energy_per_sequence": free_energy_per_sequence,
    }

    return free_energy, aux

# Just take the negative of this to do gradient desc 
def neural_loss(
    neural: NeuralParams,
    key, 
    model, 
    dynamics: LDSParams,
    counts, 
    mask = None,
    num_samples = 1, 
    jitter = 1e-6,
): 
    params = merge_params(dynamics, neural)

    elbo, aux = batch_free_energy(
        key,
        model,
        params,
        counts,
        mask = mask,
        num_samples = num_samples,
        jitter = jitter,
    )

    return -elbo, aux


### Acutal training loop

def make_optimizer(step_size = 1e-3):
    return optimizers.adam(step_size)


def initialise_train_state(
    key,
    params: ModelParams,
    opt_init,
):
    neural = split_neural_params(params)

    return TrainState(
        dynamics = params.dynamics,
        opt_state = opt_init(neural),
        key = key,
        step = jnp.array(0),
    )
    


def train_step(
    state: TrainState,
    model,
    opt_update,
    get_neural_params,
    counts,
    mask = None,
    num_samples = 1,
    dynamics_step_size = 0.05,
    jitter = 1e-6,
):
    # Extract necessary parts
    key, subkey = jax.random.split(state.key)
    neural = get_neural_params(state.opt_state)


    ##1. Generalised M-step update for neural network params
    
    # This runs the E-like-step for post quantities in forming the Free Energy, 
    # and then we backprop through the Kalman Smoothing to get grads w.r.t recognition nwk params
    (loss, aux), grads = jax.value_and_grad(
        neural_loss, has_aux = True,
    )(neural, subkey, model, state.dynamics, counts, mask, num_samples, jitter)

    # Update optimiser state
    opt_state = opt_update(state.step, grads, state.opt_state)

    ##2. Exact M-step update for dynamics parameters (damped = generalised)
    dynamics = dynamics_update_from_posterior(
        model,
        state.dynamics,
        aux["posterior"],
        dynamics_step_size = dynamics_step_size, 
        jitter = jitter,
    )

    # outputs are new state and metrics
    new_state = TrainState(
        dynamics = dynamics,
        opt_state = opt_state,
        key = key,
        step = state.step + 1,
    )
    metrics = {
        "loss": loss,
        "free_energy" : -loss,
        "mean_expected_log_likelihood" : jnp.mean(aux["expected_log_likelihood"]),
        "mean_kl" : jnp.mean(aux["kl"]),
    }

    return new_state, metrics 

## Outer Loop

# Quick helper functions 
def get_params_from_state(state: TrainState, get_neural_params):
    neural = get_neural_params(state.opt_state)
    return merge_params(state.dynamics, neural)

def sample_minibatch(key, counts, batch_size, mask = None):
    num_sequences = counts.shape[0]
    # Sample with/without replacement dep on desiredbatch size
    replace = batch_size > num_sequences

    # Choose which sequences to include
    idx = jax.random.choice(
        key, 
        num_sequences,
        shape = (batch_size,),
        replace = replace,
    )

    # Extract the selected sequences
    batch_counts = counts[idx]
    batch_mask = None if mask is None else mask[idx]

    return batch_counts, batch_mask

# model fitting
def fit_model(
    key,
    model,
    params: ModelParams, 
    counts, 
    mask = None, 
    num_steps = 1000,
    batch_size = 32,
    learning_rate = 1e-3,
    num_samples = 1,
    dynamics_step_size = 0.05, 
    jitter = 1e-6,
    log_every = 50,
):
    """
    Fit the point-estimate Poisson LDS-SVAE using minibatch variational EM.

    Model:
    p_theta,gamma(y, z) = p_theta(z_0) prod_t p_theta(z_{t+1} | z_t)
                              prod_t p_gamma(y_t | z_t)

    Structured variational family:
        q_phi,theta(z | y) propto p_theta(z) prod_t psi_phi,t(z_t; y_t)

    where the recognition network gives Gaussian sites
        psi_phi,t(z_t; y_t) propto exp(h_t^T z_t - 0.5 z_t^T R_t z_t)

    For each minibatch, inference is exact in this structured family:
        q_phi,theta(z | y) is computed by Kalman filtering and smoothing.

    The fitted objective is the SVAE free energy:
        L(theta, gamma, phi)
        = E_q[log p_gamma(y | z)] - KL(q_phi,theta(z | y) || p_theta(z))

    Each train step performs a generalised variational EM update:
        E-like step:
            form recognition sites psi_phi,t and compute marginals of q(z | y) by Kalman smoothing

        Neural M-like step:
            update recognition parameters phi and decoder parameters gamma by
            gradient ascent on L, using Monte Carlo samples from q(z | y) for
            E_q[log p_gamma(y | z)]

        Dynamics M-like step:
            update theta = (A, Q, m0, S0) from the smoothed sufficient statistics
            E_q[z_t z_t^T], E_q[z_{t+1} z_t^T], E_q[z_{t+1} z_{t+1}^T]

    The dynamics update is damped because minibatches give noisy estimates of the
    sufficient statistics.

    Returns:
        fitted_params: final ModelParams
        state: final TrainState
        history: logged scalar metrics
    """
    # setups
    opt_init, opt_update, get_neural_params = make_optimizer(learning_rate)
    key, init_key = jax.random.split(key)
    state = initialise_train_state(init_key, params, opt_init)
    history = []

    # training loop
    for step in range(num_steps):
        key, batch_key = jax.random.split(key)

        batch_counts, batch_mask = sample_minibatch(
            batch_key, counts, batch_size, mask = mask,
        )

        # training step 
        state, metrics = train_step(
            state, 
            model,
            opt_update,
            get_neural_params,
            batch_counts,
            mask = batch_mask,
            num_samples = num_samples,
            dynamics_step_size = dynamics_step_size,
            jitter = jitter,
        )

        # log metrics
        if step % log_every == 0 or step == num_steps - 1: 
            metrics = {name: float(value) for name, value in metrics.items()}
            metrics["step"] = step
            history.append(metrics)

    # extract the fitted parameters
    fitted_params = get_params_from_state(state, get_neural_params)

    return fitted_params, state, history










