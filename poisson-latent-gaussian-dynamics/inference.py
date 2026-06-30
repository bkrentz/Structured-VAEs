from typing import NamedTuple

import jax
import jax.numpy as jnp

from model import LDSParams, GaussianSites

# Package
class LDSSmoothingPosterior(NamedTuple):
    mean: jnp.ndarray
    cov: jnp.ndarray
    cross_cov: jnp.ndarray
    filtered_mean: jnp.ndarray
    filtered_cov: jnp.ndarray
    predicted_mean: jnp.ndarray
    predicted_cov: jnp.ndarray
    smoother_gain: jnp.ndarray
    log_normaliser: jnp.ndarray


class LDSSufficientStatistics(NamedTuple):
    sum_z0: jnp.ndarray
    sum_z0z0: jnp.ndarray
    sum_ztzt: jnp.ndarray
    sum_ztp1zt: jnp.ndarray
    sum_ztp1ztp1: jnp.ndarray
    num_sequences: int
    num_transitions: int


# Helpers
def symmetrise(x):
    return 0.5 * (x + jnp.swapaxes(x, -1, -2))


def diagonal_matrix(x):
    return jnp.diag(x)


# Combine a forward-pass gaussian message with a site.
def forward_gaussian_site_update(mean_pred, cov_pred, h, precision_diag, jitter=1e-6):
    """
    Combine a forward-pass gaussian message with a site.

    Inputs:
    - mean_pred: E[z_t | sites 0:t-1]
    - cov_pred: Cov[z_t | sites 0:t-1]
    - h: natural parameter of site t
    - precision_diag: diagonal precision of site t
    - jitter: small constant to add to the diagonal of the covariance matrix to prevent numerical instability

    Returns:
    - mean_filt: E[z_t | sites 0:t]
    - cov_filt: Cov[z_t | sites 0:t]
    - log_normaliser: log Z_t = log p(y_t | y1,...,y_{t-1})
    """
    # Create identity matrix for cheap inversion
    latent_dim = mean_pred.shape[-1]
    eye = jnp.eye(latent_dim)

    # predictive precision
    precision_pred = jnp.linalg.solve(cov_pred, eye)
    natural_pred = precision_pred @ mean_pred

    # convert the diagonal precision into a full precision matrix
    site_precision = diagonal_matrix(precision_diag)

    # Combine to get natural params for t | t
    # We are using h and preicsion...
    precision_filt = precision_pred + site_precision + jitter * eye
    natural_filt = natural_pred + h

    # Convert back to standard parameter form
    cov_filt = jnp.linalg.solve(precision_filt, eye)
    cov_filt = symmetrise(cov_filt)
    mean_filt = jnp.linalg.solve(precision_filt, natural_filt)

    # Compute normaliser contibution for this update
    # log Z_t = log p (y_t | y1,...,y_{t-1})
    logdet_pred_cov = jnp.linalg.slogdet(cov_pred)[1]
    logdet_filt_precision = jnp.linalg.slogdet(precision_filt)[1]
    log_normaliser = (
        -0.5 * logdet_pred_cov
        -0.5 * natural_pred @ mean_pred
        -0.5 * logdet_filt_precision
        + 0.5 * natural_filt @ mean_filt
    )

    return mean_filt, cov_filt, log_normaliser


## Filtering and Smoothing for a single observation (observation is time-series of outputs)


def filter_one_sequence(h, precision_diag, A, Q, m0, S0, jitter=1e-6):
    """
    Forward Kalman filter on one sequence with Gaussian recognition sites.

    Inference target:
        p(z_{0:T}) prod_t phi_t(z_t)
    where phi_t(z_t) propto exp(h_t^T z_t - 0.5 z_t^T Lambda_t z_t),
    Lambda_t = diag(precision_diag_t).

    Dynamics prior:
        p(z_0) = N(m0, S0)
        p(z_{t+1} | z_t) = N(A z_t, Q)

    t = 0:
        absorb site phi_0 into prior (m0, S0)

    t = 1, ..., T-1:
        Predict:  m_{t|t-1} = A m_{t-1|t-1},  Sigma_{t|t-1} = A Sigma_{t-1|t-1} A^T + Q
        Update:   absorb site phi_t into (m_{t|t-1}, Sigma_{t|t-1})

    Returns (time index t = 0, ..., T-1):
    - filtered_mean[t]   = E[z_t | sites 0:t]     = m_{t|t}
    - filtered_cov[t]    = Cov(z_t | sites 0:t)     = Sigma_{t|t}
    - predicted_mean[t]  = E[z_t | sites 0:t-1]     = m_{t|t-1};  predicted_mean[0] = m0
    - predicted_cov[t]   = Cov(z_t | sites 0:t-1)   = Sigma_{t|t-1};  predicted_cov[0] = S0
    - log_normaliser     = sum_t log Z_t            (scalar)
    """
    mean0, cov0, logz0 = forward_gaussian_site_update(
        m0,
        S0,
        h[0],
        precision_diag[0],
        jitter,
    )

    def step(carry, inputs):
        mean_prev, cov_prev = carry
        h_t, precision_diag_t = inputs

        mean_pred = A @ mean_prev
        cov_pred = A @ cov_prev @ A.T + Q
        cov_pred = symmetrise(cov_pred)

        mean_filt, cov_filt, logz = forward_gaussian_site_update(
            mean_pred,
            cov_pred,
            h_t,
            precision_diag_t,
            jitter,
        )

        carry = (mean_filt, cov_filt)
        output = (mean_pred, cov_pred, mean_filt, cov_filt, logz)
        return carry, output

    _, outputs = jax.lax.scan(
        step,
        (mean0, cov0),
        (h[1:], precision_diag[1:]),
    )

    mean_pred, cov_pred, mean_filt, cov_filt, logz = outputs

    predicted_mean = jnp.concatenate([m0[None, :], mean_pred], axis=0)
    predicted_cov = jnp.concatenate([S0[None, :, :], cov_pred], axis=0)

    filtered_mean = jnp.concatenate([mean0[None, :], mean_filt], axis=0)
    filtered_cov = jnp.concatenate([cov0[None, :, :], cov_filt], axis=0)

    log_normaliser = logz0 + jnp.sum(logz)

    return (
        filtered_mean,
        filtered_cov,
        predicted_mean,
        predicted_cov,
        log_normaliser,
    )



def smooth_one_sequence(
    filtered_mean,
    filtered_cov,
    predicted_mean,
    predicted_cov,
    A,
    jitter=1e-6,
):
    """
    RTS smoother to obtain z_t | all sites.

    Inputs:
    - filtered_mean: mean of the filtered distribution at time t
    - filtered_cov: covariance of the filtered distribution at time t
    - predicted_mean: mean of the predicted distribution at time t+1
    - predicted_cov: covariance of the predicted distribution at time t+1
    - A: transition matrix
    - jitter: small constant to add to the diagonal of the covariance matrix to prevent numerical instability

    Returns:
    - smoothed_mean: E[z_t | all sites] for all time steps
    - smoothed_cov: Cov[z_t | all sites] for all time steps
    - cross_cov: Cov[z_{t+1}, z_{t} | all sites] for all time steps
    - smoother_gain: RTS gain matrix G_t for all time steps
    """

    # shapes
    latent_dim = filtered_mean.shape[-1]
    eye = jnp.eye(latent_dim)

    # One backward update
    def step(carry, inputs):
        # mean_next_smooth is E[z_{t+1} | {0:T}], similar for cov
        mean_next_smooth, cov_next_smooth = carry

        # mean_filt is E[z_t | {0:t}], similar for cov
        # mean_pred_next is E[z_{t+1} | {0:t}]
        mean_filt, cov_filt, mean_pred_next, cov_pred_next = inputs

        cov_pred_next = symmetrise(cov_pred_next) + jitter * eye

        # this is the RTS gain matrix G_t
        gain = jnp.linalg.solve(cov_pred_next, A @ cov_filt).T

        # Obtain the smoothed mean and covariance for this time step
        # mean_smooth = E[z_t | {0:T}]
        # cov_smooth = Cov[z_t | {0:T}]
        # cross_cov = Cov[z_{t+1}, z_{t} | {0:T}]
        mean_smooth = mean_filt + gain @ (mean_next_smooth - mean_pred_next)
        cov_smooth = cov_filt + gain @ (cov_next_smooth - cov_pred_next) @ gain.T
        cov_smooth = symmetrise(cov_smooth)
        cross_cov = cov_next_smooth @ gain.T

        # package into carry
        carry = (mean_smooth, cov_smooth)
        output = (mean_smooth, cov_smooth, cross_cov, gain)
        return carry, output

    # Reverse the order of the inputs to match the order of the carry
    inputs = (
        filtered_mean[:-1][::-1],
        filtered_cov[:-1][::-1],
        predicted_mean[1:][::-1],
        predicted_cov[1:][::-1],
    )

    # Initial carry
    last = (filtered_mean[-1], filtered_cov[-1])
    # Iterate: return outputs which is (mean_smooth, cov_smooth, cross_cov, gain) for each time step
    _, outputs = jax.lax.scan(step, last, inputs)

    # these are in reverse order
    mean_rev, cov_rev, cross_cov_rev, gain_rev = outputs

    # Reverse the order and append with the last one.
    mean = jnp.concatenate([mean_rev[::-1], filtered_mean[-1:]], axis=0)
    cov = jnp.concatenate([cov_rev[::-1], filtered_cov[-1:]], axis=0)

    # Just reverse
    cross_cov = cross_cov_rev[::-1]
    smoother_gain = gain_rev[::-1]

    return mean, cov, cross_cov, smoother_gain


# Now construct the kalman filter, and allow for batched and non batched.
def kalman_smoother(
    dynamics: LDSParams,
    sites: GaussianSites,
    transition_mask=None,
    variance_floor=1e-6,
    jitter=1e-6,
):
    """
    Runs Forwards filtering and backwards smoothing on every sequence.

    Outputs:
    LDSSmoothingPosterior(
    mean = E[z_t | {0:T}],
    cov = Cov[z_t | {0:T}],
    cross_cov = Cov[z_{t+1}, z_{t} | {0:T}],
    filtered_mean = E[z_t | {0:t}],
    filtered_cov = Cov[z_t | {0:t}],
    predicted_mean[t] = E[z_t | {0:t-1}], with predicted_mean[0] = m0,
    predicted_cov[t] = Cov[z_t | {0:t-1}], with predicted_cov[0] = S0,
    smoother_gain = RTS gain matrix G_t for all time steps
    log_normaliser = sum_t log Z_t for each sequence
    """
    # Extract params
    A = dynamics.A
    if transition_mask is not None:
        A = A * transition_mask

    Q = diagonal_matrix(jnp.exp(dynamics.log_q_diag) + variance_floor)
    S0 = diagonal_matrix(jnp.exp(dynamics.log_s0_diag) + variance_floor)

    # Check if the sites are unbatched, if so then make fake for vmap
    unbatched = sites.h.ndim == 2
    if unbatched:
        sites = GaussianSites(
            mean=sites.mean[None, ...],
            precision_diag=sites.precision_diag[None, ...],
            h=sites.h[None, ...],
        )


    def infer_one(h, precision_diag):
        """
        Take in output from the recognition network, return the posterior quantities after smoothing
        """
        (
            filtered_mean,
            filtered_cov,
            predicted_mean,
            predicted_cov,
            log_normaliser,
        ) = filter_one_sequence(h, precision_diag, A, Q, dynamics.m0, S0, jitter)

        mean, cov, cross_cov, smoother_gain = smooth_one_sequence(
            filtered_mean,
            filtered_cov,
            predicted_mean,
            predicted_cov,
            A,
            jitter,
        )

        return LDSSmoothingPosterior(
            mean=mean,
            cov=cov,
            cross_cov=cross_cov,
            filtered_mean=filtered_mean,
            filtered_cov=filtered_cov,
            predicted_mean=predicted_mean,
            predicted_cov=predicted_cov,
            smoother_gain=smoother_gain,
            log_normaliser=log_normaliser,
        )

    # Run the inference for each sequence
    posterior = jax.vmap(infer_one)(sites.h, sites.precision_diag)

    # If the input was unbatched, strip the leading 1 used for vmap
    if unbatched:
        posterior = jax.tree_util.tree_map(lambda x: x[0], posterior)

    return posterior 


### Compute quantities from the posterior that we will need for the M-like-Step. 

# First the expected sufficient statistics of the prior
def expected_sufficient_stats(posterior:LDSSmoothingPosterior):
    """
    Returns:
    LDSSufficientStatistics(
    sum_z0 = sum_n E[z_{n,0} | all sites]
    sum_z0z0 = sum_n E[z_{n,0} z_{n,0}^T | all sites]
    sum_ztzt = sum_n sum_t E[z_{n,t} z_{n,t}^T | all sites], for t = 0,...,T-2
    sum_ztp1zt = sum_n sum_t E[z_{n,t+1} z_{n,t}^T | all sites], for t = 0,...,T-2
    sum_ztp1ztp1 = sum_n sum_t E[z_{n,t+1} z_{n,t+1}^T | all sites], for t = 0,...,T-2
    num_sequences = number of sequences
    num_transitions = number of transitions

    """
    # Extract the mean and covariance from the posterior
    mean = posterior.mean
    cov = posterior.cov
    cross_cov = posterior.cross_cov

    # make batching work
    if mean.ndim == 2:
        mean = mean[None, ...]
        cov = cov[None, ...]
        cross_cov = cross_cov[None, ...]

    # Compute moments
    second_moment = cov + mean[..., :, None] * mean[..., None, :]
    cross_second_moment = (
    cross_cov
    + mean[:, 1:, :, None] * mean[:, :-1, None, :]
    )

    return LDSSufficientStatistics(
        sum_z0=jnp.sum(mean[:, 0], axis=0),
        sum_z0z0=jnp.sum(second_moment[:, 0], axis=0),
        sum_ztzt=jnp.sum(second_moment[:, :-1], axis=(0, 1)),
        sum_ztp1zt=jnp.sum(cross_second_moment, axis=(0, 1)),
        sum_ztp1ztp1=jnp.sum(second_moment[:, 1:], axis=(0, 1)),
        num_sequences=mean.shape[0],
        num_transitions=mean.shape[0] * (mean.shape[1] - 1),
    )
    

# We will need  KL term between prior and posterior, for at leas tthe parameters of the recog nwk
def expected_log_sites(
    posterior: LDSSmoothingPosterior,
    sites: GaussianSites):
    """
    Compute E[sum_{t=0}^T log psi at time t]
    """
    # Extract posterior mean and covariance
    mean = posterior.mean
    cov = posterior.cov

    # batch-ready
    if mean.ndim == 2:
        mean = mean[None, ...]
        cov = cov[None, ...]
        sites = GaussianSites(
            mean = sites.mean[None, ...],
            precision_diag = sites.precision_diag[None, ...],
            h=sites.h[None, ...],
        )

    # Extract the covariance matrix which is a diagonal matrix. 
    cov_diag = jnp.diagonal(cov, axis1=-2, axis2 = -1)

    # Compute the expected log density (we are using canonical form)
    expected_quadratic = sites.precision_diag * (cov_diag + mean**2)
    expected_linear = sites.h * mean

    # Sum over all batches 
    return jnp.sum(expected_linear - 0.5 * expected_quadratic, axis=(-2, -1))

# Plug into expression for KL[q || p]: 
# Expression is E[log q - log p] = E[log prior + sum_t log psi_t - log normaliser - log prior]
# = E[sum_t log psi_t - log normaliser]
def kl_to_prior(
    posterior: LDSSmoothingPosterior,
    sites: GaussianSites,
):
    return expected_log_sites(posterior, sites) - posterior.log_normaliser

    

    












    
    
