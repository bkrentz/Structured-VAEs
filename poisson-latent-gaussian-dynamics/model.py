from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.nn import softplus
from jax.scipy.special import gammaln


# Create classes for the parameters
class LDSParams(NamedTuple):
    A: jnp.ndarray
    log_q_diag: jnp.ndarray  # Diagonal process-noise log variances for simplicity
    m0: jnp.ndarray
    log_s0_diag: jnp.ndarray

class RecognitionParams(NamedTuple):
    W1: jnp.ndarray
    b1: jnp.ndarray
    W_mean: jnp.ndarray
    b_mean: jnp.ndarray
    W_prec: jnp.ndarray
    b_prec: jnp.ndarray

class DecoderParams(NamedTuple):
    W1: jnp.ndarray
    b1: jnp.ndarray
    W2: jnp.ndarray
    b2: jnp.ndarray
    W3: jnp.ndarray
    b3: jnp.ndarray

# Bundle up all the parameters
class ModelParams(NamedTuple):
    dynamics: LDSParams
    recognition: RecognitionParams
    decoder: DecoderParams

# We use the recognition network to cast likelihoods into Gaussian sites for Kalman inference
class GaussianSites(NamedTuple):
    mean: jnp.ndarray
    precision_diag: jnp.ndarray
    h: jnp.ndarray  # Natural parameter: precision_diag * mean


# Helper functions for model building/initialisation
def glorot(key, shape):
    """
    Uniform Xavier initialisation
    """
    fan_out, fan_in = shape
    lim = jnp.sqrt(6.0 / (fan_out + fan_in))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)


def inverse_softplus(x):
    return jnp.log(jnp.expm1(jnp.asarray(x)))


def initial_A(latent_dim=2, radius=0.95):
    A = jnp.eye(latent_dim) * radius
    return A


# Initial SVAE with point estimates for all non-latent parameters
class PointEstimatePoissonLDSSVAE:
    def __init__(
        self,
        latent_dim = 2,
        decoder_hidden_dim = (32, 32),
        recognition_hidden_dim = 32,
        dt=0.05,
        transition_mask = None, # Mask A to enforce structure if wanted
        init_process_noise = 0.25, # Starting standard deviation for latent dynamics
        init_site_precision = 0.1, # Starting precision for Gaussian sites
        min_site_precision = 1e-4, # Minimum precision for Gaussian sites
        variance_floor = 1e-4, # Minimum variance for diagonal dynamics noise
        log_rate_clip = 8.0, # Symmetric clip for decoder log rates
    ):
        self.latent_dim = latent_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.recognition_hidden_dim = recognition_hidden_dim
        self.dt = dt
        self.transition_mask = transition_mask
        self.init_process_noise = init_process_noise
        self.init_site_precision = init_site_precision
        self.min_site_precision = min_site_precision
        self.variance_floor = variance_floor
        self.log_rate_clip = log_rate_clip

    def init_parameters(self, key, counts):
        num_neurons = counts.shape[-1]
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, num=6)

        # Initialise dynamics
        A = initial_A(self.latent_dim)
        if self.transition_mask is not None:
            A = A * self.transition_mask
        dynamics = LDSParams(
            A = A,
            log_q_diag = jnp.log((self.init_process_noise**2) * jnp.ones(self.latent_dim)),
            m0 = jnp.zeros(self.latent_dim),
            log_s0_diag = jnp.zeros(self.latent_dim),
        )

        # Average counts over trials and timesteps, then use the implied log rate as the decoder bias
        mean_counts = jnp.mean(counts, axis=tuple(range(counts.ndim - 1)))
        decoder_bias = jnp.log(jnp.maximum(mean_counts / self.dt, 1e-3))

        # Initialise decoder
        decoder = DecoderParams(
            W1=glorot(k1, (self.decoder_hidden_dim[0], self.latent_dim)),
            b1=jnp.zeros(self.decoder_hidden_dim[0]),
            W2=glorot(k2, (self.decoder_hidden_dim[1], self.decoder_hidden_dim[0])),
            b2=jnp.zeros(self.decoder_hidden_dim[1]),
            W3=0.05 * glorot(k3, (num_neurons, self.decoder_hidden_dim[1])),
            b3=decoder_bias,
        )

        # Initialise recognition network
        recognition = RecognitionParams(
            W1=glorot(k4, (self.recognition_hidden_dim, num_neurons)),
            b1=jnp.zeros(self.recognition_hidden_dim),
            W_mean=0.05 * glorot(k5, (self.latent_dim, self.recognition_hidden_dim)),
            b_mean=jnp.zeros(self.latent_dim),
            W_prec=0.01 * glorot(k6, (self.latent_dim, self.recognition_hidden_dim)),
            b_prec=inverse_softplus(self.init_site_precision) * jnp.ones(self.latent_dim),
        )

        return ModelParams(dynamics, recognition, decoder)

    def recognise(self, params, counts, mask = None):
        # Mask observed neurons/time bins if wanted
        if mask is None:
            mask = jnp.ones_like(counts)
        if mask.ndim == counts.ndim - 1:
            mask = mask[..., None]

        # Log-transform the counts
        x = jnp.log1p(counts) * mask

        # Pass through the recognition network
        hidden = jnp.tanh(x @ params.W1.T + params.b1)
        mean = hidden @ params.W_mean.T + params.b_mean
        raw_precision = hidden @ params.W_prec.T + params.b_prec
        precision_diag = softplus(raw_precision) + self.min_site_precision

        # Scale site precision down when only part of an observation is visible
        observed_fraction = jnp.mean(mask, axis=-1, keepdims=True)
        precision_diag = precision_diag * observed_fraction

        h = precision_diag * mean
        return GaussianSites(mean=mean, precision_diag=precision_diag, h=h)

    # Decoder network
    def decode_log_rates(self, params, z):
        hidden1 = jnp.tanh(z @ params.W1.T + params.b1)
        hidden2 = jnp.tanh(hidden1 @ params.W2.T + params.b2)
        log_rates = hidden2 @ params.W3.T + params.b3
        log_rates = jnp.clip(log_rates, -self.log_rate_clip, self.log_rate_clip)
        return log_rates

    def decode_rates(self, params, z):
        return jnp.exp(self.decode_log_rates(params, z))

    # Parts for the free-energy computation
    def poisson_log_prob(self, params, z, counts, mask = None):
        log_mean = self.decode_log_rates(params, z) + jnp.log(self.dt)
        mean = jnp.exp(log_mean)
        log_prob = counts * log_mean - mean - gammaln(counts + 1)

        if mask is not None:
            if mask.ndim == counts.ndim - 1:
                mask = mask[..., None]
            log_prob = log_prob * mask

        return log_prob

    def poisson_log_likelihood(self, params, z, counts, mask = None):
        return jnp.sum(self.poisson_log_prob(params, z, counts, mask), axis=(-2, -1))

    # In case we want to mask some of the latent dynamics
    def masked_A(self, params):
        if self.transition_mask is None:
            return params.A
        return params.A * self.transition_mask

    # Convert log noise to variance for computations
    def process_noise_diag(self, params):
        return jnp.exp(params.log_q_diag) + self.variance_floor

    def initial_cov_diag(self, params):
        return jnp.exp(params.log_s0_diag) + self.variance_floor
