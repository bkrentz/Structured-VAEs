from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
LOG_2PI = float(np.log(2.0 * np.pi))


## Optimiser containers

class AdamState(NamedTuple):
    step: Array
    first_moment: Any
    second_moment: Any


## Numerical helpers

def _symmetrise(x: Array) -> Array:
    return 0.5 * (x + jnp.swapaxes(x, -1, -2))


def _softplus_inverse(x: float | Array) -> Array:
    x = jnp.asarray(x)
    return jnp.log(jnp.expm1(x))


def _glorot(key: Array, shape: tuple[int, int]) -> Array:
    fan_out, fan_in = shape
    limit = jnp.sqrt(6.0 / float(fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def _chol_solve(matrix: Array, rhs: Array, jitter: float = 0.0) -> Array:
    if jitter:
        matrix = matrix + jitter * jnp.eye(matrix.shape[-1], dtype=matrix.dtype)
    chol = jnp.linalg.cholesky(_symmetrise(matrix))
    y = jax.scipy.linalg.solve_triangular(chol, rhs, lower=True)
    return jax.scipy.linalg.solve_triangular(jnp.swapaxes(chol, -1, -2), y, lower=False)


def _spd_inverse(matrix: Array, jitter: float = 0.0) -> Array:
    eye = jnp.eye(matrix.shape[-1], dtype=matrix.dtype)
    return _symmetrise(_chol_solve(matrix, eye, jitter))


def _logdet_spd(matrix: Array, jitter: float = 0.0) -> Array:
    if jitter:
        matrix = matrix + jitter * jnp.eye(matrix.shape[-1], dtype=matrix.dtype)
    chol = jnp.linalg.cholesky(_symmetrise(matrix))
    return 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)


def _actual_variance(log_variance: Array, floor: float) -> Array:
    return jnp.exp(log_variance) + floor


def _log_variance_parameter(variance: Array, floor: float) -> Array:
    return jnp.log(jnp.maximum(variance - floor, 1e-8))


def _tree_zeros_like(tree: Any) -> Any:
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _tree_global_norm(tree: Any) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def _clip_gradient_tree(grads: Any, max_norm: float) -> tuple[Any, Array]:
    norm = _tree_global_norm(grads)
    factor = jnp.minimum(1.0, max_norm / jnp.maximum(norm, 1e-12))
    return jax.tree_util.tree_map(lambda g: g * factor, grads), norm


def adam_init(params: Any) -> AdamState:
    return AdamState(
        step=jnp.array(0, dtype=jnp.int32),
        first_moment=_tree_zeros_like(params),
        second_moment=_tree_zeros_like(params),
    )


def adam_update(
    params: Any,
    grads: Any,
    state: AdamState,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[Any, AdamState]:
    step = state.step + 1
    first = jax.tree_util.tree_map(
        lambda m, g: beta1 * m + (1.0 - beta1) * g,
        state.first_moment,
        grads,
    )
    second = jax.tree_util.tree_map(
        lambda v, g: beta2 * v + (1.0 - beta2) * jnp.square(g),
        state.second_moment,
        grads,
    )
    correction1 = 1.0 - beta1**step
    correction2 = 1.0 - beta2**step
    new_params = jax.tree_util.tree_map(
        lambda p, m, v: p
        - learning_rate * (m / correction1) / (jnp.sqrt(v / correction2) + epsilon),
        params,
        first,
        second,
    )
    return new_params, AdamState(step, first, second)


def _assert_finite_tree(tree: Any, label: str) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        arr = np.asarray(leaf)
        if not np.all(np.isfinite(arr)):
            raise FloatingPointError(f"Non-finite values found in {label}.")


def spectral_radius(matrix: Array) -> Array:
    return jnp.max(jnp.abs(jnp.linalg.eigvals(matrix)))
