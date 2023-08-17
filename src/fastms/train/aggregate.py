from jaxtyping import Array, PyTree
from jax import numpy as jnp
from jax.tree_util import tree_map_with_path

_days_months = [
    31,
    28,
    31,
    30,
    31,
    30,
    31,
    31,
    30,
    31,
    30,
    31
]

def _aggregate_months(x: Array, f) -> Array:
    years = int(x.shape[1] / 365)
    months = jnp.split(
        x,
        jnp.cumsum(jnp.array(_days_months * years))[:-1],
        axis=1
    )
    agg_months = [f(a, axis=1, keepdims=True) for a in months]
    return jnp.concatenate(agg_months, axis=1)

def monthly(samples: PyTree) -> PyTree:
    (x, x_seq, x_t), y = samples
    x_t = x_t // 365 * 12
    y = tree_map_with_path(
        lambda key, leaf: _aggregate_months(
            leaf,
            jnp.sum if 'p_inc_clinical' in key else jnp.mean
        ),
        y
    )
    return (x, x_seq, x_t), y
