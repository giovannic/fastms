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

def aggregate_ibm_outputs(xs, ns, age_lower, age_upper, time_lower, time_upper):
    #TODO: these masks are static, pull them out
    age_lower = age_lower[:, jnp.newaxis, jnp.newaxis]
    age_upper = age_upper[:, jnp.newaxis, jnp.newaxis]
    time_lower = time_lower[:, jnp.newaxis, jnp.newaxis]
    time_upper = time_upper[:, jnp.newaxis, jnp.newaxis]
    age_mask = jnp.arange(xs.shape[2])[jnp.newaxis, jnp.newaxis, :]
    age_mask = (age_mask >= age_lower) & (age_mask <= age_upper)
    time_mask = jnp.arange(xs.shape[1])[jnp.newaxis, :, jnp.newaxis]
    time_mask = (time_mask >= time_lower) & (time_mask <= time_upper)
    mask = age_mask & time_mask
    xs = jnp.where(mask, xs, 0)
    ns = jnp.where(mask, ns, 0)
    xs_over_age = jnp.sum(xs, axis=2) #type: ignore
    ns_over_age = jnp.sum(ns, axis=2) #type: ignore
    prev_over_time = jnp.sum(
        jnp.where(jnp.squeeze(time_mask, 2), xs_over_age / ns_over_age, 0),
        axis=1
    )
    return prev_over_time / jnp.sum(time_mask, axis=(1, 2))
