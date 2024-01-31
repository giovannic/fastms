from typing import Tuple, Callable
from jaxtyping import Array
from jax import numpy as jnp
from jax.scipy.stats import truncnorm

def trunc_nll(
    y_min: Array,
    y_max: Array,
    ) -> Callable[[Tuple[Array, Array], Array], Array]:
    def f(y_hat: Tuple[Array, Array], y: Array):
        mu, logsigma = y_hat
        sigma = jnp.exp(logsigma)
        return -jnp.sum(truncnorm.logpdf(
            y,
            _standardise(y_min, mu, sigma),
            _standardise(y_max, mu, sigma),
            loc=mu, #type: ignore
            scale=sigma #type: ignore
        ))
    return f

def _standardise(x: Array, mu: Array, sigma: Array) -> Array:
    return (x - mu) / sigma
