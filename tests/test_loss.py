import jax.numpy as jnp
from fastms.density.train import trunc_nll, _standardise
from jax import value_and_grad
from jax.tree_util import tree_leaves

def test_trunc_nll_is_finite_at_zero():
    y_min = jnp.zeros(1)
    y_max = jnp.ones(1)
    y_hat = (jnp.zeros(1), jnp.zeros(1))
    y = jnp.zeros(1)
    f = trunc_nll(y_min, y_max)
    assert jnp.isfinite(f(y_hat, y))

def test_trunc_nll_is_not_finite_below_zero():
    y_min = jnp.zeros(1)
    y_max = jnp.ones(1)
    y_hat = (jnp.zeros(1), jnp.zeros(1))
    y = jnp.zeros(1) - 1
    f = trunc_nll(y_min, y_max)
    assert not jnp.isfinite(f(y_hat, y))

def test_trunc_nll_is_finite_at_standardised_bounds():
    y = jnp.arange(4) / 4
    y_mean = jnp.mean(y)
    y_std = jnp.std(y)
    y_hat = (_standardise(jnp.zeros(4), y_mean, y_std), jnp.zeros(1))
    y_scaled = _standardise(y, y_mean, y_std)
    y_min = _standardise(jnp.zeros(1), y_mean, y_std)
    y_max = _standardise(jnp.ones(1), y_mean, y_std)
    f = trunc_nll(y_min, y_max)

    assert jnp.isfinite(f(y_hat, y_scaled))

def test_trunc_nll_is_differentiable_with_infinite_bounds():
    y = jnp.arange(4) / 4
    y_mean = jnp.mean(y)
    y_std = jnp.std(y)
    y_hat = (_standardise(jnp.zeros(4), y_mean, y_std), jnp.zeros(1))
    y_scaled = _standardise(y, y_mean, y_std)
    y_min = _standardise(jnp.zeros(1), y_mean, y_std)
    y_max = _standardise(jnp.array(jnp.inf), y_mean, y_std)
    f = trunc_nll(y_min, y_max)
    x, g = value_and_grad(f)(y_hat, y_scaled)

    assert jnp.isfinite(x)
    for leaf in tree_leaves(g):
        assert jnp.isfinite(leaf).all()

def test_trunc_nll_is_differentiable_at_max_n():
    y = jnp.arange(4) / 4
    y_mean = jnp.mean(y)
    y_std = jnp.std(y)
    y_hat = (_standardise(jnp.zeros(4), y_mean, y_std), jnp.zeros(1))
    y_scaled = _standardise(y, y_mean, y_std)
    y_min = _standardise(jnp.zeros(1), y_mean, y_std)
    y_max = _standardise(jnp.array(jnp.finfo(jnp.float32).max), y_mean, y_std)
    f = trunc_nll(y_min, y_max)
    x, g = value_and_grad(f)(y_hat, y_scaled)

    assert jnp.isfinite(x)
    for leaf in tree_leaves(g):
        assert jnp.isfinite(leaf).all()
