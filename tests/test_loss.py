import jax.numpy as jnp
from fastms.density.rnn import log_prob, _standardise

def test_log_prob_is_finite_at_zero():
    y_min = jnp.zeros(1)
    y_max = jnp.ones(1)
    y_hat = (jnp.zeros(1), jnp.zeros(1))
    y = jnp.zeros(1)
    f = log_prob(y_min, y_max)
    assert jnp.isfinite(f(y_hat, y))

def test_log_prob_is_not_finite_below_zero():
    y_min = jnp.zeros(1)
    y_max = jnp.ones(1)
    y_hat = (jnp.zeros(1), jnp.zeros(1))
    y = jnp.zeros(1) - 1
    f = log_prob(y_min, y_max)
    assert not jnp.isfinite(f(y_hat, y))

def test_log_prob_is_finite_at_standardised_bounds():
    y = jnp.arange(4) / 4
    y_mean = jnp.mean(y)
    y_std = jnp.std(y)
    y_hat = (_standardise(jnp.zeros(4), y_mean, y_std), jnp.zeros(1))
    y_scaled = _standardise(y, y_mean, y_std)
    y_min = _standardise(jnp.zeros(1), y_mean, y_std)
    y_max = _standardise(jnp.ones(1), y_mean, y_std)
    f = log_prob(y_min, y_max)

    assert jnp.isfinite(f(y_hat, y_scaled))
