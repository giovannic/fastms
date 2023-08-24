from fastms.infer.parser import _aggregate
from jax import numpy as jnp

def test_aggregation_works_with_top_quarter():
    n = 2
    xs = jnp.arange(200).reshape((n, 10, 10))
    ns = jnp.arange(200).reshape((n, 10, 10))
    xs = xs.at[0,0:5,0:5].set(5)
    ns = ns.at[0,0:5,0:5].set(10)
    xs = xs.at[1,0:5,0:5].set(2)
    ns = ns.at[1,0:5,0:5].set(10)
    age_lower = jnp.zeros((n,))
    age_upper = jnp.full((n,), 4)
    time_lower = jnp.zeros((n,))
    time_upper = jnp.full((n,), 4)
    agg = _aggregate(xs, ns, age_lower, age_upper, time_lower, time_upper)
    expected = jnp.array([.5, .2])
    assert jnp.array_equal(agg, expected)

