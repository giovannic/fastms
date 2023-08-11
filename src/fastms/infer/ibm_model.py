import numpyro #type: ignore
from jax import numpy as jnp
from numpyro import distributions as dist
from jaxtyping import Array
from typing import Callable, Optional, Dict, Tuple

MIN_RATE = 1e-12

def model(
    prev_N: Array,
    impl: Callable[[Dict],Tuple[Array, Array]],
    prev: Optional[Array]=None,
    inc: Optional[Array]=None
    ):
    """
    model. A numpyro model for fitting IBM parameters to prevalence/incidence
    data

    :param prev_N: Array Counts of individuals surveyed for the prevalence
    statistics
    :param impl: Callable[[Dict],Tuple[Array, Array]] a model implementation
    which takes a dictionary of model parameters and returns projected 
    prevalence and incidence statistics in the same shape as the observed prev
    and inc arguments
    :param prev: Array an array of observed prevalence statistics
    :param inc: Array an array of observed incidence statistics
    """
    # Pre-erythrocytic immunity
    kb = numpyro.sample('kb', dist.LogNormal(0., .25))
    ub = numpyro.sample('ub', dist.LogNormal(0., .25))
    b0 = numpyro.sample('b0', dist.Beta(1., 1.))
    IB0 = numpyro.sample(
        'IB0',
        dist.LeftTruncatedDistribution(dist.Normal(50., 10.), low=0.)
    )
    
    # Clinical immunity
    kc = numpyro.sample('kc', dist.LogNormal(0., .25))
    uc = numpyro.sample('uc', dist.LogNormal(0., .25))
    phi0 = numpyro.sample('phi0', dist.Beta(5., 1.))
    phi1 = numpyro.sample('phi1', dist.Beta(1., 2.))
    IC0 = numpyro.sample(
        'IC0',
        dist.LeftTruncatedDistribution(dist.Cauchy(100., 10.), low=0.)
    )
    PM = numpyro.sample('PM', dist.Beta(1., 1.))
    dm = numpyro.sample(
        'dm',
        dist.LeftTruncatedDistribution(dist.Cauchy(200., 10.), low=0.)
    )
    
    # Detection immunity
    kd = numpyro.sample('kd', dist.LogNormal(0., .25))
    ud = numpyro.sample('ud', dist.LogNormal(0., .25))
    d1 = numpyro.sample('d1', dist.Beta(1., 2.))
    ID0 = numpyro.sample(
        'ID0',
        dist.LeftTruncatedDistribution(dist.Cauchy(25., 1.), low=0.)
    )
    fd0 = numpyro.sample('fd0', dist.Beta(1., 1.))
    gd = numpyro.sample('gd', dist.LogNormal(0., .25))
    ad0 = numpyro.sample('ad0', dist.TruncatedDistribution(
            dist.Cauchy(30. * 365., 365.),
            low=20. * 365.,
            high=40. * 365.
        ))
    
    ru = numpyro.sample('rU', dist.LogNormal(0., .25))
    
    # FOIM
    cd = numpyro.sample('cD', dist.Beta(1., 2.))
    cu = numpyro.sample('cU', dist.Beta(1., 5.))
    g_inf = numpyro.sample('g_inf', dist.LogNormal(0., .25))
    
    x = {
        'kb': kb,
        'ub': ub,
        'b0': b0,
        'IB0': IB0,
        'kc': kc,
        'uc': uc,
        'IC0': IC0,
        'phi0': phi0,
        'phi1': phi1,
        'PM': PM,
        'dm': dm,
        'kd': kd,
        'ud': ud,
        'd1': d1,
        'ID0': ID0,
        'fd0': fd0,
        'gd': gd,
        'ad0': ad0,
        'rU': ru,
        'cD': cd,
        'cU': cu,
        'g_inf': g_inf
    }
    
    prev_stats, inc_stats = impl(x)
    
    numpyro.sample(
        'obs_prev',
        dist.Independent(
            dist.Binomial(total_count=prev_N, probs=prev_stats, validate_args=True),
            1
        ),
        obs=prev
    )

    numpyro.sample(
        'obs_inc',
        dist.Independent(
            dist.Poisson(rate=jnp.maximum(inc_stats, MIN_RATE)),
            1
        ),
        obs=inc
    )
