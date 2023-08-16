from jaxtyping import Array, PyTree
from typing import Callable, Optional, Dict, Tuple
import numpyro #type: ignore
from jax import numpy as jnp
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS #type: ignore

from jax.random import PRNGKeyArray

MIN_RATE = 1e-12

def model(
    n_prev: Array,
    impl: Callable[[Dict, Array],Tuple[Array, Array]],
    prev: Optional[Array]=None,
    inc: Optional[Array]=None
    ):
    """
    model. A numpyro model for fitting IBM parameters to prevalence/incidence
    data

    :param prev_N: Array Counts of individuals surveyed for the prevalence
    statistics
    :param impl: Callable[[Dict, Array],Tuple[Array, Array]] a model implementation
    which takes a dictionary of model parameters and EIR and returns projected 
    prevalence and incidence statistics in the same shape as the observed prev
    and inc arguments
    :param prev: Array an array of observed prevalence statistics
    :param inc: Array an array of observed incidence statistics
    """
    # Pre-erythrocytic immunity
    eir = numpyro.sample('eir', dist.Uniform(0., 500.))

    kb = numpyro.sample('kb', dist.LogNormal(0., .25))
    ub = numpyro.sample('ub', dist.LogNormal(0., .25))
    b0 = numpyro.sample('b0', dist.Beta(1., 1.))
    ib0 = numpyro.sample(
        'ib0',
        dist.LeftTruncatedDistribution(dist.Normal(50., 10.), low=0.)
    )
    
    # Clinical immunity
    kc = numpyro.sample('kc', dist.LogNormal(0., .25))
    uc = numpyro.sample('uc', dist.LogNormal(0., .25))
    phi0 = numpyro.sample('phi0', dist.Beta(5., 1.))
    phi1 = numpyro.sample('phi1', dist.Beta(1., 2.))
    ic0 = numpyro.sample(
        'ic0',
        dist.LeftTruncatedDistribution(dist.Cauchy(100., 10.), low=0.)
    )
    pcm = numpyro.sample('pcm', dist.Beta(1., 1.))
    rm = numpyro.sample(
        'rm',
        dist.LeftTruncatedDistribution(dist.Cauchy(200., 10.), low=0.)
    )
    
    # Detection immunity
    kd = numpyro.sample('kd', dist.LogNormal(0., .25))
    ud = numpyro.sample('ud', dist.LogNormal(0., .25))
    d1 = numpyro.sample('d1', dist.Beta(1., 2.))
    id0 = numpyro.sample(
        'id0',
        dist.LeftTruncatedDistribution(dist.Cauchy(25., 1.), low=0.)
    )
    fd0 = numpyro.sample('fd0', dist.Beta(1., 1.))
    gammad = numpyro.sample('gammad', dist.LogNormal(0., .25))
    ad = numpyro.sample('ad', dist.TruncatedDistribution(
        dist.Cauchy(30. * 365., 365.),
        low=20. * 365.,
        high=40. * 365.
    ))
    
    du = numpyro.sample(
        'du',
        dist.LeftTruncatedDistribution(dist.Cauchy(100., 10.), low=0.)
    )
    
    # FOIM
    cd = numpyro.sample('cd', dist.Beta(1., 2.))
    cu = numpyro.sample('cu', dist.Beta(1., 5.))
    gamma1 = numpyro.sample('gamma1', dist.LogNormal(0., .25))
    
    x = {
        'kb': kb,
        'ub': ub,
        'b0': b0,
        'ib0': ib0,
        'kc': kc,
        'uc': uc,
        'ic0': ic0,
        'phi0': phi0,
        'phi1': phi1,
        'pcm': pcm,
        'rm': rm,
        'kd': kd,
        'ud': ud,
        'd1': d1,
        'id0': id0,
        'fd0': fd0,
        'gammad': gammad,
        'ad': ad,
        'du': du,
        'cd': cd,
        'cu': cu,
        'gamma1': gamma1
    }
    
    prev_stats, inc_stats = impl(x, eir)
    
    numpyro.sample(
        'obs_prev',
        dist.Independent(
            dist.Binomial(
                total_count=n_prev,
                probs=prev_stats,
                validate_args=True
            ),
            1
        ),
        obs=prev
    )

    numpyro.sample(
        'obs_inc',
        dist.Independent(
            dist.Poisson(
                rate=jnp.maximum(inc_stats, MIN_RATE),
                validate_args=True
            ),
            1
        ),
        obs=inc
    )

def surrogate_posterior(
    key: PRNGKeyArray,
    impl: Callable[[Dict, Array],Tuple[Array, Array]],
    n_prev: Array,
    prev: Array,
    inc: Array,
	n_samples: int = 100,
	n_warmup: int = 100,
	n_chains: int = 10
	) -> PyTree:
	# NOTE: Reverse mode has lead to initialisation errors for dmeq
	kernel = NUTS(model)

	mcmc = MCMC(
		kernel,
		num_samples=n_samples,
		num_warmup=n_warmup,
		num_chains=n_chains,
		chain_method='vectorized' #pmap leads to segfault for some reason (https://github.com/google/jax/issues/13858)
	)
	mcmc.run(key, n_prev, impl, prev, inc)
	return mcmc.get_samples()
