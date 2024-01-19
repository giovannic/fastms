from jaxtyping import Array
from typing import Callable, Optional, Dict, Tuple
import numpyro #type: ignore
from jax import numpy as jnp, lax
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import arviz as az

from jax import random

MIN_RATE = 1e-12

def model(
    n_sites: int,
    n_prev: Array,
    prev_index: Array,
    inc_risk_time: Array,
    inc_index: Array,
    impl: Callable[[Dict, Array],Tuple[Array, Array]],
    prev: Optional[Array]=None,
    inc: Optional[Array]=None,
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
    with numpyro.plate('sites', n_sites):
        eir = numpyro.sample(
            'eir',
            dist.Uniform(0., 500.)
        )

        # Overdispersion variables
        q = numpyro.sample(
            'q',
            dist.Beta(10., 1.)
        )
        mu = numpyro.sample(
            'mu',
            dist.Gamma(2., 2.)
        )
        z = numpyro.sample(
            'z',
            dist.Beta(10., 1.)
        )

    kb = numpyro.sample('kb', dist.LogNormal(0., 1.))
    ub = numpyro.sample('ub', dist.LogNormal(0., 1.))
    b0 = numpyro.sample('b0', dist.Beta(1., 1.))
    ib0 = numpyro.sample(
        'ib0',
        dist.TruncatedDistribution(dist.Cauchy(50., 10.), low=0., high=100.)
    )
    
    # Clinical immunity
    kc = numpyro.sample('kc', dist.LogNormal(0., 1.))
    uc = numpyro.sample('uc', dist.LogNormal(0., 1.))
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
    ) #TODO: Should this be inverse?
    
    # Detection immunity
    kd = numpyro.sample('kd', dist.LogNormal(0., 1.))
    ud = numpyro.sample('ud', dist.LogNormal(0., 1.))
    d1 = numpyro.sample('d1', dist.Beta(1., 5.))
    id0 = numpyro.sample(
        'id0',
        dist.LeftTruncatedDistribution(dist.Cauchy(25., 1.), low=5.)
    )
    fd0 = numpyro.sample('fd0', dist.Beta(1., 1.))
    gammad = numpyro.sample('gammad', dist.LogNormal(0., 1.))
    ad = numpyro.sample('ad', dist.TruncatedDistribution(
        dist.Cauchy(50. * 365., 365.),
        low=20. * 365.,
        high=80. * 365.
    ))
    
    ru = numpyro.sample('ru', dist.LogNormal(0., 1.))
    du = 1 / ru #type: ignore
    
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
    
    prev_stats, inc_stats = impl(x, eir) #type: ignore

    round_n = lax.convert_element_type(
        straight_through(jnp.ceil, n_prev * z[prev_index]),
        jnp.int64
    )

    numpyro.sample(
        'obs_prev',
        dist.Independent(
            dist.Binomial(
                total_count=round_n, #type: ignore
                probs=jnp.minimum(prev_stats * z[prev_index], 1.),
                validate_args=True
            ),
            1
        ),
        obs=prev
    )

    mean = straight_through(
        lambda x: jnp.maximum(x, MIN_RATE),
        inc_stats * inc_risk_time * mu[inc_index]
    )

    numpyro.sample(
        'obs_inc',
        dist.Independent(
            dist.GammaPoisson(
                mean * q[inc_index],
                q[inc_index],
                validate_args=True
            ),
            1
        ),
        obs=inc
    )

def surrogate_posterior(
        key: Array,
        n_samples: int = 100,
        n_warmup: int = 100,
        n_chains: int = 10,
        **model_args
        ):
    # NOTE: Reverse mode has lead to initialisation errors for dmeq
    kernel = NUTS(model)

    prior_key, key = random.split(key, 2)
    prior = Predictive(model, num_samples=500)(
        prior_key,
        **model_args
    )

    sample_key, key = random.split(key, 2)
    mcmc = MCMC(
            kernel,
            num_samples=n_samples,
            num_warmup=n_warmup,
            num_chains=n_chains,
            chain_method='vectorized' #pmap leads to segfault for some reason (https://github.com/google/jax/issues/13858)
            )
    mcmc.run(sample_key, **model_args)

    post_key, key = random.split(key, 2)
    posterior_predictive = Predictive(model, mcmc.get_samples())(
        post_key,
        **model_args
    )
    data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive
    )
    return data

def straight_through(f, x):
    # Create an exactly-zero expression with Sterbenz lemma that has
    # an exactly-one gradient.
    zero = x - lax.stop_gradient(x)
    return zero + lax.stop_gradient(f(x))
