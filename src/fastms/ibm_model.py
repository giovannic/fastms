from jaxtyping import Array
from typing import Callable, Optional, Dict
import arviz as az
import jax
from jax import numpy as jnp, lax
from jax import random
import numpyro
from numpyro import distributions as dist, optim
from numpyro.infer import (
    MCMC,
    NUTS,
    Predictive,
    SVI,
    Trace_ELBO,
    log_likelihood
)
from numpyro.infer.autoguide import (
    AutoDelta,
    AutoBNAFNormal,
    AutoGuideList
)
from numpyro.infer.reparam import NeuTraReparam
from numpyro.infer.initialization import init_to_median
from numpyro.contrib.tfp.mcmc import TFPKernel
import tensorflow_probability.substrates.jax as tfp
from numpyro.handlers import seed, block
import arviz as az
from functools import partial

import logging

cpu_device = jax.devices('cpu')[0]

MIN_RATE = 1e-12

def model(
    n_sites: int,
    n_prev: Array,
    prev_index: Array,
    inc_pop: Array,
    inc_index: Array,
    prev_impl: Callable[[Dict, Array],Array],
    inc_impl: Callable[[Dict, Array],Array],
    prev: Optional[Array]=None,
    inc: Optional[Array]=None,
    prev_subsample: Optional[int]=None,
    inc_subsample: Optional[int]=None,
    alpha=1.
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

    #eir = numpyro.sample(
    #    'eir',
    #    dist.Uniform(jnp.full(n_sites, 0.), jnp.full(n_sites, 1000.))
    #)

    with numpyro.plate('sites', n_sites):
        # Bug in neutra https://github.com/pyro-ppl/numpyro/issues/1694
        eir = numpyro.sample(
            'eir',
            dist.Uniform(0., 1000.)
        )

        # Overdispersion variables
        q = numpyro.sample(
            'q',
            dist.Exponential(.05**-2)
        )
        inv_phi = numpyro.sample(
            'inv_phi',
            dist.Exponential(.05**-2)
        )

    # Pre-erythrocytic immunity
    kb = numpyro.sample('kb', dist.LogNormal(0., .25))
    ub = numpyro.sample('ub', dist.Gamma(7., 1.))
    b0 = numpyro.sample('b0', dist.Beta(1., 1.))
    ib0 = numpyro.sample(
        'ib0',
        dist.TruncatedDistribution(dist.Normal(50., 20.), low=25., high=75.)
    )
    
    # Clinical immunity
    kc = numpyro.sample('kc', dist.LogNormal(0., .25))
    uc = numpyro.sample('uc', dist.Gamma(7., 1.))
    phi0 = numpyro.sample('phi0', dist.Beta(2., 1.))
    phi1 = numpyro.sample('phi1', dist.Beta(1., 2.))
    ic0 = numpyro.sample(
        'ic0',
        dist.TruncatedDistribution(dist.Normal(25., 10.), low=5., high=50.)
    )
    pcm = numpyro.sample('pcm', dist.Beta(1., 1.))
    rm = numpyro.sample(
        'rm',
        dist.TruncatedDistribution(dist.Normal(50., 20.), low=5., high=100.)
    )
    
    # Detection immunity
    kd = numpyro.sample('kd', dist.LogNormal(0., .25))
    ud = numpyro.sample('ud', dist.Gamma(7., 1.))
    d1 = numpyro.sample('d1', dist.Beta(1., 5.))
    id0 = numpyro.sample(
        'id0',
        dist.TruncatedDistribution(dist.Normal(25., 10.), low=5., high=50.)
    )
    fd0 = numpyro.sample('fd0', dist.Beta(1., 1.))
    gammad = numpyro.sample('gammad', dist.LogNormal(0., 2.))
    ad = numpyro.sample('ad', dist.TruncatedDistribution(
        dist.Normal(70. * 365., 365.),
        low=40. * 365.,
        high=100. * 365.
    ))
    
    ru = numpyro.sample('ru', dist.LogNormal(0., 1.))
    
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
        'ru': ru,
        'cd': cd,
        'cu': cu,
        'gamma1': gamma1
    }

    with numpyro.plate(
        'prev_data',
        len(prev_index),
        subsample_size=prev_subsample) as ind:

        prev_ind = ind
        prev_sites = prev_index[ind]
        obs_prev = prev if prev is None else prev[ind]

    prev_stats = prev_impl( #type: ignore
        x,
        eir[prev_sites],
        prev_sites,
        prev_ind
    )
    alpha = straight_through(
        lambda x: jnp.maximum(x, MIN_RATE),
        (prev_stats) / inv_phi[prev_sites]
    )
    beta = straight_through(
        lambda x: jnp.maximum(x, MIN_RATE),
        (1. - prev_stats) / inv_phi[prev_sites]
    )

    prev_scale = 1. if prev_subsample is None else (
        len(prev_sites)/prev_subsample
    )
    with numpyro.handlers.scale(scale=alpha * prev_scale):
        numpyro.sample(
            'obs_prev',
            dist.Independent(
                dist.BetaBinomial(
                    concentration1=alpha,
                    concentration0=beta,
                    total_count=n_prev[ind], #type: ignore
                    validate_args=True
                ),
                1
            ),
            obs=obs_prev
        )

    with numpyro.plate(
        'inc_data',
        len(inc_index),
        subsample_size=inc_subsample) as ind:

        inc_ind = ind
        inc_sites = inc_index[ind]
        obs_inc = inc if inc is None else inc[ind]

    inc_stats = inc_impl( #type: ignore
        x,
        eir[inc_sites],
        inc_sites,
        inc_ind
    )

    mean = straight_through(
        lambda x: jnp.maximum(x, MIN_RATE),
        inc_stats * inc_pop[inc_ind]
    )

    inc_scale = 1. if inc_subsample is None else (
        len(inc_sites)/inc_subsample
    )

    with numpyro.handlers.scale(scale=alpha * inc_scale):
        numpyro.sample(
            'obs_inc',
            dist.Independent(
                dist.GammaPoisson(
                    mean / q[inc_sites],
                    1. / q[inc_sites], #type: ignore
                    validate_args=True
                ),
                1
            ),
            obs=obs_inc
        )

def surrogate_posterior_svi(
        key: Array,
        autoguide,
        n_train_samples: int = 10000,
        n_samples: int = 100,
        block_stochastic: bool = False,
        **model_args
    ):
    
    # sample prior
    prior_key, key = random.split(key, 2)
    prior_args = {
        k: v for k, v in model_args.items()
        if k not in {
            'prev',
            'inc',
            'prev_subsample',
            'inc_subsample'
        }
    }


    logging.info('Sampling prior')
    prior = Predictive(model, num_samples=n_samples)(
        prior_key,
        **prior_args
    )
    prior = _remove_stoch_variables(prior)

    # initialise SVI
    if block_stochastic:
        stoch_sites = {
            'n_detect',
            'n_detect_n',
            'inc',
            'inc_n'
        }

        guide = AutoGuideList(model)
        guide.append(
            AutoDelta(
                block(
                    seed(model, key),
                    expose=stoch_sites
                ),
                prefix='auto_stoch_' # stop conflicts
            )
        )
        guide.append(
            autoguide(
                block(
                    seed(model, key),
                    hide=stoch_sites
                )
            )
        )
    else:
        guide = autoguide(model)

    svi = SVI(
        model,
        guide,
        optim.ClippedAdam(1e-4),
        loss=Trace_ELBO(num_particles=32),
        **model_args
    )

    # train SVI
    logging.info('Training SVI')
    sample_key, key = random.split(key, 2)
    svi_result = svi.run(sample_key, n_train_samples)
    svi_params = svi_result.params

    # sample posterior
    logging.info('Sampling posterior')
    post_key, key = random.split(key, 2)
    posterior_samples = Predictive(
        guide,
        params=svi_params,
        num_samples=n_samples
    )(post_key)
    #TODO, run predictives first to get likelihood without subsampling
    seeded_model = seed(model, post_key)
    log_likelihoods = log_likelihood( 
        seeded_model,
        posterior_samples,
        **model_args
    )
    posterior_samples = _remove_stoch_variables(posterior_samples)

    # sample posterior predictive
    logging.info('Sampling posterior predictive')
    post_predictive = Predictive(
        model,
        posterior_samples,
        num_samples=n_samples
    )(post_key, **prior_args)
    post_predictive = _remove_stoch_variables(post_predictive)

    logging.info('Compiling results to save')
    data = az.from_dict(
        posterior=_to_arviz_dict(posterior_samples),
        posterior_predictive=_to_arviz_dict(post_predictive),
        prior=_to_arviz_dict({
            k: v
            for k, v in prior.items()
            if k not in {'obs_prev', 'obs_inc'}
        }),
        prior_predictive=_to_arviz_dict({
            k: v
            for k, v in prior.items()
            if k in {'obs_prev', 'obs_inc'}
        }),
        observed_data={
            'obs_prev': model_args['prev'],
            'obs_inc': model_args['inc']
        },
        log_likelihood=_to_arviz_dict(log_likelihoods)
    )
    return data

def surrogate_posterior_neutra(
        key: Array,
        n_train_samples: int = 10000,
        n_samples: int = 100,
        n_warmup: int = 100,
        n_chains: int = 10,
        **model_args
    ):
    logging.info('Sampling prior')
    prior_key, key = random.split(key, 2)
    prior_args = {
        k: v for k, v in model_args.items()
        if k not in ['prev', 'inc']
    }
    prior = Predictive(model, num_samples=n_samples)(
        prior_key,
        **prior_args
    )

    logging.info('Training SVI')
    bound_model = partial(model, **model_args)
    guide = AutoBNAFNormal(bound_model, num_flows=2)
    svi = SVI(
        bound_model,
        guide,
        optim.ClippedAdam(1e-4),
        loss=Trace_ELBO(num_particles=128),
    )
    svi_key, key = random.split(key, 2)
    svi_result = svi.run(svi_key, n_train_samples, stable_update=True)
    svi_params = svi_result.params

    neutra = NeuTraReparam(guide, svi_params)
    neutra_model = neutra.reparam(bound_model)

    kernel = NUTS(neutra_model)
    logging.info('Sampling posterior')
    mcmc = MCMC(
        kernel,
        num_samples=n_samples,
        num_warmup=n_warmup,
        num_chains=n_chains,
        chain_method='vectorized'
    )
    mcmc_key, key = random.split(key, 2)
    mcmc.run(mcmc_key)

    posterior_samples = mcmc.get_samples()
    log_likelihoods = log_likelihood(
        model,
        posterior_samples,
        **model_args
    )

    logging.info('Sampling posterior predictive')
    post_key, key = random.split(key, 2)
    posterior_predictive = Predictive(model, posterior_samples)(
        post_key,
        **prior_args
    )

    logging.info('Compiling results to save')
    data = az.from_dict(
        posterior=_to_arviz_dict(posterior_samples),
        posterior_predictive=_to_arviz_dict(posterior_predictive),
        prior=_to_arviz_dict({
            k: v
            for k, v in prior.items()
            if k not in {'obs_prev', 'obs_inc'}
        }),
        prior_predictive=_to_arviz_dict({
            k: v
            for k, v in prior.items()
            if k in {'obs_prev', 'obs_inc'}
        }),
        observed_data={
            'obs_prev': model_args['prev'],
            'obs_inc': model_args['inc']
        },
        log_likelihood=_to_arviz_dict(log_likelihoods)
    )
    return data

def surrogate_posterior(
        key: Array,
        n_samples: int = 100,
        n_warmup: int = 100,
        n_chains: int = 10,
        kernel_type: str = 'nuts',
        **model_args
    ):
    # NOTE: Reverse mode has lead to initialisation errors for dmeq
    if kernel_type == 'nuts':
        kernel = NUTS(
            model,
            dense_mass=True,
            #dense_mass=[
            #    ('kb', 'ub', 'b0', 'ib0'),
            #    ('kc', 'uc', 'ic0', 'phi0', 'phi1', 'pcm', 'rm'),
            #    ('kd', 'ud', 'd1', 'id0', 'fd0', 'gammad', 'ad', 'ru'),
            #    ('cd', 'cu', 'gamma1'),
            #],
            target_accept_prob=.90,
            max_tree_depth=15,
            init_strategy=init_to_median
        )
    else:
        assert kernel_type == 'pt'
        swap_fn = tfp.mcmc.even_odd_swap_proposal_fn(1)
        inverse_temperatures = .2 ** jnp.arange(n_chains)
        def make_kernel_fn(target_log_prob_fn):
            return tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                step_size=1e-3 / jnp.sqrt(0.5 ** jnp.arange(n_chains)[..., None]),
                num_leapfrog_steps=100
            )

        kernel = TFPKernel[tfp.mcmc.ReplicaExchangeMC](
            model,
            inverse_temperatures=inverse_temperatures,
            make_kernel_fn=make_kernel_fn,
            swap_proposal_fn=swap_fn
        )

    logging.info('Sampling prior')
    prior_key, key = random.split(key, 2)
    prior_args = {
        k: v for k, v in model_args.items()
        if k not in ['prev', 'inc']
    }
    prior = Predictive(model, num_samples=n_samples)(
        prior_key,
        **prior_args
    )

    logging.info('Sampling posterior')
    sample_key, key = random.split(key, 2)
    mcmc = MCMC(
        kernel,
        num_samples=n_samples,
        num_warmup=n_warmup,
        num_chains=n_chains,
        chain_method='vectorized' if kernel_type == 'nuts' else 'parallel'
    )
    mcmc.run(sample_key, **model_args)

    logging.info('Sampling posterior predictive')
    post_key, key = random.split(key, 2)
    posterior_predictive = Predictive(model, mcmc.get_samples())(
        post_key,
        **prior_args
    )

    logging.info('Compiling outputs for saving')
    data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive
    )
    return data

def sample_fake_data(key: Array, **model_args):
    sample_key, key = random.split(key, 2)
    truth = Predictive(model, num_samples=1)(
        sample_key,
        **model_args
    )
    return truth

def straight_through(f, x):
    # Create an exactly-zero expression with Sterbenz lemma that has
    # an exactly-one gradient.
    zero = x - lax.stop_gradient(x)
    return zero + lax.stop_gradient(f(x))

def _to_arviz_dict(samples):
    return {
        k: v[None, ...]
        for k, v in samples.items()
    }

def _remove_stoch_variables(samples):
    return {
        k: v
        for k, v in samples.items()
        if not k in {'inc', 'inc_n', 'n_detect', 'n_detect_n'}
    }
