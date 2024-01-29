from mox.sampling import DistStrategy, sample
from numpyro import distributions as dist
from jaxtyping import Array, PyTree
from jax import numpy as jnp
from .sites import import_sites, pad_sites, sample_sites, sites_to_tree
from .ibm import run_ibm

_prior_intrinsic_space = {
    'kb': DistStrategy(dist.LogNormal(0., .1)),
    'ub': DistStrategy(dist.LogNormal(0., 1.)),
    'b0': DistStrategy(dist.Beta(1., 1.)),
    'ib0': DistStrategy(
        dist.LeftTruncatedDistribution(dist.Normal(50., 10.), low=0.)
    ),
    'kc': DistStrategy(dist.LogNormal(0., .1)),
    'uc': DistStrategy(dist.LogNormal(0., 1.)),
    'ic0': DistStrategy(
        dist.LeftTruncatedDistribution(dist.Cauchy(100., 10.), low=0.)
    ),
    'phi0': DistStrategy(dist.Beta(5., 1.)),
    'phi1': DistStrategy(dist.Beta(1., 2.)),
    'pcm': DistStrategy(dist.Beta(1., 1.)),
    'rm': DistStrategy(
        dist.LeftTruncatedDistribution(dist.Cauchy(200., 10.), low=0.)
    ),
    'kd': DistStrategy(dist.LogNormal(0., .1)),
    'ud': DistStrategy(dist.LogNormal(0., 1.)),
    'd1': DistStrategy(dist.Beta(1., 2.)),
    'id0': DistStrategy(
        dist.LeftTruncatedDistribution(dist.Cauchy(25., 1.), low=0.)
    ),
    'fd0': DistStrategy(dist.Beta(1., 1.)),
    'gammad': DistStrategy(dist.LogNormal(0., .1)),
    'ad': DistStrategy(dist.TruncatedDistribution(
        dist.Cauchy(70. * 365., 365.),
        low=40. * 365.,
        high=100. * 365.
    )),
    'ru': DistStrategy(dist.LogNormal(0., 1.)),
    'cd': DistStrategy(dist.Beta(1., 2.)),
    'cu': DistStrategy(dist.Beta(1., 5.)),
    'gamma1': DistStrategy(dist.LogNormal(0., .1))
}

def sample_prior(
    site_path: str,
    n: int,
    key: Array,
    burnin,
    cores: int = 1,
    start_year: int = 1985,
    end_year: int = 2018,
    population: int = 100000
    ) -> PyTree:
    EIR = DistStrategy(dist.Uniform(0., 1000.))
    X_intrinsic, init_EIR = sample(
        [_prior_intrinsic_space, EIR], # type: ignore
        n,
        key
    )
    sites = import_sites(site_path)
    sites = pad_sites(sites, start_year, end_year)
    site_samples = sample_sites(sites, n, key)
    X_sites = sites_to_tree(site_samples, sites)
    y = run_ibm(
        X_intrinsic,
        sites,
        site_samples,
        init_EIR,
        burnin,
        cores,
        population=population,
    )
    X = {
        'intrinsic': X_intrinsic,
        'init_EIR': init_EIR,
        'seasonality': X_sites['seasonality'],
        'vector_composition': X_sites['vectors']
    }
    X_seq = {
        'interventions': X_sites['interventions'],
        'demography': X_sites['demography']
    }
    X_t = jnp.arange(0, end_year - start_year + 1) * 365
    return (X, X_seq, X_t), y
