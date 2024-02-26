from mox.sampling import DistStrategy, sample
from numpyro import distributions as dist
from numpyro.distributions import transforms as trans
from jaxtyping import Array, PyTree
from jax import numpy as jnp
from .sites import import_sites, pad_sites, sample_sites, sites_to_tree
from .ibm import run_ibm

_prior_intrinsic_space = {
    'kb': DistStrategy(dist.TransformedDistribution(
        dist.LogNormal(0., .25),
        trans.AffineTransform(
            1.,
            1.,
            domain=dist.constraints.positive
        )
    )),
    'ub': DistStrategy(dist.Gamma(7., 1.)),
    'b0': DistStrategy(dist.Beta(1., 1.)),
    'ib0': DistStrategy(
        dist.TruncatedDistribution(dist.Normal(50., 20.), low=25., high=75.)
    ),
    'kc': DistStrategy(dist.TransformedDistribution(
        dist.LogNormal(0., .25),
        trans.AffineTransform(
            1.,
            1.,
            domain=dist.constraints.positive
        )
    )),
    'uc': DistStrategy(dist.Gamma(7., 1.)),
    'ic0': DistStrategy(
        dist.TruncatedDistribution(dist.Normal(25., 10.), low=5., high=50.)
    ),
    'phi0': DistStrategy(dist.Beta(10., 1.)),
    'phi1': DistStrategy(dist.Beta(1., 10.)),
    'pcm': DistStrategy(dist.Beta(1., 1.)),
    'rm': DistStrategy(
        dist.TruncatedDistribution(dist.Normal(50., 20.), low=5., high=100.)
    ),
    'kd': DistStrategy(dist.LogNormal(0., .25)),
    'ud': DistStrategy(dist.Gamma(7., 1.)),
    'd1': DistStrategy(dist.Beta(1., 10.)),
    'id0': DistStrategy(
        dist.TruncatedDistribution(dist.Normal(25., 10.), low=5., high=50.)
    ),
    'fd0': DistStrategy(dist.Beta(1., 1.)),
    'gammad': DistStrategy(dist.LogNormal(0., 2.)),
    'ad': DistStrategy(dist.TruncatedDistribution(
        dist.Normal(50. * 365., 365.),
        low=20. * 365.,
        high=80. * 365.
    )),
    'ru': DistStrategy(dist.LogNormal(0., 1.)),
    'cd': DistStrategy(dist.Beta(1., 2.)),
    'cu': DistStrategy(dist.Beta(1., 5.)),
    'gamma1': DistStrategy(dist.LogNormal(0., .25))
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
    EIR = DistStrategy(dist.Uniform(0., 400.))
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
