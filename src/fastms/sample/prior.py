from mox.sampling import DistStrategy, sample #type: ignore
from numpyro import distributions as dist #type: ignore
from jaxtyping import PyTree
from jax import random
from .sites import import_sites, sample_sites, sites_to_tree
from .ibm import run_ibm

_prior_intrinsic_space = {
    'kb': DistStrategy(dist.LogNormal(0., .25)),
    'ub': DistStrategy(dist.LogNormal(0., .25)),
    'b0': DistStrategy(dist.Beta(1., 1.)),
    'ib0': DistStrategy(
        dist.LeftTruncatedDistribution(dist.Normal(50., 10.), low=0.)
    ),
    'kc': DistStrategy(dist.LogNormal(0., .25)),
    'uc': DistStrategy(dist.LogNormal(0., .25)),
    'ic0': DistStrategy(
        dist.LeftTruncatedDistribution(dist.Cauchy(100., 10.), low=0.)
    ),
    'phi0': DistStrategy(dist.Beta(5., 1.)),
    'phi1': DistStrategy(dist.Beta(1., 2.)),
    'pcm': DistStrategy(dist.Beta(1., 1.)),
    'rm': DistStrategy(
        dist.LeftTruncatedDistribution(dist.Cauchy(200., 10.), low=0.)
    ),
    'kd': DistStrategy(dist.LogNormal(0., .25)),
    'ud': DistStrategy(dist.LogNormal(0., .25)),
    'd1': DistStrategy(dist.Beta(1., 2.)),
    'id0': DistStrategy(
        dist.LeftTruncatedDistribution(dist.Cauchy(25., 1.), low=0.)
    ),
    'fd0': DistStrategy(dist.Beta(1., 1.)),
    'gammad': DistStrategy(dist.LogNormal(0., .25)),
    'ad': DistStrategy(dist.TruncatedDistribution(
        dist.Cauchy(30. * 365., 365.),
        low=20. * 365.,
        high=40. * 365.
    )),
    'du': DistStrategy(
        dist.LeftTruncatedDistribution(dist.Cauchy(100., 10.), low=0.)
    ),
    'cd': DistStrategy(dist.Beta(1., 2.)),
    'cu': DistStrategy(dist.Beta(1., 5.)),
    'gamma1': DistStrategy(dist.LogNormal(0., .25))
}

def sample_prior(
    site_path: str,
    n: int,
    key: random.PRNGKeyArray,
    cores: int = 1
    ) -> PyTree:
    EIR = DistStrategy(dist.Uniform(0., 500.))
    X_intrinsic, X_eir = sample(
        [_prior_intrinsic_space, EIR],
        n,
        key
    )
    sites = import_sites(site_path)
    #TODO: fix years beyond files
    site_samples = sample_sites(sites, n, key, 1985, 2018)
    X_sites = sites_to_tree(site_samples, sites, 1985, 2018)
    y = run_ibm(
        X_intrinsic,
        sites,
        site_samples,
        X_eir,
        cores
    )
    X = [X_intrinsic, X_eir, X_sites['seasonality'], X_sites['vectors']]
    X_seq = [X_sites['interventions'], X_sites['demography']]
    return (X, X_seq), y
