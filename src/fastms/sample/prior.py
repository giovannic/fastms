from mox.sampling import DistStrategy, sample #type: ignore
from numpyro import distributions as dist #type: ignore
from jaxtyping import PyTree
from jax import random
from .sites import sample_sites
from .ibm import run_ibm

_prior_intrinsic_space = {
    'kb': DistStrategy(dist.LogNormal(0., .25)),
    'ub': DistStrategy(dist.LogNormal(0., .25)),
    'b0': DistStrategy(dist.Beta(1., 1.)),
    'IB0': DistStrategy(
        dist.LeftTruncatedDistribution(dist.Normal(50., 10.), low=0.)
    ),
    'kc': DistStrategy(dist.LogNormal(0., .25)),
    'uc': DistStrategy(dist.LogNormal(0., .25)),
    'IC0': DistStrategy(
        dist.LeftTruncatedDistribution(dist.Cauchy(100., 10.), low=0.)
    ),
    'phi0': DistStrategy(dist.Beta(5., 1.)),
    'phi1': DistStrategy(dist.Beta(1., 2.)),
    'PM': DistStrategy(dist.Beta(1., 1.)),
    'dm': DistStrategy(
        dist.LeftTruncatedDistribution(dist.Cauchy(200., 10.), low=0.)
    ),
    'kd': DistStrategy(dist.LogNormal(0., .25)),
    'ud': DistStrategy(dist.LogNormal(0., .25)),
    'd1': DistStrategy(dist.Beta(1., 2.)),
    'ID0': DistStrategy(
        dist.LeftTruncatedDistribution(dist.Cauchy(25., 1.), low=0.)
    ),
    'fd0': DistStrategy(dist.Beta(1., 1.)),
    'gd': DistStrategy(dist.LogNormal(0., .25)),
    'ad0': DistStrategy(dist.TruncatedDistribution(
        dist.Cauchy(30. * 365., 365.),
        low=20. * 365.,
        high=40. * 365.
    )),
    'rU': DistStrategy(dist.LogNormal(0., .25)),
    'cD': DistStrategy(dist.Beta(1., 2.)),
    'cU': DistStrategy(dist.Beta(1., 5.)),
    'g_inf': DistStrategy(dist.LogNormal(0., .25))
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
    X_sites = sample_sites(site_path, n, key, 1985, 2018)
    y = run_ibm(
        X_intrinsic,
        X_sites,
        X_eir,
        cores
    )
    X = [X_intrinsic, X_eir, X_sites['seasonality'], X_sites['vectors']]
    X_seq = [X_sites['interventions'], X_sites['demography']]
    return (X, X_seq), y
