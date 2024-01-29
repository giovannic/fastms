from jax import random, numpy as jnp
from jaxtyping import PyTree
from numpyro import distributions as dist #type: ignore
from .sites import import_sites, pad_sites, sites_to_tree, sample_sites
from .ibm import run_ibm


def sample_calibration(
    site_path: str,
    n: int,
    key: random.PRNGKeyArray,
    burnin: int,
    cores: int = 1,
    start_year: int = 1985,
    end_year: int = 2018,
    population: int = 100000
    ) -> PyTree:
    init_EIR = dist.Uniform(0., 500.).sample(key, (n,))
    sites = import_sites(site_path)
    sites = pad_sites(sites, start_year, end_year)
    site_samples = sample_sites(sites, n, key)
    X_sites = sites_to_tree(site_samples, sites)
    y = run_ibm(
        None,
        sites,
        site_samples,
        init_EIR,
        burnin,
        cores,
        population=population
    )
    X = {
        'baseline_eir': init_EIR,
        'seasonality': X_sites['seasonality'],
        'vector_composition': X_sites['vectors']
    }
    X_seq = {
        'interventions': X_sites['interventions'],
        'demography': X_sites['demography']
    }
    X_t = jnp.arange(0, end_year - start_year + 1) * 365
    return (X, X_seq, X_t), y
