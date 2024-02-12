from typing import Tuple, Dict
import arviz as az
import numpy as np
from numpy.typing import NDArray
from jaxtyping import Array
from jax.tree_util import tree_map
from jax import random
import jax.numpy as jnp

from ..sites import make_site_inference_data
from ..sample.ibm import run_ibm
from ..sample.prior import _prior_intrinsic_space

def sample_from_data(
    key: Array,
    data_path: str,
    sites_path: str,
    burnin: int,
    sample_start: int,
    n_samples: int,
    cores: int=1,
    start_year: int=1985,
    end_year: int=2018,
    population: int=100000,
    ) -> Tuple[Tuple[Tuple[Dict, Dict, NDArray], NDArray], Array, Array]:
    """Sample from the posterior distribution of the IBM model given data.

    Args:
        data_path: Path to the data file.
        sites_path: Path to the sites file.
        burnin: Number of burnin iterations.
        sample_start: Start index for sampling from the data.
        n_samples: Number of samples to take.
        cores: Number of cores to use.
        start_year: Start year of the data.
        end_year: End year of the data.
        population: Population size.
        dynamic_burnin: Whether to use dynamic burnin.

    Returns:
        Samples from the posterior distribution of the IBM model.
    """
    # Load inference data
    data = az.from_netcdf(data_path)

    # Load site data
    sites = make_site_inference_data(sites_path, start_year, end_year)

    # Extract samples from the posterior distribution
    sample_end = sample_start + n_samples
    init_EIR = np.array(
        az.extract(
            data,
            var_names='eir',
            rng=False
        )
    )
    sample_space = init_EIR.shape[0] * init_EIR.shape[1] # sites * intrinsic samples
    samples = random.permutation(
        key,
        jnp.arange(sample_space)
    )[sample_start:sample_end]
    site_samples = samples // init_EIR.shape[1]
    intrinsic_samples = samples % init_EIR.shape[1]
    init_EIR = init_EIR[site_samples, intrinsic_samples].reshape(-1)
    intrinsic_vars = list(_prior_intrinsic_space.keys())
    intrinsic_draws = az.extract(
        data,
        var_names=intrinsic_vars,
        rng=False
    )
    X_intrinsic = {
        x: np.array(intrinsic_draws[x][intrinsic_samples])
        for x in intrinsic_vars
    }
    site_df = sites.site_index.loc[
        sites.site_index.iloc[site_samples].index.repeat(n_samples)
    ]
    y = run_ibm(
        X_intrinsic,
        sites.site_df_dict,
        site_df,
        init_EIR,
        burnin,
        cores,
        population=population
    )
    X_sites = tree_map(
        lambda leaf: leaf[site_samples],
        sites.x_sites
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
    X_t = np.arange(0, end_year - start_year + 1) * 365
    return ((X, X_seq, X_t), y), site_samples, intrinsic_samples
