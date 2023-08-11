from rpy2.robjects.packages import importr #type: ignore
import rpy2.robjects as ro #type: ignore
from rpy2.robjects import pandas2ri #type: ignore
from multiprocessing import Pool
from jaxtyping import PyTree, Array
import pandas as pd
from jax.tree_util import tree_map
from jax import numpy as jnp

BURNIN = 50

def run_ibm(
    X_intrinsic: PyTree,
    sites: dict,
    site_samples: pd.DataFrame,
    X_eir: Array,
    cores: int
    ) -> PyTree:
    n = X_eir.shape[0]
    with Pool(cores) as pool:
        args = (
            (
                _extract_from_tree(X_intrinsic, i),
                _extract_site(sites, site_samples, i),
                X_eir[i]
            )
            for i in range(n)
        )
        outputs = pool.starmap(_run_ibm, args)
        return jnp.stack(outputs)

def _extract_from_tree(tree: PyTree, i: int) -> PyTree:
    return tree_map(lambda leaf: leaf[i], tree)

def _extract_site(
    sites: dict,
    samples: pd.DataFrame,
    i: int
    ) -> dict:
    site = samples.iloc[i]
    ints = sites['interventions']
    ints = ints[(ints.iso3c == site.iso3c) &
                (ints.name_1 == site.name_1) &
                (ints.urban_rural == site.urban_rural)]
    dem = sites['demography']
    dem = dem[(dem.iso3c == site.iso3c)]
    seas = sites['seasonality']
    seas = seas[(seas.iso3c == site.iso3c) &
                (seas.name_1 == site.name_1)]
    vec = sites['vectors']
    vec = vec[
        (vec.iso3c == site.iso3c) &
        (vec.name_1 == site.name_1)
    ]

    return {
        'interventions': ints,
        'demography': dem,
        'seasonality': seas,
        'vectors': vec
    }

def _run_ibm(
    X_intrinsic: dict,
    X_site: dict,
    X_eir: float
    ) -> PyTree:
    site = importr('site')
    ms = importr('malariasimulation')
    params = site.site_parameters(
        interventions = site.burnin_interventions(
            _convert_pandas_df(X_site['interventions']),
            BURNIN
        ),
        demography = site.burnin_demography(
            _convert_pandas_df(X_site['demography']),
            BURNIN
        ),
        vectors = _convert_pandas_df(X_site['vectors']),
        seasonality = _convert_pandas_df(X_site['seasonality']),
        eir = float(X_eir),
        overrides = _parse_overrides(X_intrinsic)
    )
    output = ms.run_simulation(
        timesteps = params.rx2['timesteps'],
        parameters = params
    )
    df = _convert_r_df(output)
    return df.values

def _convert_pandas_df(df):
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().py2rpy(df)

def _convert_r_df(df):
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().rpy2py(df)

def _parse_overrides(params):
    params = ro.vectors.ListVector(
        (name, float(value))
        for name, value in params.items()
    )
    params.rx2['human_population'] = 100000
    return params
