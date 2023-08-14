from typing import List
from jaxtyping import PyTree, Array
from rpy2.robjects.packages import importr #type: ignore
import rpy2.robjects as ro #type: ignore
from rpy2.robjects import pandas2ri #type: ignore
from multiprocessing import Pool
import pandas as pd
from jax.tree_util import tree_map
from jax import numpy as jnp

BURNIN = 50

_min_ages = list(range(0, 100 * 365, 365))
_max_ages = [a + 365 for a in _min_ages]
_species = ['arabiensis', 'funestus', 'gambiae']
_immunity = ['ica_mean', 'icm_mean', 'ib_mean', 'id_mean']
_states = ['S', 'A', 'D', 'U', 'Tr']
_vector_states = ['E', 'L', 'P', 'Sm', 'Pm', 'Im']
_EIRs = [f'EIR_{s}' for s in _species]

def run_ibm(
    X_intrinsic: PyTree,
    sites: dict,
    site_samples: pd.DataFrame,
    init_EIR: Array,
    cores: int
    ) -> PyTree:
    n = init_EIR.shape[0]
    with Pool(cores) as pool:
        args = (
            (
                _extract_from_tree(X_intrinsic, i),
                _extract_site(sites, site_samples, i),
                init_EIR[i]
            )
            for i in range(n)
        )
        outputs = pool.starmap(_run_ibm, args)
    model_outputs, eirs = zip(*outputs)
    return _stack_trees(model_outputs), jnp.array(eirs)

def _extract_from_tree(tree: PyTree, i: int) -> PyTree:
    return tree_map(lambda leaf: leaf[i], tree)

def _stack_trees(trees: List[PyTree]) -> PyTree:
    return tree_map(lambda *leaves: jnp.stack(leaves), *trees)

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

    # parameterise site with burnin
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

    # set prev/inc age ranges
    min_ages = ro.vectors.FloatVector(_min_ages)
    max_ages = ro.vectors.FloatVector(_max_ages)
    params.rx2['prevalence_rendering_min_ages'] = min_ages
    params.rx2['prevalence_rendering_max_ages'] = max_ages
    params.rx2['clinical_incidence_rendering_min_ages'] = min_ages
    params.rx2['clinical_incidence_rendering_max_ages'] = max_ages
    output = ms.run_simulation(
        timesteps = params.rx2['timesteps'],
        parameters = params
    )
    df = _convert_r_df(output)

    # calculate baseline
    baseline_eir = _baseline_eir(df)

    # remove burnin
    df = df.iloc[BURNIN * 365:]

    # format the outputs
    model_outputs = {
        'n': df[[f'n_{a}_{b}' for a, b in zip(_min_ages, _max_ages)]].values,
        'p_detect': df[
            [f'p_detect_{a}_{b}' for a, b in zip(_min_ages, _max_ages)]
        ].values,
        'p_inc_clinical': df[
            [f'p_inc_clinical_{a}_{b}' for a, b in zip(_min_ages, _max_ages)]
        ].values,
        'total_M': df[[f'total_M_{s}' for s in _species]].values,
        'EIR': df[_EIRs].values,
        'human_states': df[[f'{s}_count' for s in _states]].values,
        'vector_states': df[
            [f'{s}_{v}_count' for v in _species for s in _vector_states]
        ].values,
        'immunity': df[_immunity].values
    }
    return model_outputs, baseline_eir

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

def _baseline_eir(df: pd.DataFrame):
    final_burnin = df.iloc[(BURNIN - 1)*365:BURNIN*365]
    return final_burnin[_EIRs].sum(axis=1).mean()
