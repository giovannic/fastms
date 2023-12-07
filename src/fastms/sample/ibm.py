from typing import List, Optional
from jaxtyping import PyTree, Array
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import rpy2.rinterface as ri
from multiprocessing import Pool
import pandas as pd
from jax.tree_util import tree_map
from jax import numpy as jnp
import numpy as np

_min_ages = list(range(0, 100 * 365, 365))
_max_ages = [a + 365 for a in _min_ages]
_species = ['arabiensis', 'funestus', 'gambiae']
_immunity = ['ica_mean', 'icm_mean', 'ib_mean', 'id_mean']
_states = ['S', 'A', 'D', 'U', 'Tr']
_vector_states = ['E', 'L', 'P', 'Sm', 'Pm', 'Im']
_EIRs = [f'EIR_{s}' for s in _species]
_vector_counts = [f'{s}_{v}_count' for v in _species for s in _vector_states]

def run_ibm(
    X_intrinsic: PyTree,
    sites: dict,
    site_samples: pd.DataFrame,
    init_EIR: Array,
    burnin,
    cores: int,
    population: int = 100000,
    dynamic_burnin: bool = False,
    ) -> PyTree:
    n = init_EIR.shape[0]
    with Pool(cores) as pool:
        args = (
            (
                _extract_from_tree(X_intrinsic, i),
                _extract_site(sites, site_samples, i),
                init_EIR[i],
                population,
                burnin
            )
            for i in range(n)
        )
        if dynamic_burnin:
            outputs = pool.starmap(_run_ibm_until_stable, args)
        else:
            outputs = pool.starmap(_run_ibm_fixed_burnin, args)
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

def _run_ibm_fixed_burnin(
    X_intrinsic: Optional[dict],
    X_site: dict,
    X_eir: float,
    population: int = 100000,
    burnin: int = 50
    ) -> PyTree:
    site = importr('site')
    ms = importr('malariasimulation')

    if (X_intrinsic is None):
        X_intrinsic = {}

    params = site.site_parameters(
        interventions = site.burnin_interventions(
            _convert_pandas_df(X_site['interventions']),
            burnin
        ),
        demography = site.burnin_demography(
            _convert_pandas_df(X_site['demography']),
            burnin
        ),
        vectors = _convert_pandas_df(X_site['vectors']),
        seasonality = _convert_pandas_df(X_site['seasonality']),
        eir = float(X_eir),
        overrides = _parse_overrides(X_intrinsic, population)
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

    # fill in missing EIRs
    for column in _EIRs + _vector_counts:
        if column not in df.columns:
            df[column] = 0

    # calculate baseline
    baseline_eir = jnp.array(_baseline_eir(df, burnin))

    # remove burnin
    df = df.iloc[burnin * 365:]

    # convert outputs to jax
    model_outputs = format_outputs(df)
    return model_outputs, baseline_eir

def _run_ibm_until_stable(
    X_intrinsic: Optional[dict],
    X_site: dict,
    X_eir: float,
    population: int = 100000,
    burnin: int = 50
    ) -> PyTree:
    site = importr('site')
    ms = importr('malariasimulation')

    if (X_intrinsic is None):
        X_intrinsic = {}

    warmup_params = site.site_parameters(
        interventions = site.burnin_interventions(
            _convert_pandas_df(X_site['interventions']),
            burnin
        ),
        demography = site.burnin_demography(
            _convert_pandas_df(X_site['demography']),
            burnin
        ),
        vectors = _convert_pandas_df(X_site['vectors']),
        seasonality = _convert_pandas_df(X_site['seasonality']),
        eir = float(X_eir),
        overrides = _parse_overrides(X_intrinsic, population)
    )

    # set prev/inc age ranges
    min_ages = ro.vectors.FloatVector(_min_ages)
    max_ages = ro.vectors.FloatVector(_max_ages)
    warmup_params.rx2['prevalence_rendering_min_ages'] = min_ages
    warmup_params.rx2['prevalence_rendering_max_ages'] = max_ages
    warmup_params.rx2['clinical_incidence_rendering_min_ages'] = min_ages
    warmup_params.rx2['clinical_incidence_rendering_max_ages'] = max_ages

    # function for creating model parameters after burnin
    @ri.rternalize
    def post_warmup_parameters(t_converged):
        post_params = site.site_parameters(
            interventions = site.burnin_interventions(
                _convert_pandas_df(X_site['interventions']),
                t_converged / 365
            ),
            demography = site.burnin_demography(
                _convert_pandas_df(X_site['demography']),
                t_converged / 365
            ),
            vectors = _convert_pandas_df(X_site['vectors']),
            seasonality = _convert_pandas_df(X_site['seasonality']),
            eir = float(X_eir),
            overrides = _parse_overrides(X_intrinsic)
        )
        post_params.rx2['prevalence_rendering_min_ages'] = min_ages
        post_params.rx2['prevalence_rendering_max_ages'] = max_ages
        post_params.rx2['clinical_incidence_rendering_min_ages'] = min_ages
        post_params.rx2['clinical_incidence_rendering_max_ages'] = max_ages
        return post_params

    output = ms.run_simulation_until_stable(
        parameters = warmup_params,
        post_parameters = post_warmup_parameters,
        tolerance = 1e-1,
        max_t = burnin * 365,
        post_t = int((np.ptp(X_site['interventions'].year) + 1) * 365)
    )
    df = _convert_r_df(output.rx2['post'])
    pre_df = _convert_r_df(output.rx2['pre'])

    # fill in missing EIRs
    for column in _EIRs + _vector_counts:
        if column not in df.columns:
            df[column] = 0
            pre_df[column] = 0

    # calculate baseline
    baseline_eir = jnp.array(_baseline_eir(pre_df))

    # format the outputs
    model_outputs = format_outputs(df)

    return model_outputs, baseline_eir

def format_outputs(df: pd.DataFrame):
    outputs = {
        'n': df[[f'n_{a}_{b}' for a, b in zip(_min_ages, _max_ages)]].values,
        'p_detect': df[
            [f'p_detect_{a}_{b}' for a, b in zip(_min_ages, _max_ages)]
        ].values,
        'p_inc_clinical': df[
            [f'p_inc_clinical_{a}_{b}' for a, b in zip(_min_ages, _max_ages)]
        ].values,
        'EIR': df[_EIRs].values,
        'human_states': df[[f'{s}_count' for s in _states]].values,
        'vector_states': df[_vector_counts].values,
        'immunity': df[_immunity].values
    }
    return { k: jnp.array(v) for k, v in outputs.items() }


def _convert_pandas_df(df):
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().py2rpy(df)

def _convert_r_df(df):
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().rpy2py(df)

def _parse_overrides(params, population=100000):
    params = ro.vectors.ListVector(
        (name, float(value))
        for name, value in params.items()
    )
    params.rx2['human_population'] = population
    return params

def _baseline_eir(df: pd.DataFrame, burnin=0):
    final_burnin = df.iloc[burnin-365:]
    return final_burnin[_EIRs].sum(axis=1).mean()
