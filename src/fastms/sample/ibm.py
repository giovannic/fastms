from typing import List, Optional
from jaxtyping import PyTree, Array
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import rpy2.rinterface as ri
import multiprocessing as mp
import pandas as pd
from jax.tree_util import tree_map
from jax import numpy as jnp
import numpy as np

mp.set_start_method('spawn')

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
    calibrate_to_EIR: bool = False
    ) -> PyTree:
    n = init_EIR.shape[0]
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
        outputs = _apply(_run_ibm_until_stable, args, cores)
    if calibrate_to_EIR:
        outputs = _apply(_run_ibm_calibrated_to_EIR, args, cores)
    else:
        outputs = _apply(_run_ibm_fixed_burnin, args, cores)
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
    params.rx2['render_grid'] = ro.vectors.StrVector(['p_detect_', 'p_inc_clinical_'])

    output = ms.run_simulation(
        timesteps = params.rx2['timesteps'],
        parameters = params
    )
    df = _convert_r_df(output)
    n = remove_burnin(jnp.array(df[[f'grid_n_{i}' for i in range(100)]].to_numpy()), burnin)
    n_detect = remove_burnin(
        jnp.array(df[[f'grid_p_detect_{i}' for i in range(100)]].to_numpy()),
        burnin
    )
    n_inc_clinical = remove_burnin(
        jnp.array(df[[f'grid_p_inc_clinical_{i}' for i in range(100)]].to_numpy()),
        burnin
    )

    # fill in missing EIRs
    for column in _EIRs + _vector_counts:
        if column not in df.columns:
            df[column] = 0

    # calculate baseline
    baseline_eir = jnp.array(_baseline_eir(df, burnin))

    # remove burnin
    df = df.iloc[burnin * 365:]

    # convert outputs to jax
    model_outputs = format_outputs(df, n, n_detect, n_inc_clinical)
    return model_outputs, baseline_eir

def _run_ibm_calibrated_to_EIR(
    X_intrinsic: dict,
    X_site: dict,
    X_eir: float,
    population: int = 100000,
    burnin: int = 50
    ) -> PyTree:
    site = importr('site')
    cali = importr('cali')

    cali_params = site.site_parameters(
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
    cali_params.rx2['timesteps'] = burnin * 365

    rcolumns = ','.join([f'"{s}"' for s in _EIRs])
    last_year_EIR = ro.r(
        f'function(x) mean(rowSums(tail(x[c({rcolumns})], 365)))'
    )

    init_EIR = cali.calibrate(
        parameters = cali_params,
        target = float(X_eir),
        summary_function = last_year_EIR,
        tolerance = 1,
        low = 1,
        high = 2000
    )

    return _run_ibm_fixed_burnin(
        X_intrinsic,
        X_site,
        float(init_EIR[0]),
        population,
        burnin
    )

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
    warmup_params.rx2['render_grid'] = True

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
        post_params.rx2['render_grid'] = True
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
    if not 'grid' in output:
        raise NotImplementedError('TODO: format outputs')

    model_outputs = format_outputs(df, None, None, None)

    return model_outputs, baseline_eir

def remove_burnin(x: Array, burnin: int) -> Array:
    return x[burnin*365:]

def format_outputs(df: pd.DataFrame, n, n_detect, n_inc_clinical):
    outputs = {
        'n': n,
        'n_detect': n_detect,
        'n_inc_clinical': n_inc_clinical,
        'EIR': jnp.array(df[_EIRs].values),
        'human_states': jnp.array(
            df[[f'{s}_count' for s in _states]].values
        ),
        'vector_states': jnp.array(df[_vector_counts].values),
        'immunity': jnp.array(df[_immunity].values)
    }
    return outputs

def _convert_pandas_df(df):
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().py2rpy(df)

def _convert_r_df(df):
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().rpy2py(df)

def _parse_overrides(params, population=100000):
    params = ro.vectors.ListVector(
        (name, float(value)) if name != 'ru' else ('du', 1. / float(value))
        for name, value in params.items()
    )
    params.rx2['human_population'] = population
    return params

def _baseline_eir(df: pd.DataFrame, burnin=0):
    final_burnin = df.iloc[burnin-365:]
    return final_burnin[_EIRs].sum(axis=1).mean()

def _apply(f, args, cores):
    if cores == 1:
        return [f(*a) for a in args]
    else:
        with mp.Pool(cores) as pool:
            return pool.starmap(f, args)
