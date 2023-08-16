from typing import Dict
from jaxtyping import PyTree, Array
from os.path import join
from jax import numpy as jnp
from jax import random
import pandas as pd

def sample_sites(
    sites: dict,
    n: int,
    key: random.PRNGKeyArray
    ) -> pd.DataFrame:
    return _sample_site_dfs(key, n, sites)

_intervention_columns = {
    'country': str,
    'iso3c': str,
    'urban_rural': str,
    'name_1': str,
    'year': int,
    'itn_input_dist': float,
    'tx_cov': float,
    'prop_act': float,
    'irs_cov': float,
    'smc_cov': float,
    'smc_min_age': float,
    'smc_max_age': float,
    'smc_n_rounds': int,
    'smc_drug': str,
    'rtss_cov': float,
    'dn0': float,
    'rn0': float,
    'gamman': float,
    'rnm': float,
    'ls_theta': float,
    'ls_gamma': float,
    'ks_theta': float,
    'ks_gamma': float,
    'ms_theta': float,
    'ms_gamma': float
}

_demography_columns = {
    'iso3c': str,
    'country': str,
    'age_upper': float,
    'year': int,
    'mortality_rate': float
}

_seasonality_columns = {
    'country': str,
    'iso3c': str,
    'name_1': str,
    'g0': float,
    'g1': float,
    'g2': float,
    'g3': float,
    'h1': float,
    'h2': float,
    'h3': float
}

_vector_columns = {
    'country': str,
    'iso3c': str,
    'name_1': str,
    'species': str,
    'prop': float,
    'blood_meal_rates': float,
    'foraging_time': float,
    'Q0': float,
    'phi_bednets': float,
    'phi_indoors': float,
    'mum': float
}

def import_sites(path: str) -> Dict:
    return {
        'interventions': pd.read_csv(
            join(path, 'interventions.csv'),
            usecols = list(_intervention_columns.keys()),
            dtype = _intervention_columns
        ),
        'demography': pd.read_csv(
            join(path, 'demography.csv'),
            usecols = list(_demography_columns.keys()),
            dtype = _demography_columns
        ),
        'seasonality': pd.read_csv(
            join(path, 'seasonality.csv'),
            usecols = list(_seasonality_columns.keys()),
            dtype = _seasonality_columns
        ),
        'vectors': pd.read_csv(
            join(path, 'vectors.csv'),
            usecols = list(_vector_columns.keys()),
            dtype = _vector_columns
        )
    }

def sites_to_tree(
    site_samples: pd.DataFrame,
    sites: Dict,
    start_year: int,
    end_year: int
    ) -> PyTree:
    return {
        'interventions': _parse_interventions(
            site_samples,
            sites['interventions'],
            start_year,
            end_year
        ),
        'demography': _parse_demography(
            site_samples,
            sites['demography'],
            start_year,
            end_year
        ),
        'seasonality': _parse_seasonality(site_samples, sites['seasonality']),
        'vectors': _parse_vectors(site_samples, sites['vectors'])
    }

def _sample_site_dfs(
    key: random.PRNGKeyArray,
    n: int,
    sites: Dict
    ) -> pd.DataFrame:
    possible_sites = sites['interventions'][['iso3c', 'name_1',
                                             'urban_rural']].drop_duplicates()
    index = random.choice(key, len(possible_sites), (n,), replace=True)
    return possible_sites.iloc[index]

def _parse_interventions(
    samples: pd.DataFrame,
    df: pd.DataFrame,
    start_year: int,
    end_year: int
    ) -> Dict:
    df = df[df.year.between(start_year, end_year)]
    min_year = df.year.min()
    if start_year < min_year:
        df_min = df[df.year == min_year]
        padding = pd.concat([
            df_min.assign(year=y)
            for y in range(start_year, min_year)
        ])
        df = pd.concat([padding, df]).sort_values('year')
    int_values = df.pivot(
        index=['iso3c', 'name_1', 'urban_rural'],
        columns='year',
        values=[
            'tx_cov',
            'prop_act',
            'itn_input_dist',
            'dn0',
            'rn0',
            'rnm',
            'irs_cov',
            'ls_theta',
            'ls_gamma',
            'ks_theta',
            'ks_gamma',
            'ms_theta',
            'ms_gamma',
            'smc_cov',
            'rtss_cov'
        ]
    )
    sampled_values = int_values.loc[list(samples.itertuples(index=False))]
    # PyTree with (batch, time, features)
    return {
        'itn': {
            'coverage': jnp.array(
                sampled_values.itn_input_dist.values
            )[:,:,jnp.newaxis],
            'net_params': jnp.stack([
                jnp.array(sampled_values.dn0.values),
                jnp.array(sampled_values.rn0.values),
                jnp.array(sampled_values.rnm.values)
            ], axis = -1)
        },
        'irs': {
            'coverage': jnp.array(sampled_values.irs_cov.values)[:,:,jnp.newaxis],
            'spray_params': jnp.stack([
                jnp.array(sampled_values.ls_theta.values),
                jnp.array(sampled_values.ls_gamma.values),
                jnp.array(sampled_values.ks_theta.values),
                jnp.array(sampled_values.ks_gamma.values),
                jnp.array(sampled_values.ms_theta.values),
                jnp.array(sampled_values.ms_gamma.values)
            ], axis = -1)
        },
        'smc': jnp.array(sampled_values.smc_cov.values)[:,:,jnp.newaxis],
        'rtss': jnp.array(sampled_values.rtss_cov.values)[:,:,jnp.newaxis]
    }

def _parse_demography(
    samples: pd.DataFrame,
    df: pd.DataFrame,
    start_year: int,
    end_year: int
    ) -> Array:
    df = df[df.year.between(start_year, end_year)]
    min_year = df.year.min()
    if start_year < min_year:
        df_min = df[df.year == min_year]
        padding = pd.concat([
            df_min.assign(year=y)
            for y in range(start_year, min_year)
        ])
        df = pd.concat([padding, df]).sort_values('year')
    dem_values = df.pivot(
        index=['iso3c', 'year'],
        columns='age_upper',
        values=[
            'mortality_rate'
        ]
    )
    # (batch, time, mortality)
    country_groups = dem_values.groupby('iso3c')
    dem_3d_array = jnp.stack(
        [
            mortality.to_numpy()
            for _, mortality in country_groups
        ]
    )
    country_index = pd.Series(
        range(dem_3d_array.shape[0]),
        index=[i for i, _ in country_groups]
    )
    return dem_3d_array[country_index.loc[samples.iso3c].values]

def _parse_seasonality(samples: pd.DataFrame, df: pd.DataFrame) -> Array:
    df = df.set_axis(pd.MultiIndex.from_frame(df[['iso3c', 'name_1']]), axis=0)
    return jnp.array(
        df.loc[list(samples[['iso3c', 'name_1']].itertuples(index=False))][
            ['g0', 'g1', 'g2', 'g3', 'h1', 'h2', 'h3']
        ].to_numpy()
    )

def _parse_vectors(samples: pd.DataFrame, df: pd.DataFrame) -> Array:
    vector_values = df.pivot(
        index=['iso3c', 'name_1'],
        columns='species',
        values=[
            'prop'
        ]
    )
    sampled_values = vector_values.loc[
        list(samples[['iso3c', 'name_1']].itertuples(index=False))
    ]
    return jnp.array(sampled_values.fillna(0).to_numpy())
