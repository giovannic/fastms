import pandas as pd
from .sample.sites import import_sites, pad_sites, sites_to_tree
from jax import numpy as jnp
import dataclasses
from jaxtyping import Array

@dataclasses.dataclass
class SiteData:
    prev_lar: Array
    prev_uar: Array
    inc_lar: Array
    inc_uar: Array
    prev_start_time: Array
    prev_end_time: Array
    inc_start_time: Array
    inc_end_time: Array
    prev_index: Array
    n_prev: Array
    prev: Array
    inc_index: Array
    inc_risk_time: Array
    inc: Array
    x_sites: Array
    site_df_dict: dict
    site_index: pd.DataFrame
    n_sites: int


def make_site_inference_data(sites_path, start_year, end_year) -> SiteData:
    """Make inference data from site data.

    Args:
        sites_path: Path to the sites file.
        start_year: Start year of the data.
        end_year: End year of the data.

    Returns:
        SiteData object.
    """
    # Loaed prevalence and incidence data
    prev_path = sites_path + '/prev.csv'
    inc_path = sites_path + '/inc.csv'
    prev = pd.read_csv(prev_path)
    inc = pd.read_csv(inc_path)

    # Load site data
    sites = import_sites(sites_path)

    # Merge site data with prevalence and incidence data
    site_description = ['iso3c', 'name_1', 'urban_rural']
    urban_sites = sites['interventions'][site_description].sort_values(
        'urban_rural',
        ascending=False # prefer urban
    ).drop_duplicates(['iso3c', 'name_1'])
    prev = pd.merge(
        prev,
        urban_sites,
        how='left'
    )
    inc = pd.merge(
        inc,
        urban_sites,
        how='left'
    )

    # Only keep sites with both prevalence and incidence data
    site_samples = pd.merge(
        prev[site_description],
        inc[site_description]
    ).drop_duplicates()

    prev = pd.merge(prev, site_samples)
    inc = pd.merge(inc, site_samples)

    # Create parameters for surrogate modelling
    start_year, end_year = 1985, 2018
    sites = pad_sites(sites, start_year, end_year)
    x_sites = sites_to_tree(site_samples, sites)

    # Create site, prevalence and incidence indices
    site_index = site_samples.reset_index(drop=True).reset_index().set_index(
        site_description
    )
    prev_index = jnp.array(site_index.loc[
        list(prev[site_description].itertuples(index=False))
    ]['index'].values)
    inc_index = jnp.array(site_index.loc[
        list(inc[site_description].itertuples(index=False))
    ]['index'].values)

    # Create indices for aggregation
    #NOTE: truncating very small ages
    prev_lar = jnp.array(prev.PR_LAR, dtype=jnp.int64)
    prev_uar = jnp.array(prev.PR_UAR, dtype=jnp.int64)
    inc_lar = jnp.array(inc.INC_LAR, dtype=jnp.int64)
    inc_uar = jnp.array(inc.INC_UAR, dtype=jnp.int64)
    prev_start_time = jnp.array(
        (prev.START_YEAR - start_year),
        dtype=jnp.int64
    ) * 12
    prev_end_time = jnp.array(
        (prev.END_YEAR - start_year),
        dtype=jnp.int64
    ) * 12
    inc_start_time = jnp.array(
        (inc.START_YEAR.values - start_year), #type: ignore
        dtype=jnp.int64
    ) * 12 + inc.START_MONTH.values
    inc_end_time = jnp.array(
        (inc.END_YEAR.values - start_year), #type: ignore
        dtype=jnp.int64
    ) * 12 + inc.END_MONTH.values

    n_sites = len(site_samples)

    return SiteData(
        prev_lar=prev_lar,
        prev_uar=prev_uar,
        inc_lar=inc_lar,
        inc_uar=inc_uar,
        prev_start_time=prev_start_time,
        prev_end_time=prev_end_time,
        inc_start_time=inc_start_time,
        inc_end_time=inc_end_time,
        prev_index=prev_index,
        n_prev=jnp.array(prev.N.values),
        prev=jnp.array(prev.N_POS.values),
        inc_index=inc_index,
        inc_risk_time=jnp.array(inc.PYO.values),
        inc=jnp.array(inc.d.values, dtype=jnp.int64),
        x_sites=x_sites,
        site_df_dict=sites,
        site_index=site_samples,
        n_sites=n_sites
    )
