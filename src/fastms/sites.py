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
    n_prev: Array
    prev: Array
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
    prev = pd.merge(
        prev,
        sites['interventions'][site_description],
        how='left'
    ).sort_values(
        'urban_rural',
        ascending=False # prefer urban
    ).drop_duplicates(site_description)
    inc = pd.merge(
        inc,
        sites['interventions'][site_description],
        how='left'
    ).sort_values(
        'urban_rural',
        ascending=False # prefer urban
    ).drop_duplicates(site_description)

    # Get sites which have both prevalence and incidence data
    site_samples = pd.merge(
        prev,
        inc,
        on=site_description,
        suffixes=('_prev', '_inc')
    ).reset_index(
        drop=True
    ).reset_index().set_index(site_description)

    # Create model parameters for each site
    n_sites = len(site_samples)
    start_year, end_year = 1985, 2018
    sites = pad_sites(sites, start_year, end_year)
    site_index = site_samples.reset_index()[site_description]
    x_sites = sites_to_tree(
        site_index,
        sites
    )
    
    # Calculate indices for age and time ranges for each study
    #NOTE: truncating very small ages
    prev_lar = jnp.array(site_samples.PR_LAR, dtype=jnp.int64)
    prev_uar = jnp.array(site_samples.PR_UAR, dtype=jnp.int64)
    inc_lar = jnp.array(site_samples.INC_LAR, dtype=jnp.int64)
    inc_uar = jnp.array(site_samples.INC_UAR, dtype=jnp.int64)
    prev_start_time = jnp.array(
        (site_samples.START_YEAR_prev - start_year),
        dtype=jnp.int64
    ) * 12
    prev_end_time = jnp.array(
        (site_samples.END_YEAR_prev - start_year),
        dtype=jnp.int64
    ) * 12
    inc_start_time = jnp.array(
        (site_samples.START_YEAR_inc.values - start_year), #type: ignore
        dtype=jnp.int64
    ) * 12 + site_samples.START_MONTH.values
    inc_end_time = jnp.array(
        (site_samples.END_YEAR_inc.values - start_year), #type: ignore
        dtype=jnp.int64
    ) * 12 + site_samples.END_MONTH.values

    # Make this all into a site data object

    return SiteData(
        prev_lar=prev_lar,
        prev_uar=prev_uar,
        inc_lar=inc_lar,
        inc_uar=inc_uar,
        prev_start_time=prev_start_time,
        prev_end_time=prev_end_time,
        inc_start_time=inc_start_time,
        inc_end_time=inc_end_time,
        n_prev=jnp.array(site_samples.N.values),
        prev=jnp.array(site_samples.N_POS.values),
        inc_risk_time=jnp.array(site_samples.PYO.values) * 365.,
        inc=jnp.array(site_samples.INC.values, dtype=jnp.int64),
        x_sites=x_sites,
        site_df_dict=sites,
        site_index=site_index,
        n_sites=n_sites
    )
