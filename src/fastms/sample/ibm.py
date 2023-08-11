from rpy2.robjects.packages import importr
from multiprocessing import Pool
from jaxtyping import PyTree, Array
from jax.tree_util import tree_map
from jax import numpy as jnp

def run_ibm(
    X_intrinsic: PyTree,
    X_sites: PyTree,
    X_eir: Array,
    cores: int
    ) -> PyTree:
    with Pool(cores) as pool:
        n = X_eir.shape[0]
        args = (
            (
                _extract_from_tree(X_intrinsic, i),
                _extract_from_tree(X_sites, i),
                X_eir[i]
            )
            for i in range(n)
        )
        outputs = pool.starmap(_run_ibm, args)
        return tree_map(jnp.concatenate, *outputs)

def _extract_from_tree(tree: PyTree, i: int) -> PyTree:
    return tree_map(lambda leaf: leaf[i], tree)

def _run_ibm(
    X_intrinsic,
    X_site,
    X_eir
    ) -> PyTree:
    site = importr('site')
    ms = importr('malariasimulation')
    params = site.site_parameters(
        interventions = _parse_interventions(X_site),
        demorgraphy = _parse_demography(X_site),
        vectors = _parse_vectors(X_site),
        seasonality = _parse_seasonality(X_site),
        eir = X_eir,
        overrides = _parse_overrides(X_intrinsic)
    )
    output = ms.run_simulation(
        timesteps = params.rx2['timesteps'],
        parameters = params
    )
    return _parse_output(output)
