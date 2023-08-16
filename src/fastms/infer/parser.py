import orbax.checkpoint #type: ignore
import pandas as pd
from ..sample.sites import import_sites, sites_to_tree
from jax import numpy as jnp
from jax import vmap
from jax.random import PRNGKey
from .ibm_model import surrogate_posterior
from ..train.rnn import build, init
import pickle

def add_parser(subparsers):
    """add_parser. Adds the inference parser to the main ArgumentParser
    :param subparsers: the subparsers to modify
    """
    sample_parser = subparsers.add_parser(
        'infer',
        help='Infers model parameters'
    )
    sample_parser.add_argument(
        'target',
        choices=['EIR', 'intrinsic'],
        help='Parameters to infer'
    )
    sample_parser.add_argument(
        'model',
        choices=['eq', 'det', 'ibm'],
        help='Model to use for inference'
    )
    sample_parser.add_argument(
        'output',
        type=str,
        help='Path to save the posterior estimates in'
    )
    sample_parser.add_argument(
        '--prevalence',
        '-p',
        type=str,
        help='Path to observational prevalence'
    )
    sample_parser.add_argument(
        '--incidence',
        '-i',
        type=str,
        help='Path to observational incidence'
    )
    sample_parser.add_argument(
        '--sites',
        type=str,
        help='Path to site data'
    )
    sample_parser.add_argument(
        '--samples',
        type=str,
        help='Samples used for training the surrogate'
    )
    sample_parser.add_argument(
        '--surrogate',
        '-s',
        type=str,
        help='Path to a surrogate model to use'
    )
    sample_parser.add_argument(
        '--seed',
        type=int,
        help='Random number generation seed',
        default=42
    )

def _aggregate(xs, ns, age_lower, age_upper, time_lower, time_upper):
    return vmap(
        lambda x, n, a_l, a_u, t_l, t_u: jnp.mean(
            jnp.sum(x[t_l:t_u, a_l:a_u], axis=1)/
            jnp.sum(n[t_l:t_u, a_l:a_u], axis=1)
        ),
        in_axes=[0, 0, 0, 0, 0, 0]
    )(xs, ns, age_lower, age_upper, time_lower, time_upper)

def run(args):
    if args.model == 'ibm':
        if args.surrogate is None:
            raise NotImplementedError(
                'Only surrogate based inference is implemented'
            )
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        if args.samples is None:
            raise ValueError('Samples required')
        with open(args.samples, 'rb') as f:
            samples = pickle.load(f)
        model = build(samples)
        params = init(model)
        ckpt = { 'surrogate': model, 'params': params }
        ckpt = orbax_checkpointer.restore(args.surrogate, item=ckpt)
        model, params = ckpt['surrogate'], ckpt['params']
        if args.prevalence is None or args.incidence is None:
            raise ValueError('Both prevalence and incidence required')
        prev = pd.read_csv(args.prevalence)
        inc = pd.read_csv(args.incidence)
        if args.sites is None:
            raise ValueError('Site files are required for IBM')
        sites = import_sites(args.sites)
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
        site_samples = pd.concat(
            [prev[site_description], inc[site_description]]
        ).drop_duplicates()
        start_year = 1985
        x_sites = sites_to_tree(site_samples, sites, start_year, 2018)
        site_index = site_samples.reset_index().set_index(
            site_description
        )
        prev_index = site_index.loc[
            list(prev[site_description].itertuples(index=False))
        ].index
        inc_index = site_index.loc[
            list(inc[site_description].itertuples(index=False))
        ].index
        prev_lar, prev_uar = jnp.array(prev.PR_LAR), jnp.array(prev.PR_UAR)
        inc_lar, inc_uar = jnp.array(inc.INC_LAR), jnp.array(inc.INC_UAR)
        prev_start_time = jnp.array(
            prev.START_YEAR - start_year * 365,
            dtype=jnp.int32
        )
        prev_end_time = jnp.array(
            prev.END_YEAR - start_year * 365,
            dtype=jnp.int32
        )
        inc_start_time = jnp.floor(
            inc.START_YEAR.values - start_year * 365 +
            inc.START_MONTH.values * (365/12)
        )
        inc_end_time = jnp.floor(
            inc.END_YEAR.values - start_year * 365 +
            inc.END_MONTH.values * (365/12)
        )
        def impl(x_intrinsic, x_eir):
            x = [
                x_intrinsic,
                x_eir,
                x_sites['seasonality'],
                x_sites['vectors']
            ]
            x_seq = [x_sites['interventions'], x_sites['demography']]
            model_outputs = model.apply(params, x, x_seq)
            site_prev = _aggregate(
                model_outputs['p_detect'][prev_index],
                model_outputs['n'][prev_index],
                prev_lar,
                prev_uar,
                prev_start_time,
                prev_end_time
            )
            site_inc = _aggregate(
                model_outputs['p_detect'][inc_index],
                model_outputs['n'][inc_index],
                inc_lar,
                inc_uar,
                inc_start_time,
                inc_end_time
            ) * inc.POP.values
            return site_prev, site_inc
        key = PRNGKey(args.seed)
        posterior_samples = surrogate_posterior(
            key,
            impl,
            prev.N.values,
            prev.N_POS.values,
            inc.INC.values / 1000 * inc.POP.values
        )
        with open(args.output, 'wb') as f:
            pickle.dump(posterior_samples, f)
    else:
        raise NotImplementedError('Model not implemented yet')
