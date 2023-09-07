import orbax.checkpoint #type: ignore
import pandas as pd
from ..sample.sites import import_sites, pad_sites, sites_to_tree
from jax import numpy as jnp, random
from jax.tree_util import tree_map
from .ibm_model import surrogate_posterior
from ..train.rnn import build, init
from ..train.aggregate import monthly
from ..samples import load_samples
import pickle
from flax.linen.module import _freeze_attr

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
        nargs='*',
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
    sample_parser.add_argument(
        '--warmup',
        type=int,
        help='Number of warmup inference samples',
        default=100
    )
    sample_parser.add_argument(
        '--n_samples',
        type=int,
        help='Number of inference samples',
        default=100
    )
    sample_parser.add_argument(
        '--cores',
        type=int,
        default=1,
        help='Number of cores to use for sample injestion'
    )

def _aggregate(xs, ns, age_lower, age_upper, time_lower, time_upper):
    age_lower = age_lower[:, jnp.newaxis, jnp.newaxis]
    age_upper = age_upper[:, jnp.newaxis, jnp.newaxis]
    time_lower = time_lower[:, jnp.newaxis, jnp.newaxis]
    time_upper = time_upper[:, jnp.newaxis, jnp.newaxis]
    age_mask = jnp.arange(xs.shape[2])[jnp.newaxis, jnp.newaxis, :]
    age_mask = (age_mask >= age_lower) & (age_mask <= age_upper)
    time_mask = jnp.arange(xs.shape[1])[jnp.newaxis, :, jnp.newaxis]
    time_mask = (time_mask >= time_lower) & (time_mask <= time_upper)
    mask = age_mask & time_mask
    xs = jnp.where(mask, xs, 0)
    ns = jnp.where(mask, ns, 0)
    xs_over_age = jnp.sum(xs, axis=2)
    ns_over_age = jnp.sum(ns, axis=2)
    prev_over_time = jnp.sum(
        jnp.where(jnp.squeeze(time_mask, 2), xs_over_age / ns_over_age, 0),
        axis=1
    )
    return prev_over_time / jnp.sum(time_mask, axis=(1, 2))

def run(args):
    if args.model == 'ibm':
        if args.surrogate is None:
            raise NotImplementedError(
                'Only surrogate based inference is implemented'
            )
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        if args.samples is None:
            raise ValueError('Samples required')
        samples = load_samples(args.samples, args.cores)
        model = build(monthly(samples))
        key = random.PRNGKey(args.seed)
        params = init(model, samples, key)
        empty = { 'surrogate': model, 'params': params }
        restored = orbax_checkpointer.restore(args.surrogate, item=empty)
        params = restored['params']
        #NOTE: there's a bug in restoring the model
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
        n_sites = len(site_samples)
        start_year, end_year = 1985, 2018
        sites = pad_sites(sites, start_year, end_year)
        x_sites = sites_to_tree(site_samples, sites)
        site_index = site_samples.reset_index(drop=True).reset_index().set_index(
            site_description
        )
        prev_index = site_index.loc[
            list(prev[site_description].itertuples(index=False))
        ]['index'].values
        inc_index = site_index.loc[
            list(inc[site_description].itertuples(index=False))
        ]['index'].values
        #NOTE: truncating very small ages
        prev_lar = jnp.array(prev.PR_LAR, dtype=jnp.int32)
        prev_uar = jnp.array(prev.PR_UAR, dtype=jnp.int32)
        inc_lar = jnp.array(inc.INC_LAR, dtype=jnp.int32)
        inc_uar = jnp.array(inc.INC_UAR, dtype=jnp.int32)
        prev_start_time = jnp.array(
            (prev.START_YEAR - start_year),
            dtype=jnp.int32
        ) * 12
        prev_end_time = jnp.array(
            (prev.END_YEAR - start_year),
            dtype=jnp.int32
        ) * 12
        inc_start_time = jnp.array(
            (inc.START_YEAR.values - start_year),
            dtype=jnp.int32
        ) * 12 + inc.START_MONTH.values
        inc_end_time = jnp.array(
            (inc.END_YEAR.values - start_year),
            dtype=jnp.int32
        ) * 12 + inc.END_MONTH.values
        def impl(x_intrinsic, x_eir):
            x = [
                tree_map(lambda leaf: jnp.full((n_sites,), leaf), x_intrinsic),
                x_eir,
                x_sites['seasonality'],
                x_sites['vectors']
            ]
            x_seq = [x_sites['interventions'], x_sites['demography']]
            model_outputs = model.apply(params, _freeze_attr((x, x_seq)))
            site_prev = _aggregate(
                model_outputs['p_detect'][prev_index],
                model_outputs['n'][prev_index],
                prev_lar,
                prev_uar,
                prev_start_time,
                prev_end_time
            )
            site_inc = _aggregate(
                model_outputs['p_inc_clinical'][inc_index],
                model_outputs['n'][inc_index],
                inc_lar,
                inc_uar,
                inc_start_time,
                inc_end_time
            ) * inc.POP.values
            return site_prev, site_inc
        key_i, key = random.split(key)
        posterior_samples = surrogate_posterior(
            key_i,
            impl,
            prev.N.values,
            prev.N_POS.values,
            jnp.round(inc.INC.values / 1000 * inc.POP.values),
            n_sites,
            n_warmup=args.warmup,
            n_samples=args.n_samples
        )
        with open(args.output, 'wb') as f:
            pickle.dump(posterior_samples, f)
    else:
        raise NotImplementedError('Model not implemented yet')
