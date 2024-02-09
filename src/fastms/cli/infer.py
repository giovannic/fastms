import pandas as pd
from ..sample.sites import import_sites, pad_sites, sites_to_tree
from jax import numpy as jnp, random
from jax.tree_util import tree_map
from ..ibm_model import surrogate_posterior, surrogate_posterior_svi
from ..sample.save import load_samples
from ..density.rnn import load
from ..density.transformer import load as load_transformer
from mox.seq2seq.rnn import apply_surrogate
import numpyro
import numpyro.distributions as dist
import logging

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
        '--samples_def',
        type=str,
        help='PyTree definition for the samples'
    )
    sample_parser.add_argument(
        '--surrogate',
        '-s',
        choices=['rnn', 'transformer'],
        help='Surrogate Model to use for inference'
    )
    sample_parser.add_argument(
        '--surrogate_path',
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
    sample_parser.add_argument(
        '--mcmc',
        type=bool,
        default=False,
        help='Whether to use MCMC'
    )
    sample_parser.add_argument(
        '--n_train_svi',
        type=int,
        default=10000,
        help='Number of training samples for SVI'
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
    xs_over_age = jnp.sum(xs, axis=2) #type: ignore
    ns_over_age = jnp.sum(ns, axis=2) #type: ignore
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
        if args.samples is None:
            raise ValueError('Samples required')
        samples = load_samples(
            args.samples,
            args.samples_def,
            args.cores
        )
        if args.surrogate == 'rnn':
            surrogate, net, params = load(args.surrogate_path, samples)
        elif args.surrogate == 'transformer':
            surrogate, net, params = load_transformer(args.surrogate_path, samples)
        else:
            raise NotImplementedError(
                'Only RNN and Transformer surrogates are implemented'
            )
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
            (inc.START_YEAR.values - start_year), #type: ignore
            dtype=jnp.int32
        ) * 12 + inc.START_MONTH.values
        inc_end_time = jnp.array(
            (inc.END_YEAR.values - start_year), #type: ignore
            dtype=jnp.int32
        ) * 12 + inc.END_MONTH.values

        def impl(x_intrinsic, x_eir):
            x = {
                'intrinsic': tree_map(lambda leaf: jnp.full((n_sites,), leaf), x_intrinsic),
                'init_EIR': x_eir,
                'seasonality': x_sites['seasonality'],
                'vector_composition': x_sites['vectors']
            }
            x_seq = {
                'interventions': x_sites['interventions'],
                'demography': x_sites['demography']
            }
            x_in = (x, x_seq)

            mu, log_sigma = apply_surrogate(
                surrogate,
                net,
                params,
                x_in
            )
            sigma = tree_map(jnp.exp, log_sigma)

            #n_detect = mu['n_detect'][prev_index]
            #n_detect_n = mu['n'][prev_index]
            #n_inc_clinical = mu['n_inc_clinical'][inc_index]
            #inc_n = mu['n'][inc_index]
            n_detect = numpyro.sample(
                'n_detect',
                dist.LeftTruncatedDistribution(
                    dist.Normal(
                        mu['n_detect'][prev_index],
                        sigma['n_detect'][prev_index],
                    ),
                    0
                )
            )
            n_detect_n = numpyro.sample(
                'n_detect_n',
                dist.LeftTruncatedDistribution(
                    dist.Normal(
                    mu['n'][prev_index],
                        sigma['n'][prev_index],
                    ),
                0
                )
            )
            n_inc_clinical = numpyro.sample(
                'inc',
                dist.LeftTruncatedDistribution(
                    dist.Normal(
                        mu['n_inc_clinical'][inc_index],
                        sigma['n_inc_clinical'][inc_index]
                    ),
                    0
                )
            )
            inc_n = numpyro.sample(
                'inc_n',
                dist.LeftTruncatedDistribution(
                    dist.Normal(
                        mu['n'][inc_index],
                        sigma['n'][inc_index],
                    ),
                    0
                )
            )

            site_prev = _aggregate(
                n_detect,
                n_detect_n,
                prev_lar,
                prev_uar,
                prev_start_time,
                prev_end_time
            )
            site_inc = _aggregate(
                n_inc_clinical,
                inc_n,
                inc_lar,
                inc_uar,
                inc_start_time,
                inc_end_time
            )
            return site_prev, site_inc
        key = random.PRNGKey(args.seed)
        key_i, key = random.split(key)
        if not args.mcmc:
            i_data = surrogate_posterior_svi(
                key_i,
                n_train_samples=args.n_train_svi,
                n_samples=args.n_samples,
                impl=impl,
                n_sites=n_sites,
                n_prev=prev.N.values,
                prev=prev.N_POS.values,
                prev_index=prev_index,
                inc_risk_time=inc.PYO.values * 365., #type: ignore
                inc=jnp.array(jnp.round(inc.INC.values), dtype=jnp.int64), #type: ignore
                inc_index=inc_index
            )
        else:
            i_data = surrogate_posterior(
                key_i,
                impl=impl,
                n_warmup=args.warmup,
                n_samples=args.n_samples,
                n_sites=n_sites,
                n_prev=prev.N.values,
                prev=prev.N_POS.values,
                prev_index=prev_index,
                inc_risk_time=inc.PYO.values * 365., #type: ignore
                inc=jnp.array(jnp.round(inc.INC.values), dtype=jnp.int64), #type: ignore
                inc_index=inc_index
            )

        logging.info('Saving results')
        i_data.to_netcdf(args.output)
    else:
        raise NotImplementedError('Model not implemented yet')
