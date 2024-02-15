import pickle
from jax import numpy as jnp, random
from jax.tree_util import tree_map
from ..ibm_model import (
    surrogate_posterior,
    surrogate_posterior_svi,
    sample_fake_data
)
from ..sample.save import load_samples
from ..density.rnn import load
from ..density.transformer import load as load_transformer
from ..sites import make_site_inference_data
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
        '--fake',
        type=bool,
        default=False,
        help='Whether to use fake data for validation'
    )
    sample_parser.add_argument(
        '--stoch',
        type=bool,
        default=False,
        help='Whether to model stochasticity in the surrogate'
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

        # Load site data
        if args.sites is None:
            raise ValueError('Site files are required for IBM')
        start_year, end_year = 1985, 2018
        sites = make_site_inference_data(args.sites, start_year, end_year)

        def mean_impl(x_intrinsic, x_eir):
            x = {
                'intrinsic': tree_map(
                    lambda leaf: jnp.full((sites.n_sites,), leaf),
                    x_intrinsic
                ),
                'init_EIR': x_eir,
                'seasonality': sites.x_sites['seasonality'],
                'vector_composition': sites.x_sites['vectors']
             }
            x_seq = {
                'interventions': sites.x_sites['interventions'],
                'demography': sites.x_sites['demography']
            }
            x_in = (x, x_seq)

            mu, _ = apply_surrogate(
                surrogate,
                net,
                params,
                x_in
            )

            n_detect = mu['n_detect'][sites.prev_index]
            n_detect_n = mu['n'][sites.prev_index]
            n_inc_clinical = mu['n_inc_clinical'][sites.inc_index]
            inc_n = mu['n'][sites.inc_index]

            site_prev = _aggregate(
                n_detect,
                n_detect_n,
                sites.prev_lar,
                sites.prev_uar,
                sites.prev_start_time,
                sites.prev_end_time
            )
            site_inc = _aggregate(
                n_inc_clinical,
                inc_n,
                sites.inc_lar,
                sites.inc_uar,
                sites.inc_start_time,
                sites.inc_end_time
            )
            return site_prev, site_inc


        def stoch_impl(x_intrinsic, x_eir):
            x = {
                'intrinsic': tree_map(
                    lambda leaf: jnp.full((sites.n_sites,), leaf),
                    x_intrinsic
                ),
                'init_EIR': x_eir,
                'seasonality': sites.x_sites['seasonality'],
                'vector_composition': sites.x_sites['vectors']
            }
            x_seq = {
                'interventions': sites.x_sites['interventions'],
                'demography': sites.x_sites['demography']
            }
            x_in = (x, x_seq)

            mu, log_sigma = apply_surrogate(
                surrogate,
                net,
                params,
                x_in
            )
            sigma = tree_map(jnp.exp, log_sigma)

            n_detect = numpyro.sample(
                'n_detect',
                dist.LeftTruncatedDistribution(
                    dist.Normal(
                        mu['n_detect'][sites.prev_index],
                        sigma['n_detect'][sites.prev_index]
                    ),
                    0
                )
            )
            n_detect_n = numpyro.sample(
                'n_detect_n',
                dist.LeftTruncatedDistribution(
                    dist.Normal(
                        mu['n'][sites.prev_index],
                        sigma['n'][sites.prev_index]
                    ),
                    0
                )
            )
            n_inc_clinical = numpyro.sample(
                'inc',
                dist.LeftTruncatedDistribution(
                    dist.Normal(
                        mu['n_inc_clinical'][sites.inc_index],
                        sigma['n_inc_clinical'][sites.inc_index]
                    ),
                    0
                )
            )
            inc_n = numpyro.sample(
                'inc_n',
                dist.LeftTruncatedDistribution(
                    dist.Normal(
                        mu['n'][sites.inc_index],
                        sigma['n'][sites.inc_index]
                    ),
                    0
                )
            )

            site_prev = _aggregate(
                n_detect,
                n_detect_n,
                sites.prev_lar,
                sites.prev_uar,
                sites.prev_start_time,
                sites.prev_end_time
            )
            site_inc = _aggregate(
                n_inc_clinical,
                inc_n,
                sites.inc_lar,
                sites.inc_uar,
                sites.inc_start_time,
                sites.inc_end_time
            )
            return site_prev, site_inc

        if args.stoch:
            impl = stoch_impl
        else:
            impl = mean_impl

        # Make fake data for validation
        key = random.PRNGKey(args.seed)
        if args.fake:
            truth = sample_fake_data(
                key,
                impl=impl,
                n_sites=sites.n_sites,
                n_prev=sites.n_prev,
                prev_index=sites.prev_index,
                inc_risk_time=sites.inc_risk_time,
                inc_index=sites.inc_index
            )
            sites.prev = truth['obs_prev']
            sites.inc = truth['obs_inc']

            logging.info('Saving fake data')
            truth_path = args.output + '_truth.pkl'
            with open(truth_path, 'wb') as f:
                pickle.dump(truth, f)

        key_i, key = random.split(key)
        if not args.mcmc:
            i_data = surrogate_posterior_svi(
                key_i,
                n_train_samples=args.n_train_svi,
                n_samples=args.n_samples,
                impl=impl,
                n_sites=sites.n_sites,
                n_prev=sites.n_prev,
                prev_index=sites.prev_index,
                prev=sites.prev,
                inc_risk_time=sites.inc_risk_time,
                inc=sites.inc,
                inc_index=sites.inc_index
            )
        else:
            i_data = surrogate_posterior(
                key_i,
                impl=impl,
                n_warmup=args.warmup,
                n_samples=args.n_samples,
                n_sites=sites.n_sites,
                n_prev=sites.n_prev,
                prev_index=sites.prev_index,
                prev=sites.prev,
                inc_risk_time=sites.inc_risk_time,
                inc=sites.inc,
                inc_index=sites.inc_index
            )

        logging.info('Saving results')
        i_data.to_netcdf(args.output)
    else:
        raise NotImplementedError('Model not implemented yet')
