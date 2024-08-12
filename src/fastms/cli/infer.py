import pickle
from functools import partial
from jax import numpy as jnp, random
from jax.lax import dynamic_slice
from jax.tree_util import tree_map
from ..ibm_model import (
    surrogate_posterior,
    surrogate_posterior_svi,
    surrogate_posterior_neutra,
    sample_fake_data
)
from ..sample.save import load_samples
from ..density.rnn import load
from ..density.transformer import load as load_transformer
from ..sites import make_site_inference_data
from ..aggregate import aggregate_prev, aggregate_inc
from mox.seq2seq.rnn import apply_surrogate
import numpyro
import numpyro.distributions as dist
from numpyro.infer.autoguide import (
    AutoNormal,
    AutoBNAFNormal,
    AutoLaplaceApproximation
)
import numpy as np

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
        '--n_chains',
        type=int,
        help='Number of chains for MCMC inference',
        default=4
    )
    sample_parser.add_argument(
        '--cores',
        type=int,
        default=1,
        help='Number of cores to use for sample injestion'
    )
    sample_parser.add_argument(
        '--inf_model',
        choices=['la', 'normal', 'bnaf', 'nuts', 'pt', 'neutra'],
        default='la',
        help='Which inference model to use'
    )
    sample_parser.add_argument(
        '--fake',
        type=bool,
        default=False,
        help='Whether to use fake data for validation'
    )
    sample_parser.add_argument(
        '--stoch',
        choices=['mean', 'mask', 'slice'],
        default='mean',
        help='How to handle stochasticity in the surrogate'
    )
    sample_parser.add_argument(
        '--n_train_svi',
        type=int,
        default=10000,
        help='Number of training samples for SVI'
    )
    sample_parser.add_argument(
        '--alpha_rate',
        type=float,
        default=0,
        help='Rate for annealing'
    )
    sample_parser.add_argument(
        '--alpha_round',
        type=int,
        default=0,
        help='Round for annealing calculations'
    )
    sample_parser.add_argument(
        '--alpha_max_round',
        type=int,
        default=10,
        help='Max round for annealing calculations'
    )

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

        def impl_output(x_intrinsic, x_eir, site_ind):
            n_runs = x_eir.shape[0]
            x = {
                'intrinsic': tree_map(
                    lambda leaf: jnp.full((n_runs,), leaf),
                    x_intrinsic
                ),
                'init_EIR': x_eir,
                'seasonality': sites.x_sites['seasonality'][site_ind],
                'vector_composition': sites.x_sites['vectors'][site_ind]
            }
            x_seq = {
                'interventions': tree_map(
                    lambda leaf: leaf[site_ind],
                    sites.x_sites['interventions']
                ),
                'demography': sites.x_sites['demography'][site_ind]
            }
            x_in = (x, x_seq)

            return apply_surrogate(
                surrogate,
                net,
                params,
                x_in
            )

        def mean_prev_impl(x_intrinsic, x_eir, site_ind, stat_ind):
            mu, _ = impl_output(x_intrinsic, x_eir, site_ind)

            n_detect = mu['n_detect']
            n_detect_n = mu['n']

            return aggregate_prev(
                n_detect,
                n_detect_n,
                sites.prev_lar[stat_ind],
                sites.prev_uar[stat_ind],
                sites.prev_start_time[stat_ind],
                sites.prev_end_time[stat_ind]
            )

        def mean_inc_impl(x_intrinsic, x_eir, site_ind, stat_ind):
            mu, _ = impl_output(x_intrinsic, x_eir, site_ind)

            n_inc_clinical = mu['n_inc_clinical']
            inc_n = mu['n']

            return aggregate_inc(
                n_inc_clinical,
                inc_n,
                sites.inc_lar[stat_ind],
                sites.inc_uar[stat_ind],
                sites.inc_start_time[stat_ind],
                sites.inc_end_time[stat_ind]
            )

        def stoch_prev_impl(x_intrinsic, x_eir, site_ind, stat_ind):
            mu, log_sigma = impl_output(x_intrinsic, x_eir, site_ind)
            sigma = tree_map(jnp.exp, log_sigma)
            n_detect = numpyro.sample(
                'n_detect',
                dist.LeftTruncatedDistribution(
                    dist.Normal(
                        mu['n_detect'],
                        sigma['n_detect']
                    ),
                    0
                )
            )
            n_detect_n = numpyro.sample(
                'n_detect_n',
                dist.LeftTruncatedDistribution(
                    dist.Normal(
                        mu['n'],
                        sigma['n']
                    ),
                    0
                )
            )

            return aggregate_prev(
                n_detect,
                n_detect_n,
                sites.prev_lar[stat_ind],
                sites.prev_uar[stat_ind],
                sites.prev_start_time[stat_ind],
                sites.prev_end_time[stat_ind]
            )

        def stoch_inc_impl(x_intrinsic, x_eir, site_ind, stat_ind):
            mu, log_sigma = impl_output(x_intrinsic, x_eir, site_ind)
            sigma = tree_map(jnp.exp, log_sigma)

            n_inc_clinical = numpyro.sample(
                'inc',
                dist.LeftTruncatedDistribution(
                    dist.Normal(
                        mu['n_inc_clinical'],
                        sigma['n_inc_clinical']
                    ),
                    0
                )
            )
            inc_n = numpyro.sample(
                'inc_n',
                dist.LeftTruncatedDistribution(
                    dist.Normal(
                        mu['n'],
                        sigma['n']
                    ),
                    0
                )
            )

            return aggregate_inc(
                n_inc_clinical,
                inc_n,
                sites.inc_lar[stat_ind],
                sites.inc_uar[stat_ind],
                sites.inc_start_time[stat_ind],
                sites.inc_end_time[stat_ind]
            )


        prev_start_time = np.array(sites.prev_start_time, dtype=np.int64)
        prev_n_time = np.array(sites.prev_end_time - sites.prev_start_time + 1, dtype=np.int64)
        prev_lar = np.array(sites.prev_lar, dtype=np.int64)
        prev_n_age = np.array(sites.prev_uar - sites.prev_lar + 1, dtype=np.int64)
        prev_index = np.array(sites.prev_index, dtype=np.int64)
        inc_start_time = np.array(sites.inc_start_time, dtype=np.int64)
        inc_n_time = np.array(sites.inc_end_time - sites.inc_start_time + 1, dtype=np.int64)
        inc_lar = np.array(sites.inc_lar, dtype=np.int64)
        inc_n_age = np.array(sites.inc_uar - sites.inc_lar + 1, dtype=np.int64)
        inc_index = np.array(sites.inc_index, dtype=np.int64)
        def _slice_model_output(
            x,
            i,
            p_i,
            start_time,
            n_time,
            start_age,
            n_age
            ):
            return dynamic_slice(
                x[p_i],
                (start_time[i], start_age[i]),
                (n_time[i], n_age[i])
            )


        def _sample_surrogate_stat(
            sample_name,
            stat,
            p_i,
            i,
            start_time,
            n_time,
            start_age,
            n_age
            ):
            mu_stat = _slice_model_output(
                mu[stat],
                i,
                p_i,
                start_time,
                n_time,
                start_age,
                n_age
            )
            sigma_stat = _slice_model_output(
                sigma[stat],
                i,
                p_i,
                start_time,
                n_time,
                start_age,
                n_age
            )
            return numpyro.sample(
                f'{sample_name}_{i}',
                dist.LeftTruncatedDistribution(
                    dist.Normal(mu_stat, sigma_stat), #type: ignore
                    0
                )
            )
        if args.stoch == 'mask':
            prev_impl = stoch_prev_impl
            inc_impl = stoch_inc_impl
        else:
            assert args.stoch == 'mean'
            prev_impl = mean_prev_impl
            inc_impl = mean_inc_impl

        # Make fake data for validation
        key = random.PRNGKey(args.seed)

        if args.fake:
            truth = sample_fake_data(
                key,
                prev_impl=prev_impl,
                inc_impl=inc_impl,
                n_sites=sites.n_sites,
                n_prev=sites.n_prev,
                prev_index=sites.prev_index,
                inc_risk_time=sites.inc_risk_time,
                inc_index=sites.inc_index
            )
            sites.prev = truth['obs_prev'][0]
            sites.inc = truth['obs_inc'][0]

            logging.info('Saving fake data')
            truth_path = args.output + '_truth.pkl'
            with open(truth_path, 'wb') as f:
                pickle.dump(truth, f)

        key_i, key = random.split(key)
        if args.inf_model in {'la', 'normal', 'bnaf'}:
            if args.inf_model == 'la':
                autoguide = AutoLaplaceApproximation
            elif args.inf_model == 'normal':
                autoguide = AutoNormal
            else:
                autoguide = partial(AutoBNAFNormal, num_flows=2)

            alpha = args.alpha_rate ** (
                args.alpha_max_round - args.alpha_round - 1
            )
            i_data = surrogate_posterior_svi(
                key_i,
                autoguide=autoguide,
                n_train_samples=args.n_train_svi,
                n_samples=args.n_samples,
                block_stochastic=(args.inf_model == 'bnaf'),
                prev_impl=prev_impl,
                inc_impl=inc_impl,
                n_sites=sites.n_sites,
                n_prev=sites.n_prev,
                prev_index=sites.prev_index,
                prev=sites.prev,
                inc_pop=sites.inc_pop,
                inc=sites.inc,
                inc_index=sites.inc_index,
                prev_subsample=10,
                inc_subsample=10,
                alpha=alpha
            )
        elif args.inf_model == 'neutra':
            i_data = surrogate_posterior_neutra(
                key_i,
                impl=impl,
                n_train_samples=args.n_train_svi,
                n_warmup=args.warmup,
                n_samples=args.n_samples,
                n_chains=args.n_chains,
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
                n_chains=args.n_chains,
                kernel_type=args.inf_model,
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
