from ..sample.prior import sample_prior
from ..sample.calibration import sample_calibration
from ..sample.data import sample_from_data
from ..sample.save import save_compressed_pytree, save_pytree_def
from ..aggregate import monthly
from jax.random import PRNGKey

def add_parser(subparsers):
    """add_parser. Adds the sample parser to the main ArgumentParser
    :param subparsers: the subparsers to modify
    """
    sample_parser = subparsers.add_parser(
        'sample',
        help='Samples input parameters and model outputs for training'
    )
    sample_parser.add_argument(
        'model',
        choices=['eq', 'det', 'ibm'],
        help='Model to sample from'
    )
    sample_parser.add_argument(
        'intrinsic_strategy',
        choices=['prior', 'lhs', 'none', 'data'],
        help='Strategy for modelling intrinsic parameters'
    )
    sample_parser.add_argument(
        'output',
        type=str,
        help='Path for the samples to be saved in'
    )
    sample_parser.add_argument(
        'output_def',
        type=str,
        help='Path for the sample pytree def to be saved in'
    )
    sample_parser.add_argument(
        '--aggregate',
        '-a',
        choices=['monthly', 'none'],
        default='monthly',
        help='Aggregate the samples for quicker/easier training'
    )
    sample_parser.add_argument(
        '--sites',
        type=str,
        help='Path to site parameters'
    )
    sample_parser.add_argument(
        '--number',
        '-n',
        type=int,
        default=100,
        help='Number of samples to make'
    )
    sample_parser.add_argument(
        '--n_sites',
        type=int,
        default=-1,
        help='Number of sites to use for data sampling'
    )
    sample_parser.add_argument(
        '--site_start',
        type=int,
        default=0,
        help='Index for the first site to use for data sampling'
    )
    sample_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed to use for pseudo random number generation'
    )
    sample_parser.add_argument(
        '--cores',
        type=int,
        default=1,
        help='Number of cores to use'
    )
    sample_parser.add_argument(
        '--start',
        type=int,
        default=1985,
        help='Start year for sites'
    )
    sample_parser.add_argument(
        '--end',
        type=int,
        default=2018,
        help='End year for sites'
    )
    sample_parser.add_argument(
        '--population',
        type=int,
        default=100000,
        help='Total population for the model runs'
    )
    sample_parser.add_argument(
        '--burnin',
        type=int,
        default=50,
        help='Number of years to run the burnin for'
    )
    sample_parser.add_argument(
        '--data',
        type=str,
        help='Path to az InferenceData to use for model sampling'
    )
    sample_parser.add_argument(
        '--data_start',
        type=int,
        default=0,
        help='Start index for sampling from the data'
    )

def run(args):
    if args.model == 'ibm':
        if args.intrinsic_strategy == 'prior':
            if args.sites is None:
                raise ValueError('--sites must be set')
            samples = sample_prior(
                args.sites,
                args.number,
                PRNGKey(args.seed),
                args.burnin,
                cores=args.cores,
                start_year=args.start,
                end_year=args.end,
                population=args.population
            )
            if args.aggregate == 'monthly':
                samples = monthly(samples)
            save_compressed_pytree(samples, args.output)
            save_pytree_def(samples, args.output_def)
        elif args.intrinsic_strategy == 'none':
            # EIR sampling strategy
            if args.sites is None:
                raise ValueError('--sites must be set')
            samples = sample_calibration(
                args.sites,
                args.number,
                PRNGKey(args.seed),
                args.burnin,
                cores=args.cores,
                start_year=args.start,
                end_year=args.end,
                population=args.population
            )
            if args.aggregate == 'monthly':
                samples = monthly(samples)
            save_compressed_pytree(samples, args.output)
            save_pytree_def(samples, args.output_def)
        elif args.intrinsic_strategy == 'data':
            if args.data is None:
                raise ValueError('--data must be set')
            samples = sample_from_data(
                args.data,
                args.sites,
                args.burnin,
                n_samples=args.number,
                sample_start=args.data_start,
                site_start=args.site_start,
                n_sites=args.n_sites,
                cores=args.cores,
                start_year=args.start,
                end_year=args.end,
                population=args.population,
            )
            if args.aggregate == 'monthly':
                samples = monthly(samples)
            save_compressed_pytree(samples, args.output)
            save_pytree_def(samples, args.output_def)
        else:
            raise NotImplementedError('Sampling strategy not implemented yet')
    else:
        raise NotImplementedError('Model not implemented yet')
