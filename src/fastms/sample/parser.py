from .prior import sample_prior
from jax.random import PRNGKey
import pickle

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
        choices=['prior', 'lhs'],
        help='Strategy for modelling intrinsic parameters'
    )
    sample_parser.add_argument(
        'output',
        type=str,
        help='Path for the samples to be saved in'
    )
    sample_parser.add_argument(
        '--sites',
        type=str,
        help='Path to site parameters. If not set, sites are sampled using the ' + 
            'LHS strategy'
    )
    sample_parser.add_argument(
        '--number',
        '-n',
        type=int,
        default=100,
        help='Number of samples to make'
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

def run(args):
    if args.model == 'ibm':
        if args.intrinsic_strategy == 'prior':
            if args.sites is None:
                raise ValueError('--sites must be set')
            samples = sample_prior(
                args.sites,
                args.number,
                PRNGKey(args.seed),
                cores=args.cores
            )
            with open(args.output, 'wb') as f:
                pickle.dump(samples, f)
        else:
            raise NotImplementedError('Sampling strategy not implemented yet')
    else:
        raise NotImplementedError('Model not implemented yet')