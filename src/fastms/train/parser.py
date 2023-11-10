from jax import random
from .rnn import build, init, train, save
from .aggregate import monthly
from ..samples import load_samples

def add_parser(subparsers):
    """add_parser. Adds the training parser to the main ArgumentParser
    :param subparsers: the subparsers to modify
    """
    sample_parser = subparsers.add_parser(
        'train',
        help='Trains a surrogate model'
    )
    sample_parser.add_argument(
        'model',
        choices=['mlp', 'rnn', 'transformer'],
        help='Surrogate model to use'
    )
    sample_parser.add_argument(
        'output',
        type=str,
        help='Path to save the model in'
    )
    sample_parser.add_argument(
        '--samples',
        nargs='*',
        help='Paths for the samples to use for training'
    )
    sample_parser.add_argument(
        '--epochs',
        '-e',
        type=int,
        default=100,
        help='Number of epochs for training'
    )
    sample_parser.add_argument(
        '--n_batches',
        '-b',
        type=int,
        default=100,
        help='Number of minibatches'
    )
    sample_parser.add_argument(
        '--aggregate',
        '-a',
        choices=['monthly'],
        help='Aggregate the samples for quicker/easier training'
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
        help='Number of cores to use for sample injestion'
    )

def run(args):
    samples = load_samples(args.samples, args.cores)

    if args.model == 'rnn':
        if args.aggregate == 'monthly':
            samples = monthly(samples)

        model = build(samples)
        key = random.PRNGKey(args.seed)
        params = init(model, samples, key)
        key_i, key = random.split(key)
        params = train(
            model,
            params,
            samples,
            key_i,
            args.epochs,
            args.n_batches
        )
        save(args.output, model, params)
    else:
        raise NotImplementedError('Model not implemented yet')
