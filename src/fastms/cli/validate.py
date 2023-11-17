from ..rnn import load
from ..aggregate import monthly
from ..samples import load_samples
from mox.seq2seq.rnn import apply_surrogate
import pickle
from jax.tree_util import tree_map
from jax import numpy as jnp
import jax

cpu_device = jax.devices('cpu')[0]

def add_parser(subparsers):
    """add_parser. Adds the validation parser to the main ArgumentParser
    :param subparsers: the subparsers to modify
    """
    sample_parser = subparsers.add_parser(
        'validate',
        help='Validates a surrogate model'
    )
    sample_parser.add_argument(
        'model',
        choices=['mlp', 'rnn', 'transformer'],
        help='Surrogate model to use'
    )
    sample_parser.add_argument(
        'model_path',
        type=str,
        help='Path of the saved surrogate model'
    )
    sample_parser.add_argument(
        'output',
        type=str,
        help='Path to save the predictions in'
    )
    sample_parser.add_argument(
        '--samples',
        nargs='*',
        help='Paths for the samples to use for validation'
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
    with jax.default_device(cpu_device):
        samples = load_samples(args.samples, args.cores)

    if args.model == 'rnn':
        if args.aggregate == 'monthly':
            with jax.default_device(cpu_device):
                samples = monthly(samples)

        surrogate, net, params = load(args.model_path, samples)
        (x, x_seq, _), y = samples
        y_pred = apply_surrogate(surrogate, net, params, (x, x_seq))
        error = tree_map(lambda x, y: jnp.mean(jnp.square(x - y)), y_pred, y)
        print(error)
        with open(args.output, 'wb') as f:
            pickle.dump(y_pred, f)
    else:
        raise NotImplementedError('Model not implemented yet')
