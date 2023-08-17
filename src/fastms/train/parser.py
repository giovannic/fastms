import pickle
from jax import random
from flax.training import orbax_utils
import orbax.checkpoint #type: ignore
from .rnn import build, init, train
from .aggregate import monthly

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
        'samples',
        type=str,
        help='Path for the samples to use for training'
    )
    sample_parser.add_argument(
        'output',
        type=str,
        help='Path to save the model in'
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

def run(args):
    with open(args.samples, 'rb') as f:
        samples = pickle.load(f)
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
        ckpt = { 'surrogate': model, 'params': params }
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(
            args.output,
            ckpt,
            force=True,
            save_args=save_args
        )
    else:
        raise NotImplementedError('Model not implemented yet')
