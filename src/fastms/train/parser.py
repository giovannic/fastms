import pickle
from jax import random
from jax import numpy as jnp
from jax.tree_util import tree_map
from mox.loss import mse
from mox.seq2seq.rnn import make_rnn_surrogate
from mox.seq2seq.training import train_seq2seq_surrogate
from flax.linen.module import _freeze_attr
from flax.training import orbax_utils
import orbax.checkpoint #type: ignore

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
        '--seed',
        type=int,
        default=42,
        help='Seed to use for pseudo random number generation'
    )

def run(args):
    with open(args.samples, 'rb') as f:
        samples = pickle.load(f)
    if args.model == 'rnn':
        (x, x_seq), y = _freeze_attr(samples)
        n_steps = y['immunity'].shape[1]
        x_t = jnp.array(range(0, n_steps, 365))
        x_std = None
        x_seq_std = _freeze_attr([
            tree_map(lambda _: (0, 1), x_seq[0]), # interventions
            None # demography
        ])
        y_std = None
        y_min = _freeze_attr(tree_map(lambda _: 0, y))
        max_n = jnp.finfo(jnp.float32).max
        y_max = _freeze_attr(tree_map(lambda _: max_n, y))
        model = make_rnn_surrogate(
            x,
            x_seq,
            x_t,
            n_steps,
            y,
            x_std,
            x_seq_std,
            y_std,
            y_min,
            y_max
        )
        key = random.PRNGKey(args.seed)
        params = model.init(key, (x, x_seq))
        key_i, key = random.split(key)
        params = train_seq2seq_surrogate(
            (x, x_seq),
            y,
            model,
            params,
            mse,
            key,
            epochs = args.epochs,
            n_batches = args.n_batches
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
