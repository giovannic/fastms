from jax import random
from ..rnn import build, init, train, save, make_rnn
from ..sample.save import load_samples
from ..density.train import (
    make_rnn as make_density_rnn,
    train as train_density
)
import jax
from jax import numpy as jnp

cpu_device = jax.devices('cpu')[0]

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
        '--samples_def',
        type=str,
        help='PyTree definition for the samples'
    )
    sample_parser.add_argument(
        '--epochs',
        '-e',
        type=int,
        default=100,
        help='Number of epochs for training'
    )
    sample_parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        default=100,
        help='Size of batches'
    )
    sample_parser.add_argument(
        '--n',
        '-n',
        type=int,
        default=-1,
        help='Number of samples to use'
    )
    sample_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed to use for pseudo random number generation'
    )
    sample_parser.add_argument(
        '--density',
        type=bool,
        default=True,
        help='Whether to model probabilistic model outputs'
    )
    sample_parser.add_argument(
        '--cores',
        type=int,
        default=1,
        help='Number of cores to use for sample injestion'
    )
    sample_parser.add_argument(
        '--f64',
        type=bool,
        default=True,
        help='Execute in 64 bit precision'
    )

def run(args):
    if args.f64:
        dtype = jnp.float64
    else:
        dtype = jnp.float32
    with jax.default_device(cpu_device):
        samples = load_samples(
            args.samples,
            args.samples_def,
            cores=args.cores,
            n=args.n,
            dtype=dtype
        )

    if args.model == 'rnn':
        model = build(samples, dtype=dtype)
        key = random.PRNGKey(args.seed)
        if args.density:
            net = make_density_rnn(model, samples, dtype=dtype)
        else:
            net = make_rnn(model, samples)
        with jax.default_device(cpu_device):
            params = init(model, net, samples, key)
        key_i, key = random.split(key)
        if args.density:
            state = train_density(
                model,
                net,
                params,
                samples,
                key_i,
                args.epochs,
                args.batch_size
            )
        else:
            state = train(
                model,
                net,
                params,
                samples,
                key_i,
                args.epochs,
                args.batch_size
            )
        save(args.output, model, net, state.params)
    else:
        raise NotImplementedError('Model not implemented yet')