from abc import ABC, abstractmethod
from jaxtyping import PyTree
import jax
from jax import random, numpy as jnp
import flax.linen as nn

from ..rnn import build, init, train, save, make_rnn
from ..sample.save import load_samples
from ..density.rnn import make_rnn as make_density_rnn
from ..density.transformer import(
    init as init_transformer,
    save as save_transformer,
    make_transformer as make_density_transformer
)
from ..density.rnn import train as train_density
from ..density.transformer import train as train_transformer

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
        interface = RNNTrainingInterface(args.density, dtype)
    elif args.model == 'transformer':
        interface = TransformerTrainingInterface()
    else:
        raise NotImplementedError('Model not implemented yet')

    model = interface.make_model(samples)
    key = random.PRNGKey(args.seed)
    net = interface.make_net(model, samples)
    with jax.default_device(cpu_device):
        params = interface.init_net(model, net, samples, key)
    state = interface.train(
        model,
        net,
        params,
        samples,
        key,
        args.epochs,
        args.batch_size
    )
    interface.save(args.output, model, net, state.params)

class TrainingInterface(ABC):

    @abstractmethod
    def make_model(self, samples):
        pass

    @abstractmethod
    def make_net(self, model, samples) -> nn.Module:
        pass

    @abstractmethod
    def init_net(self, model, net, samples, key) -> PyTree:
        pass

    @abstractmethod
    def train(self, model, net, params, samples, key, epochs, batch_size) -> PyTree:
        pass

    @abstractmethod
    def save(self, output, model, net, params):
        pass


class RNNTrainingInterface(TrainingInterface):

    def __init__(self, density=False, dtype=jnp.float32):
        self.density = density
        self.dtype = dtype

    def make_model(self, samples):
        return build(samples, dtype=self.dtype)

    def make_net(self, model, samples) -> nn.Module:
        if self.density:
            return make_density_rnn(model, samples)
        return make_rnn(model, samples)

    def init_net(self, model, net, samples, key) -> PyTree:
        return init(model, net, samples, key)

    def train(self, model, net, params, samples, key, epochs, batch_size) -> PyTree:
        if self.density:
            return train_density(model, net, params, samples, key, epochs, batch_size)
        return train(model, net, params, samples, key, epochs, batch_size)

    def save(self, output, model, net, params):
        save(output, model, net, params)

class TransformerTrainingInterface(RNNTrainingInterface):

    def __init__(self):
        super().__init__(density=True)

    def make_net(self, model, samples) -> nn.Module:
        return make_density_transformer(model, samples)

    def init_net(self, model, net, samples, key) -> PyTree:
        return init_transformer(model, net, samples, key)

    def train(
        self,
        model,
        net,
        params,
        samples,
        key,
        epochs,
        batch_size
    ) -> PyTree:
        return train_transformer(
            model,
            net,
            params,
            samples,
            key,
            epochs,
            batch_size
        )

    def save(self, output, model, net, params):
        save_transformer(output, model, net, params)
