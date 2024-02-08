import dataclasses
import orbax.checkpoint
from typing import Tuple
from jaxtyping import PyTree, Array
from flax import linen as nn
from flax.training.train_state import TrainState
from flax.training import orbax_utils
from jax import random, numpy as jnp
from jax.tree_util import tree_map
from mox.seq2seq.rnn import RNNSurrogate, RNNDensitySurrogate
from mox.seq2seq.training import train_rnn_surrogate
from ..rnn import build, init
from .train import trunc_nll

LSTMCarry = Tuple[Array, Array]

def make_rnn(model, samples, n_layers=2, units=255, dtype=jnp.float32):
    y = tree_map(lambda x: x[0], samples[1])
    y_zero = tree_map(lambda x: jnp.zeros_like(x), y)
    y_vec = model.vectorise_output(y)
    y_min = model.vectorise_output(y_zero)[0:1,:]
    feature_size = y_vec.shape[-1]
    return DensityDecoderRNN(
        n_layers,
        units,
        feature_size,
        y_min,
        dtype
    )

class DensityDecoderRNN(nn.Module):
    """A multilayer LSTM model for density estimation."""
    n_layers: int
    units: int
    feature_size: int
    y_min: Array
    dtype: jnp.float32

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers):
            x = nn.RNN(
                nn.LSTMCell(self.units, param_dtype=self.dtype)
            )(x) #type: ignore
        mu = nn.Dense(features=self.feature_size)(x)
        mu = nn.softplus(mu) + self.y_min
        log_sigma = nn.Dense(features=self.feature_size)(x)
        return mu, log_sigma

def load(path: str, dummy_samples: PyTree) -> Tuple[RNNSurrogate, nn.Module, PyTree]:
    dtype = dummy_samples[1]['immunity'].dtype
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    empty_model = build(dummy_samples)
    empty_net = make_rnn(empty_model, dummy_samples, dtype=dtype)
    empty = {
        'surrogate': dataclasses.asdict(empty_model),
        'net': dataclasses.asdict(empty_net),
        'params': init(
            empty_model,
            empty_net,
            dummy_samples,
            random.PRNGKey(0)
        )
    }
    ckpt = orbax_checkpointer.restore(path, item=empty)
    model = RNNDensitySurrogate(**ckpt['surrogate'])
    net = DensityDecoderRNN(**ckpt['net'])
    params = ckpt['params']
    return model, net, params

def save(path: str, model: RNNSurrogate, net: nn.Module, params: PyTree):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = {
        'surrogate': dataclasses.asdict(model),
        'net': dataclasses.asdict(net),
        'params': params
    }
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(
        path,
        ckpt,
        save_args=save_args
    )

def train(
    model: RNNSurrogate,
    net: nn.RNN,
    params: PyTree,
    samples: PyTree,
    key: Array,
    epochs: int,
    batch_size: int,
    vectorising_device=None
    ) -> TrainState:
    (x, x_seq, _), y = samples
    n_batches = y['immunity'].shape[0] // batch_size
    y_min = model.vectorise_output(
        tree_map(lambda leaf: jnp.zeros_like(leaf[0]), y)
    )

    big_n = 1e16
    y_max = model.vectorise_output(
        tree_map(lambda leaf: jnp.full(leaf[0].shape, big_n), y)
    )
    loss = trunc_nll(y_min, y_max)
    return train_rnn_surrogate(
        (x, x_seq),
        y,
        model,
        net,
        params,
        loss,
        key,
        epochs = epochs,
        batch_size = n_batches,
        vectorising_device = vectorising_device
    )
