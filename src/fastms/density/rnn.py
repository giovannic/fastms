import dataclasses
import orbax.checkpoint
from typing import Tuple
from jaxtyping import PyTree, Array
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import random, numpy as jnp
from jax.tree_util import tree_map
from mox.seq2seq.rnn import RNNSurrogate, RNNDensitySurrogate
from mox.seq2seq.training import train_rnn_surrogate
from ..rnn import build, init
from .train import trunc_nll

LSTMCarry = Tuple[Array, Array]

def make_rnn(model, samples, units=255, dtype=jnp.float32):
    y = tree_map(lambda x: x[0], samples[1])
    y_zero = tree_map(lambda x: jnp.zeros_like(x), y)
    y_vec = model.vectorise_output(y)
    y_min = model.vectorise_output(y_zero)[0:1,:]
    feature_size = y_vec.shape[-1]
    return nn.RNN(
        DensityDecoderLSTMCell(units, feature_size, y_min, dtype=dtype)
    )

class DensityDecoderLSTMCell(nn.RNNCellBase):
    """DecoderLSTM Module wrapped in a lifted scan transform.
    feature_size: Feature size of the output sequence
    """
    units: int
    feature_size: int
    y_min: Array
    dtype: jnp.float32

    def setup(self):
        self.lstm = nn.LSTMCell(self.units, param_dtype=self.dtype)
        self.dense_mu = nn.Dense(features=self.feature_size)
        self.dense_log_std = nn.Dense(features=self.feature_size)

    def __call__(
          self,
          carry: LSTMCarry,
          x: Array
          ) -> Tuple[LSTMCarry, Tuple[Array, Array]]:
        """Applies the DecoderLSTM model."""
        carry, y = self.lstm(carry, x)
        mu = self.dense_mu(y)
        mu = nn.softplus(mu) + self.y_min
        log_sigma = self.dense_log_std(y)
        return carry, (mu, log_sigma)

    def initialize_carry(self, rng, input_shape) -> LSTMCarry:
        return self.lstm.initialize_carry(rng, input_shape)

    @property
    def num_feature_axes(self) -> int:
        return 1

def load(path: str, dummy_samples: PyTree) -> Tuple[RNNSurrogate, nn.Module, PyTree]:
    dtype = dummy_samples[1]['immunity'].dtype
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    empty_model = build(dummy_samples)
    empty_net = make_rnn(empty_model, dummy_samples, dtype=dtype)
    empty = {
        'surrogate': dataclasses.asdict(empty_model),
        'cell': dataclasses.asdict(empty_net.cell),
        'params': init(
            empty_model,
            empty_net,
            dummy_samples,
            random.PRNGKey(0)
        )
    }
    ckpt = orbax_checkpointer.restore(path, item=empty)
    model = RNNDensitySurrogate(**ckpt['surrogate'])
    net = nn.RNN(DensityDecoderLSTMCell(**ckpt['cell']))
    params = ckpt['params']
    return model, net, params

def train(
    model: RNNSurrogate,
    net: nn.RNN,
    params: PyTree,
    samples: PyTree,
    key: Array,
    epochs: int,
    batch_size: int
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
        batch_size = n_batches
    )
