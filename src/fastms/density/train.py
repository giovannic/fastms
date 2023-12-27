import dataclasses
import orbax.checkpoint
from typing import Tuple
from flax.training.train_state import TrainState
from flax import linen as nn
from jaxtyping import PyTree, Array
from jax import random, numpy as jnp
from jax.tree_util import tree_map
from .rnn import DensityDecoderLSTMCell, log_prob
from ..rnn import build, init
from mox.seq2seq.rnn import RNNSurrogate
from mox.seq2seq.training import train_rnn_surrogate

def make_rnn(model, samples, units=255):
    feature_size = model.vectorise_output(samples[1]).shape[-1]
    return nn.RNN(
        DensityDecoderLSTMCell(units, feature_size)
    )

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
    state = train_rnn_surrogate(
        (x, x_seq),
        y,
        model,
        net,
        params,
        log_prob(y_min, jnp.inf),
        key,
        epochs = epochs,
        batch_size = n_batches
    )
    return state

def load(path: str, dummy_samples: PyTree) -> Tuple[RNNSurrogate, nn.Module, PyTree]:
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    empty_model = build(dummy_samples)
    empty_net = make_rnn(empty_model, dummy_samples)
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
    model = RNNSurrogate(**ckpt['surrogate'])
    net = nn.RNN(DensityDecoderLSTMCell(**ckpt['cell']))
    params = ckpt['params']
    return model, net, params
