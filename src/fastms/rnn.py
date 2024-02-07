import dataclasses
import orbax.checkpoint
from typing import Tuple
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from flax import linen as nn
from jaxtyping import PyTree, Array
from jax import numpy as jnp, random
from jax.tree_util import tree_map
from mox.loss import mse
from mox.seq2seq.rnn import DecoderLSTMCell, RNNSurrogate, make_rnn_surrogate, init_surrogate
from mox.seq2seq.training import train_rnn_surrogate

def make_rnn(model, samples, units=255):
    feature_size = model.vectorise_output(samples[1]).shape[-1]
    return nn.RNN(
        DecoderLSTMCell(units, feature_size)
    )

def build(samples: PyTree, dtype=jnp.float32):
    (x, x_seq, x_t), y = samples
    n_steps = y['immunity'].shape[1]
    x_std = None
    x_seq_std = {
        'demography': None,
        'interventions': tree_map(lambda _: (0, 1), x_seq['interventions'])
    }
    y_std = None
    y_min = tree_map(lambda _: 0, y)
    max_n = jnp.finfo(dtype).max
    y_max = tree_map(lambda _: max_n, y)
    return make_rnn_surrogate(
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

def init(model: RNNSurrogate, net: nn.Module, samples, key):
    (x, x_seq, _), _ = samples
    return init_surrogate(key, model, net, (x, x_seq))

def train(
    model: RNNSurrogate,
    net: nn.RNN,
    params: PyTree,
    samples: PyTree,
    key: Array,
    epochs: int,
    batch_size: int,
    vectorising_device = None
    ) -> TrainState:
    (x, x_seq, _), y = samples
    n_batches = y['immunity'].shape[0] // batch_size
    state = train_rnn_surrogate(
        (x, x_seq),
        y,
        model,
        net,
        params,
        mse,
        key,
        epochs = epochs,
        batch_size = n_batches,
        vectorising_device=vectorising_device
    )
    return state

def save(path: str, model: RNNSurrogate, net: nn.RNN, params: PyTree):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = {
        'surrogate': dataclasses.asdict(model),
        'cell': dataclasses.asdict(net.cell),
        'params': params
    }
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(
        path,
        ckpt,
        save_args=save_args
    )

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
    net = nn.RNN(DecoderLSTMCell(**ckpt['cell']))
    params = ckpt['params']
    return model, net, params
