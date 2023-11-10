import dataclasses
import orbax.checkpoint
from flax.training import orbax_utils
from flax import linen as nn
from jaxtyping import PyTree
from jax import numpy as jnp, random
from jax.tree_util import tree_map
from mox.loss import mse
from mox.seq2seq.rnn import DecoderLSTMCell, RNNSurrogate, make_rnn_surrogate, init_surrogate
from mox.seq2seq.training import train_rnn_surrogate

def build(samples: PyTree):
    (x, x_seq, x_t), y = samples
    n_steps = y['immunity'].shape[1]
    x_std = None
    x_seq_std = [
        tree_map(lambda _: (0, 1), x_seq[0]), # interventions
        None # demography
    ]
    y_std = None
    y_min = tree_map(lambda _: 0, y)
    max_n = jnp.finfo(jnp.float64).max
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

def init(model: RNNSurrogate, samples, key):
    (x, x_seq, _), _ = samples
    return init_surrogate(key, model, (x, x_seq))

def train(model, params, samples, key, epochs, n_batches):
    (x, x_seq, _), y = samples
    params = train_rnn_surrogate(
        (x, x_seq),
        y,
        model,
        params,
        mse,
        key,
        epochs = epochs,
        batch_size = x[1].shape[0] // n_batches
    )
    return params

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

def load(path: str, dummy_samples: PyTree):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    empty_model = build(dummy_samples)
    empty = {
        'surrogate': dataclasses.asdict(empty_model),
        'cell': dataclasses.asdict(DecoderLSTMCell(1, 1)),
        'params': init(empty_model, dummy_samples, random.PRNGKey(0))
    }
    ckpt = orbax_checkpointer.restore(path, item=empty)
    model = RNNSurrogate(**ckpt['surrogate'])
    net = nn.RNN(**ckpt['cell'])
    params = ckpt['params']
    return model, net, params
