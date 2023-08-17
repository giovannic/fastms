from jaxtyping import PyTree
from flax.linen.module import _freeze_attr
from jax import numpy as jnp
from jax.tree_util import tree_map
from mox.loss import mse
from mox.seq2seq.rnn import make_rnn_surrogate
from mox.seq2seq.training import train_seq2seq_surrogate

def build(samples: PyTree):
    (x, x_seq, x_t), y = _freeze_attr(samples)
    n_steps = int(y['immunity'].shape[1])
    x_std = None
    x_seq_std = _freeze_attr([
        tree_map(lambda _: (0, 1), x_seq[0]), # interventions
        None # demography
    ])
    y_std = None
    y_min = _freeze_attr(tree_map(lambda _: 0, y))
    max_n = jnp.finfo(jnp.float32).max
    y_max = _freeze_attr(tree_map(lambda _: max_n, y))
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

def init(model, samples, key):
    (x, x_seq, _), _ = _freeze_attr(samples)
    return model.init(key, (x, x_seq))

def train(model, params, samples, key, epochs, n_batches):
    (x, x_seq, _), y = _freeze_attr(samples)
    params = train_seq2seq_surrogate(
        (x, x_seq),
        y,
        model,
        params,
        mse,
        key,
        epochs = epochs,
        n_batches = n_batches
    )
    return params
