from jaxtyping import PyTree
from jax import numpy as jnp
from jax.tree_util import tree_map
from mox.loss import mse
from mox.seq2seq.rnn import RNNSurrogate, make_rnn_surrogate, init_surrogate
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
