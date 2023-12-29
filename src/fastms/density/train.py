import dataclasses
import orbax.checkpoint
from typing import Tuple, Any
from flax.training.train_state import TrainState
from flax import linen as nn
from jaxtyping import PyTree, Array
from jax import random, numpy as jnp, vmap
from jax.tree_util import tree_map, tree_structure
from .rnn import DensityDecoderLSTMCell
from ..rnn import build, init
from mox.seq2seq.rnn import RNNSurrogate, _recover_sequence
from mox.seq2seq.training import train_rnn_surrogate
from mox.surrogates import (
    minrelu,
    maxrelu,
    _inverse_standardise
)
from mox.utils import unbatch_tree
from jax.scipy.stats import norm

def make_rnn(model, samples, units=255):
    y = tree_map(lambda x: x[0], samples[1])
    y_zero = tree_map(lambda x: jnp.zeros_like(x), y)
    y_vec = model.vectorise_output(y)
    y_min = model.vectorise_output(y_zero)[0:1,:]
    feature_size = y_vec.shape[-1]
    return nn.RNN(
        DensityDecoderLSTMCell(units, feature_size, y_min)
    )

class RNNDensitySurrogate(RNNSurrogate):

    def recover(self, y: Any) -> PyTree:
        mu, log_sigma = y
        mu = _recover_sequence(
            mu,
            self.y_shapes,
            tree_structure(self.y_mean),
            self.y_boundaries,
            self.n_steps
        )

        mu = unbatch_tree(
            tree_map(_inverse_standardise, mu, self.y_mean, self.y_var)
        )

        if self.y_min is not None:
            mu = tree_map(
                lambda y, y_min: minrelu(y, y_min),
                mu,
                self.y_min
            )

        if self.y_max is not None:
            mu = tree_map(
                lambda y, y_max: maxrelu(y, y_max),
                mu,
                self.y_max
            )

        log_sigma = _recover_sequence(
            log_sigma,
            self.y_shapes,
            tree_structure(self.y_mean),
            self.y_boundaries,
            self.n_steps
        )

        log_sigma = unbatch_tree(
            tree_map(
                lambda leaf, y_var: leaf + .5 * jnp.log(y_var),
                log_sigma,
                self.y_var
            )
        )

        return mu, log_sigma

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
    state = train_rnn_surrogate(
        (x, x_seq),
        y,
        model,
        net,
        params,
        log_prob,
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
    model = RNNDensitySurrogate(**ckpt['surrogate'])
    net = nn.RNN(DensityDecoderLSTMCell(**ckpt['cell']))
    params = ckpt['params']
    return model, net, params

def log_prob(y_hat: Tuple[Array, Array], y: Array) -> Array:
    mu, log_sigma = y_hat
    sigma = jnp.exp(log_sigma)
    return jnp.sum(norm.logpdf(y, mu, sigma))
