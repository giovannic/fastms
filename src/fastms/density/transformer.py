from typing import Tuple
from jaxtyping import Array, PyTree
import dataclasses
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import orbax.checkpoint
from jax import random, numpy as jnp
from jax.tree_util import tree_map
import flax.linen as nn

from mox.seq2seq.transformer.transformer import DensityTransformer
from mox.seq2seq.transformer.surrogate import init_surrogate
from mox.seq2seq.transformer.training import train_transformer
from mox.seq2seq.rnn import RNNSurrogate, RNNDensitySurrogate

from ..rnn import build
from .train import trunc_nll

def make_transformer(
    model,
    samples,
    num_layers=2,
    latent_dim=256,
    num_heads=4,
    dim_feedforward=256,
    dropout_prob=0.1,
    ):
    return DensityTransformer(
        num_layers=num_layers,
        latent_dim=latent_dim,
        output_dim=model.get_output_dim(samples[1]),
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout_prob=dropout_prob
    )

def init(model: RNNSurrogate, net: DensityTransformer, samples, key):
    (x, x_seq, _), _ = samples
    return init_surrogate(key, model, net, (x, x_seq))

def save(
    path: str,
    model: RNNSurrogate,
    net: DensityTransformer,
    params: PyTree
    ):
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

def load(path: str, dummy_samples: PyTree) -> Tuple[RNNSurrogate, nn.Module, PyTree]:
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    empty_model = build(dummy_samples)
    empty_net = make_transformer(empty_model, dummy_samples)
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
    net = DensityTransformer(**ckpt['net'])
    params = ckpt['params']
    return model, net, params

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
    y_min = model.vectorise_output(
        tree_map(lambda leaf: jnp.zeros_like(leaf[0]), y)
    )

    big_n = 1e16
    y_max = model.vectorise_output(
        tree_map(lambda leaf: jnp.full(leaf[0].shape, big_n), y)
    )
    loss = trunc_nll(y_min, y_max)
    return train_transformer(
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
