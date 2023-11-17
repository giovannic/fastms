from jax import numpy as jnp, random
from jax.tree_util import tree_map, tree_flatten
from fastms.train.rnn import build, init, load, make_rnn, save
from numpy.testing import assert_equal
import dataclasses

def assert_dataclass_equal(a, b):
    assert_equal(dataclasses.asdict(a), dataclasses.asdict(b))

def test_rnn_surrogate_can_be_saved_and_loaded(tmp_path):
    path = tmp_path / 'checkpoint'
    x = {
        'b0': jnp.zeros(5),
        'b1': jnp.zeros(5)
    }
    x_seq = [
        {
            'nets': jnp.zeros((5, 5, 5))
        },
        jnp.zeros((5, 5, 5))
    ]
    x_t = jnp.arange(5)
    y = {
        'prev': jnp.zeros((5, 5, 10)),
        'inc': jnp.zeros((5, 5, 10)),
        'immunity': jnp.zeros((5, 5, 10))
    }
    samples = ((x, x_seq, x_t), y)
    model = build(samples)
    net = make_rnn(model, samples)
    key = random.PRNGKey(0)
    params = init(model, net, samples, key)
    save(path, model, net, params)
    loaded_model, loaded_net, loaded_params = load(path, samples)
    assert_dataclass_equal(loaded_model, model)
    assert_dataclass_equal(loaded_net, net)
    assert_equal(loaded_params, params)
