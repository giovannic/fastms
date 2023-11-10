from typing import List, Tuple
from jaxtyping import Array
import pickle
from glob import glob
from multiprocessing.pool import Pool
from jax import numpy as jnp
from jax.tree_util import tree_map

SAMPLE_TYPE = Tuple[Tuple[Array, Array, Array], Array]

def _load_pickle(path: str) -> SAMPLE_TYPE:
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_samples(expressions: List[str], cores: int = 1) -> SAMPLE_TYPE:
    paths = [
        path
        for expression in expressions
        for path in glob(expression)
    ]

    pickles: List[SAMPLE_TYPE]
    if cores == 1:
        pickles = [_load_pickle(path) for path in paths]
    else:
        with Pool(cores) as p:
            pickles = p.map(_load_pickle, paths)

    x_t = pickles[0][0][2]
    samples = tree_map(lambda *leaves: jnp.concatenate(leaves), *pickles)

    # fix x_t
    (x, x_seq, _), y = samples
    return (x, x_seq, x_t), y
