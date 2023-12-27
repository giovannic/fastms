import numpy as np
import pickle
import os.path
from glob import glob
from typing import List
from jaxtyping import PyTree, PyTreeDef
from multiprocessing.pool import Pool
from jax import numpy as jnp
from jax.tree_util import (
    tree_flatten,
    tree_unflatten,
    tree_structure,
    tree_map
)

def save_compressed_pytree(x: PyTree, path: str) -> None:
    """Saves a PyTree to a compressed file.

    Args:
        x: Pytree to save.
        path: Path to save to.
    """
    if not os.path.isfile(path):
        leaves, _ = tree_flatten(x)
        np.savez_compressed(path, *leaves)

def save_pytree_def(x: PyTree, path: str) -> None:
    """Saves a PyTree definition to a compressed file.

    Args:
        x: Pytree to save.
        path: Path to save to.
    """
    if not os.path.isfile(path):
        with open(path, 'wb') as f:
            pickle.dump(tree_structure(x), f)

def load_pytree(treedef: PyTreeDef, path: str) -> PyTree:
    with np.load(path) as data:
        return tree_unflatten(treedef, data.values())

def load_samples(
    sample_path_expr: List[str],
    def_path: str,
    cores: int = 1,
    n: int = -1
    ) -> PyTree:
    paths = [
        path
        for expr in sample_path_expr
        for path in glob(expr)
    ]

    with open(def_path, 'rb') as f:
        treedef = pickle.load(f)

    pickles: List[PyTree]
    if cores == 1:
        pickles = [load_pytree(treedef, path) for path in paths]
    else:
        def _load_pickle(path: str) -> PyTree:
            return load_pytree(treedef, path)
        with Pool(cores) as p:
            pickles = p.map(_load_pickle, paths)

    x_t = pickles[0][0][2]
    samples = tree_map(lambda *leaves: jnp.concatenate(leaves), *pickles)

    # fix x_t
    (x, x_seq, _), y = samples

    if (n > -1):
        x = tree_map(lambda l: l[:n], x)
        x_seq = tree_map(lambda l: l[:n], x_seq)
        y = tree_map(lambda l: l[:n], y)

    return (x, x_seq, x_t), y
