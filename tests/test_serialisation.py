import pytest
import numpy as np
import os
from fastms.prob_model import create_prob_model, create_ensemble
from fastms.hyperparameters import default_params
from fastms.export import save_model
from fastms.interface import load_model

def test_default_prob_model_can_be_saved_and_loaded(tmp_path):
    params = default_params(10, 5, 365)
    model = create_prob_model(**params)
    path = os.path.join(tmp_path, 'model')
    save_model(model, path)
    try:
        load_model(path)
    except Exception as exc:
        assert False, f"'load_model raised an exception {exc}"
