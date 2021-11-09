import pytest
import numpy as np
from fastms.interface import vectorise

def test_vectorise_produces_expected_vectors():
    spec = {
        "parameters": ["alpha", "beta"],
        "timed_parameters": ["delta", "gamma"]
    }
    parameters = {
        "alpha": 1.,
        "beta": 2.,
        "delta": [3., 4.],
        "gamma": [5., 6.]
    }
    expected = np.array([
        [
            [1., 2., 3., 5.],
            [1., 2., 4., 6.]
        ]
    ])
    np.testing.assert_array_equal(vectorise(parameters, spec), expected)
