from typing import Tuple, Callable, Union
from jaxtyping import Array
from flax import linen as nn
from jax import numpy as jnp
from jax.scipy.stats import truncnorm

LSTMCarry = Tuple[Array, Array]

class DensityDecoderLSTMCell(nn.RNNCellBase):
    """DecoderLSTM Module wrapped in a lifted scan transform.
    feature_size: Feature size of the output sequence
    """
    units: int
    feature_size: int

    def setup(self):
        self.lstm = nn.LSTMCell(self.units)
        self.dense_mu = nn.Dense(features=self.feature_size)
        self.dense_log_std = nn.Dense(features=self.feature_size)

    def __call__(
          self,
          carry: LSTMCarry,
          x: Array
          ) -> Tuple[LSTMCarry, Tuple[Array, Array]]:
        """Applies the DecoderLSTM model."""
        carry, y = self.lstm(carry, x)
        mu = self.dense_mu(y)
        log_sigma = self.dense_log_std(y)
        return carry, (mu, log_sigma)

    def initialize_carry(self, rng, input_shape) -> LSTMCarry:
        return self.lstm.initialize_carry(rng, input_shape)

    @property
    def num_feature_axes(self) -> int:
        return 1

def log_prob(
    y_min: Array,
    y_max: Array,
    ) -> Callable[[Tuple[Array, Array], Array], Array]:
    def f(y_hat: Tuple[Array, Array], y: Array):
        mu, logsigma = y_hat
        sigma = jnp.exp(logsigma)
        return jnp.sum(truncnorm.logpdf(
            _standardise(y, mu, sigma),
            _standardise(y_min, mu, sigma),
            _standardise(y_max, mu, sigma),
        ))
    return f

def _standardise(x: Array, mu: Array, sigma: Array) -> Array:
    return (x - mu) / sigma
