from typing import Tuple
from jaxtyping import Array
from flax import linen as nn
from jax import numpy as jnp

LSTMCarry = Tuple[Array, Array]

class DensityDecoderLSTMCell(nn.RNNCellBase):
    """DecoderLSTM Module wrapped in a lifted scan transform.
    feature_size: Feature size of the output sequence
    """
    units: int
    feature_size: int
    y_min: Array
    dtype: jnp.float32

    def setup(self):
        self.lstm = nn.LSTMCell(self.units, param_dtype=self.dtype)
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
        mu = nn.softplus(mu) + self.y_min
        log_sigma = self.dense_log_std(y)
        return carry, (mu, log_sigma)

    def initialize_carry(self, rng, input_shape) -> LSTMCarry:
        return self.lstm.initialize_carry(rng, input_shape)

    @property
    def num_feature_axes(self) -> int:
        return 1
