from tensorflow.config import list_physical_devices
from tensorflow.distribute import MirroredStrategy, get_strategy
from tensorflow.keras import layers, Model, Input, losses
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from .model import RepeatLayer

EPSILON = 1e-6

@tf.function
def beta_negative_log_likelihood(y_true, y_pred):
    '''
    beta distribution is only valid between 0 and 1 exclusive
    so we softclip
    '''
    softclip = tfp.bijectors.SoftClip(0, 1)
    return -y_pred.log_prob(softclip(y_true))

def beta_distribution_from_tensor(t, boundary):
    '''
    splits the tensor in half at `boundary` for the 0 and 1 concentration
    parameters
    '''
    return tfd.Beta(
        tf.math.softplus(t[..., :boundary]),
        tf.math.softplus(t[..., boundary:]),
        validate_args=True,
        allow_nan_stats=False
    )

class EnsemblingLayer(layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(EnsemblingLayer, self).__init__(**kwargs)

    def call(self, inputs):
        mus, sigmas = inputs
        mu = tf.math.reduce_mean(mus, axis=0)
        sigma = tf.math.sqrt(
            tf.math.reduce_mean(
                sigmas + tf.math.square(mus), axis=0
            ) - tf.square(mu)
        )
        return [mu, sigma]

    def get_config(self):
        config = super(EnsemblingLayer, self).get_config()
        config.update({ 'output_dim': self.output_dim })
        return config

def create_prob_model(
    optimiser,
    rnn_layer,
    n_layer,
    n_static_features,
    n_seq_features,
    dropout,
    loss,
    n_dense_layer,
    n_outputs,
    dense_activation,
    dense_initialiser,
    **kwargs
    ):
    if list_physical_devices('GPU') and kwargs.get('multigpu', False):
      strategy = MirroredStrategy()
    else:  # Use the Default Strategy
      strategy = get_strategy()
    with strategy.scope():
        static_input = Input(shape=n_static_features, dtype='float32')
        seq_input = Input(shape=(None, n_seq_features), dtype='float32')

        # combine static and sequential outputs
        repeated_static_input = RepeatLayer()([static_input, seq_input])
        combined_inputs = layers.Concatenate()(
            [seq_input, repeated_static_input]
        )

        # apply sequential layers
        recurrent_model = combined_inputs
        for n in n_layer:
            recurrent_model = rnn_layer(
                n,
                dropout=dropout,
                return_sequences=True,
            )(recurrent_model)

        model_output = recurrent_model

        # apply dense layers
        dense_specs = zip(
            n_dense_layer,
            dense_activation,
            dense_initialiser
        )

        for n, activation, initialiser in dense_specs:
            model_output = layers.TimeDistributed(
                layers.Dense(
                    n,
                    activation=activation,
                    kernel_initializer=initialiser
                )
            )(model_output)

        # apply a beta distribution
        prob = tfp.layers.DistributionLambda(
            lambda t: beta_distribution_from_tensor(t, n_outputs),
            convert_to_tensor_fn = lambda d: d.mean()
        )(model_output)

        model = Model(
            inputs = [static_input, seq_input],
            outputs = [prob]
        )

        # compile a negative log likelihood, but output mse too
        model.compile(
            loss=beta_negative_log_likelihood,
            optimizer=optimiser,
            metrics=['mean_squared_error']
        )

    print(model.summary())
    return model

# Not possible!
def create_ensemble(models, n_static_features, n_seq_features):
    static_input = Input(shape=n_static_features, dtype='float32')
    seq_input = Input(shape=(None, n_seq_features), dtype='float32')
    inputs = [model.input for model in models]

    mu = tf.stack([m.output[0] for m in models])
    sigma = tf.stack([m.output[1] for m in models])
    mu, sigma = EnsemblingLayer(mu.shape[-1])([mu, sigma])

    for i, model in enumerate(models):
        for layer in model.layers:
            layer._name = f'ensemble_{i}_{layer.name}'

    model = Model(
        inputs = inputs,
        outputs = [mu, sigma]
    )
    print(model.summary())
    return model

def prob_model_predict(model, X_test, X_seq_test, scaler, n):
    predictions = model.predict((X_test, X_seq_test) * n)
    return scaler.inverse_transform(predictions)
