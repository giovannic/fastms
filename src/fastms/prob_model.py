from tensorflow.config import list_physical_devices
from tensorflow.distribute import MirroredStrategy, get_strategy
from tensorflow.keras import layers, Model, Input, losses
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from .model import RepeatLayer

EPSILON = 1e-6

def negative_log_likelihood(y_true, y_pred):
    return -y_pred.log_prob(y_true)

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
    disable_eager_execution()
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

        # make the outputs positive for the beta distribution
        model_output = tfp.bijectors.Softplus()(model_output)

        # apply a beta distribution
        prob = tfp.layers.DistributionLambda(
            lambda i: tfd.Beta(i[..., :n_outputs],i[..., n_outputs:]),
            convert_to_tensor_fn = lambda d: d.mean()
        )(model_output)

        model = Model(
            inputs = [static_input, seq_input],
            outputs = [prob]
        )

        # compile a negative log likelihood, but output mse too
        model.compile(
            loss=negative_log_likelihood,
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
    tf.keras.utils.plot_model(model, show_shapes=True)
    return model

def prob_model_predict(model, X_test, X_seq_test, scaler, n):
    mu, sigma = model.predict((X_test, X_seq_test) * n)
    return scaler.inverse_transform(mu)
