from tensorflow.config import list_physical_devices
from tensorflow.distribute import MirroredStrategy, get_strategy
from tensorflow.keras import layers, Model, Input, losses
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow.keras.backend as K
import tensorflow as tf

EPSILON = 1e-6

class GaussianLoss(losses.Loss):
    """NOTE: Not serializable"""

    def __init__(self, sigma, **kwargs):
        self.sigma = sigma
        super(GaussianLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        return tf.math.reduce_mean(
            0.5 * tf.math.log(self.sigma) +
            0.5 * tf.math.divide(tf.math.square(y_true - y_pred), self.sigma)
        ) + EPSILON

class GaussianLayer(layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(GaussianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mu_kernel = self.add_weight(
          name='mu_kernel', 
          shape=(input_shape[-1], self.output_dim),
          initializer='glorot_normal',
          trainable=True
        )
        self.sig_kernel = self.add_weight(
            name='sig_kernel', 
            shape=(input_shape[-1], self.output_dim),
            initializer='glorot_normal',
            trainable=True
        )
        self.mu_bias = self.add_weight(
            name='mu_bias',
            shape=(self.output_dim, ),
            initializer='glorot_normal',
            trainable=True
        )
        self.sig_bias = self.add_weight(
            name='sig_bias',
            shape=(self.output_dim, ),
            initializer='glorot_normal',
            trainable=True
        )
        super(GaussianLayer, self).build(input_shape) 

    def call(self, x):
        output_mu  = K.dot(x, self.mu_kernel) + self.mu_bias
        output_sig = K.dot(x, self.sig_kernel) + self.sig_bias
        output_sig_pos = K.log(1 + K.exp(output_sig)) + EPSILON
        return [output_mu, output_sig_pos]

    def get_config(self):
        config = super(GaussianLayer, self).get_config()
        config.update({ 'output_dim': self.output_dim })
        return config

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

        def repeat(args):
            return layers.RepeatVector(K.shape(args[1])[1])(args[0])

        repeated_static_input = layers.Lambda(repeat)([static_input, seq_input])
        combined_inputs = layers.Concatenate()(
            [seq_input, repeated_static_input]
        )
        recurrent_model = combined_inputs
        for n in n_layer:
            recurrent_model = rnn_layer(
                n,
                dropout=dropout,
                return_sequences=True,
            )(recurrent_model)

        model_output = recurrent_model

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

        mu, sigma = GaussianLayer(n_dense_layer[-1], name='probs')(model_output)

        model = Model(
            inputs = [static_input, seq_input],
            outputs = [mu, sigma]
        )

        # have a loss only for the mu output
        model.compile(
            loss={ 'probs': GaussianLoss(sigma) },
            optimizer=optimiser,
            metrics={ 'probs': 'mean_squared_error' }
        )

    print(model.summary())
    return model

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
