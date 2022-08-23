from tensorflow.config import list_physical_devices
from tensorflow.distribute import MirroredStrategy, get_strategy
from tensorflow.keras import layers, Model, Input, losses
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from .model import RepeatLayer

@tf.function
def negative_log_likelihood(y_true, y_pred):
    return -y_pred.log_prob(y_true)

class ClippedNegativeLogLikelihood(losses.Loss):
    def __init__(self, **kwargs):
        self.softclip = tfp.bijectors.SoftClip(0, 1, hinge_softness=.01)
        super(ClippedNegativeLogLikelihood, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        return -y_pred.log_prob(self.softclip(y_true))

def normal_distribution_from_tensor(t, boundary):
    '''
    splits the tensor in half at `boundary` for the 0 and 1 concentration
    parameters
    '''
    return tfd.Normal(
        t[..., :boundary],
        tf.math.softplus(t[..., boundary:]),
        validate_args=True,
        allow_nan_stats=False
    )

def logit_normal_distribution_from_tensor(t, boundary):
    '''
    splits the tensor in half at `boundary` for the 0 and 1 concentration
    parameters
    '''
    return tfd.LogitNormal(
        t[..., :boundary],
        tf.math.softplus(t[..., boundary:]),
        validate_args=True,
        allow_nan_stats=False
    )

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
    n_dense_prob_layer,
    prob,
    regulariser,
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
                kernel_regularizer=regulariser,
                return_sequences=True
            )(recurrent_model)

        model_output = recurrent_model

        # apply dense layers
        for n in n_dense_layer:
            model_output = layers.TimeDistributed(
                layers.Dense(
                    n,
                    activation=dense_activation,
                    kernel_initializer=dense_initialiser,
                    kernel_regularizer=regulariser
                )
            )(model_output)


        if prob == 'beta':
            alpha = beta = model_output

            for n in n_dense_prob_layer:
                alpha = layers.Dense(
                    n,
                    activation=dense_activation,
                    kernel_initializer=dense_initialiser,
                    kernel_regularizer=regulariser
                )(alpha)

                beta = layers.Dense(
                    n,
                    activation=dense_activation,
                    kernel_initializer=dense_initialiser,
                    kernel_regularizer=regulariser
                )(beta)

            prob_params = layers.Concatenate()([alpha, beta])

            # apply a beta distribution
            prob = tfp.layers.DistributionLambda(
                lambda t: beta_distribution_from_tensor(t, n_outputs),
                convert_to_tensor_fn = lambda d: d.mean()
            )(prob_params)

            loss = ClippedNegativeLogLikelihood()
        elif prob == 'normal':
            mu = sigma = model_output
            for n in n_dense_prob_layer:
                mu = layers.Dense(
                    n,
                    activation=dense_activation,
                    kernel_initializer=dense_initialiser
                )(mu)

                sigma = layers.Dense(
                    n,
                    activation=dense_activation,
                    kernel_initializer=dense_initialiser
                )(sigma)

            prob_params = layers.Concatenate()([mu, sigma])

            # apply a beta distribution
            prob = tfp.layers.DistributionLambda(
                lambda t: normal_distribution_from_tensor(t, n_outputs),
                convert_to_tensor_fn = lambda d: d.mean()
            )(prob_params)

            loss = negative_log_likelihood
        elif prob == 'logit_normal':
            mu = sigma = model_output
            for n in n_dense_prob_layer:
                mu = layers.Dense(
                    n,
                    activation=dense_activation,
                    kernel_initializer=dense_initialiser
                )(mu)

                sigma = layers.Dense(
                    n,
                    activation=dense_activation,
                    kernel_initializer=dense_initialiser
                )(sigma)

            prob_params = layers.Concatenate()([mu, sigma])

            # apply a beta distribution
            prob = tfp.layers.DistributionLambda(
                lambda t: logit_normal_distribution_from_tensor(t, n_outputs),
                convert_to_tensor_fn = lambda d: d.mean_approx()
            )(prob_params)

            loss = ClippedNegativeLogLikelihood()
        else:
            raise ValueError(f'Unknown value of prob {prob}')


        model = Model(
            inputs = [static_input, seq_input],
            outputs = [prob]
        )

        # compile a negative log likelihood, but output mse too
        model.compile(
            loss=loss,
            optimizer=optimiser,
            metrics=['mean_squared_error']
        )

    print(model.summary())
    return model

def prob_model_predict(model, X_test, X_seq_test, scaler, n):
    predictions = model.predict((X_test, X_seq_test) * n)
    return scaler.inverse_transform(predictions)
