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

def mean_field_posterior(n_kernel, n_bias, dtype):
    n = n_kernel + n_bias
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n], scale=tf.math.softplus(t[..., n:])),
            reinterpreted_batch_ndims=1
        ))
    ])

def normal_prior(n_kernel, n_bias, dtype):
    n = n_kernel + n_bias
    return tfp.layers.DistributionLambda(lambda t: tfd.Independent(
        tfd.Normal(loc=tf.zeros(n), scale=1),
        reinterpreted_batch_ndims=1
    ))

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
    output_activation,
    output_initialiser,
    n_dense_prob_layer,
    prob,
    regulariser,
    variational,
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
                recurrent_regularizer=regulariser,
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

        # apply prob layers
        param_1 = param_2 = model_output
        for i, n in enumerate(n_dense_prob_layer):
            if i == len(n_dense_prob_layer) - 1:
                activation = output_activation
                initialiser = output_initialiser
                layer_regulariser = None
            else:
                activation = dense_activation
                initialiser = dense_initialiser
                layer_regulariser = regulariser

            if variational:
                param_1 = tfp.layers.DenseVariational(
                    n,
                    mean_field_posterior,
                    normal_prior
                )(param_1)

                param_2 = tfp.layers.DenseVariational(
                    n,
                    mean_field_posterior,
                    normal_prior
                )(param_2)
            else:
                param_1 = layers.Dense(
                    n,
                    activation=activation,
                    kernel_initializer=initialiser,
                    kernel_regularizer=layer_regulariser
                )(param_1)

                param_2 = layers.Dense(
                    n,
                    activation=activation,
                    kernel_initializer=initialiser,
                    kernel_regularizer=layer_regulariser
                )(param_2)

        prob_params = layers.Concatenate()([param_1, param_2])

        if prob == 'beta':
            prob = tfp.layers.DistributionLambda(
                lambda t: beta_distribution_from_tensor(t, n_outputs),
                convert_to_tensor_fn = lambda d: d.mean()
            )(prob_params)

            loss = ClippedNegativeLogLikelihood()
        elif prob == 'normal':
            prob = tfp.layers.DistributionLambda(
                lambda t: normal_distribution_from_tensor(t, n_outputs),
                convert_to_tensor_fn = lambda d: d.mean()
            )(prob_params)

            loss = negative_log_likelihood
        elif prob == 'logit_normal':
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
