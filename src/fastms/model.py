from tensorflow.config import list_physical_devices
from tensorflow.distribute import MirroredStrategy, get_strategy
from tensorflow.random import set_seed
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
import tensorflow.keras.backend as K
from .attention import BahdanauAttention, LuongAttention, AttentionDecoder
from tensorflow.keras.callbacks import TensorBoard
from .log import ExtendedTensorBoard

class RepeatLayer(layers.Layer):

    def call(self, inputs):
        return layers.RepeatVector(K.shape(inputs[1])[1])(inputs[0])

def create_model(
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
    if list_physical_devices('GPU') and kwargs.get('multigpu', False):
      strategy = MirroredStrategy()
    else:  # Use the Default Strategy
      strategy = get_strategy()
    with strategy.scope():
        static_input = Input(shape=n_static_features, dtype='float32')
        seq_input = Input(shape=(None, n_seq_features), dtype='float32')

        repeated_static_input = RepeatLayer()([static_input, seq_input])

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
        for n in n_dense_layer:
            model_output = layers.TimeDistributed(
                layers.Dense(
                    n,
                    activation=dense_activation,
                    kernel_initializer=dense_initialiser
                )
            )(model_output)

        model = Model(
            inputs = [static_input, seq_input],
            outputs = [model_output]
        )

    model.compile(loss=loss, optimizer=optimiser, metrics=['mean_squared_error'])
    print(model.summary())
    return model

def create_ed_model(
    optimiser,
    rnn_layer,
    n_outputs,
    n_latent,
    dropout,
    loss,
    n_timesteps,
    **kwargs):
    if list_physical_devices('GPU') and kwargs.get('multigpu', False):
      strategy = MirroredStrategy()
    else:  # Use the Default Strategy
      strategy = get_strategy()
    with strategy.scope():
        model = keras.Sequential()
        model.add(rnn_layer(n_latent, dropout=dropout))
        model.add(layers.RepeatVector(n_timesteps))
        model.add(rnn_layer(n_outputs, dropout=dropout, return_sequences=True))
        model.add(layers.TimeDistributed(layers.Dense(n_outputs)))
    model.compile(loss=loss, optimizer=optimiser, metrics=['mean_squared_error'])
    return model

def create_attention_model(
    optimiser,
    rnn_layer,
    n_latent,
    n_outputs,
    n_features,
    dropout,
    loss,
    dense_activation,
    dense_initialiser,
    **kwargs
    ):
    if list_physical_devices('GPU') and kwargs.get('multigpu', False):
      strategy = MirroredStrategy()
    else:  # Use the Default Strategy
      strategy = get_strategy()
    with strategy.scope():
        # encoder
        encoder_input = Input(shape=(None, n_features), dtype='float32')
        encoder = rnn_layer(
            n_latent,
            dropout=dropout,
            return_sequences=True,
            return_state=True
        )
        encoder_output, h, c = encoder(encoder_input)

        # decoder
        decoder = AttentionDecoder(
            n_latent,
            n_outputs,
            rnn_layer,
            LuongAttention,
            dense_activation,
            dense_initialiser
        )
        output, attention, state = decoder(encoder_input, encoder_output, [h, c])
        model = Model(encoder_input, output)

    model.compile(loss=loss, optimizer=optimiser, metrics=['mean_squared_error'])
    return model

def train_model(model, gen, epochs, seed, verbose=True, log=False):
    set_seed(seed)
    if log:
        model.fit(
            gen,
            epochs = epochs,
            verbose = verbose,
            callbacks = [
                TensorBoard(log_dir=log, histogram_freq=1)
                # ExtendedTensorBoard(gen, log_dir=log, histogram_freq=1)
            ]
        )
    else:
        model.fit(gen, epochs=epochs, verbose=verbose)

def model_predict(model, X_test, X_seq_test, scaler):
    predictions = model.predict((X_test, X_seq_test))
    return scaler.inverse_transform(predictions)
