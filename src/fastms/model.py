from tensorflow.config import list_physical_devices
from tensorflow.distribute import MirroredStrategy, get_strategy
from tensorflow.random import set_seed
from tensorflow import keras, make_ndarray
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from .attention import BahdanauAttention, LuongAttention, AttentionDecoder

def create_model(
    optimiser,
    rnn_layer,
    n_layer,
    n_features,
    n_outputs,
    dropout,
    loss,
    n_dense_layer,
    dense_activation,
    **kwargs
    ):
    if list_physical_devices('GPU') and kwargs.get('multigpu', False):
      strategy = MirroredStrategy()
    else:  # Use the Default Strategy
      strategy = get_strategy()
    with strategy.scope():
        input_layer = Input(shape=(None, n_features), dtype='float32')
        recurrent_layers = [
            rnn_layer(
                n,
                dropout=dropout,
                return_sequences=True,
            )
            for n in n_layer
        ]

        dense_layers = [
            layers.TimeDistributed(
                layers.Dense(
                    n,
                    activation=activation
                )
            )
            for n, activation in zip(n_dense_layer, dense_activation)
        ]

        model = keras.Sequential(
            [input_layer] + recurrent_layers + dense_layers
        )

    model.compile(loss=loss, optimizer=optimiser, metrics=['mean_squared_error'])
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
            LuongAttention
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
            callbacks = [TensorBoard(log_dir=log, histogram_freq=1)]
        )
    else:
        model.fit(gen, epochs=epochs, verbose=verbose)

def model_predict(model, X_test, scaler):
    predictions = model.predict(X_test)
    return scaler.inverse_transform(predictions)
