from tensorflow.config import list_physical_devices
from tensorflow.distribute import MirroredStrategy, get_strategy
from tensorflow.random import set_seed
from tensorflow import keras, make_ndarray
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from .attention import BahdanauAttention, LuongAttention, AttentionDecoder

def create_model(optimiser, rnn_layer, n_layer, dropout, loss, **kwargs):
    if list_physical_devices('GPU') and kwargs.get('multigpu', False):
      strategy = MirroredStrategy()
    else:  # Use the Default Strategy
      strategy = get_strategy()
    with strategy.scope():
        model = keras.Sequential()
        model.add(rnn_layer(n_layer[0], dropout=dropout, return_sequences=True))
        model.add(rnn_layer(n_layer[1], dropout=dropout, return_sequences=True))
        model.add(layers.TimeDistributed(layers.Dense(n_layer[1])))
    model.compile(loss=loss, optimizer=optimiser, metrics=['mean_squared_error'])
    return model

def create_ed_model(
    optimiser,
    rnn_layer,
    n_layer,
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
        model.add(rnn_layer(n_layer[0], dropout=dropout))
        model.add(layers.RepeatVector(n_timesteps))
        model.add(rnn_layer(n_layer[1], dropout=dropout, return_sequences=True))
        model.add(layers.TimeDistributed(layers.Dense(n_layer[1])))
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

def train_model(model, gen, epochs, seed, verbose=True):
    set_seed(seed)
    model.fit(gen, epochs=epochs, verbose=verbose)

def model_predict(model, X_test, scaler):
    predictions = model.predict(X_test)
    return scaler.inverse_transform(predictions)
