from tensorflow.random import set_seed
from tensorflow import keras, make_ndarray
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

def create_model(optimiser, n_layer, dropout, loss, **kwargs):
    model = keras.Sequential()
    model.add(layers.LSTM(n_layer[0], dropout=dropout, return_sequences=True))
    model.add(layers.LSTM(n_layer[1], dropout=dropout, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(n_layer[1])))
    model.compile(loss=loss, optimizer=optimiser)
    return model

def train_model(model, gen, epochs, seed):
    set_seed(seed)
    model.fit(gen, epochs=epochs)

def model_predict(model, gen, scaler):
    predictions = model.predict(gen)
    return scaler.inverse_transform(predictions).reshape(predictions.shape[0], -1)
