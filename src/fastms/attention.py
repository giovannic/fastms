from tensorflow.keras import layers
from tensorflow import concat

class LuongAttention(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.attention = layers.Attention()

    def call(self, query, value):
        context_vector, attention_weights = self.attention(
            inputs = [query, value],
            return_attention_scores = True,
        )

        return context_vector, attention_weights

class BahdanauAttention(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.w1 = layers.Dense(units, use_bias=False)
        self.w2 = layers.Dense(units, use_bias=False)
        self.attention = layers.AdditiveAttention()

    def call(self, query, value):
        w1_query = self.w1(query)
        w2_key = self.w2(value)

        context_vector, attention_weights = self.attention(
            inputs = [w1_query, value, w2_key],
            return_attention_scores = True,
        )

        return context_vector, attention_weights

class AttentionDecoder(layers.Layer):
    def __init__(self, units, n_outputs, RNNLayer, AttentionLayer):
        super().__init__()
        self.rnn_layer = RNNLayer(
            units,
            return_sequences = True,
            return_state = True,
            recurrent_initializer='glorot_uniform'
        )
        self.attention = AttentionLayer(units)
        self.w1 = layers.Dense(units, activation='tanh', use_bias=False)
        self.w2 = layers.TimeDistributed(
            layers.Dense(n_outputs, activation='tanh')
        )

    def call(self, inputs, enc_output, state):
        rnn_output, h, c = self.rnn_layer(inputs, initial_state = state)
        context_vector, attention_weights = self.attention(
            query=rnn_output,
            value=enc_output
        )
        context_and_rnn_output = concat([context_vector, rnn_output], axis=-1)

        attention_vector = self.w1(context_and_rnn_output)
        output = self.w2(attention_vector)

        return output, attention_weights, [h, c]
