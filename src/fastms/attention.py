from tensorflow.keras import layers

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
        self.w2 = layers.Dense(n_outputs, activation='softmax')

    def call(self, query, value):
        w1_query = self.w1(query)
        w2_key = self.w2(value)

        context_vector, attention_weights = self.attention(
            inputs = [w1_query, value, w2_key],
            return_attention_scores = True,
        )

        return context_vector, attention_weights


