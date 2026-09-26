import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, LSTM, Dense
from attention import BahdanauAttention

class CaptionDecoder(Layer):
    def __init__(self, vocab_size, embedding_dim, units):
        super(CaptionDecoder, self).__init__()
        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(units, return_sequences=True, return_state=True)
        self.fc1 = Dense(units)
        self.fc2 = Dense(vocab_size)
        self.attention = BahdanauAttention(units)

    def call(self, x, features, hidden_state, cell_state):
        context_vector, attention_weights = self.attention(features, hidden_state)

        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, hidden_state, cell_state = self.lstm(x, initial_state=[hidden_state, cell_state])

        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)

        return x, hidden_state, cell_state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))