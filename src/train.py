import tensorflow as tf
import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.dirname(__file__))
from decoder import CaptionDecoder

def build_dataset(captions_df, sequences, image_features, word_to_idx, batch_size=64):
    image_names = captions_df["image"].tolist()

    feature_list = []
    valid_indices = []
    for i, img_name in enumerate(image_names):
        img_path = os.path.join("../data/Images", img_name)
        if img_path in image_features:
            feature_list.append(image_features[img_path])
            valid_indices.append(i)

    feature_array = np.array(feature_list, dtype=np.float32)
    sequence_array = sequences[valid_indices]

    dataset = tf.data.Dataset.from_tensor_slices((feature_array, sequence_array))
    dataset = dataset.shuffle(1000).batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def train_step(features, target, decoder, optimizer, loss_object, units, batch_size):
    loss = 0
    hidden, cell = decoder.reset_state(batch_size=batch_size)
    dec_input = tf.expand_dims(target[:, 0], 1)

    with tf.GradientTape() as tape:
        for t in range(1, target.shape[1]):
            predictions, hidden, cell, _ = decoder(dec_input, features, hidden, cell)

            mask = tf.math.logical_not(tf.math.equal(target[:, t], 0))
            loss_t = loss_object(target[:, t], predictions)
            mask = tf.cast(mask, dtype=loss_t.dtype)
            loss_t *= mask
            loss += tf.reduce_mean(loss_t)

            dec_input = tf.expand_dims(target[:, t], 1)

    total_loss = loss / int(target.shape[1])
    trainable_vars = decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))

    return total_loss

def train_model(captions_df, sequences, image_features, word_to_idx, vocab_size,
                 embedding_dim=256, units=512, batch_size=64, epochs=20):

    dataset = build_dataset(captions_df, sequences, image_features, word_to_idx, batch_size)

    decoder = CaptionDecoder(vocab_size, embedding_dim, units)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch_features, batch_target in dataset:
            batch_loss = train_step(batch_features, batch_target, decoder, optimizer, loss_object, units, batch_size)
            total_loss += batch_loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        loss_history.append(float(avg_loss))
        print(f"Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

    return decoder, loss_history