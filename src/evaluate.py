import tensorflow as tf
import numpy as np
import pickle
import sys
import os

try:
    sys.path.append(os.path.dirname(__file__))
except NameError:
    sys.path.append("./src")

from decoder import CaptionDecoder
from encoder import build_encoder, preprocess_image

def load_trained_model(checkpoint_dir, vocab_size, embedding_dim=256, units=512):
    decoder = CaptionDecoder(vocab_size, embedding_dim, units)
    optimizer = tf.keras.optimizers.Adam()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, decoder=decoder)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
        print(f"Loaded checkpoint: {checkpoint_manager.latest_checkpoint}")
    else:
        raise FileNotFoundError("No checkpoint found in " + checkpoint_dir)

    return decoder

def generate_caption_greedy(image_path, encoder, decoder, word_to_idx, idx_to_word, max_len):
    img_array = preprocess_image(image_path)
    img_array = np.expand_dims(img_array, axis=0)

    features = encoder.predict(img_array, verbose=0)
    features = features.reshape(1, -1, features.shape[-1])

    hidden, cell = decoder.reset_state(batch_size=1)
    dec_input = tf.expand_dims([word_to_idx["startseq"]], 0)

    result = []
    attention_plot = []

    for i in range(max_len):
        predictions, hidden, cell, attention_weights = decoder(dec_input, features, hidden, cell)
        attention_plot.append(tf.reshape(attention_weights, (-1,)).numpy())

        predicted_id = tf.argmax(predictions[0]).numpy()
        predicted_word = idx_to_word.get(predicted_id, "")

        if predicted_word == "endseq":
            break

        result.append(predicted_word)
        dec_input = tf.expand_dims([predicted_id], 0)

    return " ".join(result), attention_plot

def get_blocked_bigrams(seq, block_ngram_size=2):
    blocked = set()
    if len(seq) < block_ngram_size:
        return blocked
    for i in range(len(seq) - block_ngram_size + 1):
        ngram = tuple(seq[i:i + block_ngram_size])
        blocked.add(ngram)
    return blocked

def generate_caption_beam_search(image_path, encoder, decoder, word_to_idx, idx_to_word, max_len,
                                  beam_width=3, block_ngram_size=2):
    img_array = preprocess_image(image_path)
    img_array = np.expand_dims(img_array, axis=0)

    features = encoder.predict(img_array, verbose=0)
    features = features.reshape(1, -1, features.shape[-1])

    start_hidden, start_cell = decoder.reset_state(batch_size=1)
    start_token = word_to_idx["startseq"]
    end_token = word_to_idx["endseq"]

    sequences = [([start_token], 0.0, start_hidden, start_cell)]

    for step in range(max_len):
        candidates = []

        for seq, score, hidden, cell in sequences:
            last_word = seq[-1]

            if last_word == end_token:
                candidates.append((seq, score, hidden, cell))
                continue

            dec_input = tf.expand_dims([last_word], 0)
            predictions, new_hidden, new_cell, _ = decoder(dec_input, features, hidden, cell)

            log_probs = tf.nn.log_softmax(predictions[0]).numpy()

            blocked_ngrams = get_blocked_bigrams(seq, block_ngram_size)
            if len(seq) >= block_ngram_size - 1:
                prefix = tuple(seq[-(block_ngram_size - 1):])
                for candidate_word in range(len(log_probs)):
                    would_be_ngram = prefix + (candidate_word,)
                    if would_be_ngram in blocked_ngrams:
                        log_probs[candidate_word] = -1e9

            top_indices = np.argsort(log_probs)[-beam_width:]

            for idx in top_indices:
                if log_probs[idx] <= -1e9:
                    continue
                new_seq = seq + [int(idx)]
                new_score = score + log_probs[idx]
                candidates.append((new_seq, new_score, new_hidden, new_cell))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[1], reverse=True)
        sequences = candidates[:beam_width]

        if all(seq[-1] == end_token for seq, _, _, _ in sequences):
            break

    best_seq = sequences[0][0]
    words = [idx_to_word.get(idx, "") for idx in best_seq if idx not in (start_token, end_token)]
    return " ".join(words), None

def generate_caption(image_path, encoder, decoder, word_to_idx, idx_to_word, max_len,
                      use_beam_search=True, beam_width=3, block_ngram_size=2):
    if use_beam_search:
        return generate_caption_beam_search(
            image_path, encoder, decoder, word_to_idx, idx_to_word, max_len,
            beam_width, block_ngram_size
        )
    else:
        return generate_caption_greedy(image_path, encoder, decoder, word_to_idx, idx_to_word, max_len)
