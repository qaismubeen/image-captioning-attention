import pandas as pd
import re
import string
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_captions(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = ["image", "caption"]
    return df

def clean_caption(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def add_start_end_tokens(text: str) -> str:
    return f"startseq {text} endseq"

def preprocess_captions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_caption"] = df["caption"].apply(clean_caption)
    df["clean_caption"] = df["clean_caption"].apply(add_start_end_tokens)
    return df

def build_vocabulary(captions: list, min_word_freq: int = 5):
    word_counts = {}
    for caption in captions:
        for word in caption.split():
            word_counts[word] = word_counts.get(word, 0) + 1

    vocab = [word for word, count in word_counts.items() if count >= min_word_freq]
    vocab = sorted(vocab)

    word_to_idx = {word: idx + 1 for idx, word in enumerate(vocab)}
    word_to_idx["<pad>"] = 0
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    return word_to_idx, idx_to_word

def get_max_caption_length(captions: list) -> int:
    return max(len(caption.split()) for caption in captions)

def captions_to_sequences(captions: list, word_to_idx: dict, max_len: int):
    sequences = []
    for caption in captions:
        seq = [word_to_idx[word] for word in caption.split() if word in word_to_idx]
        sequences.append(seq)

    padded = pad_sequences(sequences, maxlen=max_len, padding="post", value=word_to_idx["<pad>"])
    return padded