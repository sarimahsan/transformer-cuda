import os
import urllib.request
import numpy as np


def prepare_tinyshakespeare(data_dir: str = "data"):
    os.makedirs(data_dir, exist_ok=True)
    txt_path = os.path.join(data_dir, "tinyshakespeare.txt")
    bin_path = os.path.join(data_dir, "input.bin")

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    if not os.path.exists(txt_path):
        print(f"[Data] Downloading TinyShakespeare dataset from {url}...")
        try:
            urllib.request.urlretrieve(url, txt_path)
            print(f"[Data] Downloaded to {txt_path}")
        except Exception as e:
            print(f"[Data] Download failed ({e}). Generating synthetic text corpus...")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("To be, or not to be, that is the question.\n" * 5000)

    with open(txt_path, "r", encoding="utf-8") as f:
        data = f.read()

    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"[Data] Dataset length: {len(data):,} characters. Vocabulary size: {vocab_size}")

    # Map characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Save vocab mapping
    with open(os.path.join(data_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        for ch, idx in stoi.items():
            f.write(f"{idx}\t{repr(ch)}\n")

    # Encode to int32 tokens
    tokens = np.array([stoi[ch] for ch in data], dtype=np.int32)
    tokens.tofile(bin_path)

    print(f"[Data] Serialized {len(tokens):,} int32 tokens to '{bin_path}' ({os.path.getsize(bin_path):,} bytes).")
    return vocab_size


if __name__ == "__main__":
    prepare_tinyshakespeare()
