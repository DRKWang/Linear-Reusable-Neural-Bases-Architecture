import os

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from config import TOT_VOCAB_SIZE, SEQ_LEN


CACHE_DIR = "./local_wikitext2"
TOKENIZER_PATH = "wikitext2_bpe_tokenizer.json"

BASE_SEED = 43
BATCH_SIZE = 16

NUM_TEST_BATCHES = 128
NUM_TRAIN_BATCHES = 512


def load_wikitext2_raw():
    """
    Load the raw WikiText-2 dataset and cache it locally.
    """
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        cache_dir=CACHE_DIR,
    )

    print("-" * 40)
    print("Dataset downloaded/loaded locally.")
    print(dataset)
    print("Train rows:", len(dataset["train"]))

    return dataset


def build_tokenizer(
    dataset,
    tokenizer_path=TOKENIZER_PATH,
    vocab_size=TOT_VOCAB_SIZE,
):
    """
    Load an existing tokenizer or train a byte-level BPE tokenizer
    on the WikiText-2 training split.
    """

    if os.path.exists(tokenizer_path):
        print(f"Tokenizer already exists: {tokenizer_path}")
        return Tokenizer.from_file(tokenizer_path)

    def batch_iterator(batch_size=2**10):
        for i in range(0, len(dataset["train"]), batch_size):
            yield dataset["train"][i : i + batch_size]["text"]

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(
        add_prefix_space=False
    )

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "[UNK]",
            "[PAD]",
            "[SOS]",
            "[EOS]",
        ],
    )

    print("Training tokenizer...")

    tokenizer.train_from_iterator(
        batch_iterator(),
        trainer=trainer,
    )

    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.save(tokenizer_path)

    print(f"Tokenizer trained and saved to: {tokenizer_path}")

    test_texts = [
        "hello world",
        "Valkyria Chronicles III",
        "emoji test 😊",
        "rare unicode: 𠜎",
    ]

    for text in test_texts:
        encoded = tokenizer.encode(text)

        print("\nText:", text)
        print("Tokens:", encoded.tokens)
        print("IDs:", encoded.ids)
        print("Has [UNK]?", "[UNK]" in encoded.tokens)

    return tokenizer


class dataset_maker(Dataset):
    """
    Dataset containing all valid contiguous token windows.

    Random sampling is handled by RandomSampler in the DataLoader.
    """

    def __init__(
        self,
        split="train",
        chunk_size=SEQ_LEN,
        cache_dir=CACHE_DIR,
        tokenizer_path=TOKENIZER_PATH,
    ):
        super().__init__()

        self.split = split
        self.chunk_size = chunk_size

        dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            cache_dir=cache_dir,
        )[split]

        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer file not found: {tokenizer_path}. "
                "Run build_tokenizer() first."
            )

        tokenizer = Tokenizer.from_file(tokenizer_path)

        print(f"Tokenizing and flattening the {split} split...")

        all_ids = []

        for row in dataset:
            text = row["text"]
            text = text.strip()
            if not text:
                continue

            encoded = tokenizer.encode(text)
            all_ids.extend(encoded.ids)

        self.tokens = torch.tensor(
            all_ids,
            dtype=torch.long,
        )

        # A valid sample requires chunk_size input tokens and one
        # additional token for the shifted target.
        self.num_valid_starts = len(self.tokens) - self.chunk_size

        if self.num_valid_starts <= 0:
            raise ValueError(
                f"The {split} split contains {len(self.tokens)} tokens, "
                f"which is insufficient for chunk_size={self.chunk_size}."
            )

        print(
            f"Created {split} dataset from {len(self.tokens):,} tokens."
        )
        print(
            f"Valid starting positions: {self.num_valid_starts:,}"
        )
        print(f"Sequence length: {self.chunk_size}")

    def __len__(self):
        return self.num_valid_starts

    def __getitem__(self, start_idx):
        """
        Return a contiguous language-modeling sample beginning at start_idx.
        """
        x = self.tokens[
            start_idx : start_idx + self.chunk_size
        ]

        y = self.tokens[
            start_idx + 1 : start_idx + self.chunk_size + 1
        ]

        return x, y


if __name__ == "__main__":
    raw_dataset = load_wikitext2_raw()

    tokenizer = build_tokenizer(
        dataset=raw_dataset,
    )

    train_dataset = dataset_maker(
        split="train",
        chunk_size=SEQ_LEN,
    )

    test_dataset = dataset_maker(
        split="test",
        chunk_size=SEQ_LEN,
    )

    x, y = train_dataset[0]

    print("\nExample sample")
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("x:", x)
    print("y:", y)