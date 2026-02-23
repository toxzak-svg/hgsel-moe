"""
Data loading utilities for language modeling.

Provides simple tokenization and batch loading.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import numpy as np


class SimpleTokenizer:
    """Simple character-level tokenizer for testing."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.char_to_id = {chr(i): i for i in range(vocab_size)}
        self.id_to_char = {i: chr(i) for i in range(vocab_size)}

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return "".join(self.id_to_char.get(id, "<unk>") for id in ids)


class LanguageModelDataset(Dataset):
    """
    Simple language modeling dataset.

    Args:
        text: Raw text string
        seq_len: Sequence length for LM
        stride: Step size between sequences (for overlapping windows)
        tokenizer: Tokenizer instance
    """

    def __init__(
        self,
        text: str,
        seq_len: int = 128,
        stride: int = 1,
        tokenizer=None,
    ):
        self.text = text
        self.seq_len = seq_len
        self.stride = stride

        if tokenizer is None:
            self.tokenizer = SimpleTokenizer()
        else:
            self.tokenizer = tokenizer

        # Tokenize
        self.tokens = self.tokenizer.encode(text)

        # Create sliding windows
        self.sequences = []
        for i in range(0, len(self.tokens) - seq_len, stride):
            self.sequences.append(i)

        if len(self.sequences) == 0:
            raise ValueError(f"Text too short for seq_len={seq_len}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (input_ids, labels) for language modeling.

        input_ids: tokens[start:start+seq_len]
        labels: tokens[start+1:start+seq_len+1]  (shifted by 1)
        """
        start = self.sequences[idx]

        input_ids = torch.tensor(
            self.tokens[start : start + self.seq_len],
            dtype=torch.long,
        )

        labels = torch.tensor(
            self.tokens[start + 1 : start + self.seq_len + 1],
            dtype=torch.long,
        )

        return input_ids, labels


class DummyDataLoader:
    """
    Dummy data loader for quick testing.

    Generates random sequences of token IDs.
    """

    def __init__(
        self,
        num_batches: int = 10,
        batch_size: int = 32,
        seq_len: int = 128,
        vocab_size: int = 256,
    ):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __iter__(self):
        for _ in range(self.num_batches):
            input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
            labels = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
            yield input_ids, labels

    def __len__(self) -> int:
        return self.num_batches


def get_dummy_loaders(
    batch_size: int = 32,
    val_batch_size: int = 64,
    num_train_batches: int = 100,
    num_val_batches: int = 20,
    seq_len: int = 128,
    vocab_size: int = 256,
):
    """Get dummy train and validation loaders for testing."""
    train_loader = DummyDataLoader(
        num_batches=num_train_batches,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )

    val_loader = DummyDataLoader(
        num_batches=num_val_batches,
        batch_size=val_batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )

    return train_loader, val_loader


def create_wiki_dataset(seq_len: int = 128, train_split: float = 0.9):
    """
    Create dataset from WikiText-2 (if available).

    Falls back to dummy data if not available.

    Returns:
        (train_loader, val_loader)
    """
    try:
        from datasets import load_dataset

        # Load WikiText-2 (small sample for testing)
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

        # Concatenate all text
        text = " ".join([doc["text"] for doc in dataset if doc["text"].strip()])

        if len(text) < 1000:
            print("⚠ WikiText loaded but very small, using dummy data instead")
            return get_dummy_loaders()

        print(f"✓ Loaded WikiText-2: {len(text):,} characters")

        # Split
        split_idx = int(len(text) * train_split)
        train_text = text[:split_idx]
        val_text = text[split_idx:]

        # Create datasets
        train_dataset = LanguageModelDataset(train_text, seq_len=seq_len, stride=seq_len)
        val_dataset = LanguageModelDataset(val_text, seq_len=seq_len, stride=seq_len)

        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        return train_loader, val_loader

    except Exception as e:
        print(f"⚠ Could not load WikiText: {e}")
        print("  Using dummy data instead")
        return get_dummy_loaders()


if __name__ == "__main__":
    print("Testing data loading...")

    # Test dummy loader
    train_loader, val_loader = get_dummy_loaders(
        batch_size=4,
        num_train_batches=5,
        num_val_batches=2,
    )

    print(f"Train batches: {len(train_loader)}")
    for i, (input_ids, labels) in enumerate(train_loader):
        print(f"  Batch {i}: input {input_ids.shape}, labels {labels.shape}")
        if i >= 2:
            break

    print(f"\nVal batches: {len(val_loader)}")
    for i, (input_ids, labels) in enumerate(val_loader):
        print(f"  Batch {i}: input {input_ids.shape}, labels {labels.shape}")
        if i >= 1:
            break

    print("\n✓ Data loading test complete!")
