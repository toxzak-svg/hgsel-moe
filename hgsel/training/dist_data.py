"""
Distributed data loader helpers for Phase 4 DDP parity validation.

Uses DistributedSampler and deterministic per-epoch shuffling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class DummyLanguageModelDataset(Dataset):
    """Deterministic synthetic language-model dataset for parity/smoke runs."""

    def __init__(
        self,
        num_samples: int,
        seq_length: int,
        vocab_size: int,
        seed: int = 1234,
    ) -> None:
        self.num_samples = int(num_samples)
        self.seq_length = int(seq_length)
        self.vocab_size = int(vocab_size)

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        self.input_ids = torch.randint(
            0,
            vocab_size,
            (num_samples, seq_length),
            generator=g,
            dtype=torch.long,
        )
        self.labels = torch.randint(
            0,
            vocab_size,
            (num_samples, seq_length),
            generator=g,
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.labels[idx]


class TokenSequenceDataset(Dataset):
    """Language-model dataset over a 1D token stream.

    Builds (input_ids, labels) pairs using a sliding window:
    - input_ids = tokens[start:start+seq_length]
    - labels    = tokens[start+1:start+seq_length+1]
    """

    def __init__(
        self,
        token_ids: torch.Tensor,
        seq_length: int,
        stride: Optional[int] = None,
    ) -> None:
        if token_ids.dim() != 1:
            raise ValueError("token_ids must be a 1D tensor")
        self.token_ids = token_ids.to(dtype=torch.long, device="cpu").contiguous()
        self.seq_length = int(seq_length)
        self.stride = int(stride) if stride is not None else int(seq_length)

        if self.seq_length <= 0:
            raise ValueError("seq_length must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.token_ids.numel() <= self.seq_length:
            raise ValueError("token_ids is too short for seq_length")

        self.starts = list(range(0, self.token_ids.numel() - self.seq_length, self.stride))
        if not self.starts:
            raise ValueError("No valid windows produced; increase token count or reduce seq_length/stride")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        input_ids = self.token_ids[start : start + self.seq_length]
        labels = self.token_ids[start + 1 : start + self.seq_length + 1]
        return input_ids, labels


@dataclass
class DistributedLoaderInfo:
    train_loader: DataLoader
    val_loader: DataLoader
    train_sampler: Optional[DistributedSampler]
    val_sampler: Optional[DistributedSampler]
    per_rank_batch_size: int
    global_batch_size: int
    world_size: int
    rank: int


def _build_distributed_loaders(
    *,
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    rank: int,
    world_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
) -> DistributedLoaderInfo:
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=drop_last,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=seed + 1,
            drop_last=drop_last,
        )
        train_shuffle = False
        val_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True
        val_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=val_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return DistributedLoaderInfo(
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        per_rank_batch_size=batch_size,
        global_batch_size=batch_size * max(world_size, 1),
        world_size=world_size,
        rank=rank,
    )


def create_distributed_dummy_loaders(
    *,
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    num_train_batches: int,
    num_val_batches: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 1234,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
) -> DistributedLoaderInfo:
    """Create deterministic loaders using DistributedSampler when world_size > 1."""
    train_num_samples = int(num_train_batches * batch_size * max(world_size, 1))
    val_num_samples = int(num_val_batches * batch_size * max(world_size, 1))

    train_dataset = DummyLanguageModelDataset(
        num_samples=train_num_samples,
        seq_length=seq_length,
        vocab_size=vocab_size,
        seed=seed,
    )
    val_dataset = DummyLanguageModelDataset(
        num_samples=val_num_samples,
        seq_length=seq_length,
        vocab_size=vocab_size,
        seed=seed + 1,
    )

    return _build_distributed_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def _tokenize_text_to_ids(text: str, vocab_size: int, tokenizer_mode: str) -> torch.Tensor:
    if tokenizer_mode == "byte":
        token_values = list(text.encode("utf-8", errors="ignore"))
        if not token_values:
            raise ValueError("No tokens produced from text")
        ids = torch.tensor(token_values, dtype=torch.long)
        if vocab_size < 256:
            ids = ids % vocab_size
        return ids

    if tokenizer_mode == "char":
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive for char tokenization")
        ids = torch.tensor([ord(ch) % vocab_size for ch in text], dtype=torch.long)
        if ids.numel() == 0:
            raise ValueError("No tokens produced from text")
        return ids

    raise ValueError(f"Unsupported tokenizer_mode: {tokenizer_mode}")


def create_distributed_text_loaders_from_text(
    *,
    train_text: str,
    val_text: str,
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 1234,
    tokenizer_mode: str = "byte",
    stride: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
) -> DistributedLoaderInfo:
    """Create distributed LM loaders from raw train/val text."""
    train_ids = _tokenize_text_to_ids(train_text, vocab_size=vocab_size, tokenizer_mode=tokenizer_mode)
    val_ids = _tokenize_text_to_ids(val_text, vocab_size=vocab_size, tokenizer_mode=tokenizer_mode)

    train_dataset = TokenSequenceDataset(train_ids, seq_length=seq_length, stride=stride)
    val_dataset = TokenSequenceDataset(val_ids, seq_length=seq_length, stride=stride)

    return _build_distributed_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def _limit_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    return text[:max_chars]


def _join_hf_text_column(dataset, text_column: str, max_chars: int = 0) -> str:
    parts = []
    chars = 0
    for row in dataset:
        value = row.get(text_column, "")
        if not isinstance(value, str):
            continue
        if not value.strip():
            continue
        parts.append(value)
        chars += len(value)
        if max_chars > 0 and chars >= max_chars:
            break
    joined = "\n".join(parts)
    return _limit_text(joined, max_chars)


def create_distributed_hf_text_loaders(
    *,
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    train_split: str = "train",
    val_split: str = "validation",
    text_column: str = "text",
    rank: int = 0,
    world_size: int = 1,
    seed: int = 1234,
    tokenizer_mode: str = "byte",
    stride: Optional[int] = None,
    max_train_chars: int = 0,
    max_val_chars: int = 0,
    cache_dir: str = "",
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
) -> DistributedLoaderInfo:
    """Create distributed LM loaders from a Hugging Face text dataset."""
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets library is required for HF text loaders") from exc

    ds_kwargs = {"split": train_split}
    if cache_dir:
        ds_kwargs["cache_dir"] = cache_dir
    train_ds = load_dataset(dataset_name, dataset_config, **ds_kwargs)

    ds_kwargs = {"split": val_split}
    if cache_dir:
        ds_kwargs["cache_dir"] = cache_dir
    val_ds = load_dataset(dataset_name, dataset_config, **ds_kwargs)

    train_text = _join_hf_text_column(train_ds, text_column=text_column, max_chars=max_train_chars)
    val_text = _join_hf_text_column(val_ds, text_column=text_column, max_chars=max_val_chars)

    if len(train_text) == 0 or len(val_text) == 0:
        raise ValueError(
            f"Loaded empty text from dataset={dataset_name}/{dataset_config} "
            f"(train_split={train_split}, val_split={val_split}, text_column={text_column})"
        )

    return create_distributed_text_loaders_from_text(
        train_text=train_text,
        val_text=val_text,
        batch_size=batch_size,
        seq_length=seq_length,
        vocab_size=vocab_size,
        rank=rank,
        world_size=world_size,
        seed=seed,
        tokenizer_mode=tokenizer_mode,
        stride=stride,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def set_distributed_epoch(loader: DataLoader, epoch: int) -> None:
    """Call DistributedSampler.set_epoch(epoch) when present."""
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)
