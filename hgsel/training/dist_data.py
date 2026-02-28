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


def set_distributed_epoch(loader: DataLoader, epoch: int) -> None:
    """Call DistributedSampler.set_epoch(epoch) when present."""
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)
