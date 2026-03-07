"""Tests for distributed data loader helpers (no process group required)."""

import sys
from pathlib import Path

import torch
from torch.utils.data.distributed import DistributedSampler

current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from hgsel.training.dist_data import (  # noqa: E402
    create_distributed_dummy_loaders,
    create_distributed_text_loaders_from_text,
    set_distributed_epoch,
)


def test_create_distributed_dummy_loaders_single_rank():
    info = create_distributed_dummy_loaders(
        batch_size=4,
        seq_length=8,
        vocab_size=32,
        num_train_batches=3,
        num_val_batches=2,
        rank=0,
        world_size=1,
        seed=42,
    )

    assert info.global_batch_size == 4
    assert info.per_rank_batch_size == 4
    assert info.world_size == 1
    assert info.train_sampler is None
    assert info.val_sampler is None
    assert len(info.train_loader) == 3
    assert len(info.val_loader) == 2

    batch = next(iter(info.train_loader))
    input_ids, labels = batch
    assert input_ids.shape == (4, 8)
    assert labels.shape == (4, 8)

    # No-op in single-rank mode should not raise.
    set_distributed_epoch(info.train_loader, 0)


def test_create_distributed_dummy_loaders_multi_rank_sampler_math():
    info = create_distributed_dummy_loaders(
        batch_size=4,
        seq_length=8,
        vocab_size=32,
        num_train_batches=5,
        num_val_batches=3,
        rank=0,
        world_size=2,
        seed=123,
    )

    assert info.global_batch_size == 8
    assert info.per_rank_batch_size == 4
    assert info.world_size == 2
    assert isinstance(info.train_sampler, DistributedSampler)
    assert isinstance(info.val_sampler, DistributedSampler)

    # Dataset size is chosen so each rank sees exactly num_*_batches batches at this batch size.
    assert len(info.train_loader) == 5
    assert len(info.val_loader) == 3

    # DistributedSampler.set_epoch should be callable.
    set_distributed_epoch(info.train_loader, 7)


def test_create_distributed_dummy_loaders_seed_reproducibility():
    info_a = create_distributed_dummy_loaders(
        batch_size=2,
        seq_length=6,
        vocab_size=16,
        num_train_batches=2,
        num_val_batches=1,
        rank=0,
        world_size=1,
        seed=999,
    )
    info_b = create_distributed_dummy_loaders(
        batch_size=2,
        seq_length=6,
        vocab_size=16,
        num_train_batches=2,
        num_val_batches=1,
        rank=0,
        world_size=1,
        seed=999,
    )

    # DataLoader may shuffle in single-rank mode; compare underlying deterministic dataset tensors.
    dataset_a = info_a.train_loader.dataset
    dataset_b = info_b.train_loader.dataset

    assert torch.equal(dataset_a.input_ids, dataset_b.input_ids)
    assert torch.equal(dataset_a.labels, dataset_b.labels)


def test_create_distributed_text_loaders_from_text_byte_tokenization():
    info = create_distributed_text_loaders_from_text(
        train_text=("hello world\n" * 200),
        val_text=("goodbye world\n" * 100),
        batch_size=4,
        seq_length=16,
        vocab_size=256,
        rank=0,
        world_size=1,
        seed=7,
        tokenizer_mode="byte",
    )

    assert info.global_batch_size == 4
    assert info.per_rank_batch_size == 4
    assert len(info.train_loader) > 0
    assert len(info.val_loader) > 0
    batch = next(iter(info.train_loader))
    input_ids, labels = batch
    assert input_ids.shape == (4, 16)
    assert labels.shape == (4, 16)
    assert torch.all(input_ids >= 0)
    assert torch.all(input_ids < 256)


def test_create_distributed_text_loaders_from_text_distributed_sampler():
    info = create_distributed_text_loaders_from_text(
        train_text=("abcdefg " * 200),
        val_text=("hijklmn " * 100),
        batch_size=2,
        seq_length=8,
        vocab_size=64,
        rank=0,
        world_size=2,
        seed=11,
        tokenizer_mode="char",
    )

    assert isinstance(info.train_sampler, DistributedSampler)
    assert isinstance(info.val_sampler, DistributedSampler)
    assert info.global_batch_size == 4
    assert info.per_rank_batch_size == 2
    assert len(info.train_loader) > 0
