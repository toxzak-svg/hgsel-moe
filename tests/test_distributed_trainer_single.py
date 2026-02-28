"""Single-rank tests for DistributedTrainer API behavior."""

import random
import sys
from pathlib import Path

import pytest
import torch

current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from experiments.baselines.dense_transformer import TransformerModel  # noqa: E402
from hgsel.training.distributed_trainer import DistributedTrainer  # noqa: E402
from hgsel.training.trainer import TrainingConfig  # noqa: E402


def _tiny_model() -> TransformerModel:
    return TransformerModel(
        vocab_size=64,
        d_model=16,
        d_ff=64,
        n_layers=1,
        n_heads=2,
        max_seq_len=8,
        dropout=0.0,
    )


def _tiny_batch() -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    labels = torch.randint(0, 64, (2, 8), dtype=torch.long)
    return input_ids, labels


def test_distributed_trainer_single_rank_train_step():
    config = TrainingConfig(
        batch_size=2,
        num_epochs=1,
        learning_rate=1e-3,
        warmup_steps=0,
        use_wandb=False,
        device="cpu",
        clip_grad_norm=1.0,
    )
    trainer = DistributedTrainer(
        model=_tiny_model(),
        config=config,
        device=torch.device("cpu"),
        auto_init_from_env=False,
    )

    metrics = trainer.train_step(_tiny_batch())
    assert "loss" in metrics
    assert "learning_rate" in metrics
    assert metrics["loss"] > 0
    assert metrics["learning_rate"] > 0

    trainer.cleanup()


def test_distributed_trainer_checkpoint_roundtrip_single_rank(tmp_path: Path):
    random.seed(1234)
    torch.manual_seed(1234)

    config = TrainingConfig(
        batch_size=2,
        num_epochs=1,
        learning_rate=1e-3,
        warmup_steps=0,
        use_wandb=False,
        device="cpu",
        checkpoint_dir=tmp_path / "ckpts",
    )
    trainer = DistributedTrainer(
        model=_tiny_model(),
        config=config,
        device=torch.device("cpu"),
        auto_init_from_env=False,
    )

    # Advance state a bit so checkpoint restore is meaningful.
    _ = trainer.train_step(_tiny_batch())
    trainer.global_step = 3
    trainer.global_epoch = 1

    ckpt_path = tmp_path / "roundtrip.pt"
    trainer.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()

    new_trainer = DistributedTrainer(
        model=_tiny_model(),
        config=config,
        device=torch.device("cpu"),
        auto_init_from_env=False,
    )
    new_trainer.load_checkpoint(ckpt_path)

    assert new_trainer.global_step == 3
    assert new_trainer.global_epoch == 1

    # Verify RNG state restore is meaningful (not just metadata restore).
    expected_python_random = random.random()
    expected_batch = _tiny_batch()
    expected_metrics = trainer.train_step(expected_batch)

    observed_python_random = random.random()  # Consumes current global RNG before restore path check below

    # Re-create trainer and reload checkpoint again so RNG state is reset to checkpoint point.
    replay_trainer = DistributedTrainer(
        model=_tiny_model(),
        config=config,
        device=torch.device("cpu"),
        auto_init_from_env=False,
    )
    replay_trainer.load_checkpoint(ckpt_path)

    replay_python_random = random.random()
    replay_batch = _tiny_batch()
    replay_metrics = replay_trainer.train_step(replay_batch)

    assert replay_python_random == expected_python_random
    assert replay_python_random != observed_python_random
    assert torch.equal(replay_batch[0], expected_batch[0])
    assert torch.equal(replay_batch[1], expected_batch[1])
    assert replay_metrics["loss"] == pytest.approx(expected_metrics["loss"], rel=1e-6, abs=1e-6)

    trainer.cleanup()
    new_trainer.cleanup()
    replay_trainer.cleanup()
