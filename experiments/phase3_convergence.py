"""
Phase 3 Experiment: Real training run with convergence tracking.

This script trains HGSEL and Dense baselines for comparison.

Usage:
    python experiments/phase3_convergence.py --use-hgsel --epochs 5
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
import torch.nn as nn
from pathlib import Path

from hgsel.layer import HGSELLayer
from hgsel.training.hgsel_trainer import HGSELTrainer
from hgsel.training.trainer import Trainer, TrainingConfig
from hgsel.training.data import get_dummy_loaders
from hgsel.training.losses import LoadBalancingLoss
from experiments.baselines.dense_transformer import TransformerModel, DenseMLPBlock


def run_experiment(
    use_hgsel: bool = True,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    aux_loss_weight: float = 0.05,
    device: str = "cuda",
):
    """Run a single training experiment."""
    print("=" * 80)
    print(f"Phase 3 Experiment: {'HGSEL' if use_hgsel else 'Dense Baseline'} Training")
    print("=" * 80)

    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        device = "cpu"

    # Create model
    model = TransformerModel(
        vocab_size=256,
        d_model=256,
        d_ff=1024,
        n_layers=4,
        n_heads=4,
        mlp_class=HGSELLayer if use_hgsel else DenseMLPBlock,
    )

    n_params = model.count_parameters()
    print(f"\nModel: {n_params:,} parameters")
    print(f"Type: {'HGSEL' if use_hgsel else 'Dense'}")

    # Create config
    config = TrainingConfig(
        batch_size=batch_size,
        val_batch_size=batch_size * 2,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        warmup_steps=200,
        aux_loss_weight=aux_loss_weight if use_hgsel else 0.0,
        aux_loss_schedule="constant",
        salt_tuning_interval=50,
        device=device,
        log_interval=10,
        val_interval=50,
        save_interval=500,
        use_wandb=False,
        checkpoint_dir=f"./checkpoints/phase3_{'hgsel' if use_hgsel else 'dense'}",
    )

    # Get data
    train_loader, val_loader = get_dummy_loaders(
        batch_size=config.batch_size,
        val_batch_size=config.val_batch_size,
        num_train_batches=200,
        num_val_batches=50,
    )

    print(f"\nData: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # Create auxiliary loss for HGSEL
    aux_loss_fn = None
    if use_hgsel:
        aux_loss_fn = LoadBalancingLoss(n_experts=64, initial_weight=aux_loss_weight)
        print(f"Auxiliary loss weight: {aux_loss_weight}")

    # Create trainer
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    if use_hgsel:
        trainer = HGSELTrainer(model, config, aux_loss_fn=aux_loss_fn)
    else:
        trainer = Trainer(model, config, aux_loss_fn=aux_loss_fn)

    # Train
    trainer.train(train_loader, val_loader)

    # Summary
    print("\n" + "=" * 80)
    print(f"Training Summary: {'HGSEL' if use_hgsel else 'Dense'}")
    print("=" * 80)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Total training steps: {trainer.global_step}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Convergence Experiment")
    parser.add_argument("--use-hgsel", action="store_true", help="Train HGSEL model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--aux-loss-weight", type=float, default=0.05, help="Aux loss weight")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")

    args = parser.parse_args()

    # Run HGSEL
    print("\n\n" + "=" * 80)
    print("EXPERIMENT 1: HGSEL MODEL")
    print("=" * 80 + "\n")
    trainer_hgsel = run_experiment(
        use_hgsel=True,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        aux_loss_weight=args.aux_loss_weight,
        device=args.device,
    )

    # Run Dense baseline
    print("\n\n" + "=" * 80)
    print("EXPERIMENT 2: DENSE BASELINE")
    print("=" * 80 + "\n")
    trainer_dense = run_experiment(
        use_hgsel=False,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )

    # Comparison
    print("\n\n" + "=" * 80)
    print("COMPARISON: HGSEL vs Dense")
    print("=" * 80)
    print(f"HGSEL best val loss: {trainer_hgsel.best_val_loss:.4f}")
    print(f"Dense best val loss: {trainer_dense.best_val_loss:.4f}")
    print(f"Difference: {trainer_dense.best_val_loss - trainer_hgsel.best_val_loss:+.4f}")
    if trainer_hgsel.best_val_loss < trainer_dense.best_val_loss:
        improvement = (1 - trainer_hgsel.best_val_loss / trainer_dense.best_val_loss) * 100
        print(f"HGSEL improvement: {improvement:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
