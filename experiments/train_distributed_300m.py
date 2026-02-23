#!/usr/bin/env python
"""
Distributed HGSEL training script for multi-GPU setup.

Usage with torchrun:
    torchrun --nproc_per_node=4 --master_addr=localhost --master_port=12355 \
        experiments/train_distributed_300m.py \
        --batch-size 32 --num-epochs 5 --use-hgsel

Usage with single GPU (no distributed):
    python experiments/train_distributed_300m.py \
        --batch-size 32 --num-epochs 5 --use-hgsel
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Add parent directory to path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from hgsel.distributed import dist_utils
from hgsel.training.distributed_trainer import DistributedTrainer
from hgsel.training.data import DummyDataLoader, LanguageModelDataset
from hgsel.training.trainer import TrainingConfig
from experiments.baselines.dense_transformer import TransformerModel


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Distributed HGSEL Training")
    
    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max training steps")
    
    # Model configuration
    parser.add_argument("--use-hgsel", action="store_true", help="Use HGSEL layer")
    parser.add_argument("--num-experts", type=int, default=64, help="Number of experts")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--d-ff", type=int, default=1024, help="FFN hidden dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    
    # Data
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size")
    parser.add_argument("--num-batches", type=int, default=100, help="Number of batches (dummy data)")
    
    # Loss configuration
    parser.add_argument("--auxiliary-loss-weight", type=float, default=0.001, help="Auxiliary loss weight")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    
    # Distributed configuration
    parser.add_argument("--backend", type=str, default="nccl", help="Distributed backend")
    parser.add_argument("--master-addr", type=str, default="localhost", help="Master address")
    parser.add_argument("--master-port", type=str, default="12355", help="Master port")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--val-interval", type=int, default=1, help="Validation interval")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/phase4_distributed",
                       help="Checkpoint directory")
    
    return parser.parse_args()


def create_model(args):
    """Create model based on arguments."""
    # For now, always use dense baseline
    # TODO: Support HGSEL layers in distributed setup
    model = TransformerModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_seq_length=args.seq_length,
    )
    return model


def create_data_loaders(args, rank, world_size):
    """Create training and validation data loaders."""
    # Use dummy data for now (same across all ranks)
    train_dataset = DummyDataLoader(
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
    )
    
    val_dataset = DummyDataLoader(
        num_batches=max(10, args.num_batches // 10),
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
    )
    
    return train_dataset, val_dataset


def main():
    """Main training loop."""
    args = parse_args()
    
    # Resolve distributed environment
    env = dist_utils.resolve_dist_env()
    rank = env.rank
    world_size = env.world_size
    is_master = rank == 0
    
    if is_master:
        print(f"Starting distributed training: rank={rank}, world_size={world_size}")
        print(f"Batch size: {args.batch_size}, Epochs: {args.num_epochs}")
        print(f"Model: d_model={args.d_model}, num_layers={args.num_layers}")
    
    # Create model
    model = create_model(args)
    
    # Get device
    device = dist_utils.get_device() if world_size > 1 else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)
    
    if is_master:
        print(f"Model on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Create training config
    config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        auxiliary_loss_weight=args.auxiliary_loss_weight,
        clip_grad_norm=args.clip_grad_norm,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Create distributed trainer
    trainer = DistributedTrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        device=device,
    )
    
    # Setup distributed if needed
    if world_size > 1:
        trainer.setup_distributed(
            rank=rank,
            world_size=world_size,
            backend=args.backend,
            master_addr=args.master_addr,
            master_port=args.master_port,
        )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args, rank, world_size)
    
    # Train
    if is_master:
        print("\nStarting training...")
    
    results = trainer.train(train_loader, val_loader)
    
    # Log final results
    if is_master:
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        if results["train_loss"]:
            print(f"Final train loss: {results['train_loss'][-1]:.4f}")
        if results["val_loss"]:
            print(f"Final val loss: {results['val_loss'][-1]:.4f}")
            print(f"Final val perplexity: {results['val_perplexity'][-1]:.2f}")
        print("="*60)
    
    # Cleanup
    trainer.cleanup()


if __name__ == "__main__":
    main()
