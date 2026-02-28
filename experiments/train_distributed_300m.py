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
import json
import os
import sys
from pathlib import Path

import torch

# Add parent directory to path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from hgsel.distributed import dist_utils
from hgsel.layer import HGSELLayer
from hgsel.training.distributed_trainer import DistributedTrainer
from hgsel.training.dist_data import create_distributed_dummy_loaders
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
    parser.add_argument("--backend", type=str, default="auto", help="Distributed backend (auto|nccl|gloo)")
    parser.add_argument("--master-addr", type=str, default="localhost", help="Master address")
    parser.add_argument("--master-port", type=str, default="12355", help="Master port")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--val-interval", type=int, default=1, help="Validation interval")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/phase4_distributed",
                       help="Checkpoint directory")
    parser.add_argument("--results-json", type=str, default="",
                       help="Optional path to write training results JSON (rank 0 only)")
    
    return parser.parse_args()


def create_model(args):
    """Create model based on arguments."""
    mlp_class = HGSELLayer if args.use_hgsel else None
    mlp_kwargs = None
    if args.use_hgsel:
        mlp_kwargs = {"n_experts": args.num_experts}

    model = TransformerModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.num_heads,
        d_ff=args.d_ff,
        n_layers=args.num_layers,
        max_seq_len=args.seq_length,
        mlp_class=mlp_class,
        mlp_kwargs=mlp_kwargs,
    )
    return model


def create_data_loaders(args, rank, world_size):
    """Create training and validation data loaders."""
    loader_info = create_distributed_dummy_loaders(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
        num_train_batches=args.num_batches,
        num_val_batches=max(10, args.num_batches // 10),
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        pin_memory=torch.cuda.is_available(),
    )
    return loader_info


def resolve_backend(args, device: torch.device) -> str:
    if args.backend != "auto":
        return args.backend
    return "nccl" if device.type == "cuda" else "gloo"


def main():
    """Main training loop."""
    args = parse_args()

    # Resolve env first; initialize process group before model/device selection in multi-rank runs.
    env = dist_utils.resolve_dist_env()
    if env.world_size > 1 and not dist_utils.is_dist_initialized():
        backend = args.backend if args.backend != "auto" else ("nccl" if torch.cuda.is_available() else "gloo")
        env = dist_utils.resolve_dist_env(default_backend=backend)
        dist_utils.init_distributed(env)

    rank = dist_utils.get_rank()
    world_size = dist_utils.get_world_size()
    is_master = rank == 0

    device = dist_utils.get_device() if world_size > 1 else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    if is_master:
        print(f"Starting distributed training: rank={rank}, world_size={world_size}")
        print(f"Mode: {'DDP-only parity (HGSEL local dispatch)' if args.use_hgsel else 'DDP-only parity (dense baseline)'}")
        print(f"Batch size (per-rank): {args.batch_size}, Epochs: {args.num_epochs}")
        print(f"Model: d_model={args.d_model}, num_layers={args.num_layers}, use_hgsel={args.use_hgsel}")
        print(f"Device: {device}, Backend: {dist_utils.get_backend() or resolve_backend(args, device)}")

    # Create model after device/backend are resolved
    model = create_model(args).to(device)

    if is_master:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
        use_wandb=False,
    )

    # Create optimizer after model is on correct device.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Create distributed trainer (auto_init_from_env handles torchrun envs)
    trainer = DistributedTrainer(
        model=model,
        config=config,
        optimizer=optimizer,
        device=device,
        auto_init_from_env=False,  # Already handled above.
    )

    # Ensure wrapper is active in non-torchrun manual setups as well.
    if world_size > 1:
        trainer.setup_distributed(
            rank=rank,
            world_size=world_size,
            backend=resolve_backend(args, device),
            master_addr=args.master_addr,
            master_port=args.master_port,
        )

    # Create data loaders
    loader_info = create_data_loaders(args, rank, world_size)
    train_loader, val_loader = loader_info.train_loader, loader_info.val_loader

    if is_master:
        print(
            f"Global batch size check: {loader_info.global_batch_size} = "
            f"{loader_info.per_rank_batch_size} x {loader_info.world_size}"
        )

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

        if args.results_json:
            out_path = Path(args.results_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "metadata": {
                    "script": "experiments/train_distributed_300m.py",
                    "rank": rank,
                    "world_size": world_size,
                    "device": str(device),
                    "backend": dist_utils.get_backend(),
                    "use_hgsel": bool(args.use_hgsel),
                    "global_batch_size": int(loader_info.global_batch_size),
                    "per_rank_batch_size": int(loader_info.per_rank_batch_size),
                    "seed": int(args.seed),
                },
                "config": vars(args),
                "results": results,
            }
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Wrote results JSON: {out_path}")
        print("="*60)
    
    # Cleanup
    trainer.cleanup()


if __name__ == "__main__":
    main()
