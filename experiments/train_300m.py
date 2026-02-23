"""
Phase 2 training script: Single-GPU training of HGSEL models.

Example usage:
    python experiments/train_300m.py --config experiments/configs/baseline_300m.yaml
    
    Or with command-line overrides:
    python experiments/train_300m.py \
        --batch-size 32 \
        --num-epochs 5 \
        --learning-rate 0.001
"""

import sys
import os
import argparse
from pathlib import Path

# Add hgsel to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("⚠ PyYAML not installed, config files will be skipped")

import torch
import torch.nn as nn

from hgsel.layer import HGSELLayer
from hgsel.training.trainer import Trainer, TrainingConfig
from hgsel.training.data import get_dummy_loaders, create_wiki_dataset
from hgsel.training.losses import LoadBalancingLoss
from experiments.baselines.dense_transformer import TransformerModel, DenseMLPBlock


def load_config(config_path: str = None, **kwargs):
    """Load config from YAML or use defaults."""
    config_dict = {}

    if config_path and Path(config_path).exists():
        if HAS_YAML:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f) or {}
        else:
            print(f"⚠ Config file {config_path} specified but PyYAML not installed")

    # Override with command-line args
    config_dict.update({k: v for k, v in kwargs.items() if v is not None})

    return config_dict


def create_model(
    vocab_size: int = 256,
    d_model: int = 512,
    d_ff: int = 2048,
    n_layers: int = 6,
    n_heads: int = 8,
    use_hgsel: bool = True,
) -> nn.Module:
    """Create a Transformer model with HGSEL or dense MLPs."""
    mlp_class = HGSELLayer if use_hgsel else DenseMLPBlock

    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_class=mlp_class,
    )

    n_params = model.count_parameters()
    print(f"Created model: {n_params:,} parameters")
    print(f"  Model type: {'HGSEL' if use_hgsel else 'Dense'}")
    print(f"  Embedding: {d_model}")
    print(f"  FFN hidden: {d_ff}")
    print(f"  Layers: {n_layers}")
    print(f"  Heads: {n_heads}")

    return model


def get_data_loaders(
    batch_size: int = 32,
    val_batch_size: int = 64,
    use_wiki: bool = False,
    seq_len: int = 128,
):
    """Get training and validation data loaders."""
    if use_wiki:
        print("Loading WikiText-2 dataset...")
        train_loader, val_loader = create_wiki_dataset(seq_len=seq_len)
    else:
        print("Using dummy data loaders...")
        train_loader, val_loader = get_dummy_loaders(
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            seq_len=seq_len,
        )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Phase 2 HGSEL Training")

    # Config file
    parser.add_argument("--config", type=str, default=None, help="Config YAML file")

    # Model params
    parser.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size")
    parser.add_argument("--d-model", type=int, default=256, help="Model embedding dimension")
    parser.add_argument("--d-ff", type=int, default=1024, help="FFN hidden dimension")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--use-hgsel", action="store_true", help="Use HGSEL instead of dense MLPs")

    # Training params
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=32, help="Validation batch size")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps")

    # Load balancing (HGSEL specific)
    parser.add_argument("--aux-loss-weight", type=float, default=0.05, help="Auxiliary loss weight")
    parser.add_argument("--aux-loss-schedule", type=str, default="constant", help="Loss schedule")
    parser.add_argument("--salt-tuning-interval", type=int, default=100, help="Salt tuning interval")

    # Data
    parser.add_argument("--use-wiki", action="store_true", help="Use WikiText dataset")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")

    # Logging
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval")
    parser.add_argument("--val-interval", type=int, default=100, help="Validation interval")
    parser.add_argument("--save-interval", type=int, default=500, help="Checkpoint save interval")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases logging")

    # Other
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")

    args = parser.parse_args()

    # Print config
    print("=" * 70)
    print("Phase 2: Single-GPU HGSEL Training")
    print("=" * 70)

    # Create model
    model = create_model(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        use_hgsel=args.use_hgsel,
    )

    # Create config
    config = TrainingConfig(
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        aux_loss_weight=args.aux_loss_weight,
        aux_loss_schedule=args.aux_loss_schedule,
        salt_tuning_interval=args.salt_tuning_interval,
        device=args.device,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
    )

    print(f"\nTraining config:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")

    # Get data
    print("\n" + "=" * 70)
    train_loader, val_loader = get_data_loaders(
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        use_wiki=args.use_wiki,
        seq_len=args.seq_len,
    )

    print(f"✓ Data loaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create auxiliary loss (if HGSEL)
    aux_loss_fn = None
    if args.use_hgsel:
        aux_loss_fn = LoadBalancingLoss(n_experts=64, initial_weight=args.aux_loss_weight)
        print(f"✓ Auxiliary loss initialized with weight {args.aux_loss_weight}")

    # Create trainer
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")

    trainer = Trainer(model, config, aux_loss_fn=aux_loss_fn)

    # Train
    trainer.train(train_loader, val_loader)

    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print(f"  Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"  Checkpoints saved to: {config.checkpoint_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
