"""
Quick validation of Phase 2 training setup.

Tests:
- Model creation (HGSEL)
- Data loading
- Training loop
- Single forward pass with loss
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from hgsel.layer import HGSELLayer
from hgsel.training.trainer import Trainer, TrainingConfig
from hgsel.training.data import get_dummy_loaders
from experiments.baselines.dense_transformer import TransformerModel


def test_training_setup():
    """Validate Phase 2 training pipeline."""
    print("=" * 70)
    print("Phase 2 Training Setup Validation")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # 1. Create model
    print("\n1. Creating HGSEL model...")
    model = TransformerModel(
        vocab_size=256,
        d_model=128,
        d_ff=512,
        n_layers=2,
        n_heads=2,
        mlp_class=HGSELLayer,
    )
    n_params = model.count_parameters()
    print(f"   ✓ Model created: {n_params:,} parameters")

    # 2. Create config
    print("\n2. Creating training config...")
    config = TrainingConfig(
        batch_size=4,
        val_batch_size=8,
        num_epochs=1,
        learning_rate=0.001,
        device=device,
        use_wandb=False,
    )
    print(f"   ✓ Config created")
    print(f"     Batch size: {config.batch_size}")
    print(f"     Learning rate: {config.learning_rate}")

    # 3. Get data loaders
    print("\n3. Creating data loaders...")
    train_loader, val_loader = get_dummy_loaders(
        batch_size=config.batch_size,
        val_batch_size=config.val_batch_size,
        num_train_batches=5,
        num_val_batches=2,
    )
    print(f"   ✓ Data loaders created")
    print(f"     Train batches: {len(train_loader)}")
    print(f"     Val batches: {len(val_loader)}")

    # 4. Create trainer
    print("\n4. Creating trainer...")
    trainer = Trainer(model, config)
    print(f"   ✓ Trainer created")

    # 5. Single training step
    print("\n5. Running single training step...")
    model.train()
    batch = next(iter(train_loader))
    metrics = trainer.train_step(batch)
    print(f"   ✓ Training step completed")
    print(f"     Loss: {metrics['loss']:.4f}")
    print(f"     LR: {metrics['lr']:.2e}")

    # 6. Validation step
    print("\n6. Running validation...")
    val_loss = trainer.validate(val_loader)
    print(f"   ✓ Validation completed")
    print(f"     Val loss: {val_loss:.4f}")

    # 7. Multiple steps
    print("\n7. Running 10 training steps...")
    for step in range(10):
        batch = next(iter(train_loader))
        metrics = trainer.train_step(batch)
        if (step + 1) % 5 == 0:
            print(f"   Step {step + 1}: loss={metrics['loss']:.4f}")

    print(f"   ✓ Multi-step training completed")
    print(f"     Global step: {trainer.global_step}")

    # 8. Check HGSEL layer statistics
    print("\n8. Checking HGSEL layer statistics...")
    for name, module in model.named_modules():
        if isinstance(module, HGSELLayer):
            stats = module.get_expert_load_stats()
            print(f"   Layer: {name}")
            print(f"     Mean expert load: {stats['mean_load']:.4f}")
            print(f"     Std expert load: {stats['std_load']:.4f}")
            print(f"     Entropy (normalized): {stats['entropy']:.4f}")
            break

    print("\n" + "=" * 70)
    print("✓ Phase 2 training setup validation complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_training_setup()
