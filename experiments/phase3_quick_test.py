"""
Phase 3 Quick Test: Abbreviated convergence experiment for fast iteration.

Reduces model size and epoch count to enable quick CPU testing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hgsel.layer import HGSELLayer
from hgsel.training.hgsel_trainer import HGSELTrainer
from hgsel.training.trainer import Trainer, TrainingConfig
from hgsel.training.data import get_dummy_loaders
from hgsel.training.losses import LoadBalancingLoss
from experiments.baselines.dense_transformer import TransformerModel


def compare_models(num_epochs=1):
    """Quick comparison of Dense vs HGSEL training."""
    print("=" * 70)
    print("Phase 3 Quick Test: Dense vs HGSEL Convergence")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: 4 (CPU-friendly)\n")

    # Reduce model size for faster CPU training
    config = TrainingConfig(
        batch_size=4,
        val_batch_size=8,
        num_epochs=num_epochs,
        learning_rate=0.001,
        warmup_steps=2,  # Minimal warmup
        gradient_accumulation_steps=1,
    )

    # Create data loaders (small, fast)
    train_loader, val_loader = get_dummy_loaders(
        num_train_batches=10,  # Small for quick test
        num_val_batches=3,
        batch_size=4,
        seq_len=32,
        vocab_size=256,
    )

    results = {}

    # Test 1: Dense Baseline
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Dense Baseline")
    print("=" * 70)

    model_dense = TransformerModel(
        vocab_size=256,
        d_model=64,  # Smaller for speed
        d_ff=256,
        n_layers=2,
        n_heads=2,
        mlp_class=None,  # Use default dense MLP
    )
    model_dense.to(device)
    print(f"Dense model: {model_dense.count_parameters():,} parameters")

    trainer_dense = Trainer(
        model=model_dense,
        config=config,
    )

    print("\nTraining dense model...")
    trainer_dense.train(train_loader, val_loader)
    dense_best_loss = trainer_dense.best_val_loss
    results['dense'] = dense_best_loss
    print(f"Dense best validation loss: {dense_best_loss:.4f}")

    # Test 2: HGSEL
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: HGSEL Model")
    print("=" * 70)

    model_hgsel = TransformerModel(
        vocab_size=256,
        d_model=64,  # Same as dense for fair comparison
        d_ff=256,
        n_layers=2,
        n_heads=2,
        mlp_class=HGSELLayer,
    )
    model_hgsel.to(device)
    print(f"HGSEL model: {model_hgsel.count_parameters():,} parameters")

    aux_loss_fn = LoadBalancingLoss(
        n_experts=64,
        initial_weight=0.01,
        strategy="utilization"
    )

    trainer_hgsel = HGSELTrainer(
        model=model_hgsel,
        config=config,
        aux_loss_fn=aux_loss_fn,
    )

    print("\nTraining HGSEL model with load balancing...")
    trainer_hgsel.train(train_loader, val_loader)
    hgsel_best_loss = trainer_hgsel.best_val_loss
    results['hgsel'] = hgsel_best_loss
    print(f"HGSEL best validation loss: {hgsel_best_loss:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 3 QUICK TEST RESULTS")
    print("=" * 70)
    print(f"\nDense baseline:")
    print(f"  Best validation loss: {dense_best_loss:.4f}")
    print(f"\nHGSEL with load balancing:")
    print(f"  Best validation loss: {hgsel_best_loss:.4f}")

    delta = dense_best_loss - hgsel_best_loss
    pct_change = (delta / dense_best_loss) * 100 if dense_best_loss != 0 else 0
    print(f"\nDifference (HGSEL vs Dense):")
    print(f"  Delta: {delta:+.4f}")
    print(f"  Percentage: {pct_change:+.2f}%")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    if abs(pct_change) < 2:
        print("✓ HGSEL converges at similar quality to Dense")
        print("  → Phase 3 architecture is sound")
        print("  → Proceed to Phase 4 (multi-GPU distribution)")
    elif pct_change < 0:
        print("✓ HGSEL outperforms Dense")
        print("  → Load balancing helps convergence")
        print("  → Phase 3 is working as intended")
    else:
        print("⚠ HGSEL slightly underperforms Dense")
        print("  → Auxiliary loss weight may need tuning")
        print("  → Consider Phase 3 iteration before Phase 4")

    return results


if __name__ == "__main__":
    results = compare_models(num_epochs=1)
    print("\nDone!")
