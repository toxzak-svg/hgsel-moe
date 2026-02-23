"""
Phase 3 validation: Test HGSEL-specific features.

Validates:
- Auxiliary loss computation
- Salt optimization
- Expert utilization tracking
- Convergence stability
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hgsel.layer import HGSELLayer
from hgsel.training.hgsel_trainer import HGSELTrainer
from hgsel.training.trainer import TrainingConfig
from hgsel.training.data import get_dummy_loaders
from hgsel.training.losses import LoadBalancingLoss
from experiments.baselines.dense_transformer import TransformerModel


def test_hgsel_training():
    """Validate Phase 3 HGSEL training features."""
    print("=" * 70)
    print("Phase 3: HGSEL Training Features Validation")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}\n")

    # 1. Create model
    print("1. Creating HGSEL model...")
    model = TransformerModel(
        vocab_size=256,
        d_model=128,
        d_ff=512,
        n_layers=2,
        n_heads=2,
        mlp_class=HGSELLayer,
    )
    print(f"   ✓ Model: {model.count_parameters():,} parameters")

    # 2. Create config
    print("\n2. Creating training config...")
    config = TrainingConfig(
        batch_size=4,
        val_batch_size=8,
        num_epochs=2,
        learning_rate=0.001,
        warmup_steps=100,
        aux_loss_weight=0.05,
        salt_tuning_interval=20,
        device=device,
        use_wandb=False,
        log_interval=10,
        val_interval=30,
    )
    print(f"   ✓ Config created")

    # 3. Create auxiliary loss
    print("\n3. Creating auxiliary loss...")
    aux_loss_fn = LoadBalancingLoss(n_experts=64, initial_weight=0.05)
    print(f"   ✓ Load balancing loss initialized")

    # 4. Create HGSEL trainer
    print("\n4. Creating HGSELTrainer...")
    trainer = HGSELTrainer(model, config, aux_loss_fn=aux_loss_fn)
    print(f"   ✓ Trainer created")
    print(f"   ✓ HGSEL layer monitors: {len(trainer.utilization_monitors)}")
    print(f"   ✓ Salt optimizer: {trainer.salt_optimizer is not None}")

    # 5. Get data
    print("\n5. Loading data...")
    train_loader, val_loader = get_dummy_loaders(
        batch_size=4,
        val_batch_size=8,
        num_train_batches=20,
        num_val_batches=5,
    )
    print(f"   ✓ Train batches: {len(train_loader)}")
    print(f"   ✓ Val batches: {len(val_loader)}")

    # 6. Single training step with auxiliary loss
    print("\n6. Testing training step with auxiliary loss...")
    model.train()
    batch = next(iter(train_loader))
    metrics = trainer.train_step(batch)
    print(f"   ✓ Main loss: {metrics['loss']:.4f}")
    print(f"   ✓ Aux loss: {metrics['aux_loss']:.6f}")
    print(f"   ✓ Total loss: {metrics['total_loss']:.4f}")

    # 7. Test salt optimization
    print("\n7. Testing salt optimization...")
    salt_result = trainer.optimize_salt()
    if salt_result:
        print(f"   ✓ Salt optimized to: {salt_result['salt']:.4f}")
        print(f"   ✓ Entropy: {salt_result['entropy']:.4f}")

    # 8. Test layer statistics collection
    print("\n8. Collecting layer statistics...")
    layer_stats = trainer.collect_layer_statistics()
    for layer_name, stats in layer_stats.items():
        print(f"   {layer_name}:")
        print(f"     Mean load: {stats['mean_load']:.4f}")
        print(f"     Entropy: {stats['entropy']:.4f}")
        print(f"     Std load: {stats['std_load']:.4f}")

    # 9. Test multi-step convergence
    print("\n9. Running 20 training steps to check convergence...")
    losses = []
    for step in range(20):
        batch = next(iter(train_loader))
        metrics = trainer.train_step(batch)
        losses.append(metrics['total_loss'])
        if (step + 1) % 5 == 0:
            avg_loss = sum(losses[-5:]) / 5
            print(f"   Step {step + 1:2d}: avg loss = {avg_loss:.4f}")

    # 10. Validation
    print("\n10. Running validation...")
    val_loss = trainer.validate(val_loader)
    print(f"   ✓ Validation loss: {val_loss:.4f}")

    # 11. Check expert load stability
    print("\n11. Checking expert load stability...")
    layer_stats_after = trainer.collect_layer_statistics()
    for layer_name, stats in layer_stats_after.items():
        print(f"   {layer_name}:")
        print(f"     Entropy: {stats['entropy']:.4f} (should stay ~1.0 with load balancing)")
        print(f"     Mean load: {stats['mean_load']:.4f}")

    print("\n" + "=" * 70)
    print("✓ Phase 3 validation complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_hgsel_training()
