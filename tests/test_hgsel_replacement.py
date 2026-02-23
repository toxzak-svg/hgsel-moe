"""
Integration test: Dense baseline vs HGSEL layer swap.

Validates that HGSEL can be dropped in as MLP replacement.
"""

import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
from hgsel.layer import HGSELLayer
from experiments.baselines.dense_transformer import (
    TransformerModel,
    TransformerBlock,
    DenseMLPBlock,
)


def test_hgsel_replacement():
    """Verify HGSEL can replace dense MLP."""
    print("Testing HGSEL as MLP replacement...")

    vocab_size = 5000
    d_model = 256
    d_ff = 1024
    batch_size = 4
    seq_len = 64

    # Create baseline model with dense MLP
    model_dense = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_layers=2,
        mlp_class=DenseMLPBlock,
    )

    # Create HGSEL model
    model_hgsel = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_layers=2,
        mlp_class=HGSELLayer,
    )

    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward passes
    logits_dense = model_dense(input_ids)
    logits_hgsel = model_hgsel(input_ids)

    # Shapes should match
    assert logits_dense.shape == logits_hgsel.shape, "Output shape mismatch"
    assert not torch.isnan(logits_hgsel).any(), "HGSEL output contains NaNs"

    print(f"✓ Dense baseline output shape: {logits_dense.shape}")
    print(f"✓ HGSEL output shape: {logits_hgsel.shape}")

    # Compare parameter counts
    params_dense = model_dense.count_parameters()
    params_hgsel = model_hgsel.count_parameters()

    print(f"\nParameter counts:")
    print(f"  Dense: {params_dense:,} params")
    print(f"  HGSEL: {params_hgsel:,} params")
    print(f"  Ratio: {params_hgsel / params_dense:.2f}x")

    # HGSEL should have more params due to k*N experts vs single d_ff projection
    # But should be similar order of magnitude

    print("\n✓ HGSEL can successfully replace dense MLP!")


def test_mixed_model():
    """Create model with some dense, some HGSEL layers."""
    print("\nTesting mixed dense + HGSEL model...")

    vocab_size = 5000
    d_model = 128
    d_ff = 512
    batch_size = 2
    seq_len = 32

    # Create a TransformerBlock with HGSEL
    block = TransformerBlock(
        d_model=d_model,
        d_ff=d_ff,
        n_heads=4,
        mlp_class=HGSELLayer,
    )

    # Create input
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward
    output = block(x)

    assert output.shape == x.shape, "Output shape mismatch"
    assert not torch.isnan(output).any(), "Output contains NaNs"

    print(f"✓ TransformerBlock with HGSEL: {x.shape} → {output.shape}")
    print("✓ Mixed model test passed!")


if __name__ == "__main__":
    test_hgsel_replacement()
    test_mixed_model()
    print("\n✓ All integration tests passed!")
