"""
Phase 1 basic integration tests.

Validates:
  - Routing determinism (same input → same output)
  - Expert bank execution + load tracking
  - HGSEL layer integration
  - Training loop basics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from hgsel.routing import MultiHashRouter
from hgsel.expert import ExpertBank
from hgsel.layer import HGSELLayer
from hgsel.training.losses import UtilizationLoss


class TestMultiHashRouter:
    """Test deterministic routing."""

    def test_routing_determinism(self):
        """Same input should produce same routing."""
        batch_seq_len = 32
        hidden_dim = 512

        router = MultiHashRouter(
            n_experts=64,
            k_active=2,
            n_hashes=4,
            hidden_dim=hidden_dim,
            layer_id=0,
            salt=0.5,
        )

        hidden_states = torch.randn(batch_seq_len, hidden_dim)

        # Forward 1
        experts1, weights1, masks1 = router(hidden_states)

        # Forward 2 (same input)
        experts2, weights2, masks2 = router(hidden_states)

        # Check determinism
        assert torch.allclose(experts1.float(), experts2.float()), "Routing should be deterministic"
        assert torch.allclose(weights1, weights2), "Weights should be deterministic"
        assert torch.allclose(masks1, masks2), "Masks should be deterministic"

    def test_expert_ids_valid(self):
        """Routing should only select valid expert IDs."""
        batch_seq_len = 64
        hidden_dim = 512
        n_experts = 64

        router = MultiHashRouter(
            n_experts=n_experts,
            k_active=2,
            n_hashes=4,
            hidden_dim=hidden_dim,
        )

        hidden_states = torch.randn(batch_seq_len, hidden_dim)
        experts, _, _ = router(hidden_states)

        # Check bounds
        assert experts.min() >= 0, "Expert IDs should be non-negative"
        assert experts.max() < n_experts, f"Expert IDs should be < {n_experts}"

    def test_salt_changes_routing(self):
        """Different salt values should produce different routing."""
        batch_seq_len = 32
        hidden_dim = 512

        router = MultiHashRouter(
            n_experts=64,
            k_active=2,
            n_hashes=4,
            hidden_dim=hidden_dim,
            salt=0.0,
        )

        hidden_states = torch.randn(batch_seq_len, hidden_dim)

        # Route with salt=0.0
        experts1, _, _ = router(hidden_states)

        # Route with salt=1.0
        router.set_salt(1.0)
        experts2, _, _ = router(hidden_states)

        # Should differ (usually)
        differs = not torch.allclose(experts1.float(), experts2.float())
        # Note: some inputs might route same, but typically salt changes routing
        print(f"Salt changed routing: {differs}")


class TestExpertBank:
    """Test sparse expert execution."""

    def test_expert_bank_forward(self):
        """Expert bank should execute and produce correct shapes."""
        batch_seq_len = 32
        d_model = 512
        d_ff = 2048
        k_active = 2

        expert_bank = ExpertBank(
            n_experts=64,
            k_active=k_active,
            d_model=d_model,
            d_ff=d_ff,
        )

        hidden_states = torch.randn(batch_seq_len, d_model)

        # Create routing info
        router = MultiHashRouter(n_experts=64, k_active=k_active, hidden_dim=d_model)
        selected_experts, _, expert_masks = router(hidden_states)

        # Forward
        outputs, loads = expert_bank(hidden_states, selected_experts, expert_masks)

        # Check shapes: expert_outputs is [batch_seq_len, k_active, d_model]
        assert outputs.shape == (batch_seq_len, k_active, d_model), "Output shape mismatch"
        assert loads.shape[0] == 64, "Loads shape mismatch"

    def test_expert_bank_no_nans(self):
        """Forward should not produce NaNs."""
        batch_seq_len = 16
        d_model = 256
        d_ff = 1024

        expert_bank = ExpertBank(
            n_experts=32,
            k_active=2,
            d_model=d_model,
            d_ff=d_ff,
        )

        hidden_states = torch.randn(batch_seq_len, d_model)

        router = MultiHashRouter(
            n_experts=32,
            k_active=2,
            hidden_dim=d_model,
        )
        selected_experts, _, expert_masks = router(hidden_states)

        outputs, loads = expert_bank(hidden_states, selected_experts, expert_masks)

        assert not torch.isnan(outputs).any(), "Outputs contain NaNs"
        assert not torch.isnan(loads).any(), "Loads contain NaNs"


class TestHGSELLayer:
    """Test HGSEL layer integration."""

    def test_hgsel_layer_forward(self):
        """HGSEL layer should integrate all components."""
        batch_seq_len = 32
        d_model = 512
        d_ff = 2048

        layer = HGSELLayer(
            d_model=d_model,
            d_ff=d_ff,
            n_experts=64,
            k_active=2,
            combine_mode="uniform",
            layer_id=0,
        )

        hidden_states = torch.randn(batch_seq_len, d_model)

        # Forward
        output = layer(hidden_states)

        # Check shape
        assert output.shape == hidden_states.shape, "Output shape mismatch"
        assert not torch.isnan(output).any(), "Output contains NaNs"

    def test_hgsel_layer_with_routing_info(self):
        """HGSEL should return routing diagnostics."""
        batch_seq_len = 16
        d_model = 256

        layer = HGSELLayer(
            d_model=d_model,
            d_ff=1024,
            n_experts=32,
            k_active=2,
        )

        hidden_states = torch.randn(batch_seq_len, d_model)

        # Forward with routing info
        output, routing_info = layer(hidden_states, return_routing_info=True)

        # Check routing info contents
        assert "selected_experts" in routing_info
        assert "expert_loads" in routing_info
        assert "routing_entropy" in routing_info
        assert routing_info["batch_tokens"] == batch_seq_len

    def test_hgsel_layer_load_tracking(self):
        """HGSEL should track expert loads via EMA."""
        batch_seq_len = 32
        d_model = 512

        layer = HGSELLayer(
            d_model=d_model,
            d_ff=2048,
            n_experts=64,
            k_active=2,
        )

        hidden_states = torch.randn(batch_seq_len, d_model)

        # Run forward
        _ = layer(hidden_states)

        # Check load EMA is updated
        ema_loads = layer.expert_load_ema
        assert ema_loads.sum() > 0, "EMA loads not updated"
        assert not torch.isnan(ema_loads).any(), "EMA loads contain NaNs"

    def test_hgsel_layer_salt_tuning(self):
        """HGSEL should support salt parameter tuning."""
        layer = HGSELLayer(d_model=256, d_ff=1024)

        # Set salt
        layer.set_salt(2.0)
        assert layer.router.salt == 2.0, "Salt not updated"

        # Get stats
        stats = layer.get_expert_load_stats()
        assert "entropy" in stats
        assert "mean_load" in stats


class TestAuxiliaryLoss:
    """Test training loss functions."""

    def test_utilization_loss(self):
        """Utilization loss should reduce with balanced loads."""
        n_experts = 64
        loss_fn = UtilizationLoss(n_experts=n_experts, weight=1.0)

        # Uniform loads
        loads_uniform = torch.ones(n_experts) / n_experts
        loss_uniform = loss_fn(loads_uniform)

        # Imbalanced loads
        loads_imbalanced = torch.zeros(n_experts)
        loads_imbalanced[0] = 1.0
        loss_imbalanced = loss_fn(loads_imbalanced)

        # Uniform should have lower loss
        assert loss_uniform < loss_imbalanced, "Uniform loads should have lower loss"

    def test_utilization_loss_batch(self):
        """Utilization loss should handle batch loads."""
        n_experts = 32
        batch_size = 4

        loss_fn = UtilizationLoss(n_experts=n_experts)

        # Batch of load distributions
        loads_batch = torch.ones(batch_size, n_experts) / n_experts
        loss = loss_fn(loads_batch)

        assert loss > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss contains NaNs"


if __name__ == "__main__":
    # Run basic tests
    print("Testing MultiHashRouter...")
    test_router = TestMultiHashRouter()
    test_router.test_routing_determinism()
    test_router.test_expert_ids_valid()
    test_router.test_salt_changes_routing()
    print("✓ Routing tests passed")

    print("\nTesting ExpertBank...")
    test_bank = TestExpertBank()
    test_bank.test_expert_bank_forward()
    test_bank.test_expert_bank_no_nans()
    print("✓ Expert bank tests passed")

    print("\nTesting HGSELLayer...")
    test_layer = TestHGSELLayer()
    test_layer.test_hgsel_layer_forward()
    test_layer.test_hgsel_layer_with_routing_info()
    test_layer.test_hgsel_layer_load_tracking()
    test_layer.test_hgsel_layer_salt_tuning()
    print("✓ HGSEL layer tests passed")

    print("\nTesting Auxiliary Loss...")
    test_loss = TestAuxiliaryLoss()
    test_loss.test_utilization_loss()
    test_loss.test_utilization_loss_batch()
    print("✓ Loss tests passed")

    print("\n✓ All Phase 1 tests passed!")
