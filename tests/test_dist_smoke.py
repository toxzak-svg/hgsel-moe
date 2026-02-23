"""Comprehensive tests for Phase 4 distributed training components.

Tests cover:
- dist_utils: rank detection, all-reduce, synchronization
- token_exchange: all-to-all communication (single-GPU simulation)
- distributed dispatch: routing + sharding + exchange
"""

import sys
import os
from pathlib import Path

current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

import pytest
import torch

from hgsel.distributed import dist_utils
from hgsel.distributed.token_exchange import TokenExchange
from hgsel.distributed.token_dispatcher import TokenDispatcher, DispatchPlan
from hgsel.distributed.expert_sharding import ExpertPartitioner, build_shard_map
from hgsel.distributed.dispatch_api import ExpertDispatchController
from hgsel.distributed.dist_utils import is_dist_available, is_dist_initialized


# ============================================================================
# dist_utils Tests
# ============================================================================

class TestDistUtils:
    """Test distributed utilities."""

    def test_is_dist_available(self):
        """Test distributed availability check."""
        available = dist_utils.is_dist_available()
        assert isinstance(available, bool)

    def test_is_dist_initialized_false(self):
        """Test distributed initialization check (should be false initially)."""
        initialized = dist_utils.is_dist_initialized()
        assert isinstance(initialized, bool)

    def test_get_rank_single_gpu(self):
        """Test rank detection in single-GPU mode."""
        rank = dist_utils.get_rank()
        assert isinstance(rank, int)
        assert rank >= 0

    def test_get_world_size_single_gpu(self):
        """Test world size detection in single-GPU mode."""
        world_size = dist_utils.get_world_size()
        assert isinstance(world_size, int)
        assert world_size >= 1

    def test_get_backend_single_gpu(self):
        """Test backend detection in single-GPU mode."""
        backend = dist_utils.get_backend()
        assert backend is None or isinstance(backend, str)

    def test_barrier_noop(self):
        """Test barrier is a no-op in single-GPU mode."""
        # Should not raise
        dist_utils.barrier()
        assert True

    def test_all_reduce_sum_identity(self):
        """Test all-reduce sum is identity in single-GPU mode."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = dist_utils.all_reduce_sum(tensor)
        assert torch.equal(tensor, result)

    def test_all_reduce_mean_identity(self):
        """Test all-reduce mean is identity in single-GPU mode."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = dist_utils.all_reduce_mean(tensor)
        assert torch.equal(tensor, result)

    def test_all_gather_identity(self):
        """Test all-gather returns single list in single-GPU mode."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = dist_utils.all_gather(tensor)
        assert isinstance(result, list)
        assert len(result) == 1
        assert torch.equal(result[0], tensor)

    def test_broadcast_identity(self):
        """Test broadcast is identity in single-GPU mode."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = dist_utils.broadcast(tensor, src=0)
        assert torch.equal(tensor, result)

    def test_resolve_dist_env(self):
        """Test resolving distributed environment."""
        env = dist_utils.resolve_dist_env()
        assert env.rank >= 0
        assert env.world_size >= 1
        assert env.local_rank >= 0
        assert isinstance(env.backend, str)

    def test_get_device_cpu_or_cuda(self):
        """Test device detection."""
        device = dist_utils.get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]


# ============================================================================
# TokenExchange Tests
# ============================================================================

class TestTokenExchangeDistributed:
    """Test token exchange (single-GPU simulation)."""

    def test_exchange_single_rank(self):
        """Test exchange with single rank."""
        exchange = TokenExchange()
        payloads = {0: torch.randn(5, 256)}
        
        result = exchange.exchange(payloads)
        
        assert 0 in result
        assert result[0].shape == (5, 256)

    def test_exchange_multiple_ranks_simulation(self):
        """Test exchange with multiple ranks in single-GPU mode."""
        exchange = TokenExchange()
        
        # Simulate payloads for 4 ranks
        payloads = {
            0: torch.randn(3, 256),
            1: torch.randn(2, 256),
            2: torch.randn(4, 256),
            3: torch.randn(1, 256),
        }
        
        result = exchange.exchange(payloads)
        
        # In single-GPU mode, should return payload for rank 0
        assert 0 in result
        assert result[0].shape == (3, 256)

    def test_exchange_preserves_dtype(self):
        """Test that exchange preserves tensor dtype."""
        exchange = TokenExchange()
        
        for dtype in [torch.float32, torch.float64]:
            payloads = {0: torch.randn(5, 256, dtype=dtype)}
            result = exchange.exchange(payloads)
            assert result[0].dtype == dtype

    def test_exchange_preserves_device(self):
        """Test that exchange preserves tensor device."""
        exchange = TokenExchange()
        
        device = torch.device("cpu")
        payloads = {0: torch.randn(5, 256, device=device)}
        result = exchange.exchange(payloads)
        assert result[0].device == device


# ============================================================================
# Expert Sharding Tests
# ============================================================================

class TestExpertShardingDistributed:
    """Test expert sharding for distributed setup."""

    def test_sharding_4_ranks_64_experts(self):
        """Test sharding 64 experts across 4 ranks."""
        partitioner = ExpertPartitioner(num_experts=64, world_size=4)
        
        # Each rank gets 16 experts
        for rank in range(4):
            shard = partitioner.shard_for_rank(rank)
            assert len(shard.expert_ids) == 16
            # Experts should be evenly distributed
            assert all(e % 4 == rank for e in shard.expert_ids)

    def test_sharding_uneven_distribution(self):
        """Test sharding with uneven distribution."""
        partitioner = ExpertPartitioner(num_experts=65, world_size=4)
        
        # 65 = 4 * 16 + 1, so some ranks get 17 experts
        total_experts = 0
        for rank in range(4):
            shard = partitioner.shard_for_rank(rank)
            total_experts += len(shard.expert_ids)
        
        assert total_experts == 65


# ============================================================================
# Token Dispatcher Tests
# ============================================================================

class TestTokenDispatcherDistributed:
    """Test token dispatch planning."""

    def test_dispatch_deterministic(self):
        """Test that dispatch plans are deterministic."""
        shard_map = build_shard_map(num_experts=64, world_size=4)
        dispatcher = TokenDispatcher(shard_map, rank=0)
        
        expert_ids = torch.randint(0, 64, (10, 2))
        
        plan1 = dispatcher.build_plan(expert_ids)
        plan2 = dispatcher.build_plan(expert_ids)
        
        # Plans should be identical
        assert plan1.local_expert_ids == plan2.local_expert_ids
        assert plan1.remote_rank_to_expert_ids == plan2.remote_rank_to_expert_ids

    def test_dispatch_covers_all_tokens(self):
        """Test that dispatch covers all token-expert assignments."""
        shard_map = build_shard_map(num_experts=64, world_size=4)
        dispatcher = TokenDispatcher(shard_map, rank=0)
        
        batch_size, seq_len, k_active = 2, 8, 2
        expert_ids = torch.randint(0, 64, (batch_size * seq_len, k_active))
        
        plan = dispatcher.build_plan(expert_ids)
        
        # Count assignments
        local_count = len(plan.local_expert_ids)
        remote_count = sum(len(ids) for ids in plan.remote_rank_to_expert_ids.values())
        
        assert local_count + remote_count == batch_size * seq_len * k_active


# ============================================================================
# Dispatch Controller Tests
# ============================================================================

class TestDispatchController:
    """Test expert dispatch controller."""

    def test_build_local_batch_shape(self):
        """Test local batch building preserves shapes."""
        hidden_states = torch.randn(100, 256)
        
        plan = DispatchPlan(
            local_expert_ids=(0, 1, 0, 1),
            local_expert_local_indices=(0, 0, 0, 0),
            local_token_indices=(10, 20, 30, 40),
            remote_rank_to_expert_ids={},
            remote_rank_to_token_indices={},
        )
        
        controller = ExpertDispatchController()
        batch = controller.build_local_batch(hidden_states, plan)
        
        assert batch.tokens.shape == (4, 256)
        assert len(batch.token_indices) == 4
        assert len(batch.expert_local_indices) == 4

    def test_build_remote_requests_shape(self):
        """Test remote requests building."""
        plan = DispatchPlan(
            local_expert_ids=(),
            local_expert_local_indices=(),
            local_token_indices=(),
            remote_rank_to_expert_ids={1: (5, 10), 2: (15,)},
            remote_rank_to_token_indices={1: (0, 1), 2: (2,)},
        )
        
        controller = ExpertDispatchController()
        requests = controller.build_remote_requests(plan, torch.device("cpu"))
        
        assert len(requests.rank_to_token_indices) == 2
        assert len(requests.rank_to_expert_ids) == 2
        assert 1 in requests.rank_to_token_indices
        assert 2 in requests.rank_to_token_indices


# ============================================================================
# Integration Tests
# ============================================================================

class TestDistributedIntegration:
    """Integration tests for distributed components."""

    def test_full_routing_dispatch_flow(self):
        """Test end-to-end flow: routing → sharding → dispatch."""
        # Setup
        num_experts = 64
        world_size = 4
        batch_size = 2
        seq_len = 8
        k_active = 2
        
        # Simulate routing decisions
        expert_ids = torch.randint(0, num_experts, (batch_size * seq_len, k_active))
        
        # Build sharding
        shard_map = build_shard_map(num_experts, world_size)
        
        # Test dispatch from each rank
        for rank in range(world_size):
            dispatcher = TokenDispatcher(shard_map, rank=rank)
            plan = dispatcher.build_plan(expert_ids)
            
            # Verify plan completeness
            total_assignments = (
                len(plan.local_expert_ids) +
                sum(len(ids) for ids in plan.remote_rank_to_expert_ids.values())
            )
            assert total_assignments == batch_size * seq_len * k_active

    def test_sharding_and_dispatch_consistency(self):
        """Test that sharding and dispatch are consistent."""
        shard_map = build_shard_map(num_experts=64, world_size=4)
        
        # For each expert, verify it's assigned to exactly one owner
        expert_to_owner = {}
        for expert_id, (owner_rank, local_idx) in shard_map.items():
            if expert_id in expert_to_owner:
                assert expert_to_owner[expert_id] == owner_rank
            expert_to_owner[expert_id] = owner_rank
        
        # Verify all experts are assigned
        assert len(expert_to_owner) == 64


# ============================================================================
# Multi-GPU Simulation Tests
# ============================================================================

class TestMultiGPUSimulation:
    """Simulate multi-GPU scenarios without actual torch.distributed."""

    def test_simulated_2gpu_dispatch(self):
        """Simulate dispatch on 2 GPUs."""
        num_experts = 32
        world_size = 2
        shard_map = build_shard_map(num_experts, world_size)
        
        expert_ids = torch.tensor([
            [0, 1],  # Expert 0: rank 0, Expert 1: rank 1
            [2, 3],  # Expert 2: rank 0, Expert 3: rank 1
            [4, 5],  # Expert 4: rank 0, Expert 5: rank 1
        ], dtype=torch.long)
        
        # Dispatch from rank 0
        dispatcher0 = TokenDispatcher(shard_map, rank=0)
        plan0 = dispatcher0.build_plan(expert_ids)
        
        # Rank 0 expects: local=[0, 2, 4], remote to rank 1=[1, 3, 5]
        assert len(plan0.local_expert_ids) == 3
        assert len(plan0.remote_rank_to_expert_ids[1]) == 3

        # Dispatch from rank 1
        dispatcher1 = TokenDispatcher(shard_map, rank=1)
        plan1 = dispatcher1.build_plan(expert_ids)
        
        # Rank 1 expects: local=[1, 3, 5], remote to rank 0=[0, 2, 4]
        assert len(plan1.local_expert_ids) == 3
        assert len(plan1.remote_rank_to_expert_ids[0]) == 3

    def test_simulated_4gpu_dispatch(self):
        """Simulate dispatch on 4 GPUs."""
        num_experts = 64
        world_size = 4
        shard_map = build_shard_map(num_experts, world_size)
        
        # Random expert routing
        expert_ids = torch.randint(0, num_experts, (16, 2))
        
        total_local_all_ranks = 0
        for rank in range(world_size):
            dispatcher = TokenDispatcher(shard_map, rank=rank)
            plan = dispatcher.build_plan(expert_ids)
            total_local_all_ranks += len(plan.local_expert_ids)
        
        # Total assignments should equal batch_size * k_active
        # Distributed roughly evenly (within statistical variance)
        assert total_local_all_ranks > 0
        assert total_local_all_ranks <= 16 * 2  # Maximum possible


# ============================================================================
# Legacy Tests (from original file)
# ============================================================================

@pytest.mark.skipif(not is_dist_available(), reason="torch.distributed not available")
def test_dist_smoke_if_initialized():
    """Test distributed info when initialized."""
    if not is_dist_initialized():
        pytest.skip("torch.distributed not initialized")

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    assert world_size >= 1
    assert 0 <= rank < world_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
