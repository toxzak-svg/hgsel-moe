"""Unit tests for dispatch API batch building."""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
import pytest

from hgsel.distributed.dispatch_api import ExpertDispatchController, LocalDispatchBatch
from hgsel.distributed.dispatch_pipeline import DispatchPipeline
from hgsel.distributed.expert_sharding import (
    ExpertPartitioner,
    build_shard_map,
    ExpertShardMetadata,
)
from hgsel.distributed.token_dispatcher import TokenDispatcher, DispatchPlan
from hgsel.distributed.token_exchange import TokenExchange


def test_dispatch_pipeline_empty_local_batch():
    shard_map = build_shard_map(num_experts=2, world_size=2)
    pipeline = DispatchPipeline(shard_map=shard_map, rank=0)

    hidden_states = torch.randn(2, 4)
    expert_ids = torch.tensor([[1], [1]], dtype=torch.long)

    result = pipeline.build(hidden_states, expert_ids)

    assert result.local_batch.tokens.shape == (0, 4)
    assert result.local_batch.token_indices.numel() == 0
    assert result.remote_requests.rank_to_token_indices[1].tolist() == [0, 1]


def test_dispatch_pipeline_local_batch_content():
    shard_map = build_shard_map(num_experts=2, world_size=2)
    pipeline = DispatchPipeline(shard_map=shard_map, rank=0)

    hidden_states = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    expert_ids = torch.tensor([[0], [1]], dtype=torch.long)

    result = pipeline.build(hidden_states, expert_ids)

    assert result.local_batch.token_indices.tolist() == [0]
    assert result.local_batch.tokens.tolist() == [[1.0, 1.0]]


def test_dispatch_pipeline_accepts_3d_expert_ids():
    shard_map = build_shard_map(num_experts=2, world_size=2)
    pipeline = DispatchPipeline(shard_map=shard_map, rank=1)

    hidden_states = torch.randn(2, 3)
    expert_ids = torch.tensor([[[1], [0]]], dtype=torch.long)

    result = pipeline.build(hidden_states, expert_ids)

    assert result.local_batch.token_indices.tolist() == [0]
    assert result.remote_requests.rank_to_token_indices[0].tolist() == [1]

# ============================================================================
# Expert Sharding Tests
# ============================================================================

class TestExpertSharding:
    """Test expert partitioning across ranks."""

    def test_expert_partitioner_basic(self):
        """Verify round-robin expert distribution."""
        partitioner = ExpertPartitioner(num_experts=64, world_size=4)
        
        # Each rank should get 16 experts
        for rank in range(4):
            shard = partitioner.shard_for_rank(rank)
            assert len(shard.expert_ids) == 16
            assert shard.rank == rank
            assert shard.world_size == 4
        
        # All experts should be covered exactly once
        all_expert_ids = set()
        for rank in range(4):
            shard = partitioner.shard_for_rank(rank)
            all_expert_ids.update(shard.expert_ids)
        
        assert all_expert_ids == set(range(64))

    def test_expert_partitioner_owner_rank(self):
        """Verify owner_rank returns correct rank."""
        partitioner = ExpertPartitioner(num_experts=64, world_size=4)
        
        for expert_id in range(64):
            owner_rank = partitioner.owner_rank(expert_id)
            assert owner_rank == expert_id % 4
            
            # Verify the shard for owner_rank contains this expert
            shard = partitioner.shard_for_rank(owner_rank)
            assert expert_id in shard.expert_ids

    def test_expert_partitioner_local_index(self):
        """Verify local_index maps expert_id to position within shard."""
        partitioner = ExpertPartitioner(num_experts=64, world_size=4)
        
        for rank in range(4):
            shard = partitioner.shard_for_rank(rank)
            for local_idx, expert_id in enumerate(shard.expert_ids):
                assert shard.local_index[expert_id] == local_idx

    def test_shard_map(self):
        """Test build_shard_map function."""
        shard_map = build_shard_map(num_experts=64, world_size=4)
        
        assert len(shard_map) == 64
        
        for expert_id in range(64):
            rank, local_idx = shard_map[expert_id]
            assert rank == expert_id % 4
            assert 0 <= local_idx < 16
        
        # Verify all (rank, local_idx) pairs are unique
        pairs = set(shard_map.values())
        assert len(pairs) == 64

    def test_expert_partitioner_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            ExpertPartitioner(num_experts=-1, world_size=4)
        
        with pytest.raises(ValueError):
            ExpertPartitioner(num_experts=64, world_size=-1)
        
        partitioner = ExpertPartitioner(num_experts=64, world_size=4)
        with pytest.raises(ValueError):
            partitioner.shard_for_rank(5)
        
        with pytest.raises(ValueError):
            partitioner.owner_rank(-1)


# ============================================================================
# Token Dispatcher Tests
# ============================================================================

class TestTokenDispatcher:
    """Test token dispatch planning for expert-parallel execution."""

    def test_token_dispatcher_local_only(self):
        """Test dispatch plan when all experts are local."""
        shard_map = build_shard_map(num_experts=64, world_size=4)
        dispatcher = TokenDispatcher(shard_map, rank=0)
        
        # Create expert IDs all owned by rank 0
        # Rank 0 owns experts 0, 4, 8, ..., 60
        expert_ids = torch.tensor([
            [[0, 4]],  # Batch 1, token 1: experts 0, 4 (both rank 0)
        ], dtype=torch.long)
        
        plan = dispatcher.build_plan(expert_ids)
        
        assert len(plan.local_expert_ids) == 2
        assert len(plan.remote_rank_to_expert_ids) == 0
        assert len(plan.local_token_indices) == 2

    def test_token_dispatcher_remote_only(self):
        """Test dispatch plan when all experts are remote."""
        shard_map = build_shard_map(num_experts=64, world_size=4)
        dispatcher = TokenDispatcher(shard_map, rank=0)
        
        # Create expert IDs all owned by rank 1
        # Rank 1 owns experts 1, 5, 9, ..., 61
        expert_ids = torch.tensor([
            [[1, 5]],  # Batch 1, token 1: experts 1, 5 (both rank 1)
        ], dtype=torch.long)
        
        plan = dispatcher.build_plan(expert_ids)
        
        assert len(plan.local_expert_ids) == 0
        assert len(plan.remote_rank_to_expert_ids) == 1
        assert 1 in plan.remote_rank_to_expert_ids
        assert len(plan.remote_rank_to_expert_ids[1]) == 2

    def test_token_dispatcher_mixed(self):
        """Test dispatch plan with mixed local and remote experts."""
        shard_map = build_shard_map(num_experts=64, world_size=4)
        dispatcher = TokenDispatcher(shard_map, rank=0)
        
        # Batch of 2 tokens, each with 2 experts
        # Token 1: experts 0 (local), 1 (remote rank 1)
        # Token 2: experts 2 (remote rank 2), 3 (remote rank 3)
        expert_ids = torch.tensor([
            [[0, 1], [2, 3]],  # Shape: [batch=1, seq=2, k=2]
        ], dtype=torch.long)
        
        plan = dispatcher.build_plan(expert_ids)
        
        assert len(plan.local_expert_ids) == 1  # Only expert 0
        assert len(plan.remote_rank_to_expert_ids) == 3  # Ranks 1, 2, 3


# ============================================================================
# Token Exchange Tests
# ============================================================================

class TestTokenExchange:
    """Test all-to-all token exchange."""

    def test_exchange_single_gpu_mode(self):
        """Test token exchange in single-GPU fallback mode."""
        exchange = TokenExchange()
        
        payloads = {0: torch.randn(5, 256)}  # 5 tokens to rank 0
        
        received = exchange.exchange(payloads)
        
        assert 0 in received
        assert received[0].shape == (5, 256)
        assert torch.equal(received[0], payloads[0])

    def test_exchange_empty_payloads_error(self):
        """Test that empty payloads raise an error."""
        exchange = TokenExchange()
        
        payloads = {}
        
        with pytest.raises(ValueError):
            exchange.exchange(payloads)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests for distributed dispatch."""

    def test_full_dispatch_pipeline(self):
        """Test complete dispatch flow: routing → local/remote split → exchange."""
        # Setup
        batch_size = 2
        seq_len = 4
        d_model = 256
        num_experts = 64
        world_size = 4
        
        hidden_states = torch.randn(batch_size * seq_len, d_model)
        
        # Simulate routing: [batch*seq, k_active]
        expert_ids = torch.randint(0, num_experts, (batch_size * seq_len, 2))
        
        # Build shard map and dispatcher for rank 0
        shard_map = build_shard_map(num_experts, world_size)
        dispatcher = TokenDispatcher(shard_map, rank=0)
        plan = dispatcher.build_plan(expert_ids)
        
        # Build dispatch controller
        controller = ExpertDispatchController()
        local_batch = controller.build_local_batch(hidden_states, plan)
        remote_requests = controller.build_remote_requests(plan, torch.device("cpu"))
        
        # Verify plan splits correctly
        total_slots = len(plan.local_expert_ids) + sum(
            len(ids) for ids in plan.remote_rank_to_expert_ids.values()
        )
        assert total_slots == batch_size * seq_len * 2  # k_active = 2

    def test_dispatch_with_multiple_ranks(self):
        """Test dispatch from multiple ranks' perspective."""
        num_experts = 64
        world_size = 4
        shard_map = build_shard_map(num_experts, world_size)
        
        # Create same expert IDs but dispatch from each rank
        expert_ids = torch.tensor([
            [0, 4, 8, 12],  # All local to rank 0
            [1, 5, 9, 13],  # All local to rank 1
            [2, 6, 10, 14],  # All local to rank 2
            [3, 7, 11, 15],  # All local to rank 3
        ], dtype=torch.long)
        
        for rank in range(world_size):
            dispatcher = TokenDispatcher(shard_map, rank=rank)
            plan = dispatcher.build_plan(expert_ids[[rank]])
            
            # For rank i, all experts should be local
            assert len(plan.local_expert_ids) == 4
            assert len(plan.remote_rank_to_expert_ids) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])