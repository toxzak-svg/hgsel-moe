"""Integration test scaffold for distributed token dispatch planning."""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch

from hgsel.distributed.expert_sharding import build_shard_map
from hgsel.distributed.token_dispatcher import TokenDispatcher


def test_dispatch_plan_splits_local_and_remote():
    shard_map = build_shard_map(num_experts=4, world_size=2)
    expert_ids = torch.tensor(
        [
            [0, 1],
            [2, 3],
            [1, 2],
        ],
        dtype=torch.long,
    )

    dispatcher_rank0 = TokenDispatcher(shard_map=shard_map, rank=0)
    plan_rank0 = dispatcher_rank0.build_plan(expert_ids)

    assert plan_rank0.local_expert_ids == (0, 2, 2)
    assert plan_rank0.local_token_indices == (0, 1, 2)
    assert plan_rank0.remote_rank_to_expert_ids[1] == (1, 3, 1)
    assert plan_rank0.remote_rank_to_token_indices[1] == (0, 1, 2)

    dispatcher_rank1 = TokenDispatcher(shard_map=shard_map, rank=1)
    plan_rank1 = dispatcher_rank1.build_plan(expert_ids)

    assert plan_rank1.local_expert_ids == (1, 3, 1)
    assert plan_rank1.local_token_indices == (0, 1, 2)
    assert plan_rank1.remote_rank_to_expert_ids[0] == (0, 2, 2)
    assert plan_rank1.remote_rank_to_token_indices[0] == (0, 1, 2)


def test_dispatch_plan_3d_consistency():
    shard_map = build_shard_map(num_experts=4, world_size=2)
    expert_ids = torch.tensor(
        [
            [[0, 1], [2, 3], [1, 2]],
        ],
        dtype=torch.long,
    )

    dispatcher_rank0 = TokenDispatcher(shard_map=shard_map, rank=0)
    plan_rank0 = dispatcher_rank0.build_plan(expert_ids)

    assert plan_rank0.local_expert_ids == (0, 2, 2)
    assert plan_rank0.local_token_indices == (0, 1, 2)
    assert plan_rank0.remote_rank_to_expert_ids[1] == (1, 3, 1)
    assert plan_rank0.remote_rank_to_token_indices[1] == (0, 1, 2)

    dispatcher_rank1 = TokenDispatcher(shard_map=shard_map, rank=1)
    plan_rank1 = dispatcher_rank1.build_plan(expert_ids)

    assert plan_rank1.local_expert_ids == (1, 3, 1)
    assert plan_rank1.local_token_indices == (0, 1, 2)
    assert plan_rank1.remote_rank_to_expert_ids[0] == (0, 2, 2)
    assert plan_rank1.remote_rank_to_token_indices[0] == (0, 1, 2)
