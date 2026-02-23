"""Unit tests for expert sharding utilities."""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from hgsel.distributed.expert_sharding import ExpertPartitioner, build_shard_map


def test_partitioner_round_robin():
    partitioner = ExpertPartitioner(num_experts=6, world_size=3)
    shard = partitioner.shard_for_rank(1)

    assert shard.expert_ids == (1, 4)
    assert shard.local_index[1] == 0
    assert shard.local_index[4] == 1

    assert partitioner.owner_rank(4) == 1


def test_build_shard_map():
    shard_map = build_shard_map(num_experts=4, world_size=2)
    assert shard_map[0] == (0, 0)
    assert shard_map[1] == (1, 0)
    assert shard_map[2] == (0, 1)
    assert shard_map[3] == (1, 1)
