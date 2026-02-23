"""
Expert sharding utilities.

Provides deterministic partitioning of expert IDs across ranks.
This is a Phase 4 skeleton; communication is handled elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class ExpertShardMetadata:
    """Metadata describing a single rank's expert shard."""

    rank: int
    world_size: int
    expert_ids: Tuple[int, ...]
    local_index: Dict[int, int]


class ExpertPartitioner:
    """Deterministically partition experts across ranks.

    This class does not require torch.distributed to be initialized; it can
    compute expected partitions for planning or tests.
    """

    def __init__(self, num_experts: int, world_size: int) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        self.num_experts = num_experts
        self.world_size = world_size

    def shard_for_rank(self, rank: int) -> ExpertShardMetadata:
        """Return the expert IDs owned by a given rank.

        Uses round-robin assignment for balanced counts.
        """
        if rank < 0 or rank >= self.world_size:
            raise ValueError("rank out of range")

        expert_ids = tuple(eid for eid in range(self.num_experts) if eid % self.world_size == rank)
        local_index = {eid: idx for idx, eid in enumerate(expert_ids)}
        return ExpertShardMetadata(
            rank=rank,
            world_size=self.world_size,
            expert_ids=expert_ids,
            local_index=local_index,
        )

    def all_shards(self) -> List[ExpertShardMetadata]:
        """Return metadata for all ranks."""
        return [self.shard_for_rank(rank) for rank in range(self.world_size)]

    def owner_rank(self, expert_id: int) -> int:
        """Return the owning rank for a given expert ID."""
        if expert_id < 0 or expert_id >= self.num_experts:
            raise ValueError("expert_id out of range")
        return expert_id % self.world_size


def build_shard_map(num_experts: int, world_size: int) -> Dict[int, Tuple[int, int]]:
    """Build a mapping from expert ID to (rank, local_index)."""
    partitioner = ExpertPartitioner(num_experts=num_experts, world_size=world_size)
    shard_map: Dict[int, Tuple[int, int]] = {}
    for shard in partitioner.all_shards():
        for local_idx, expert_id in enumerate(shard.expert_ids):
            shard_map[expert_id] = (shard.rank, local_idx)
    return shard_map
