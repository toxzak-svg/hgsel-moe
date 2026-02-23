"""
Token dispatch planning for expert-parallel execution.

This module builds routing plans but does not perform communication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class DispatchPlan:
    """Plan for routing tokens to local and remote experts."""

    local_expert_ids: Tuple[int, ...]
    local_expert_local_indices: Tuple[int, ...]
    local_token_indices: Tuple[int, ...]
    remote_rank_to_expert_ids: Dict[int, Tuple[int, ...]]
    remote_rank_to_token_indices: Dict[int, Tuple[int, ...]]


class TokenDispatcher:
    """Build token dispatch plans given expert IDs and a shard map."""

    def __init__(self, shard_map: Dict[int, Tuple[int, int]], rank: int) -> None:
        self.shard_map = shard_map
        self.rank = rank

    @staticmethod
    def _flatten_expert_ids(expert_ids: torch.Tensor) -> torch.Tensor:
        if expert_ids.dim() == 3:
            batch, seq, k = expert_ids.shape
            return expert_ids.reshape(batch * seq, k)
        if expert_ids.dim() == 2:
            return expert_ids
        raise ValueError("expert_ids must be 2D or 3D")

    def build_plan(self, expert_ids: torch.Tensor) -> DispatchPlan:
        """Create a plan that splits local vs remote expert assignments."""
        flat_expert_ids = self._flatten_expert_ids(expert_ids)
        num_tokens, k = flat_expert_ids.shape

        local_expert_ids: List[int] = []
        local_expert_local_indices: List[int] = []
        local_token_indices: List[int] = []

        remote_rank_to_expert_ids: Dict[int, List[int]] = {}
        remote_rank_to_token_indices: Dict[int, List[int]] = {}

        for token_idx in range(num_tokens):
            for slot in range(k):
                expert_id = int(flat_expert_ids[token_idx, slot].item())
                owner_rank, local_idx = self.shard_map[expert_id]

                if owner_rank == self.rank:
                    local_expert_ids.append(expert_id)
                    local_expert_local_indices.append(local_idx)
                    local_token_indices.append(token_idx)
                else:
                    remote_rank_to_expert_ids.setdefault(owner_rank, []).append(expert_id)
                    remote_rank_to_token_indices.setdefault(owner_rank, []).append(token_idx)

        return DispatchPlan(
            local_expert_ids=tuple(local_expert_ids),
            local_expert_local_indices=tuple(local_expert_local_indices),
            local_token_indices=tuple(local_token_indices),
            remote_rank_to_expert_ids={
                rank: tuple(ids) for rank, ids in remote_rank_to_expert_ids.items()
            },
            remote_rank_to_token_indices={
                rank: tuple(indices) for rank, indices in remote_rank_to_token_indices.items()
            },
        )
