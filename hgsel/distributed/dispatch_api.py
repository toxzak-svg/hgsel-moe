"""
Dispatch API stubs for expert-parallel execution.

These helpers orchestrate local batching and define placeholders
for future all-to-all communication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from hgsel.distributed.token_dispatcher import DispatchPlan


@dataclass(frozen=True)
class LocalDispatchBatch:
    """Local tokens grouped for local expert execution."""

    tokens: torch.Tensor
    token_indices: torch.Tensor
    expert_local_indices: torch.Tensor


@dataclass(frozen=True)
class RemoteDispatchRequests:
    """Remote requests by destination rank (placeholder)."""

    rank_to_token_indices: Dict[int, torch.Tensor]
    rank_to_expert_ids: Dict[int, torch.Tensor]


class ExpertDispatchController:
    """Build local batches and remote requests from a dispatch plan.

    This class intentionally does not perform distributed communication.
    """

    def build_local_batch(self, hidden_states: torch.Tensor, plan: DispatchPlan) -> LocalDispatchBatch:
        """Gather local tokens and indices according to the plan."""
        if hidden_states.dim() != 2:
            raise ValueError("hidden_states must be 2D [tokens, dim]")

        token_indices = torch.tensor(plan.local_token_indices, device=hidden_states.device)
        expert_indices = torch.tensor(plan.local_expert_local_indices, device=hidden_states.device)
        tokens = hidden_states.index_select(0, token_indices) if token_indices.numel() > 0 else hidden_states[:0]

        return LocalDispatchBatch(
            tokens=tokens,
            token_indices=token_indices,
            expert_local_indices=expert_indices,
        )

    def build_remote_requests(self, plan: DispatchPlan, device: torch.device) -> RemoteDispatchRequests:
        """Prepare remote dispatch requests (no communication)."""
        rank_to_token_indices = {
            rank: torch.tensor(indices, device=device)
            for rank, indices in plan.remote_rank_to_token_indices.items()
        }
        rank_to_expert_ids = {
            rank: torch.tensor(ids, device=device)
            for rank, ids in plan.remote_rank_to_expert_ids.items()
        }

        return RemoteDispatchRequests(
            rank_to_token_indices=rank_to_token_indices,
            rank_to_expert_ids=rank_to_expert_ids,
        )
