"""
Dispatch pipeline helper that ties planning with batch preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from hgsel.distributed.dispatch_api import (
    ExpertDispatchController,
    LocalDispatchBatch,
    RemoteDispatchRequests,
)
from hgsel.distributed.token_dispatcher import DispatchPlan, TokenDispatcher


@dataclass(frozen=True)
class DispatchPipelineResult:
    """Results of building a dispatch plan and local/remote batches."""

    plan: DispatchPlan
    local_batch: LocalDispatchBatch
    remote_requests: RemoteDispatchRequests


class DispatchPipeline:
    """Convenience wrapper for planning and preparing dispatch batches."""

    def __init__(self, shard_map: Dict[int, Tuple[int, int]], rank: int) -> None:
        self.dispatcher = TokenDispatcher(shard_map=shard_map, rank=rank)
        self.controller = ExpertDispatchController()

    def build(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> DispatchPipelineResult:
        if hidden_states.dim() != 2:
            raise ValueError("hidden_states must be 2D [tokens, dim]")

        plan = self.dispatcher.build_plan(expert_ids)
        local_batch = self.controller.build_local_batch(hidden_states, plan)
        remote_requests = self.controller.build_remote_requests(plan, hidden_states.device)

        return DispatchPipelineResult(
            plan=plan,
            local_batch=local_batch,
            remote_requests=remote_requests,
        )
