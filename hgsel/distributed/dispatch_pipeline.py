"""
Dispatch pipeline helper that ties planning with batch preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional, Tuple

import torch

from hgsel.distributed.dispatch_api import (
    ExpertDispatchController,
    LocalDispatchBatch,
    RemoteDispatchRequests,
)
from hgsel.distributed.token_dispatcher import DispatchPlan, TokenDispatcher
from hgsel.distributed.phase4_trace import per_rank_shape_signature


@dataclass(frozen=True)
class DispatchPipelineResult:
    """Results of building a dispatch plan and local/remote batches."""

    plan: DispatchPlan
    local_batch: LocalDispatchBatch
    remote_requests: RemoteDispatchRequests


class DispatchPipeline:
    """Convenience wrapper for planning and preparing dispatch batches."""

    def __init__(self, shard_map: Dict[int, Tuple[int, int]], rank: int) -> None:
        self.shard_map = shard_map
        self.rank = rank
        self.world_size = max((owner_rank for owner_rank, _ in shard_map.values()), default=rank) + 1
        self.dispatcher = TokenDispatcher(shard_map=shard_map, rank=rank)
        self.controller = ExpertDispatchController()
        self.last_build_stats: Optional[Dict[str, Any]] = None

    def _summarize_build(
        self,
        hidden_states: torch.Tensor,
        plan: DispatchPlan,
        timings_ms: Dict[str, float],
    ) -> Dict[str, Any]:
        bytes_per_token = hidden_states.shape[1] * hidden_states.element_size()
        local_assignments = len(plan.local_token_indices)
        remote_rank_counts = {
            dst_rank: len(token_indices)
            for dst_rank, token_indices in plan.remote_rank_to_token_indices.items()
        }
        remote_assignments = sum(remote_rank_counts.values())

        per_rank_send_counts = {dst_rank: 0 for dst_rank in range(self.world_size)}
        for dst_rank, count in remote_rank_counts.items():
            per_rank_send_counts[dst_rank] = int(count)

        per_rank_send_bytes = {
            dst_rank: int(count * bytes_per_token)
            for dst_rank, count in per_rank_send_counts.items()
        }

        per_rank_assignment_counts = {dst_rank: 0 for dst_rank in range(self.world_size)}
        per_rank_assignment_counts[self.rank] = int(local_assignments)
        for dst_rank, count in remote_rank_counts.items():
            per_rank_assignment_counts[dst_rank] = int(count)

        return {
            **timings_ms,
            "local_assignments": int(local_assignments),
            "remote_assignments": int(remote_assignments),
            "remote_ranks": sorted(remote_rank_counts.keys()),
            "remote_rank_token_counts": remote_rank_counts,
            "per_rank_send_counts": per_rank_send_counts,
            "per_rank_send_bytes": per_rank_send_bytes,
            "per_rank_assignment_counts": per_rank_assignment_counts,
            "shape_signature": per_rank_shape_signature(
                per_rank_send_counts,
                world_size=self.world_size,
                prefix="dispatch_send",
            ),
        }

    def build(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> DispatchPipelineResult:
        if hidden_states.dim() != 2:
            raise ValueError("hidden_states must be 2D [tokens, dim]")

        t0 = time.perf_counter()
        t_plan_start = time.perf_counter()
        plan = self.dispatcher.build_plan(expert_ids)
        t_plan_ms = (time.perf_counter() - t_plan_start) * 1000.0

        t_local_start = time.perf_counter()
        local_batch = self.controller.build_local_batch(hidden_states, plan)
        t_local_ms = (time.perf_counter() - t_local_start) * 1000.0

        t_remote_start = time.perf_counter()
        remote_requests = self.controller.build_remote_requests(plan, hidden_states.device)
        t_remote_ms = (time.perf_counter() - t_remote_start) * 1000.0
        t_total_ms = (time.perf_counter() - t0) * 1000.0

        self.last_build_stats = self._summarize_build(
            hidden_states=hidden_states,
            plan=plan,
            timings_ms={
                "dispatch_plan_ms": t_plan_ms,
                "dispatch_local_batch_ms": t_local_ms,
                "dispatch_remote_request_ms": t_remote_ms,
                "dispatch_build_total_ms": t_total_ms,
            },
        )

        return DispatchPipelineResult(
            plan=plan,
            local_batch=local_batch,
            remote_requests=remote_requests,
        )
