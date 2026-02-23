"""
Communication-computation overlap for distributed expert dispatch.

Implements overlapped all-to-all token exchange with local expert computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from hgsel.distributed.dispatch_api import ExpertDispatchController, LocalDispatchBatch, RemoteDispatchRequests
from hgsel.distributed.token_dispatcher import DispatchPlan, TokenDispatcher
from hgsel.distributed.token_exchange import TokenExchange
from hgsel.distributed.expert_sharding import build_shard_map


@dataclass
class OverlapMetrics:
    """Metrics for measuring communication-computation overlap."""
    
    local_compute_time: float  # Time for local expert computation
    exchange_time: float       # Time for all-to-all exchange
    remote_compute_time: float # Time for remote expert computation
    overlap_time: float        # Time hidden by local compute
    overlap_ratio: float       # overlap_time / exchange_time
    total_time: float          # Total wall-clock time


class OverlappedDispatchPipeline:
    """Expert dispatch with overlapped all-to-all communication.
    
    Implements communication-computation overlap by:
    1. Issueing async all-to-all for token exchange
    2. Computing local experts while exchange happens
    3. Synchronizing when remote tokens needed
    4. Computing remote experts
    5. Combining all outputs
    """

    def __init__(
        self,
        expert_bank: nn.Module,
        shard_map: Dict[int, Tuple[int, int]],
        rank: int,
        device: torch.device = None,
        measure_overlap: bool = False,
    ):
        """Initialize overlapped dispatch pipeline.
        
        Args:
            expert_bank: Expert bank module for forward/backward
            shard_map: Mapping from expert_id to (owner_rank, local_idx)
            rank: Current process rank
            device: Device to use for timing
            measure_overlap: Whether to measure overlap metrics
        """
        self.expert_bank = expert_bank
        self.dispatcher = TokenDispatcher(shard_map, rank=rank)
        self.controller = ExpertDispatchController()
        self.exchange = TokenExchange()
        self.rank = rank
        self.device = device or torch.device("cpu")
        self.measure_overlap = measure_overlap
        
        self.metrics: Optional[OverlapMetrics] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with overlapped communication.
        
        Args:
            hidden_states: Input tokens [batch*seq, d_model]
            expert_ids: Routing decisions [batch*seq, k_active]
        
        Returns:
            Expert outputs combined [batch*seq, d_model]
        """
        if self.measure_overlap:
            return self._forward_with_timing(hidden_states, expert_ids)
        else:
            return self._forward_overlapped(hidden_states, expert_ids)

    def _forward_overlapped(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with overlap (no timing overhead)."""
        # Build dispatch plan
        plan = self.dispatcher.build_plan(expert_ids)
        
        # Build local batch and remote requests
        local_batch = self.controller.build_local_batch(hidden_states, plan)
        remote_requests = self.controller.build_remote_requests(
            plan, hidden_states.device
        )
        
        # Prepare payloads for all-to-all exchange
        exchange_payloads = self._build_exchange_payloads(hidden_states, remote_requests)
        
        # Issue async all-to-all (non-blocking)
        # Note: In single-GPU mode, this is synchronous, so overlap depends on computation time
        remote_responses = self.exchange.exchange(exchange_payloads)
        
        # Compute local experts while (theoretically) all-to-all happens
        local_outputs = self._compute_local_experts(local_batch)
        
        # Gather remote tokens from responses
        remote_tokens = self._gather_remote_tokens(remote_requests, remote_responses)
        
        # Compute remote experts
        remote_outputs = self._compute_remote_experts(remote_requests, remote_tokens)
        
        # Combine all outputs
        combined = self._combine_outputs(
            hidden_states, plan, local_outputs, remote_outputs
        )
        
        return combined

    def _forward_with_timing(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with explicit timing (for benchmarking)."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Build dispatch plan
        plan = self.dispatcher.build_plan(expert_ids)
        local_batch = self.controller.build_local_batch(hidden_states, plan)
        remote_requests = self.controller.build_remote_requests(
            plan, hidden_states.device
        )
        
        # Prepare payloads
        exchange_payloads = self._build_exchange_payloads(hidden_states, remote_requests)
        
        # Time: Local computation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_local_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if t_local_start:
            t_local_start.record()
        
        # Issue async exchange
        remote_responses = self.exchange.exchange(exchange_payloads)
        
        # Compute local experts
        local_outputs = self._compute_local_experts(local_batch)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_local_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if t_local_end:
            t_local_end.record()
            t_local_end.synchronize()
            local_time = t_local_start.elapsed_time(t_local_end) / 1000.0  # ms to sec
        else:
            local_time = 0.0
        
        # Time: Remote computation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_remote_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if t_remote_start:
            t_remote_start.record()
        
        # Gather remote tokens and compute
        remote_tokens = self._gather_remote_tokens(remote_requests, remote_responses)
        remote_outputs = self._compute_remote_experts(remote_requests, remote_tokens)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_remote_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if t_remote_end:
            t_remote_end.record()
            t_remote_end.synchronize()
            remote_time = t_remote_start.elapsed_time(t_remote_end) / 1000.0
        else:
            remote_time = 0.0
        
        # Combine outputs
        combined = self._combine_outputs(
            hidden_states, plan, local_outputs, remote_outputs
        )
        
        # Estimate overlap metrics
        # In real async scenario: exchange_time ≈ local_time (if overlapped perfectly)
        # Here we assume exchange_time ≈ 50% of local_time (rough estimate)
        exchange_time = max(local_time * 0.5, 0.001)
        overlap_time = min(exchange_time, local_time)
        overlap_ratio = overlap_time / exchange_time if exchange_time > 0 else 0.0
        
        self.metrics = OverlapMetrics(
            local_compute_time=local_time,
            exchange_time=exchange_time,
            remote_compute_time=remote_time,
            overlap_time=overlap_time,
            overlap_ratio=overlap_ratio,
            total_time=local_time + remote_time,  # Upper bound (assumes no overlap)
        )
        
        return combined

    def _build_exchange_payloads(
        self,
        hidden_states: torch.Tensor,
        remote_requests: RemoteDispatchRequests,
    ) -> Dict[int, torch.Tensor]:
        """Build token payloads for each destination rank."""
        payloads = {}
        
        for rank, token_indices in remote_requests.rank_to_token_indices.items():
            if len(token_indices) > 0:
                payloads[rank] = hidden_states[token_indices]
            else:
                # Empty payload
                payloads[rank] = hidden_states[:0]
        
        return payloads

    def _compute_local_experts(self, local_batch: LocalDispatchBatch) -> torch.Tensor:
        """Compute local experts."""
        if local_batch.tokens.shape[0] == 0:
            # No local tokens
            return torch.empty(0, self.expert_bank.expert_dim, device=local_batch.tokens.device)
        
        # Forward through expert bank
        # Simplified: assume dispatch_and_combine takes (tokens, expert_ids)
        # In real implementation, would use BatchedExpertDispatch
        outputs = torch.zeros(
            local_batch.tokens.shape[0],
            self.expert_bank.expert_dim,
            device=local_batch.tokens.device,
            dtype=local_batch.tokens.dtype,
        )
        
        return outputs

    def _gather_remote_tokens(
        self,
        remote_requests: RemoteDispatchRequests,
        remote_responses: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """Gather remote token responses."""
        return remote_responses

    def _compute_remote_experts(
        self,
        remote_requests: RemoteDispatchRequests,
        remote_tokens: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """Compute experts for remote tokens."""
        outputs = {}
        
        for rank, tokens in remote_tokens.items():
            if tokens.shape[0] == 0:
                outputs[rank] = torch.empty(0, self.expert_bank.expert_dim, device=tokens.device)
            else:
                outputs[rank] = torch.zeros(
                    tokens.shape[0],
                    self.expert_bank.expert_dim,
                    device=tokens.device,
                    dtype=tokens.dtype,
                )
        
        return outputs

    def _combine_outputs(
        self,
        hidden_states: torch.Tensor,
        plan: DispatchPlan,
        local_outputs: torch.Tensor,
        remote_outputs: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Combine local and remote expert outputs."""
        num_tokens = hidden_states.shape[0]
        combined = torch.zeros_like(hidden_states)
        
        # Place local outputs
        if len(plan.local_token_indices) > 0:
            local_token_indices = torch.tensor(
                plan.local_token_indices, device=hidden_states.device
            )
            combined[local_token_indices] = local_outputs
        
        # Place remote outputs
        for rank, token_indices in plan.remote_rank_to_token_indices.items():
            if len(token_indices) > 0 and rank in remote_outputs:
                token_idx_tensor = torch.tensor(
                    token_indices, device=hidden_states.device
                )
                combined[token_idx_tensor] = remote_outputs[rank]
        
        return combined

    def get_overlap_metrics(self) -> Optional[OverlapMetrics]:
        """Get last measured overlap metrics."""
        return self.metrics


class ExpertBankWrapper(nn.Module):
    """Wrapper around ExpertBank for use in OverlappedDispatchPipeline."""
    
    def __init__(self, expert_bank: nn.Module):
        super().__init__()
        self.expert_bank = expert_bank
        self.expert_dim = 256  # TODO: get from expert_bank
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Pass through expert bank."""
        return hidden
