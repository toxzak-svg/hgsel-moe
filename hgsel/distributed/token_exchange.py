"""
Token exchange stubs for expert-parallel dispatch.

Provides all-to-all token exchange for routing tokens to remote experts.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import torch

from hgsel.distributed.dist_utils import (
    all_to_all,
    get_rank,
    get_world_size,
    is_dist_initialized,
)
from hgsel.distributed.dispatch_api import RemoteDispatchRequests
from hgsel.distributed.phase4_trace import per_rank_shape_signature


class TokenExchange:
    """All-to-all token exchange for expert-parallel dispatch."""

    def __init__(self, group: Optional[object] = None) -> None:
        """Initialize token exchange.
        
        Args:
            group: torch.distributed process group (default: None = default group)
        """
        self.group = group
        self.last_exchange_stats: Optional[Dict[str, Any]] = None

    def _empty_payload(self, payloads: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Create an empty tensor matching the shape of payloads."""
        if payloads:
            sample = next(iter(payloads.values()))
            return torch.empty((0,) + sample.shape[1:], device=sample.device, dtype=sample.dtype)
        return torch.empty(0)

    def _capture_stats(
        self,
        payloads: Dict[int, torch.Tensor],
        all_to_all_latency_ms: float,
        world_size: int,
        max_tokens: int,
        distributed_enabled: bool,
    ) -> None:
        per_rank_send_counts = {
            int(dst_rank): int(payloads.get(dst_rank, self._empty_payload(payloads)).shape[0])
            for dst_rank in range(world_size)
        }
        per_rank_send_bytes = {}
        for dst_rank in range(world_size):
            tensor = payloads.get(dst_rank)
            if tensor is None:
                per_rank_send_bytes[dst_rank] = 0
            else:
                per_rank_send_bytes[dst_rank] = int(tensor.numel() * tensor.element_size())

        self.last_exchange_stats = {
            "all_to_all_latency_ms": float(all_to_all_latency_ms),
            "distributed_enabled": bool(distributed_enabled),
            "world_size": int(world_size),
            "max_tokens_per_rank_padded": int(max_tokens),
            "per_rank_send_counts": per_rank_send_counts,
            "per_rank_send_bytes": per_rank_send_bytes,
            "shape_signature": per_rank_shape_signature(
                per_rank_send_counts,
                world_size=world_size,
                prefix="exchange_send",
            ),
        }

    def exchange(self, payloads: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Exchange token payloads across all ranks using all-to-all.
        
        Args:
            payloads: Dict mapping destination rank -> tokens to send to that rank
                     tokens shape: [num_tokens_for_rank, ...]
        
        Returns:
            Dict mapping source rank -> received tokens from that rank
        
        In single-GPU mode or world_size=1, returns payloads unchanged.
        """
        world_size = get_world_size()
        rank = get_rank()
        
        # Fallback for single-GPU or non-distributed
        if world_size == 1:
            self._capture_stats(
                payloads=payloads,
                all_to_all_latency_ms=0.0,
                world_size=1,
                max_tokens=int(payloads.get(0, self._empty_payload(payloads)).shape[0]),
                distributed_enabled=False,
            )
            return {0: payloads.get(0, self._empty_payload(payloads))}

        if not is_dist_initialized():
            self._capture_stats(
                payloads=payloads,
                all_to_all_latency_ms=0.0,
                world_size=world_size,
                max_tokens=max((int(p.shape[0]) for p in payloads.values()), default=0),
                distributed_enabled=False,
            )
            return {0: payloads.get(0, self._empty_payload(payloads))}
        
        # Get sample tensor for device, dtype, and batch shape
        if payloads:
            sample = next(iter(payloads.values()))
            device = sample.device
            dtype = sample.dtype
            token_shape = sample.shape[1:]  # Everything except first dim
        else:
            raise ValueError("payloads cannot be empty for distributed exchange")
        
        # Pad payloads to uniform size across all ranks
        # This is required by all_to_all_single
        max_tokens = max(p.shape[0] for p in payloads.values())
        
        # Build padded send tensor: [world_size, max_tokens, ...]
        send_list = []
        for dst_rank in range(world_size):
            if dst_rank in payloads:
                tokens = payloads[dst_rank]
                # Pad if necessary
                if tokens.shape[0] < max_tokens:
                    pad_rows = max_tokens - tokens.shape[0]
                    pad_tensor = torch.zeros(
                        (pad_rows,) + token_shape,
                        device=device,
                        dtype=dtype,
                    )
                    tokens = torch.cat([tokens, pad_tensor], dim=0)
                send_list.append(tokens)
            else:
                # Create empty padded tensor
                send_list.append(torch.zeros((max_tokens,) + token_shape, device=device, dtype=dtype))
        
        # Stack into send buffer: [world_size * max_tokens, ...]
        send_buffer = torch.cat(send_list, dim=0)  # [world_size * max_tokens, ...]
        
        # Create receive buffer (same size)
        recv_buffer = torch.zeros_like(send_buffer)
        
        # All-to-all exchange
        if torch.cuda.is_available() and send_buffer.is_cuda:
            torch.cuda.synchronize(device=send_buffer.device)
        t_all_to_all_start = time.perf_counter()
        all_to_all(send_buffer, recv_buffer, group=self.group)
        if torch.cuda.is_available() and recv_buffer.is_cuda:
            torch.cuda.synchronize(device=recv_buffer.device)
        all_to_all_latency_ms = (time.perf_counter() - t_all_to_all_start) * 1000.0
        self._capture_stats(
            payloads=payloads,
            all_to_all_latency_ms=all_to_all_latency_ms,
            world_size=world_size,
            max_tokens=int(max_tokens),
            distributed_enabled=True,
        )
        
        # Unpack received buffers
        received = {}
        for src_rank in range(world_size):
            start_idx = src_rank * max_tokens
            end_idx = (src_rank + 1) * max_tokens
            received[src_rank] = recv_buffer[start_idx:end_idx]
        
        return received

    def exchange_requests(
        self,
        requests: RemoteDispatchRequests,
        payload_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[int, torch.Tensor]:
        """Return received payloads for remote dispatch requests.
        
        This is a stub that allows end-to-end dispatch wiring without actual comms.
        Returns empty tensors matching the requested shape.
        
        Args:
            requests: Remote dispatch requests specifying destination ranks
            payload_shape: Shape of each token [d_model, ...]
            device: Device to place tensors on
            dtype: Data type of tensors
        
        Returns:
            Dict mapping rank -> empty tensors for that rank's requests
        """
        empty = torch.empty(payload_shape, device=device, dtype=dtype)
        return {rank: empty for rank in requests.rank_to_token_indices}

