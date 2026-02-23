"""
Token exchange stubs for expert-parallel dispatch.

Provides all-to-all token exchange for routing tokens to remote experts.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from hgsel.distributed.dist_utils import (
    all_to_all,
    get_rank,
    get_world_size,
    is_dist_initialized,
)
from hgsel.distributed.dispatch_api import RemoteDispatchRequests


class TokenExchange:
    """All-to-all token exchange for expert-parallel dispatch."""

    def __init__(self, group: Optional[object] = None) -> None:
        """Initialize token exchange.
        
        Args:
            group: torch.distributed process group (default: None = default group)
        """
        self.group = group

    def _empty_payload(self, payloads: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Create an empty tensor matching the shape of payloads."""
        if payloads:
            sample = next(iter(payloads.values()))
            return torch.empty((0,) + sample.shape[1:], device=sample.device, dtype=sample.dtype)
        return torch.empty(0)

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
            return {0: payloads.get(0, self._empty_payload(payloads))}
        
        if not is_dist_initialized():
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
                    padding = (0, 0) * len(token_shape) + (0, max_tokens - tokens.shape[0])
                    padding = padding[::-1]  # Reverse for torch.nn.functional.pad
                    tokens = torch.nn.functional.pad(tokens, padding, mode='constant', value=0)
                send_list.append(tokens)
            else:
                # Create empty padded tensor
                send_list.append(torch.zeros((max_tokens,) + token_shape, device=device, dtype=dtype))
        
        # Stack into send buffer: [world_size * max_tokens, ...]
        send_buffer = torch.cat(send_list, dim=0)  # [world_size * max_tokens, ...]
        
        # Create receive buffer (same size)
        recv_buffer = torch.zeros_like(send_buffer)
        
        # All-to-all exchange
        all_to_all(send_buffer, recv_buffer, group=self.group)
        
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

