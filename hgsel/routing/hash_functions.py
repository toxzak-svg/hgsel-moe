"""
Multi-hash deterministic routing engine.

Core concept: Route each token to k active experts from N total experts
using H independent hash functions applied to quantized hidden state.

Design:
  1. Quantize hidden state: Extract sign and magnitude buckets
  2. Multi-hash: Apply H hash functions (deterministic, no learned params)
  3. Expert selection: For each candidate hash, select 1 expert → k total
  4. Load balancing: Salt parameter deterministically shifts routing

This avoids learned routers (training instability) and enables compilation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class MultiHashRouter(nn.Module):
    """
    Deterministic k-expert selection via H-candidate multi-hash routing.

    Args:
        n_experts: Total number of experts (e.g., 64)
        k_active: Number of active experts per token (e.g., 2)
        n_hashes: Number of candidate hashes (e.g., 4)
        hidden_dim: Dimension of input hidden states
        layer_id: Layer index (for stability, included in routing key)
        salt: Deterministic load-balancing parameter (tuned via hill climb)
    """

    def __init__(
        self,
        n_experts: int = 64,
        k_active: int = 2,
        n_hashes: int = 4,
        hidden_dim: int = 512,
        layer_id: int = 0,
        salt: float = 0.0,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.k_active = k_active
        self.n_hashes = n_hashes
        self.hidden_dim = hidden_dim
        self.layer_id = layer_id
        self.salt = salt

        # Register hash seeds as buffer (no gradients, part of state_dict)
        hash_seeds = torch.arange(n_hashes, dtype=torch.int32)
        self.register_buffer("hash_seeds", hash_seeds, persistent=True)

    def quantize(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize hidden state to (sign, magnitude_bucket).

        Args:
            hidden_states: [batch * seq_len, hidden_dim]

        Returns:
            sign: [batch * seq_len, hidden_dim] in {-1, 0, 1}
            magnitude_bucket: [batch * seq_len, hidden_dim] in [0, 7]
        """
        # Sign: {-1, 0, 1}
        sign = torch.sign(hidden_states)

        # Magnitude bucketing: [0, 7]
        # Map |value| into buckets: 0-1→0, 1-2→1, ..., 7-8→7
        abs_val = torch.abs(hidden_states)
        magnitude_bucket = torch.clamp(abs_val.int(), 0, 7)

        return sign, magnitude_bucket

    def hash_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate H candidate expert IDs for each token via multi-hash.

        Args:
            hidden_states: [batch * seq_len, hidden_dim]

        Returns:
            expert_ids: [batch * seq_len, n_hashes] in [0, n_experts)
            hash_scores: [batch * seq_len, n_hashes] (confidence scores, for future use)
        """
        batch_seq_len, dim = hidden_states.shape

        # Quantize
        sign, magnitude_bucket = self.quantize(hidden_states)

        # Construct routing keys: (sign, magnitude, layer_id) → int64
        # Format: [sign | magnitude_bucket | layer_id]
        # Use first hidden_dim // 2 values to avoid overflow
        routing_keys = torch.zeros(
            batch_seq_len, dtype=torch.int64, device=hidden_states.device
        )

        # Combine using hash mixing: Reduce dim to scalar per token via XOR
        # XOR all dimensions together to get deterministic per-token hash key
        for d in range(0, min(dim, 64)):
            dim_value = (sign[:, d].int() & 0x3) << 2 | (magnitude_bucket[:, d] & 0x7)
            routing_keys ^= dim_value.long() * (d + 1)

        # Mix in layer_id for additional diversity
        routing_keys ^= (self.layer_id << 8)

        # Apply salt for load balancing (deterministic shift)
        routing_keys = (routing_keys + int(self.salt * 1e6)) & 0xFFFFFFFF

        # Apply H independent hash functions
        expert_ids = torch.zeros(
            batch_seq_len, self.n_hashes, dtype=torch.int64, device=hidden_states.device
        )

        for h in range(self.n_hashes):
            seed = int(self.hash_seeds[h])
            # Mix seed into key
            hash_key = routing_keys ^ seed
            # Modulo to expert count
            expert_ids[:, h] = hash_key % self.n_experts

        # Hash scores (unused for now, room for future refinements)
        hash_scores = torch.ones(
            batch_seq_len, self.n_hashes, device=hidden_states.device
        )

        return expert_ids, hash_scores

    def select_k_experts(
        self,
        expert_ids: torch.Tensor,
        hash_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select k unique expert IDs from H candidates.

        Strategy: Greedily pick top-k by hash_scores, with uniqueness constraint.

        Args:
            expert_ids: [batch * seq_len, n_hashes] in [0, n_experts)
            hash_scores: [batch * seq_len, n_hashes]

        Returns:
            selected_experts: [batch * seq_len, k_active] in [0, n_experts)
            expert_weights: [batch * seq_len, k_active] (learned later, uniform for now)
            expert_masks: [batch * seq_len, n_experts] (binary routing mask)
        """
        batch_seq_len = expert_ids.shape[0]
        device = expert_ids.device

        # Greedy K-selection with uniqueness
        selected_experts = torch.zeros(
            batch_seq_len, self.k_active, dtype=torch.int64, device=device
        )
        expert_masks = torch.zeros(
            batch_seq_len, self.n_experts, dtype=torch.float32, device=device
        )

        # For each token, pick top-k unique experts
        for seq_idx in range(batch_seq_len):
            available = set()
            for h in range(self.n_hashes):
                eid = int(expert_ids[seq_idx, h])
                if eid not in available and len(available) < self.k_active:
                    available.add(eid)

            # If we don't have k unique experts, fill with duplicates (shouldn't happen with k≤4)
            selected = list(available)
            while len(selected) < self.k_active:
                selected.append(int(expert_ids[seq_idx, 0]))

            selected_experts[seq_idx] = torch.tensor(
                selected[: self.k_active], dtype=torch.int64, device=device
            )

            # Build mask
            for k_idx, eid in enumerate(selected_experts[seq_idx]):
                expert_masks[seq_idx, eid] = 1.0 / self.k_active

        # Uniform combining weights (will be learned later via combine layer)
        expert_weights = torch.ones(
            batch_seq_len, self.k_active, dtype=torch.float32, device=device
        ) / self.k_active

        return selected_experts, expert_weights, expert_masks

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to k experts.

        Args:
            hidden_states: [batch * seq_len, hidden_dim]

        Returns:
            selected_experts: [batch * seq_len, k_active]
            expert_weights: [batch * seq_len, k_active]
            expert_masks: [batch * seq_len, n_experts]
        """
        expert_ids, hash_scores = self.hash_tokens(hidden_states)
        selected_experts, expert_weights, expert_masks = self.select_k_experts(
            expert_ids, hash_scores
        )

        return selected_experts, expert_weights, expert_masks

    def set_salt(self, salt: float):
        """Update salt parameter for load balancing tuning."""
        self.salt = salt
