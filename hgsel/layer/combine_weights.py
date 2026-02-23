"""
Output combination strategies for k expert outputs.

HGSEL supports multiple combining modes:
  1. uniform: Average k experts (no learned params)
  2. scalar: Per-expert scalar weight (k learned params)
  3. learned: Tiny learned network scoring each expert (future)
"""

import torch
import torch.nn as nn
from typing import Optional


class UniformCombine(nn.Module):
    """Average k expert outputs (uniform weights)."""

    def __init__(self, k_active: int = 2):
        super().__init__()
        self.k_active = k_active
        self.weight = 1.0 / k_active

    def forward(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expert_outputs: [batch * seq_len, k_active, d_model]

        Returns:
            combined: [batch * seq_len, d_model]
        """
        return expert_outputs.mean(dim=1)


class ScalarCombine(nn.Module):
    """
    Per-expert scalar weights (learned).

    Args:
        k_active: Number of active experts
        d_model: Embedding dimension (for context)
    """

    def __init__(self, k_active: int = 2, d_model: int = 512):
        super().__init__()
        self.k_active = k_active
        self.d_model = d_model

        # Learn a scalar weight per expert
        self.weights = nn.Parameter(torch.ones(k_active) / k_active)

    def forward(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expert_outputs: [batch * seq_len, k_active, d_model]

        Returns:
            combined: [batch * seq_len, d_model]
        """
        # Normalize weights to sum to 1
        normalized_weights = torch.softmax(self.weights, dim=0)

        # Reshape weights: [k_active] → [1, k_active, 1]
        weights_reshaped = normalized_weights.view(1, self.k_active, 1)

        # Weighted sum across expert dim
        combined = (expert_outputs * weights_reshaped).sum(dim=1)

        return combined


class LearnedCombine(nn.Module):
    """
    Learned combining network (future enhancement).

    Tiny MLP takes token embedding and produces per-expert scores.

    Args:
        k_active: Number of active experts
        d_model: Embedding dimension
        hidden_dim: Internal dimension for score network (default: d_model // 4)
    """

    def __init__(
        self,
        k_active: int = 2,
        d_model: int = 512,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.k_active = k_active
        self.d_model = d_model
        self.hidden_dim = hidden_dim or d_model // 4

        # Tiny scoring network: d_model → hidden_dim → k_active
        self.score_net = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_dim, k_active, bias=True),
        )

    def forward(
        self,
        expert_outputs: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            expert_outputs: [batch * seq_len, k_active, d_model]
            hidden_states: [batch * seq_len, d_model] (original token embedding)

        Returns:
            combined: [batch * seq_len, d_model]
        """
        # Generate per-expert scores
        scores = self.score_net(hidden_states)  # [batch * seq_len, k_active]
        scores = torch.softmax(scores, dim=1)

        # Reshape and combine
        scores_reshaped = scores.unsqueeze(2)  # [batch * seq_len, k_active, 1]
        combined = (expert_outputs * scores_reshaped).sum(dim=1)

        return combined


class CombineFactory:
    """Factory for creating combine modules."""

    @staticmethod
    def create(
        mode: str = "uniform",
        k_active: int = 2,
        d_model: int = 512,
    ) -> nn.Module:
        """
        Create appropriate combine module.

        Args:
            mode: "uniform", "scalar", or "learned"
            k_active: Number of active experts
            d_model: Embedding dimension

        Returns:
            Combine module
        """
        if mode == "uniform":
            return UniformCombine(k_active)
        elif mode == "scalar":
            return ScalarCombine(k_active, d_model)
        elif mode == "learned":
            return LearnedCombine(k_active, d_model)
        else:
            raise ValueError(f"Unknown combine mode: {mode}")
