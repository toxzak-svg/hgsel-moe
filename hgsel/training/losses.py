"""
Training loss functions for HGSEL.

Primary auxiliary loss: UtilizationLoss
  Penalizes imbalanced expert utilization (dead experts, overloaded experts).

Strategy:
  - Track expert load distribution across batch
  - Push toward uniform usage (entropy maximization)
  - Avoid hard constraints (unstable)
"""

import torch
import torch.nn as nn
from typing import Optional


class UtilizationLoss(nn.Module):
    """
    Auxiliary loss for balancing expert utilization.

    Encourages uniform expert activation across data:
    loss = E[H(load_distribution)] - H_uniform

    Where H = entropy, and higher entropy = more balanced.

    Args:
        n_experts: Number of experts
        target_entropy: Target entropy (default: -log(1/n_experts) = log(n_experts))
        weight: Loss weight in total training loss (default: 0.05)
    """

    def __init__(
        self,
        n_experts: int = 64,
        target_entropy: Optional[float] = None,
        weight: float = 0.05,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.weight = weight
        self.target_entropy = target_entropy or torch.log(torch.tensor(n_experts, dtype=torch.float32))

    def forward(self, expert_loads: torch.Tensor) -> torch.Tensor:
        """
        Compute balancing loss from expert loads.

        Args:
            expert_loads: [n_experts] or [batch, n_experts]
                - Values should be in [0, 1] representing activation rate per expert
                - Typically from HGSELLayer.expert_load_ema

        Returns:
            loss: Scalar loss value
        """
        if expert_loads.dim() == 1:
            # Single batch/global loads
            loads = expert_loads.unsqueeze(0)
        else:
            loads = expert_loads

        batch_size = loads.shape[0]

        # Normalize loads to probability distribution
        loads_normed = loads / (loads.sum(dim=1, keepdim=True) + 1e-8)
        loads_normed = loads_normed.clamp(min=1e-8)

        # Compute entropy for each batch item
        entropy = -torch.sum(loads_normed * torch.log(loads_normed), dim=1)

        # Target: uniform (maximum entropy)
        target = torch.ones_like(entropy) * torch.log(torch.tensor(self.n_experts, dtype=loads.dtype))

        # L2 loss toward target entropy
        loss = torch.mean((entropy - target) ** 2)

        return loss * self.weight


class AuxiliaryLoadLoss(nn.Module):
    """
    Simpler auxiliary loss: Penalize imbalance directly.

    loss = variance(expert_loads)

    Minimizing variance encourages uniform loads.

    Args:
        weight: Loss weight in total training loss
    """

    def __init__(self, weight: float = 0.05):
        super().__init__()
        self.weight = weight

    def forward(self, expert_loads: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expert_loads: [n_experts] with load per expert

        Returns:
            loss: Scalar loss value
        """
        # Variance of loads
        mean_load = expert_loads.mean()
        variance = torch.mean((expert_loads - mean_load) ** 2)

        return variance * self.weight


class LoadBalancingLoss(nn.Module):
    """
    Switch load-balancing aux loss during training (Phase 1-2).

    During Phase 1 (early training): Use strong utilization loss (weight=0.05)
    During Phase 2 (mid training): Gradually reduce (0.05 → 0.01)
    During Phase 3 (late training): Minimal or off

    Args:
        n_experts: Number of experts
        initial_weight: Starting loss weight (default: 0.05)
        strategy: "utilization" or "variance"
    """

    def __init__(
        self,
        n_experts: int = 64,
        initial_weight: float = 0.05,
        strategy: str = "utilization",
    ):
        super().__init__()
        self.n_experts = n_experts
        self.weight = initial_weight
        self.strategy = strategy

        if strategy == "utilization":
            self.loss_fn = UtilizationLoss(n_experts, weight=1.0)
        elif strategy == "variance":
            self.loss_fn = AuxiliaryLoadLoss(weight=1.0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def forward(self, expert_loads: torch.Tensor) -> torch.Tensor:
        """
        Compute load-balancing auxiliary loss.

        Args:
            expert_loads: [n_experts]

        Returns:
            loss: Scalar loss value * weight
        """
        return self.loss_fn(expert_loads) * self.weight

    def set_weight(self, weight: float):
        """Update loss weight (for training schedule)."""
        self.weight = weight
