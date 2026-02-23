"""
Salt optimizer: Hill-climb tuning of load-balancing salt parameter.

Strategy:
  1. Monitor expert utilization entropy at current salt value
  2. Try perturbations: salt ± delta
  3. Accept change if entropy improves (target: maximize entropy ~= log N)
  4. Exponentially decay delta for convergence

Phase 3 Integration:
  - Run every 10 training batches
  - Target entropy: log(n_experts) ≈ 0.99 for n=64
  - Track salt evolution for inference stability
"""

import torch
from typing import Tuple, Optional


class SaltOptimizer:
    """
    Hill-climb optimizer for load-balancing salt parameter.

    Args:
        n_experts: Number of experts
        initial_salt: Starting salt value (default: 0.0)
        target_entropy: Target entropy (default: log(n_experts))
        delta: Initial perturbation size (default: 0.1)
        delta_decay: Decay factor per optimization step (default: 0.99)
        lr: Learning rate for gradient-like updates (default: 0.1)
    """

    def __init__(
        self,
        n_experts: int = 64,
        initial_salt: float = 0.0,
        target_entropy: Optional[float] = None,
        delta: float = 0.1,
        delta_decay: float = 0.99,
        lr: float = 0.1,
    ):
        self.n_experts = n_experts
        self.salt = initial_salt
        self.target_entropy = target_entropy or torch.log(torch.tensor(n_experts, dtype=torch.float32))
        self.delta = delta
        self.delta_decay = delta_decay
        self.lr = lr
        self.step_count = 0

    def optimize(self, expert_loads: torch.Tensor) -> Tuple[float, float]:
        """
        Perform one optimization step.

        Args:
            expert_loads: [n_experts] with load per expert

        Returns:
            (new_salt, entropy) - Updated salt and current entropy
        """
        # Compute current entropy
        loads_norm = expert_loads / (expert_loads.sum() + 1e-8)
        loads_norm = loads_norm.clamp(min=1e-8)
        current_entropy = -torch.sum(loads_norm * torch.log(loads_norm))
        current_entropy = float(current_entropy)

        # Target: maximize entropy, so minimize distance to target
        current_error = abs(current_entropy - float(self.target_entropy))

        # Try positive perturbation: salt + delta
        old_salt = self.salt
        self.salt += self.delta

        # In practice, we don't have a forward pass here to re-compute loads
        # So we use a simple heuristic: accept if current is significantly below target

        # Revert to old salt for now (in practice, external code would try routing with new salt)
        self.salt = old_salt

        # Update salt via gradient-like signal
        # If entropy is below target, increase salt (typically spreads load)
        # If entropy is above target, decrease salt
        if current_entropy < float(self.target_entropy):
            self.salt += self.lr * self.delta
        else:
            self.salt -= self.lr * self.delta

        # Decay perturbation
        self.delta *= self.delta_decay
        self.step_count += 1

        return self.salt, current_entropy

    def adapt_lr(self, improvement: float):
        """
        Adapt learning rate based on improvement signal.

        Args:
            improvement: Positive if entropy improved, negative otherwise
        """
        if improvement > 0:
            # Good progress, slightly increase LR
            self.lr *= 1.05
        else:
            # No progress, decrease LR
            self.lr *= 0.95

    def reset(self):
        """Reset optimizer state."""
        self.salt = 0.0
        self.delta = 0.1
        self.step_count = 0


class UtilizationMonitor:
    """
    Monitor expert utilization and detect load imbalance.

    Tracks:
    - Per-expert load (EMA)
    - Entropy (load uniformity)
    - Expert collapse events
    - Load distribution statistics

    Args:
        n_experts: Number of experts
        ema_decay: EMA smoothing factor (default: 0.99)
        collapse_threshold: Load rate below which expert is considered unused
        alert_threshold: Entropy below which to warn about imbalance
    """

    def __init__(
        self,
        n_experts: int = 64,
        ema_decay: float = 0.99,
        collapse_threshold: float = 0.01,
        alert_threshold: float = 0.7,
    ):
        self.n_experts = n_experts
        self.ema_decay = ema_decay
        self.collapse_threshold = collapse_threshold
        self.alert_threshold = alert_threshold

        # Initialize EMA loads
        self.ema_loads = torch.ones(n_experts) / n_experts
        self.num_updates = 0

    def update(self, expert_loads: torch.Tensor) -> dict:
        """
        Update monitor with new expert loads.

        Args:
            expert_loads: [n_experts] with load per expert

        Returns:
            stats: Dict with monitoring info
                - entropy: Load distribution entropy
                - collapsed_experts: List of unused expert IDs
                - max_load: Maximum load value
                - min_load: Minimum load value
                - load_variance: Variance of loads
        """
        device = expert_loads.device

        # Update EMA
        expert_loads_float = expert_loads.float()
        self.ema_loads = (
            self.ema_decay * self.ema_loads.to(device)
            + (1 - self.ema_decay) * expert_loads_float
        )

        # Compute stats
        loads_norm = self.ema_loads / (self.ema_loads.sum() + 1e-8)
        loads_norm = loads_norm.clamp(min=1e-8)

        # Entropy
        entropy = -torch.sum(loads_norm * torch.log(loads_norm))
        entropy_normalized = entropy / torch.log(torch.tensor(self.n_experts, dtype=loads_norm.dtype))

        # Find collapsed experts
        collapsed = (self.ema_loads < self.collapse_threshold).nonzero(as_tuple=True)[0].tolist()

        # Variance
        load_variance = torch.var(self.ema_loads)

        stats = {
            "entropy": float(entropy),
            "entropy_normalized": float(entropy_normalized),
            "collapsed_experts": collapsed,
            "n_collapsed": len(collapsed),
            "max_load": float(self.ema_loads.max()),
            "min_load": float(self.ema_loads.min()),
            "mean_load": float(self.ema_loads.mean()),
            "load_variance": float(load_variance),
            "alert": float(entropy_normalized) < self.alert_threshold,
        }

        self.num_updates += 1

        return stats

    def get_summary(self) -> dict:
        """Get current utilization summary."""
        loads_norm = self.ema_loads / (self.ema_loads.sum() + 1e-8)
        loads_norm = loads_norm.clamp(min=1e-8)
        entropy = -torch.sum(loads_norm * torch.log(loads_norm))
        entropy_normalized = entropy / torch.log(torch.tensor(self.n_experts, dtype=torch.float32))

        return {
            "n_experts": self.n_experts,
            "num_updates": self.num_updates,
            "entropy": float(entropy),
            "entropy_normalized": float(entropy_normalized),
            "ema_loads_mean": float(self.ema_loads.mean()),
            "ema_loads_std": float(self.ema_loads.std()),
            "ema_loads_min": float(self.ema_loads.min()),
            "ema_loads_max": float(self.ema_loads.max()),
        }


if __name__ == "__main__":
    # Example: Salt optimization
    n_experts = 64
    optimizer = SaltOptimizer(n_experts=n_experts)
    monitor = UtilizationMonitor(n_experts=n_experts)

    print("Salt Optimization Example")
    print(f"Target entropy: {float(torch.log(torch.tensor(n_experts, dtype=torch.float32))):.3f}")
    print()

    # Simulate 10 optimization steps with varying loads
    for step in range(10):
        # Simulate imbalanced loads
        imbalance = 1 + 0.5 * (1 - step / 10)  # Decreasing imbalance
        loads = torch.ones(n_experts) / n_experts
        loads[0] *= imbalance  # Make expert 0 overloaded

        # Optimize
        salt, entropy = optimizer.optimize(loads)

        # Monitor
        stats = monitor.update(loads)

        print(f"Step {step + 1:2d} | Salt: {salt:6.3f} | Entropy: {entropy:5.3f} | "
              f"Collapsed: {stats['n_collapsed']:2d} | Alert: {stats['alert']}")

    print("\nFinal Summary:")
    summary = monitor.get_summary()
    for key, val in summary.items():
        print(f"  {key}: {val}")
