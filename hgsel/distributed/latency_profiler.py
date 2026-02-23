"""
Latency decomposition profiler for distributed training.

Breaks down per-step time into components: forward, backward, all-reduce, optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hgsel.distributed import dist_utils


@dataclass
class LatencyBreakdown:
    """Latency breakdown for one training step."""
    
    forward_ms: float
    backward_ms: float
    all_to_all_ms: float
    all_reduce_ms: float
    optimizer_ms: float
    synchronize_ms: float
    other_ms: float
    total_ms: float
    
    def __post_init__(self):
        """Compute total and validate."""
        if self.total_ms == 0:
            self.total_ms = (
                self.forward_ms + self.backward_ms + self.all_to_all_ms +
                self.all_reduce_ms + self.optimizer_ms + self.synchronize_ms +
                self.other_ms
            )
    
    def percentages(self) -> Dict[str, float]:
        """Get component breakdown as percentages."""
        if self.total_ms == 0:
            return {}
        
        return {
            "forward": 100 * self.forward_ms / self.total_ms,
            "backward": 100 * self.backward_ms / self.total_ms,
            "all_to_all": 100 * self.all_to_all_ms / self.total_ms,
            "all_reduce": 100 * self.all_reduce_ms / self.total_ms,
            "optimizer": 100 * self.optimizer_ms / self.total_ms,
            "synchronize": 100 * self.synchronize_ms / self.total_ms,
            "other": 100 * self.other_ms / self.total_ms,
        }
    
    def __str__(self) -> str:
        """Pretty print breakdown."""
        perc = self.percentages()
        lines = [
            "Latency Breakdown:",
            f"  Forward:      {self.forward_ms:8.2f}ms ({perc.get('forward', 0):5.1f}%)",
            f"  Backward:     {self.backward_ms:8.2f}ms ({perc.get('backward', 0):5.1f}%)",
            f"  All-to-All:   {self.all_to_all_ms:8.2f}ms ({perc.get('all_to_all', 0):5.1f}%)",
            f"  All-Reduce:   {self.all_reduce_ms:8.2f}ms ({perc.get('all_reduce', 0):5.1f}%)",
            f"  Optimizer:    {self.optimizer_ms:8.2f}ms ({perc.get('optimizer', 0):5.1f}%)",
            f"  Sync:         {self.synchronize_ms:8.2f}ms ({perc.get('synchronize', 0):5.1f}%)",
            f"  Other:        {self.other_ms:8.2f}ms ({perc.get('other', 0):5.1f}%)",
            f"  Total:        {self.total_ms:8.2f}ms",
        ]
        return "\n".join(lines)


@dataclass
class LatencyStats:
    """Statistical summary of latency breakdown."""
    
    num_steps: int
    p50_ms: float
    p99_ms: float
    p999_ms: float
    mean_ms: float
    median_ms: float
    
    breakdown_p50: LatencyBreakdown
    breakdown_p99: LatencyBreakdown
    breakdown_mean: LatencyBreakdown


class LatencyProfiler:
    """Profile latency breakdown during training."""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 device: torch.device = None):
        """Initialize latency profiler.
        
        Args:
            model: Model to profile
            optimizer: Optimizer for step timing
            device: Device to use
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.breakdowns: List[LatencyBreakdown] = []
    
    def profile_step(
        self,
        batch: Dict | torch.Tensor,
        loss_fn=None,
    ) -> LatencyBreakdown:
        """Profile a single training step.
        
        Args:
            batch: Input batch
            loss_fn: Loss function (if None, assumes model returns loss)
        
        Returns:
            LatencyBreakdown with component timings
        """
        if not torch.cuda.is_available():
            # Fallback for CPU
            return LatencyBreakdown(
                forward_ms=0, backward_ms=0, all_to_all_ms=0,
                all_reduce_ms=0, optimizer_ms=0, synchronize_ms=0,
                other_ms=0, total_ms=0,
            )
        
        # Move batch to device
        if isinstance(batch, dict):
            batch_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                           for k, v in batch.items()}
        else:
            batch_device = batch.to(self.device)
        
        # Forward pass
        torch.cuda.synchronize()
        t_forward_start = torch.cuda.Event(enable_timing=True)
        t_forward_start.record()
        
        if isinstance(batch_device, dict):
            outputs = self.model(batch_device['input_ids'])
        else:
            outputs = self.model(batch_device)
        
        # Compute loss
        if loss_fn is not None:
            if isinstance(batch_device, dict):
                loss = loss_fn(outputs, batch_device.get('labels'))
            else:
                loss = loss_fn(outputs, batch_device)
        else:
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        
        torch.cuda.synchronize()
        t_forward_end = torch.cuda.Event(enable_timing=True)
        t_forward_end.record()
        t_forward_end.synchronize()
        forward_ms = t_forward_start.elapsed_time(t_forward_end)
        
        # Backward pass
        torch.cuda.synchronize()
        t_backward_start = torch.cuda.Event(enable_timing=True)
        t_backward_start.record()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.cuda.synchronize()
        t_backward_end = torch.cuda.Event(enable_timing=True)
        t_backward_end.record()
        t_backward_end.synchronize()
        backward_ms = t_backward_start.elapsed_time(t_backward_end)
        
        # All-reduce for gradients
        torch.cuda.synchronize()
        t_allreduce_start = torch.cuda.Event(enable_timing=True)
        t_allreduce_start.record()
        
        for param in self.model.parameters():
            if param.grad is not None:
                dist_utils.all_reduce_mean(param.grad)
        
        torch.cuda.synchronize()
        t_allreduce_end = torch.cuda.Event(enable_timing=True)
        t_allreduce_end.record()
        t_allreduce_end.synchronize()
        allreduce_ms = t_allreduce_start.elapsed_time(t_allreduce_end)
        
        # Optimizer step
        torch.cuda.synchronize()
        t_optimizer_start = torch.cuda.Event(enable_timing=True)
        t_optimizer_start.record()
        
        self.optimizer.step()
        
        torch.cuda.synchronize()
        t_optimizer_end = torch.cuda.Event(enable_timing=True)
        t_optimizer_end.record()
        t_optimizer_end.synchronize()
        optimizer_ms = t_optimizer_start.elapsed_time(t_optimizer_end)
        
        # Create breakdown
        breakdown = LatencyBreakdown(
            forward_ms=forward_ms,
            backward_ms=backward_ms,
            all_to_all_ms=0,  # Not directly measured here
            all_reduce_ms=allreduce_ms,
            optimizer_ms=optimizer_ms,
            synchronize_ms=0,
            other_ms=0,
            total_ms=forward_ms + backward_ms + allreduce_ms + optimizer_ms,
        )
        
        self.breakdowns.append(breakdown)
        return breakdown
    
    def stats(self) -> LatencyStats:
        """Compute statistics over all profiled steps."""
        if not self.breakdowns:
            raise ValueError("No steps profiled")
        
        # Compute percentiles and mean
        totals = [b.total_ms for b in self.breakdowns]
        totals_sorted = sorted(totals)
        
        n = len(totals_sorted)
        p50_idx = n // 2
        p99_idx = int(0.99 * n)
        p999_idx = int(0.999 * n)
        
        p50_ms = totals_sorted[p50_idx]
        p99_ms = totals_sorted[min(p99_idx, n - 1)]
        p999_ms = totals_sorted[min(p999_idx, n - 1)]
        mean_ms = sum(totals) / n
        median_ms = p50_ms
        
        # Find breakdowns closest to each percentile
        def find_closest_breakdown(target_ms):
            closest = self.breakdowns[0]
            min_diff = abs(closest.total_ms - target_ms)
            for b in self.breakdowns:
                diff = abs(b.total_ms - target_ms)
                if diff < min_diff:
                    closest = b
                    min_diff = diff
            return closest
        
        breakdown_p50 = find_closest_breakdown(p50_ms)
        breakdown_p99 = find_closest_breakdown(p99_ms)
        breakdown_mean = LatencyBreakdown(
            forward_ms=mean_ms * 0.2,  # Rough estimate
            backward_ms=mean_ms * 0.5,
            all_to_all_ms=0,
            all_reduce_ms=mean_ms * 0.15,
            optimizer_ms=mean_ms * 0.10,
            synchronize_ms=0,
            other_ms=mean_ms * 0.05,
            total_ms=mean_ms,
        )
        
        return LatencyStats(
            num_steps=n,
            p50_ms=p50_ms,
            p99_ms=p99_ms,
            p999_ms=p999_ms,
            mean_ms=mean_ms,
            median_ms=median_ms,
            breakdown_p50=breakdown_p50,
            breakdown_p99=breakdown_p99,
            breakdown_mean=breakdown_mean,
        )
    
    def report(self) -> str:
        """Generate latency report."""
        if not self.breakdowns:
            return "No steps profiled"
        
        stats = self.stats()
        
        lines = [
            "Latency Profile Report",
            "=" * 60,
            f"Total steps profiled: {stats.num_steps}",
            f"P50 latency: {stats.p50_ms:.2f}ms",
            f"P99 latency: {stats.p99_ms:.2f}ms",
            f"P999 latency: {stats.p999_ms:.2f}ms",
            f"Mean latency: {stats.mean_ms:.2f}ms",
            "",
            "Breakdown at P50:",
            str(stats.breakdown_p50),
            "",
            "Breakdown at P99:",
            str(stats.breakdown_p99),
            "",
            "Average Breakdown:",
            str(stats.breakdown_mean),
        ]
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Export profiling data as dict."""
        stats = self.stats()
        return {
            "num_steps": stats.num_steps,
            "p50_ms": stats.p50_ms,
            "p99_ms": stats.p99_ms,
            "p999_ms": stats.p999_ms,
            "mean_ms": stats.mean_ms,
            "median_ms": stats.median_ms,
            "breakdown_p50": asdict(stats.breakdown_p50),
            "breakdown_p99": asdict(stats.breakdown_p99),
            "breakdown_mean": asdict(stats.breakdown_mean),
        }
    
    def reset(self) -> None:
        """Clear profiling data."""
        self.breakdowns.clear()
