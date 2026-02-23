"""
Memory profiling utilities for distributed training.

Tracks memory usage breakdown: parameters, gradients, activations, optimizer state.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class MemorySnapshot:
    """Memory usage at a point in time."""
    
    allocated_mb: float         # Allocated memory in MB
    reserved_mb: float          # Reserved memory in MB
    peak_allocated_mb: float    # Peak allocated in session
    
    # Breakdown (rough estimates)
    param_mb: float = 0.0
    grad_mb: float = 0.0
    activation_mb: float = 0.0
    optimizer_state_mb: float = 0.0
    buffer_mb: float = 0.0
    
    def __str__(self) -> str:
        """Pretty print memory snapshot."""
        lines = [
            f"Memory Snapshot:",
            f"  Allocated:  {self.allocated_mb:8.1f} MB",
            f"  Reserved:   {self.reserved_mb:8.1f} MB",
            f"  Peak:       {self.peak_allocated_mb:8.1f} MB",
            f"  Breakdown:",
            f"    Params:   {self.param_mb:8.1f} MB",
            f"    Grads:    {self.grad_mb:8.1f} MB",
            f"    Activations: {self.activation_mb:8.1f} MB",
            f"    Optimizer: {self.optimizer_state_mb:8.1f} MB",
            f"    Other:    {self.buffer_mb:8.1f} MB",
        ]
        return "\n".join(lines)


class MemoryProfiler:
    """Profile memory usage during training."""
    
    def __init__(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        """Initialize memory profiler.
        
        Args:
            model: Model to profile
            optimizer: Optimizer (for state estimation)
        """
        self.model = model
        self.optimizer = optimizer
        self.device = next(model.parameters()).device
        self.snapshots: list[MemorySnapshot] = []
    
    def take_snapshot(self, step_name: str = "") -> MemorySnapshot:
        """Take a memory snapshot.
        
        Args:
            step_name: Name for logging
        
        Returns:
            MemorySnapshot with current usage
        """
        if not torch.cuda.is_available():
            return MemorySnapshot(
                allocated_mb=0.0,
                reserved_mb=0.0,
                peak_allocated_mb=0.0,
            )
        
        # GPU memory
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        
        # Breakdown
        param_mb = self._estimate_param_memory()
        grad_mb = self._estimate_grad_memory()
        opt_mb = self._estimate_optimizer_state_memory()
        
        snapshot = MemorySnapshot(
            allocated_mb=allocated,
            reserved_mb=reserved,
            peak_allocated_mb=peak,
            param_mb=param_mb,
            grad_mb=grad_mb,
            activation_mb=max(0, allocated - param_mb - grad_mb - opt_mb),
            optimizer_state_mb=opt_mb,
            buffer_mb=0.0,
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def _estimate_param_memory(self) -> float:
        """Estimate total parameter memory in MB."""
        total = 0
        for param in self.model.parameters():
            total += param.numel() * param.element_size()
        return total / (1024 ** 2)
    
    def _estimate_grad_memory(self) -> float:
        """Estimate gradient memory in MB."""
        total = 0
        for param in self.model.parameters():
            if param.grad is not None:
                total += param.grad.numel() * param.grad.element_size()
        return total / (1024 ** 2)
    
    def _estimate_optimizer_state_memory(self) -> float:
        """Estimate optimizer state memory in MB.
        
        For Adam: 2 states per param (momentum, variance)
        For SGD: 1 state per param (momentum) or 0
        """
        if self.optimizer is None:
            return 0.0
        
        total = 0
        
        # Estimate based on optimizer type
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                
                # Check optimizer state
                if param in self.optimizer.state:
                    state = self.optimizer.state[param]
                    for v in state.values():
                        if isinstance(v, torch.Tensor):
                            total += v.numel() * v.element_size()
        
        return total / (1024 ** 2)
    
    def report(self) -> str:
        """Generate memory report."""
        if not self.snapshots:
            return "No snapshots taken"
        
        lines = ["Memory Profile Report", "=" * 60]
        
        # Over time
        lines.append("\nMemory Over Time:")
        for i, snap in enumerate(self.snapshots):
            lines.append(f"  Step {i}: {snap.allocated_mb:.1f} MB allocated, {snap.peak_allocated_mb:.1f} MB peak")
        
        # Summary
        if self.snapshots:
            final = self.snapshots[-1]
            max_allocated = max(s.allocated_mb for s in self.snapshots)
            max_peak = max(s.peak_allocated_mb for s in self.snapshots)
            
            lines.append(f"\nFinal memory: {final.allocated_mb:.1f} MB allocated")
            lines.append(f"Max allocated: {max_allocated:.1f} MB")
            lines.append(f"Max peak: {max_peak:.1f} MB")
            
            # Breakdown
            lines.append(f"\nMemory Breakdown (final):")
            lines.append(f"  Params:     {final.param_mb:.1f} MB ({100*final.param_mb/final.allocated_mb:.1f}%)")
            lines.append(f"  Grads:      {final.grad_mb:.1f} MB ({100*final.grad_mb/final.allocated_mb:.1f}%)")
            lines.append(f"  Activations: {final.activation_mb:.1f} MB ({100*final.activation_mb/final.allocated_mb:.1f}%)")
            lines.append(f"  Optimizer:  {final.optimizer_state_mb:.1f} MB ({100*final.optimizer_state_mb/final.allocated_mb:.1f}%)")
        
        return "\n".join(lines)
    
    def to_dict(self) -> list[Dict]:
        """Convert snapshots to list of dicts."""
        return [asdict(s) for s in self.snapshots]
    
    def reset(self) -> None:
        """Clear snapshots and reset peak memory counter."""
        self.snapshots.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def estimate_model_memory_requirements(model: nn.Module) -> Dict[str, float]:
    """Estimate memory requirements for a model.
    
    Args:
        model: Model to analyze
    
    Returns:
        Dict with memory estimates:
        - params_mb: Parameter memory
        - grads_mb: Gradient memory
        - activation_mb: Typical activation memory (estimated)
        - optimizer_adam_mb: Adam optimizer state
        - optimizer_sgd_mb: SGD optimizer state
        - total_train_mb: Total for training (params + grads + activations + optimizer)
    """
    param_size = 0
    
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    param_mb = param_size / (1024 ** 2)
    grad_mb = param_mb  # Roughly same size as params
    
    # Activation memory (rough estimate: 2-3x params for transformer)
    activation_mb = param_mb * 2.5
    
    # Optimizer state
    adam_mb = param_mb * 2  # momentum + variance
    sgd_mb = param_mb * 0.5  # momentum only (optional)
    
    total_adam = param_mb + grad_mb + activation_mb + adam_mb
    total_sgd = param_mb + grad_mb + activation_mb + sgd_mb
    
    return {
        "params_mb": param_mb,
        "grads_mb": grad_mb,
        "activation_mb": activation_mb,
        "optimizer_adam_mb": adam_mb,
        "optimizer_sgd_mb": sgd_mb,
        "total_train_with_adam_mb": total_adam,
        "total_train_with_sgd_mb": total_sgd,
    }
