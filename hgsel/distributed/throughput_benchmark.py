"""
Throughput measurement for distributed training.

Measures tokens/sec, FLOPs/sec, and scaling efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hgsel.distributed import dist_utils


@dataclass
class ThroughputMetrics:
    """Throughput and efficiency metrics."""
    
    total_steps: int
    total_tokens: int
    total_time_sec: float
    tokens_per_sec: float
    tokens_per_sec_per_gpu: float
    peak_flops_per_gpu: float
    achieved_flops_per_gpu: float
    utilization_percent: float
    world_size: int
    batch_size: int
    seq_length: int
    
    def __str__(self) -> str:
        """Pretty print metrics."""
        lines = [
            "Throughput Metrics:",
            f"  Tokens/sec (total): {self.tokens_per_sec:.1f}",
            f"  Tokens/sec/GPU:     {self.tokens_per_sec_per_gpu:.1f}",
            f"  Utilization:        {self.utilization_percent:.1f}%",
            f"  Total time:         {self.total_time_sec:.2f}s",
            f"  Total tokens:       {self.total_tokens:,}",
            f"  Steps:              {self.total_steps}",
        ]
        return "\n".join(lines)


class ThroughputBenchmark:
    """Benchmark training throughput."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        world_size: int = 1,
    ):
        """Initialize throughput benchmark.
        
        Args:
            model: Model to benchmark
            device: Device to use
            world_size: Number of distributed processes
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.world_size = world_size
    
    def run(
        self,
        data_loader: DataLoader,
        num_warmup_steps: int = 10,
        num_bench_steps: int = 100,
        measure_flops: bool = False,
    ) -> ThroughputMetrics:
        """Run throughput benchmark.
        
        Args:
            data_loader: Data loader with batches
            num_warmup_steps: Warmup steps before measurement
            num_bench_steps: Number of steps to measure
            measure_flops: Whether to estimate FLOPs
        
        Returns:
            ThroughputMetrics with measurements
        """
        self.model.eval()
        
        # Extract model config
        batch_size = data_loader.batch_size or 1
        seq_length = 128  # TODO: get from data_loader
        
        # Warm up
        iterator = iter(data_loader)
        for _ in range(num_warmup_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(data_loader)
                batch = next(iterator)
            
            with torch.no_grad():
                if isinstance(batch, dict):
                    batch_to_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                      for k, v in batch.items()}
                    self.model(batch_to_device['input_ids'])
                else:
                    self.model(batch.to(self.device))
        
        # Synchronize before benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dist_utils.barrier()
        
        # Benchmark
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start_time:
            start_time.record()
        
        total_tokens = 0
        for step in range(num_bench_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(data_loader)
                batch = next(iterator)
            
            with torch.no_grad():
                if isinstance(batch, dict):
                    batch_to_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                      for k, v in batch.items()}
                    self.model(batch_to_device['input_ids'])
                    # Count tokens
                    if 'input_ids' in batch:
                        total_tokens += batch['input_ids'].numel()
                else:
                    self.model(batch.to(self.device))
                    total_tokens += batch.numel()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if end_time:
            end_time.record()
            end_time.synchronize()
            elapsed_sec = start_time.elapsed_time(end_time) / 1000.0  # ms to sec
        else:
            elapsed_sec = 1.0  # Fallback
        
        # Aggregate across ranks
        total_tokens_tensor = torch.tensor(total_tokens, device=self.device, dtype=torch.long)
        dist_utils.all_reduce_sum(total_tokens_tensor)
        total_tokens = total_tokens_tensor.item()
        
        # Calculate metrics
        tokens_per_sec = total_tokens / elapsed_sec
        tokens_per_sec_per_gpu = tokens_per_sec / self.world_size
        
        # Estimate FLOPs (rough: 6*L*d*N*T for dense transformer)
        # N = num_tokens, T = seq_length, L = num_layers, d = d_model
        num_params = sum(p.numel() for p in self.model.parameters())
        estimated_flops = total_tokens * num_params * 2  # 2 FLOPs per param per token (rough)
        peak_flops_per_gpu = 10e12 / self.world_size  # Assume 10 TFLOPs peak per GPU (typical A100)
        achieved_flops_per_gpu = estimated_flops / elapsed_sec / self.world_size
        utilization = (achieved_flops_per_gpu / peak_flops_per_gpu) * 100 if peak_flops_per_gpu > 0 else 0
        
        return ThroughputMetrics(
            total_steps=num_bench_steps,
            total_tokens=total_tokens,
            total_time_sec=elapsed_sec,
            tokens_per_sec=tokens_per_sec,
            tokens_per_sec_per_gpu=tokens_per_sec_per_gpu,
            peak_flops_per_gpu=peak_flops_per_gpu,
            achieved_flops_per_gpu=achieved_flops_per_gpu,
            utilization_percent=utilization,
            world_size=self.world_size,
            batch_size=batch_size,
            seq_length=seq_length,
        )
    
    @staticmethod
    def estimate_peak_flops(batch_size: int, seq_length: int, d_model: int, 
                           num_layers: int, num_experts: int = 1) -> float:
        """Estimate peak FLOPs for a forward pass.
        
        Rough formula: FLOPs ≈ (batch * seq) * (d_model^2 + d_ff * d_model) * num_layers
        """
        num_tokens = batch_size * seq_length
        
        # Attention: d_model^2 per token
        attention_flops = num_tokens * (d_model * d_model)
        
        # FFN: 2 * d_model * d_ff per token (for 2 layers: linear + linear)
        d_ff = d_model * 4  # Standard expansion
        ffn_flops = num_tokens * (2 * d_model * d_ff)
        
        # Total for all layers
        total_flops = (attention_flops + ffn_flops) * num_layers
        
        # Account for expertise (FLOPs per token)
        return total_flops * 2  # 2x for forward + backward
