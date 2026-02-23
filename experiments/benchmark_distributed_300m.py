#!/usr/bin/env python
"""
Distributed benchmark script for HGSEL 300M model.

Measures throughput, memory, and latency across different configurations.

Usage:
    # Single GPU
    python experiments/benchmark_distributed_300m.py --num-gpus 1 --batch-size 32

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 experiments/benchmark_distributed_300m.py \
        --num-gpus 4 --batch-size 32

    # Configuration sweep
    python experiments/benchmark_distributed_300m.py \
        --num-gpus 4 \
        --batch-sizes 16,32,64 \
        --expert-counts 64,128 \
        --output results/benchmark_sweep
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from hgsel.distributed import dist_utils
from hgsel.distributed.memory_profiler import MemoryProfiler, estimate_model_memory_requirements
from hgsel.distributed.throughput_benchmark import ThroughputBenchmark
from hgsel.distributed.latency_profiler import LatencyProfiler
from hgsel.training.data import DummyDataLoader
from experiments.baselines.dense_transformer import TransformerModel


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="HGSEL 300M Distributed Benchmark")
    
    # Benchmark configuration
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to simulate")
    parser.add_argument("--batch-sizes", type=str, default="32",
                       help="Batch sizes to test (comma-separated)")
    parser.add_argument("--expert-counts", type=str, default="64",
                       help="Expert counts to test (comma-separated)")
    parser.add_argument("--num-warmup-steps", type=int, default=10, help="Warmup steps")
    parser.add_argument("--num-bench-steps", type=int, default=100, help="Benchmark steps")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    
    # Model configuration
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--d-ff", type=int, default=1024, help="FFN hidden dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size")
    
    # Output
    parser.add_argument("--output", type=str, default="results/benchmark",
                       help="Output directory for results")
    parser.add_argument("--profile-memory", action="store_true", help="Profile memory usage")
    parser.add_argument("--profile-latency", action="store_true", help="Profile latency")
    
    return parser.parse_args()


def parse_list_arg(arg_str: str) -> List[int]:
    """Parse comma-separated list of integers."""
    return [int(x.strip()) for x in arg_str.split(",")]


def create_model(args) -> nn.Module:
    """Create model for benchmarking."""
    return TransformerModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_seq_length=args.seq_length,
    )


def benchmark_config(
    model: nn.Module,
    batch_size: int,
    num_experts: int,
    seq_length: int,
    args: argparse.Namespace,
    rank: int = 0,
    world_size: int = 1,
) -> dict:
    """Run benchmark for a single configuration.
    
    Args:
        model: Model to benchmark
        batch_size: Batch size
        num_experts: Number of experts
        seq_length: Sequence length
        args: Command-line arguments
        rank: Process rank
        world_size: Total processes
    
    Returns:
        Dict with benchmark results
    """
    device = dist_utils.get_device() if world_size > 1 else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)
    
    results = {
        "batch_size": batch_size,
        "num_experts": num_experts,
        "seq_length": seq_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "world_size": world_size,
        "rank": rank,
    }
    
    # Create data loader
    data_loader = DummyDataLoader(
        num_batches=args.num_bench_steps + args.num_warmup_steps,
        batch_size=batch_size,
        seq_length=seq_length,
        vocab_size=args.vocab_size,
    )
    
    # Throughput benchmark
    benchmark = ThroughputBenchmark(model, device=device, world_size=world_size)
    metrics = benchmark.run(
        data_loader,
        num_warmup_steps=args.num_warmup_steps,
        num_bench_steps=args.num_bench_steps,
    )
    
    results["throughput"] = {
        "tokens_per_sec": metrics.tokens_per_sec,
        "tokens_per_sec_per_gpu": metrics.tokens_per_sec_per_gpu,
        "utilization_percent": metrics.utilization_percent,
        "total_time_sec": metrics.total_time_sec,
    }
    
    # Memory profiling
    if args.profile_memory:
        memory_profiler = MemoryProfiler(model)
        memory_profiler.take_snapshot("start")
        
        # Forward pass
        batch = next(iter(data_loader))
        if isinstance(batch, dict):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            model(batch['input_ids'])
        else:
            model(batch.to(device))
        
        memory_profiler.take_snapshot("after_forward")
        
        mem_reqs = estimate_model_memory_requirements(model)
        results["memory"] = {
            "snapshots": memory_profiler.to_dict(),
            "estimates": mem_reqs,
        }
    
    # Latency profiling
    if args.profile_latency:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        latency_profiler = LatencyProfiler(model, optimizer, device=device)
        
        # Profile a few steps
        for _ in range(5):
            batch = next(iter(data_loader))
            latency_profiler.profile_step(batch)
        
        results["latency"] = latency_profiler.to_dict()
    
    return results


def main():
    """Main benchmark loop."""
    args = parse_args()
    rank = dist_utils.get_rank()
    world_size = dist_utils.get_world_size()
    is_master = rank == 0
    
    # Parse configurations
    batch_sizes = parse_list_arg(args.batch_sizes)
    expert_counts = parse_list_arg(args.expert_counts)
    
    if is_master:
        print(f"HGSEL 300M Distributed Benchmark")
        print(f"Rank: {rank}, World Size: {world_size}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Expert counts: {expert_counts}")
        print()
    
    # Create output directory
    output_dir = Path(args.output)
    if is_master:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    dist_utils.barrier()
    
    # Run benchmark sweep
    all_results = []
    
    for batch_size in batch_sizes:
        for num_experts in expert_counts:
            if is_master:
                print(f"Benchmarking: batch_size={batch_size}, num_experts={num_experts}")
            
            # Create fresh model
            model = create_model(args)
            
            # Run benchmark
            result = benchmark_config(
                model,
                batch_size=batch_size,
                num_experts=num_experts,
                seq_length=args.seq_length,
                args=args,
                rank=rank,
                world_size=world_size,
            )
            
            all_results.append(result)
            
            if is_master:
                throughput = result["throughput"]
                print(f"  Throughput: {throughput['tokens_per_sec']:.1f} tokens/sec")
                if "memory" in result:
                    print(f"  Memory: {result['memory']['snapshots'][-1]['allocated_mb']:.1f} MB")
                print()
    
    # Save results
    if is_master:
        results_file = output_dir / "benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Results saved to {results_file}")
        
        # Print summary
        print("\nBenchmark Summary:")
        print("=" * 60)
        for result in all_results:
            batch_size = result["batch_size"]
            num_experts = result["num_experts"]
            throughput = result["throughput"]["tokens_per_sec"]
            print(f"  B={batch_size:3d}, E={num_experts:3d}: {throughput:8.1f} tokens/sec")


if __name__ == "__main__":
    main()
