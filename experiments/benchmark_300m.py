"""
Benchmarking script: Compare HGSEL vs Dense baseline.

Measures:
- Forward pass time
- Memory usage
- Training throughput
- Loss convergence
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time
from hgsel.layer import HGSELLayer
from hgsel.training.data import get_dummy_loaders
from experiments.baselines.dense_transformer import TransformerModel, DenseMLPBlock


def benchmark_forward_pass(model, batch_size=32, seq_len=128, vocab_size=256, num_iters=10):
    """Benchmark forward pass time and memory."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    model.eval()
    device = next(model.parameters()).device

    # Create dummy batch
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    input_ids = input_ids.to(device)

    # Warmup
    with torch.no_grad():
        _ = model(input_ids)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Time
    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(input_ids)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time

    throughput = (batch_size * seq_len * num_iters) / elapsed_time
    avg_time = elapsed_time / num_iters

    memory_peak = 0
    if torch.cuda.is_available():
        memory_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "avg_time_ms": avg_time * 1000,
        "throughput_tokens_per_sec": throughput,
        "memory_mb": memory_peak,
    }


def benchmark_backward_pass(model, batch_size=32, seq_len=128, vocab_size=256, num_iters=5):
    """Benchmark backward pass / training step."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    model.train()
    device = next(model.parameters()).device

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create dummy batch
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    input_ids = input_ids.to(device)
    labels = labels.to(device)

    # Warmup
    logits = model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.view(batch_size * seq_len, -1),
        labels.view(batch_size * seq_len),
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for _ in range(num_iters):
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(batch_size * seq_len, -1),
            labels.view(batch_size * seq_len),
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time

    throughput = (batch_size * seq_len * num_iters) / elapsed_time
    avg_time = elapsed_time / num_iters

    memory_peak = 0
    if torch.cuda.is_available():
        memory_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "avg_time_ms": avg_time * 1000,
        "throughput_tokens_per_sec": throughput,
        "memory_mb": memory_peak,
    }


def main():
    print("=" * 80)
    print("HGSEL vs Dense Baseline Benchmarking")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}\n")

    # Small model for fast benchmarking
    config = {
        "vocab_size": 256,
        "d_model": 256,
        "d_ff": 1024,
        "n_layers": 4,
        "n_heads": 4,
    }

    # Create models
    print("Creating models...")
    model_dense = TransformerModel(**config, mlp_class=DenseMLPBlock).to(device)
    model_hgsel = TransformerModel(**config, mlp_class=HGSELLayer).to(device)

    n_params_dense = model_dense.count_parameters()
    n_params_hgsel = model_hgsel.count_parameters()

    print(f"  Dense model: {n_params_dense:,} parameters")
    print(f"  HGSEL model: {n_params_hgsel:,} parameters")
    print(f"  Ratio: {n_params_hgsel / n_params_dense:.2f}x\n")

    # Benchmark forward pass
    print("Benchmarking forward pass (10 iterations)...")
    results_dense_fwd = benchmark_forward_pass(model_dense)
    results_hgsel_fwd = benchmark_forward_pass(model_hgsel)

    print(f"\nForward Pass Results:")
    print(f"  {'Metric':<30} {'Dense':<20} {'HGSEL':<20} {'Ratio':<10}")
    print(f"  {'-'*80}")

    metrics = ["avg_time_ms", "throughput_tokens_per_sec", "memory_mb"]
    for metric in metrics:
        dense_val = results_dense_fwd[metric]
        hgsel_val = results_hgsel_fwd[metric]

        if metric == "memory_mb":
            print(f"  {metric:<30} {dense_val:<20.2f} {hgsel_val:<20.2f} {hgsel_val/dense_val if dense_val > 0 else 0:<10.2f}x")
        elif metric == "throughput_tokens_per_sec":
            print(f"  {metric:<30} {dense_val:<20.0f} {hgsel_val:<20.0f} {dense_val/hgsel_val if hgsel_val > 0 else 0:<10.2f}x")
        else:
            print(f"  {metric:<30} {dense_val:<20.2f} {hgsel_val:<20.2f} {hgsel_val/dense_val if dense_val > 0 else 0:<10.2f}x")

    # Benchmark backward pass (training)
    print("\n\nBenchmarking backward pass / training (5 iterations)...")
    results_dense_bwd = benchmark_backward_pass(model_dense)
    results_hgsel_bwd = benchmark_backward_pass(model_hgsel)

    print(f"\nBackward Pass Results:")
    print(f"  {'Metric':<30} {'Dense':<20} {'HGSEL':<20} {'Ratio':<10}")
    print(f"  {'-'*80}")

    for metric in metrics:
        dense_val = results_dense_bwd[metric]
        hgsel_val = results_hgsel_bwd[metric]

        if metric == "memory_mb":
            print(f"  {metric:<30} {dense_val:<20.2f} {hgsel_val:<20.2f} {hgsel_val/dense_val if dense_val > 0 else 0:<10.2f}x")
        elif metric == "throughput_tokens_per_sec":
            print(f"  {metric:<30} {dense_val:<20.0f} {hgsel_val:<20.0f} {dense_val/hgsel_val if hgsel_val > 0 else 0:<10.2f}x")
        else:
            print(f"  {metric:<30} {dense_val:<20.2f} {hgsel_val:<20.2f} {hgsel_val/dense_val if dense_val > 0 else 0:<10.2f}x")

    print("\n" + "=" * 80)
    print("✓ Benchmarking complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
