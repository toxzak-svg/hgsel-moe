"""
Benchmark: Original vs Optimized HGSEL Layer.

Run: python experiments/benchmark_speedups.py
"""

import time
import torch
import argparse
from typing import Dict, List

# Original implementations
from hgsel.layer.hgsel_layer import HGSELLayer as HGSELLayerOriginal
from hgsel.routing.hash_functions import MultiHashRouter as MultiHashRouterOriginal
from hgsel.expert.expert_bank import ExpertBank as ExpertBankOriginal

# Optimized implementations  
from hgsel.layer.hgsel_layer_fast import HGSELLayerFast
from hgsel.routing.hash_functions_fast import MultiHashRouterFast, InvertedDispatchExpertBank


def benchmark_layer(
    name: str,
    layer,
    hidden_states: torch.Tensor,
    n_runs: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """Benchmark a layer."""
    
    # Warmup
    for _ in range(warmup):
        _ = layer(hidden_states)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = layer(hidden_states)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # ms
    
    return {
        "name": name,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def benchmark_routing(name: str, router, hidden_states, n_runs=100, warmup=10):
    """Benchmark routing only."""
    
    for _ in range(warmup):
        _ = router(hidden_states)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = router(hidden_states)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    return {
        "name": name,
        "mean_ms": sum(times) / len(times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--n-experts", type=int, default=64)
    parser.add_argument("--k-active", type=int, default=2)
    parser.add_argument("--n-hashes", type=int, default=4)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"HGSEL Speed Benchmark")
    print(f"{'='*60}")
    print(f"Config: batch={args.batch_size}, seq={args.seq_len}, d_model={args.d_model}")
    print(f"Experts: N={args.n_experts}, k={args.k_active}, H={args.n_hashes}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")
    
    # Create inputs
    hidden_states = torch.randn(
        args.batch_size, args.seq_len, args.d_model, 
        device=args.device
    )
    hidden_states_flat = hidden_states.view(-1, args.d_model)
    
    print("Creating layers...")
    
    # Original layers
    original_router = MultiHashRouterOriginal(
        n_experts=args.n_experts,
        k_active=args.k_active,
        n_hashes=args.n_hashes,
        hidden_dim=args.d_model,
    ).to(args.device)
    
    original_expert_bank = ExpertBankOriginal(
        n_experts=args.n_experts,
        k_active=args.k_active,
        d_model=args.d_model,
        d_ff=args.d_ff,
    ).to(args.device)
    
    original_layer = HGSELLayerOriginal(
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_experts=args.n_experts,
        k_active=args.k_active,
        n_hashes=args.n_hashes,
    ).to(args.device)
    
    # Optimized layers
    fast_router = MultiHashRouterFast(
        n_experts=args.n_experts,
        k_active=args.k_active,
        n_hashes=args.n_hashes,
        hidden_dim=args.d_model,
    ).to(args.device)
    
    fast_expert_bank = InvertedDispatchExpertBank(
        n_experts=args.n_experts,
        k_active=args.k_active,
        d_model=args.d_model,
        d_ff=args.d_ff,
    ).to(args.device)
    
    fast_layer = HGSELLayerFast(
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_experts=args.n_experts,
        k_active=args.k_active,
        n_hashes=args.n_hashes,
    ).to(args.device)
    
    fast_layer_bf16 = HGSELLayerFast(
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_experts=args.n_experts,
        k_active=args.k_active,
        n_hashes=args.n_hashes,
        use_bf16=True,
    ).to(args.device)
    
    print("Running benchmarks...\n")
    
    # Benchmark routing
    print("=== ROUTING ===")
    r_original = benchmark_routing("Original", original_router, hidden_states_flat, args.n_runs)
    r_fast = benchmark_routing("Fast", fast_router, hidden_states_flat, args.n_runs)
    speedup = r_original["mean_ms"] / r_fast["mean_ms"]
    print(f"Original: {r_original['mean_ms']:.3f} ms")
    print(f"Fast:     {r_fast['mean_ms']:.3f} ms")
    print(f"Speedup:  {speedup:.2f}x\n")
    
    # Benchmark full layers
    print("=== FULL LAYER ===")
    results = []
    
    for name, layer in [
        ("Original", original_layer),
        ("Fast (fp32)", fast_layer),
        ("Fast (bf16)", fast_layer_bf16),
    ]:
        print(f"Benchmarking {name}...", end=" ", flush=True)
        r = benchmark_layer(name, layer, hidden_states, args.n_runs)
        results.append(r)
        print(f"{r['mean_ms']:.3f} ms")
    
    print("\n=== RESULTS SUMMARY ===")
    print(f"{'Name':<15} {'Mean (ms)':<12} {'Min':<12} {'Speedup vs Original'}")
    print("-" * 55)
    original_mean = results[0]["mean_ms"]
    for r in results:
        speedup = original_mean / r["mean_ms"]
        print(f"{r['name']:<15} {r['mean_ms']:<12.3f} {r['min_ms']:<12.3f} {speedup:.2f}x")
    
    # Also test with H=2, k=2 (minimal hashing)
    print("\n\n=== EXPERIMENT: n_hashes = k_active ===")
    print("Using H=2 instead of H=4 for k=2\n")
    
    original_router_h2 = MultiHashRouterOriginal(
        n_experts=args.n_experts,
        k_active=args.k_active,
        n_hashes=2,  # Same as k_active
        hidden_dim=args.d_model,
    ).to(args.device)
    
    fast_router_h2 = MultiHashRouterFast(
        n_experts=args.n_experts,
        k_active=args.k_active,
        n_hashes=2,
        hidden_dim=args.d_model,
    ).to(args.device)
    
    r_orig_h2 = benchmark_routing("Original H=2", original_router_h2, hidden_states_flat, args.n_runs)
    r_fast_h2 = benchmark_routing("Fast H=2", fast_router_h2, hidden_states_flat, args.n_runs)
    
    print(f"Original (H=4): {r_original['mean_ms']:.3f} ms")
    print(f"Original (H=2): {r_orig_h2['mean_ms']:.3f} ms")
    print(f"Fast (H=2):     {r_fast_h2['mean_ms']:.3f} ms")
    print(f"Speedup (H=2 vs H=4): {r_original['mean_ms']/r_fast_h2['mean_ms']:.2f}x")
    
    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)


if __name__ == "__main__":
    main()
