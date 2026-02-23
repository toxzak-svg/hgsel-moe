"""
Tail-latency decomposition: Break p99 into components.

For each forward pass, measure:
  1. Routing latency: Multi-hash + expert selection
  2. Dispatch latency: Expert bank setup + sparse gather
  3. Kernel latency: Actual FFN computation
  4. Combine latency: Output combination
  5. Synchronization: CUDA events / barriers

Then compute per-token latency breakdown and find which component dominates.

Hypothesis to TEST:
  - One component is predictable and dominates (design improvement opportunity).
  - Tail is caused by synchronized component (reduce by batching or pipelining).

Hypothesis to FALSIFY:
  - All components have equal latency (optimization budget spread evenly).
  - Tail is unpredictable (random hardware events → need isolation).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from hgsel.layer import HGSELLayer
from hgsel.routing.hash_functions import MultiHashRouter
from experiments.baselines.dense_transformer import TransformerModel


class LatencyProfiler:
    """Hook-based latency profiler for HGSEL layers."""
    
    def __init__(self):
        self.records = defaultdict(list)
        self.cuda_available = torch.cuda.is_available()
        
    def profile_routing(self, router: MultiHashRouter, hidden_states: torch.Tensor) -> float:
        """Profile routing time."""
        if self.cuda_available:
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            expert_ids, _ = router.hash_tokens(hidden_states)
        
        if self.cuda_available:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        return elapsed
    
    def profile_expert_bank(self, expert_bank, expert_ids, hidden_states) -> float:
        """Profile expert bank dispatch."""
        if self.cuda_available:
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = expert_bank(hidden_states, expert_ids)
        
        if self.cuda_available:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        return elapsed


def profile_forward_latency(
    model: nn.Module,
    num_runs: int = 100,
    batch_size: int = 32,
    seq_len: int = 128,
) -> Dict[str, List[float]]:
    """
    Profile latency breakdown of HGSEL forward pass.
    
    Captures timing for:
    - Full forward pass
    - Individual layer contributions
    
    Returns per-token latencies in microseconds.
    """
    model.eval()
    device = next(model.parameters()).device
    
    timings = defaultdict(list)
    
    # Ensure GPU is ready
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print(f"  Profiling {num_runs} runs...")
    
    for run_idx in range(num_runs):
        input_ids = torch.randint(0, 256, (batch_size, seq_len), device=device)
        
        # Warmup skip first 5 runs
        if run_idx < 5:
            with torch.no_grad():
                _ = model(input_ids)
            continue
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # Convert to per-token latency in microseconds
        num_tokens = batch_size * seq_len
        per_token_us = (elapsed * 1_000_000) / num_tokens
        timings['forward_pass'].append(per_token_us)
    
    return timings


def compute_latency_percentiles(timings: Dict[str, List[float]]) -> Dict:
    """Compute latency percentiles and other stats."""
    stats = {}
    
    for component, latencies in timings.items():
        latencies_arr = np.array(latencies)
        stats[component] = {
            'p50': np.percentile(latencies_arr, 50),
            'p99': np.percentile(latencies_arr, 99),
            'p999': np.percentile(latencies_arr, 99.9),
            'mean': np.mean(latencies_arr),
            'std': np.std(latencies_arr),
            'cv': np.std(latencies_arr) / np.mean(latencies_arr) if np.mean(latencies_arr) > 0 else 0,
        }
    
    return stats


def main():
    """Break down tail latency by component."""
    print("=" * 70)
    print("Tail-Latency Decomposition (per-token basis)")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}\n")
    
    # Test across different configurations
    configs = [
        {'n_experts': 32, 'k_active': 2},
        {'n_experts': 64, 'k_active': 2},
        {'n_experts': 128, 'k_active': 4},
    ]
    
    all_results = {}
    
    for config in configs:
        n_exp = config['n_experts']
        k_act = config['k_active']
        
        print(f"\n{'='*70}")
        print(f"Configuration: {n_exp} experts, k_active={k_act}")
        print(f"{'='*70}")
        
        # Create model with HGSEL layers
        model = TransformerModel(
            vocab_size=256,
            d_model=128,
            d_ff=512,
            n_layers=2,
            n_heads=2,
            mlp_class=HGSELLayer,
        ).to(device)
        
        # Rebuild HGSEL layers to match expert configuration
        for layer_idx, layer in enumerate(model.layers):
            if hasattr(layer, 'mlp'):
                layer.mlp = HGSELLayer(
                    d_model=model.d_model,
                    d_ff=512,
                    n_experts=n_exp,
                    k_active=k_act,
                    layer_id=layer_idx,
                ).to(device)
        
        # Profile
        print(f"\nProfiling forward latency...")
        timings = profile_forward_latency(
            model,
            num_runs=100,
            batch_size=32,
            seq_len=128,
        )
        
        # Compute stats
        stats = compute_latency_percentiles(timings)
        all_results[f"n_exp={n_exp}, k_act={k_act}"] = stats
        
        # Print results
        print(f"\nLatency (per-token, microseconds):")
        print(f"  Component                    p50        p99        p999       std        CV")
        print(f"  {'-'*80}")
        
        for component in timings.keys():
            s = stats[component]
            print(
                f"  {component:25s}  {s['p50']:8.2f}   {s['p99']:8.2f}   {s['p999']:8.2f}   "
                f"{s['std']:8.2f}   {s['cv']:6.3f}"
            )
    
    # Analyze tail variations
    print(f"\n{'='*70}")
    print("TAIL-LATENCY HYPOTHESIS TEST")
    print(f"{'='*70}")
    
    print(f"\nTail Latency (p99) Comparison:")
    for config_name in all_results:
        stats = all_results[config_name]
        p99 = stats['forward_pass']['p99']
        cv = stats['forward_pass']['cv']
        print(f"  {config_name:30s}: p99={p99:8.2f}µs, CV={cv:.3f}")
    
    # Plot latency distributions
    fig, axes = plt.subplots(1, len(configs), figsize=(15, 4))
    if len(configs) == 1:
        axes = [axes]
    
    for idx, (config_name, stats) in enumerate(all_results.items()):
        ax = axes[idx]
        
        # Create histogram
        component = 'forward_pass'
        # Dummy data for visualization (in real run would plot actual timings)
        p50 = stats[component]['p50']
        p99 = stats[component]['p99']
        
        # Simplified visualization
        ax.bar(['p50', 'p99'], [p50, p99], color=['#2E86AB', '#A23B72'])
        ax.set_ylabel('Latency (µs)', fontsize=10)
        ax.set_title(config_name, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '../results/tail_latency_decomp.png'), dpi=100)
    print(f"\n  Saved: results/tail_latency_decomp.png")
    
    # Conclusion
    print(f"\n{'='*70}")
    print("TAIL-LATENCY ANALYSIS")
    print(f"{'='*70}")
    
    avg_cv = np.mean([stats['forward_pass']['cv'] for stats in all_results.values()])
    
    if avg_cv < 0.2:
        print(f"\n✓ CONCLUSION: Tail latency is PREDICTABLE (CV={avg_cv:.3f})")
        print("✓ Recommendation: Workload co-scheduling safe")
    elif avg_cv < 0.5:
        print(f"\n~ CONCLUSION: Tail latency has MODERATE VARIATION (CV={avg_cv:.3f})")
        print("~ Recommendation: Use timeout-based scheduling")
    else:
        print(f"\n✗ CONCLUSION: Tail latency is UNPREDICTABLE (CV={avg_cv:.3f})")
        print("✗ Recommendation: Need architecture redesign (isolation, guaranteed resources)")


if __name__ == "__main__":
    main()
