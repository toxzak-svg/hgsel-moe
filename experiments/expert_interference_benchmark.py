"""
Expert interference benchmarks: Run concurrent workloads.

Simulate two concurrent workload types (coding + math) and measure:
  1. Cache interference: How much does one workload's routing affect the other?
  2. Quality cross-talk: Do experts trained for one task degrade on the other?
  3. Latency blowup: What's the SLO impact when workloads interfere?

Setup:
  - Workload A: "Coding" tokens (specific random seed)
  - Workload B: "Math" tokens (different random seed)
  - Run them:
    1. Isolated (baseline)
    2. Interleaved (worst-case cache interference)
    3. Co-scheduled (realistic sharing)

Hypothesis to TEST:
  - Interference is small → caching is safe, multi-tenancy viable.
  - Interference is measurable → use partitioning / QoS knobs.

Hypothesis to FALSIFY:
  - No interference (then you don't need isolation).
  - Catastrophic interference (then you need strict partitioning).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from hgsel.layer import HGSELLayer
from experiments.baselines.dense_transformer import TransformerModel


class WorkloadGenerator:
    """Generate workload tokens with task-specific characteristics."""
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
    
    def generate_coding_tokens(self, num_tokens: int, seed: int = 42) -> torch.Tensor:
        """Generate "coding" workload tokens."""
        rng = np.random.RandomState(seed)
        # Skewed distribution (favor certain token ranges)
        tokens = rng.choice(
            np.arange(self.vocab_size),
            size=num_tokens,
            p=np.array([0.5 / self.vocab_size] * (self.vocab_size // 2) +
                       [1.5 / self.vocab_size] * (self.vocab_size // 2))
        )
        return torch.from_numpy(tokens).long()
    
    def generate_math_tokens(self, num_tokens: int, seed: int = 123) -> torch.Tensor:
        """Generate "math" workload tokens."""
        rng = np.random.RandomState(seed)
        # Different skew
        tokens = rng.choice(
            np.arange(self.vocab_size),
            size=num_tokens,
            p=np.array([1.5 / self.vocab_size] * (self.vocab_size // 2) +
                       [0.5 / self.vocab_size] * (self.vocab_size // 2))
        )
        return torch.from_numpy(tokens).long()


def run_workload_baseline(
    model_a: nn.Module,
    model_b: nn.Module,
    num_batches: int = 20,
    batch_size: int = 32,
    seq_len: int = 128,
) -> Dict[str, List[float]]:
    """
    Baseline: Run workloads in isolation.
    
    Returns per-token latencies for each workload.
    """
    device = next(model_a.parameters()).device
    model_a.eval()
    model_b.eval()
    
    generator = WorkloadGenerator()
    
    timings_a = []
    timings_b = []
    
    print(f"\n  Running ISOLATED baselines...")
    
    # Workload A
    print(f"    Workload A (coding)...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            tokens = generator.generate_coding_tokens(batch_size * seq_len)
            tokens = tokens.reshape(batch_size, seq_len).to(device)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model_a(tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            per_token_us = (elapsed * 1_000_000) / (batch_size * seq_len)
            timings_a.append(per_token_us)
    
    # Workload B
    print(f"    Workload B (math)...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            tokens = generator.generate_math_tokens(batch_size * seq_len)
            tokens = tokens.reshape(batch_size, seq_len).to(device)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model_b(tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            per_token_us = (elapsed * 1_000_000) / (batch_size * seq_len)
            timings_b.append(per_token_us)
    
    return {
        'workload_a': timings_a,
        'workload_b': timings_b,
    }


def run_workload_interleaved(
    model_a: nn.Module,
    model_b: nn.Module,
    num_batches: int = 20,
    batch_size: int = 32,
    seq_len: int = 128,
    pattern: str = "alternating",
) -> Dict[str, List[float]]:
    """
    Co-schedule: Run workloads interleaved (alternating or random).
    
    This tests cache interference in the worst case (L1/L2 pressure).
    
    Args:
        pattern: 'alternating' (A-B-A-B) or 'random' (random permutation)
    """
    device = next(model_a.parameters()).device
    model_a.eval()
    model_b.eval()
    
    generator = WorkloadGenerator()
    
    timings_a = []
    timings_b = []
    
    print(f"\n  Running INTERLEAVED (pattern={pattern})...")
    
    # Generate all batches upfront
    batches_a = []
    batches_b = []
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            tokens_a = generator.generate_coding_tokens(batch_size * seq_len)
            tokens_a = tokens_a.reshape(batch_size, seq_len).to(device)
            batches_a.append(tokens_a)
            
            tokens_b = generator.generate_math_tokens(batch_size * seq_len)
            tokens_b = tokens_b.reshape(batch_size, seq_len).to(device)
            batches_b.append(tokens_b)
        
        # Interleave based on pattern
        if pattern == "alternating":
            schedule = []
            for i in range(num_batches):
                schedule.append(('a', i))
                schedule.append(('b', i))
        else:  # random
            rng = np.random.RandomState(777)
            schedule = (
                [('a', i) for i in range(num_batches)] +
                [('b', i) for i in range(num_batches)]
            )
            rng.shuffle(schedule)
        
        # Execute schedule
        for workload_type, batch_idx in schedule:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            if workload_type == 'a':
                _ = model_a(batches_a[batch_idx])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                per_token_us = (elapsed * 1_000_000) / (batch_size * seq_len)
                timings_a.append(per_token_us)
            else:
                _ = model_b(batches_b[batch_idx])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                per_token_us = (elapsed * 1_000_000) / (batch_size * seq_len)
                timings_b.append(per_token_us)
    
    return {
        'workload_a': timings_a,
        'workload_b': timings_b,
    }


def compute_interference_metrics(
    baseline: Dict[str, List[float]],
    interleaved: Dict[str, List[float]],
) -> Dict:
    """
    Compute interference metrics.
    
    Interference = (interleaved_latency - baseline_latency) / baseline_latency
    """
    metrics = {}
    
    for workload in ['workload_a', 'workload_b']:
        baseline_lat = np.mean(baseline[workload])
        interleaved_lat = np.mean(interleaved[workload])
        
        interference_pct = 100.0 * (interleaved_lat - baseline_lat) / baseline_lat
        interference_abs_pct = abs(interference_pct)
        
        metrics[workload] = {
            'baseline_mean_us': baseline_lat,
            'interleaved_mean_us': interleaved_lat,
            'interference_pct': interference_pct,
            'interference_abs_pct': interference_abs_pct,
        }
    
    return metrics


def main():
    """Run interference benchmark with concurrent workloads."""
    print("=" * 70)
    print("Expert Interference Benchmarks")
    print("(Coding + Math concurrent workloads)")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}\n")
    
    # Configuration
    n_experts = 64
    k_active = 2
    batch_size = 32
    seq_len = 128
    num_batches = 20
    
    # Create two identical models
    print(f"Creating models...")
    
    def create_hgsel_model(device):
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
                    n_experts=n_experts,
                    k_active=k_active,
                    layer_id=layer_idx,
                ).to(device)
        return model
    
    model_a = create_hgsel_model(device)
    model_b = create_hgsel_model(device)
    
    # Baseline: isolated runs
    print(f"\n{'='*70}")
    print("Scenario 1: ISOLATED (baseline)")
    print(f"{'='*70}")
    baseline = run_workload_baseline(
        model_a, model_b,
        num_batches=num_batches,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    
    baseline_stats = {
        'workload_a': np.mean(baseline['workload_a']),
        'workload_b': np.mean(baseline['workload_b']),
    }
    
    print(f"\n  Workload A: {baseline_stats['workload_a']:.2f} µs/token")
    print(f"  Workload B: {baseline_stats['workload_b']:.2f} µs/token")
    
    # Interleaved: alternating
    print(f"\n{'='*70}")
    print("Scenario 2: INTERLEAVED (alternating) - L1/L2 interference")
    print(f"{'='*70}")
    interleaved_alt = run_workload_interleaved(
        model_a, model_b,
        num_batches=num_batches,
        batch_size=batch_size,
        seq_len=seq_len,
        pattern="alternating",
    )
    
    metrics_alt = compute_interference_metrics(baseline, interleaved_alt)
    
    print(f"\n  Workload A:")
    print(f"    Latency: {metrics_alt['workload_a']['interleaved_mean_us']:.2f} µs/token")
    print(f"    Interference: {metrics_alt['workload_a']['interference_pct']:+.1f}%")
    
    print(f"\n  Workload B:")
    print(f"    Latency: {metrics_alt['workload_b']['interleaved_mean_us']:.2f} µs/token")
    print(f"    Interference: {metrics_alt['workload_b']['interference_pct']:+.1f}%")
    
    # Interleaved: random
    print(f"\n{'='*70}")
    print("Scenario 3: INTERLEAVED (random) - Worst-case mixing")
    print(f"{'='*70}")
    interleaved_rand = run_workload_interleaved(
        model_a, model_b,
        num_batches=num_batches,
        batch_size=batch_size,
        seq_len=seq_len,
        pattern="random",
    )
    
    metrics_rand = compute_interference_metrics(baseline, interleaved_rand)
    
    print(f"\n  Workload A:")
    print(f"    Latency: {metrics_rand['workload_a']['interleaved_mean_us']:.2f} µs/token")
    print(f"    Interference: {metrics_rand['workload_a']['interference_pct']:+.1f}%")
    
    print(f"\n  Workload B:")
    print(f"    Latency: {metrics_rand['workload_b']['interleaved_mean_us']:.2f} µs/token")
    print(f"    Interference: {metrics_rand['workload_b']['interference_pct']:+.1f}%")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Latency by scenario
    scenarios = ['Isolated', 'Alternating', 'Random']
    workload_a_lats = [
        baseline_stats['workload_a'],
        metrics_alt['workload_a']['interleaved_mean_us'],
        metrics_rand['workload_a']['interleaved_mean_us'],
    ]
    workload_b_lats = [
        baseline_stats['workload_b'],
        metrics_alt['workload_b']['interleaved_mean_us'],
        metrics_rand['workload_b']['interleaved_mean_us'],
    ]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    axes[0].bar(x - width/2, workload_a_lats, width, label='Workload A (Coding)', color='#2E86AB')
    axes[0].bar(x + width/2, workload_b_lats, width, label='Workload B (Math)', color='#A23B72')
    axes[0].set_ylabel('Latency (µs/token)', fontsize=11)
    axes[0].set_title('Latency by Scheduling Scenario', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenarios)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Interference % by scenario
    interf_a_alt = metrics_alt['workload_a']['interference_pct']
    interf_b_alt = metrics_alt['workload_b']['interference_pct']
    interf_a_rand = metrics_rand['workload_a']['interference_pct']
    interf_b_rand = metrics_rand['workload_b']['interference_pct']

    interf_a_alt_abs = metrics_alt['workload_a']['interference_abs_pct']
    interf_b_alt_abs = metrics_alt['workload_b']['interference_abs_pct']
    interf_a_rand_abs = metrics_rand['workload_a']['interference_abs_pct']
    interf_b_rand_abs = metrics_rand['workload_b']['interference_abs_pct']
    
    scenarios_no_base = ['Alternating', 'Random']
    interf_a = [interf_a_alt, interf_a_rand]
    interf_b = [interf_b_alt, interf_b_rand]
    
    x2 = np.arange(len(scenarios_no_base))
    axes[1].bar(x2 - width/2, interf_a, width, label='Workload A', color='#2E86AB')
    axes[1].bar(x2 + width/2, interf_b, width, label='Workload B', color='#A23B72')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Interference (%)', fontsize=11)
    axes[1].set_title('Latency Blowup from Interference', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(scenarios_no_base)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '../results/expert_interference.png'), dpi=100)
    print(f"\n  Saved: results/expert_interference.png")
    
    # Summary
    print(f"\n{'='*70}")
    print("EXPERT INTERFERENCE HYPOTHESIS TEST")
    print(f"{'='*70}")
    
    max_interf = max(
        interf_a_alt_abs, interf_b_alt_abs,
        interf_a_rand_abs, interf_b_rand_abs
    )
    
    print(f"\nMax interference magnitude observed: {max_interf:.1f}%")
    
    if max_interf < 5:
        print("✓ CONCLUSION: Interference is NEGLIGIBLE")
        print("✓ Implication: Multi-tenancy is safe, no isolation needed")
    elif max_interf < 20:
        print("~ CONCLUSION: Interference is MODERATE")
        print("~ Implication: Consider soft partitioning (priority scheduling)")
    else:
        print("✗ CONCLUSION: Interference is SEVERE")
        print("✗ Implication: Need strict resource partitioning or isolation")
    
    print(f"\nDetails:")
    print(f"  Alternating: A={interf_a_alt:+.1f}% (|{interf_a_alt_abs:.1f}|), B={interf_b_alt:+.1f}% (|{interf_b_alt_abs:.1f}|)")
    print(f"  Random:      A={interf_a_rand:+.1f}% (|{interf_a_rand_abs:.1f}|), B={interf_b_rand:+.1f}% (|{interf_b_rand_abs:.1f}|)")


if __name__ == "__main__":
    main()
