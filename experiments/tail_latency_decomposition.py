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

import argparse
import json
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
from experiments.baselines.dense_transformer import TransformerModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tail-latency decomposition benchmark for HGSEL")
    parser.add_argument("--num-runs", type=int, default=100, help="Total runs per configuration (includes warmup)")
    parser.add_argument("--warmup-runs", type=int, default=5, help="Warmup runs skipped from stats")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument(
        "--configs",
        type=str,
        default="32:2,64:2,128:4",
        help="Comma-separated n_experts:k_active configs (example: 32:2,64:2,128:4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device (auto prefers CUDA)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/tail_latency_decomp.png",
        help="Output plot path",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default="",
        help="Optional JSON summary output path",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation (useful for CI/smoke runs)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short smoke configuration (1 config, fewer runs, smaller batch/sequence)",
    )
    return parser.parse_args()


def parse_configs(configs_arg: str) -> List[Dict[str, int]]:
    configs: List[Dict[str, int]] = []
    for part in configs_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid config '{part}'. Expected n_experts:k_active")
        n_exp_str, k_act_str = part.split(":", 1)
        n_exp = int(n_exp_str.strip())
        k_act = int(k_act_str.strip())
        if n_exp <= 0 or k_act <= 0:
            raise ValueError(f"Invalid config '{part}'. n_experts and k_active must be > 0")
        configs.append({"n_experts": n_exp, "k_active": k_act})

    if not configs:
        raise ValueError("No valid configs parsed from --configs")
    return configs


def extract_hgsel_trace_timings_per_token_us(model: nn.Module, num_tokens: int) -> Dict[str, float]:
    """Extract per-token timing components from HGSEL layer traces.

    Returns per-token microseconds for components accumulated across all HGSEL layers.
    If traces are unavailable, returns zeros.
    """
    timings = {
        "routing_trace": 0.0,
        "expert_compute_trace": 0.0,
        "combine_trace": 0.0,
        "dispatch_planning_trace": 0.0,
        "hgsel_trace_total": 0.0,
    }

    if num_tokens <= 0 or not hasattr(model, "get_phase4_routing_traces"):
        return timings

    traces = model.get_phase4_routing_traces()  # type: ignore[assignment]
    if not isinstance(traces, list):
        return timings

    total_router_ms = 0.0
    total_expert_ms = 0.0
    total_combine_ms = 0.0
    total_dispatch_ms = 0.0

    for trace in traces:
        if not isinstance(trace, dict):
            continue
        total_router_ms += float(trace.get("router_ms", 0.0) or 0.0)
        total_expert_ms += float(trace.get("expert_compute_ms", 0.0) or 0.0)
        total_combine_ms += float(trace.get("combine_ms", 0.0) or 0.0)

        dispatch_trace = trace.get("dispatch")
        if isinstance(dispatch_trace, dict):
            # Dispatch trace keys vary; sum any *_ms numeric fields.
            for k, v in dispatch_trace.items():
                if k.endswith("_ms") and isinstance(v, (int, float)):
                    total_dispatch_ms += float(v)

    per_token_scale = 1000.0 / num_tokens  # ms -> us, then divide by tokens
    timings["routing_trace"] = total_router_ms * per_token_scale
    timings["expert_compute_trace"] = total_expert_ms * per_token_scale
    timings["combine_trace"] = total_combine_ms * per_token_scale
    timings["dispatch_planning_trace"] = total_dispatch_ms * per_token_scale
    timings["hgsel_trace_total"] = (
        timings["routing_trace"]
        + timings["expert_compute_trace"]
        + timings["combine_trace"]
        + timings["dispatch_planning_trace"]
    )
    return timings


def profile_forward_latency(
    model: nn.Module,
    num_runs: int = 100,
    warmup_runs: int = 5,
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

    if num_runs <= warmup_runs:
        raise ValueError(f"num_runs ({num_runs}) must be greater than warmup_runs ({warmup_runs})")
    
    # Ensure GPU is ready
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print(f"  Profiling {num_runs} runs ({warmup_runs} warmup, {num_runs - warmup_runs} measured)...")
    
    for run_idx in range(num_runs):
        input_ids = torch.randint(0, 256, (batch_size, seq_len), device=device)
        
        # Warmup skip first N runs
        if run_idx < warmup_runs:
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

        # Pull trace-based HGSEL component timings from the most recent forward.
        trace_components = extract_hgsel_trace_timings_per_token_us(model, num_tokens)
        for component_name, component_us in trace_components.items():
            timings[component_name].append(component_us)

        # Residual captures embeddings/attention/layernorm/output/sync/measurement gap.
        residual_us = max(per_token_us - trace_components.get("hgsel_trace_total", 0.0), 0.0)
        timings["residual_non_hgsel"].append(residual_us)
    
    return timings


def compute_latency_percentiles(timings: Dict[str, List[float]]) -> Dict:
    """Compute latency percentiles and other stats."""
    stats = {}
    
    for component, latencies in timings.items():
        if not latencies:
            continue
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


def classify_tail_variation(cv: float) -> Dict[str, str]:
    if cv < 0.2:
        return {
            "level": "predictable",
            "verdict": "OK",
            "message": "Tail latency is PREDICTABLE",
            "recommendation": "Workload co-scheduling safe",
        }
    if cv < 0.5:
        return {
            "level": "moderate",
            "verdict": "WARN",
            "message": "Tail latency has MODERATE VARIATION",
            "recommendation": "Use timeout-based scheduling",
        }
    return {
        "level": "unpredictable",
        "verdict": "FAIL",
        "message": "Tail latency is UNPREDICTABLE",
        "recommendation": "Need architecture redesign (isolation, guaranteed resources)",
    }


def main():
    """Break down tail latency by component."""
    args = parse_args()

    if args.num_runs <= 0:
        raise ValueError("--num-runs must be > 0")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be > 0")

    if args.smoke:
        if args.num_runs == 100:
            args.num_runs = 8
        if args.warmup_runs == 5:
            args.warmup_runs = 2
        if args.batch_size == 32:
            args.batch_size = 4
        if args.seq_len == 128:
            args.seq_len = 32
        if args.configs == "32:2,64:2,128:4":
            args.configs = "32:2"

    if args.num_runs <= args.warmup_runs:
        raise ValueError("--num-runs must be greater than --warmup-runs")

    configs = parse_configs(args.configs)

    print("=" * 70)
    print("Tail-Latency Decomposition (per-token basis)")
    print("=" * 70)

    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}\n")
    print(
        "Run config: "
        f"num_runs={args.num_runs}, warmup_runs={args.warmup_runs}, "
        f"batch_size={args.batch_size}, seq_len={args.seq_len}, configs={args.configs}"
    )
    
    all_results = {}
    per_config_measured_runs: Dict[str, int] = {}
    
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
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
        
        # Compute stats
        stats = compute_latency_percentiles(timings)
        config_key = f"n_exp={n_exp}, k_act={k_act}"
        all_results[config_key] = stats
        per_config_measured_runs[config_key] = len(timings.get("forward_pass", []))
        
        # Print results
        print(f"\nLatency (per-token, microseconds):")
        print(f"  Component                    p50        p99        p999       std        CV")
        print(f"  {'-'*80}")
        
        ordered_components = [
            "forward_pass",
            "routing_trace",
            "expert_compute_trace",
            "combine_trace",
            "dispatch_planning_trace",
            "hgsel_trace_total",
            "residual_non_hgsel",
        ]
        for component in ordered_components:
            if component not in timings:
                continue
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
        print(f"  {config_name:30s}: p99={p99:8.2f}us, CV={cv:.3f}")
    
    rel_output = None
    if args.no_plot:
        print("\n  Plot generation skipped (--no-plot)")
    else:
        # Plot latency distributions
        fig, axes = plt.subplots(1, len(configs), figsize=(15, 4))
        if len(configs) == 1:
            axes = [axes]

        for idx, (config_name, stats) in enumerate(all_results.items()):
            ax = axes[idx]

            # Create summary bars (p50/p99) for end-to-end forward latency
            component = 'forward_pass'
            p50 = stats[component]['p50']
            p99 = stats[component]['p99']

            # Simplified visualization
            ax.bar(['p50', 'p99'], [p50, p99], color=['#2E86AB', '#A23B72'])
            ax.set_ylabel('Latency (us)', fontsize=10)
            ax.set_title(config_name, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(os.path.dirname(__file__), "..", output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=100)
        plt.close(fig)
        rel_output = os.path.relpath(output_path, os.path.join(os.path.dirname(__file__), ".."))
        print(f"\n  Saved: {rel_output}")
    
    # Conclusion
    print(f"\n{'='*70}")
    print("TAIL-LATENCY ANALYSIS")
    print(f"{'='*70}")

    print("\nNote: component rows use HGSEL layer trace timings (router/expert/combine) and a residual bucket.")
    if torch.cuda.is_available():
        print("      On CUDA, end-to-end forward_pass is the authoritative latency metric; trace components are approximate.")
    
    avg_cv = np.mean([stats['forward_pass']['cv'] for stats in all_results.values()])
    tail_class = classify_tail_variation(float(avg_cv))

    print(f"\n[{tail_class['verdict']}] CONCLUSION: {tail_class['message']} (CV={avg_cv:.3f})")
    print(f"[{tail_class['verdict']}] Recommendation: {tail_class['recommendation']}")

    # Optional JSON summary for automation/reporting.
    if args.json_output:
        json_output_path = args.json_output
        if not os.path.isabs(json_output_path):
            json_output_path = os.path.join(os.path.dirname(__file__), "..", json_output_path)
        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

        config_summaries = []
        for config_name, stats in all_results.items():
            component_p99 = {
                k: float(v["p99"])
                for k, v in stats.items()
                if isinstance(v, dict)
                and "p99" in v
                and k not in {"forward_pass", "hgsel_trace_total"}
            }
            dominant_component = None
            dominant_p99_us = None
            if component_p99:
                dominant_component, dominant_p99_us = max(component_p99.items(), key=lambda kv: kv[1])

            config_summaries.append(
                {
                    "config_name": config_name,
                    "measured_runs": int(per_config_measured_runs.get(config_name, 0)),
                    "stats": stats,
                    "forward_pass": stats.get("forward_pass", {}),
                    "dominant_component_p99": {
                        "component": dominant_component,
                        "p99_us": (float(dominant_p99_us) if dominant_p99_us is not None else None),
                    },
                }
            )

        payload = {
            "metadata": {
                "script": "experiments/tail_latency_decomposition.py",
                "device": device,
                "smoke": bool(args.smoke),
                "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "plot_generated": not bool(args.no_plot),
                "plot_output": rel_output,
            },
            "run_config": {
                "num_runs": int(args.num_runs),
                "warmup_runs": int(args.warmup_runs),
                "batch_size": int(args.batch_size),
                "seq_len": int(args.seq_len),
                "configs": args.configs,
            },
            "summary": {
                "avg_forward_cv": float(avg_cv),
                "tail_variation_level": tail_class["level"],
                "verdict": tail_class["verdict"],
                "message": tail_class["message"],
                "recommendation": tail_class["recommendation"],
            },
            "config_results": config_summaries,
        }

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        rel_json_output = os.path.relpath(json_output_path, os.path.join(os.path.dirname(__file__), ".."))
        print(f"  Wrote JSON summary: {rel_json_output}")


if __name__ == "__main__":
    main()
