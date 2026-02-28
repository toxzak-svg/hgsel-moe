"""
Trace-driven expert working-set modeling.

For each context length, collect which expert IDs are routed to and build
a "working set size vs context length" curve, similar to OS page working sets.

Hypothesis to TEST:
  - Working set size is predictable and grows sublinearly with context length.
  - Small working set → cache-friendly (L1/L2 data reuse possible).

Hypothesis to FALSIFY:
  - Working set is unpredictable (random dist vs hash-based).
  - Large working set (close to N) → caching won't help.
  - Interference is too high → cache won't be reliable.
"""

import argparse
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Set
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from hgsel.layer import HGSELLayer
from experiments.baselines.dense_transformer import TransformerModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace-driven HGSEL expert working-set benchmark")
    parser.add_argument(
        "--context-lengths",
        type=str,
        default="32,64,128,256,512,1024",
        help="Comma-separated context lengths to test",
    )
    parser.add_argument(
        "--expert-configs",
        type=str,
        default="32,64,128",
        help="Comma-separated n_experts values to test",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=512,
        help="Approximate total tokens traced per context length",
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
        default="results/workset_curve.png",
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
        help="Run a short smoke configuration",
    )
    return parser.parse_args()


def parse_int_csv(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def trace_expert_routing(model: nn.Module, num_tokens: int, context_length: int) -> Tuple[Set[int], List[int]]:
    """
    Route num_tokens through the model and collect which experts are used.
    
    Args:
        model: Transformer with HGSEL layers
        num_tokens: Total tokens to route (num_batches * batch_size * seq_len)
        context_length: Sequence length in each batch
    
    Returns:
        unique_experts: Set of expert IDs that were routed to
        expert_sequence: List of expert IDs in order (for analysis)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Track which experts were accessed
    accessed_experts = set()
    expert_sequence = []
    
    # Patch HGSEL layers to capture routing
    patched_layers = []
    
    for module in model.modules():
        if isinstance(module, HGSELLayer):
            patched_layers.append(module)
    
    # Collect routing from hash_tokens (which is deterministic)
    try:
        batch_size = max(1, num_tokens // context_length)
        vocab_size = 256
        
        with torch.no_grad():
            for batch_idx in range(batch_size):
                input_ids = torch.randint(
                    0, vocab_size, 
                    (1, context_length),
                    device=device
                )
                
                # Forward through embedding
                x = model.embedding(input_ids)
                pos = torch.arange(context_length, device=device).unsqueeze(0)
                x = x + model.pos_embedding(pos)
                
                # Forward through transformer layers and capture routing
                for layer in model.layers:
                    if hasattr(layer, 'mlp') and isinstance(layer.mlp, HGSELLayer):
                        # Capture from HGSEL layer's router
                        hgsel = layer.mlp
                        
                        # Process through layer norm
                        ln_out = layer.ln2(x)
                        
                        # Get routing decisions from router (deterministic hash-based)
                        # Reshape for router
                        batch, seq_len, d = ln_out.shape
                        ln_flat = ln_out.view(batch * seq_len, d)
                        
                        # Get expert IDs from router (deterministic)
                        expert_ids, _ = hgsel.router.hash_tokens(ln_flat)
                        
                        # Track accessed experts (expert_ids is [batch*seq, n_hashes])
                        accessed = expert_ids.cpu().flatten().unique().tolist()
                        accessed_experts.update(accessed)
                        expert_sequence.extend(expert_ids.cpu().flatten().tolist())
                    
                    # Normal forward pass
                    x = layer(x)
    finally:
        pass
    
    return accessed_experts, expert_sequence


def compute_working_set_stats(accessed_experts: Set[int], n_experts: int) -> Dict:
    """
    Compute working-set statistics.
    
    Returns:
        - working_set_size: Number of unique experts used
        - utilization: Fraction of total experts used
        - rank: Rank of working set (coverage)
    """
    ws_size = len(accessed_experts)
    utilization = ws_size / n_experts
    
    return {
        "working_set_size": ws_size,
        "utilization": utilization,
        "coverage": ws_size / n_experts,
    }


def classify_cv(cv: float, *, high_threshold: float, moderate_threshold: float) -> Dict[str, str]:
    if cv < high_threshold:
        return {
            "level": "highly_predictable",
            "verdict": "OK",
            "label": "PREDICTABLE",
        }
    if cv < moderate_threshold:
        return {
            "level": "moderately_predictable",
            "verdict": "WARN",
            "label": "MODERATE",
        }
    return {
        "level": "unpredictable",
        "verdict": "FAIL",
        "label": "UNPREDICTABLE",
    }


def main():
    """Build working-set curve across context lengths."""
    args = parse_args()

    if args.num_tokens <= 0:
        raise ValueError("--num-tokens must be > 0")

    if args.smoke:
        if args.context_lengths == "32,64,128,256,512,1024":
            args.context_lengths = "16,32,64"
        if args.expert_configs == "32,64,128":
            args.expert_configs = "32"
        if args.num_tokens == 512:
            args.num_tokens = 128

    context_lengths = parse_int_csv(args.context_lengths)
    n_experts_config = parse_int_csv(args.expert_configs)
    if not context_lengths:
        raise ValueError("No valid context lengths parsed from --context-lengths")
    if not n_experts_config:
        raise ValueError("No valid expert configs parsed from --expert-configs")

    print("=" * 70)
    print("Trace-Driven Expert Working-Set Modeling")
    print("=" * 70)

    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    k_active = 2

    # Test across context lengths
    max_ctx_len = max(context_lengths)

    print(
        f"\nDevice: {device}\n"
        f"Run config: context_lengths={context_lengths}, expert_configs={n_experts_config}, "
        f"num_tokens={args.num_tokens}"
    )
    
    results = defaultdict(list)
    per_config_predictability: Dict[str, Dict[str, float | str]] = {}
    
    for n_exp in n_experts_config:
        print(f"\n{'='*70}")
        print(f"Testing with {n_exp} experts (k_active={k_active})")
        print(f"{'='*70}")
        
        # Create model with HGSEL layers
        # Note: HGSELLayer uses default n_experts and k_active from __init__
        model = TransformerModel(
            vocab_size=256,
            d_model=128,
            d_ff=512,
            n_layers=2,
            n_heads=2,
            max_seq_len=max_ctx_len,
            mlp_class=HGSELLayer,
        ).to(device)
        model.eval()
        
        # Rebuild HGSEL layers to match expert configuration
        for layer_idx, layer in enumerate(model.layers):
            if hasattr(layer, 'mlp'):
                layer.mlp = HGSELLayer(
                    d_model=model.d_model,
                    d_ff=512,
                    n_experts=n_exp,
                    k_active=k_active,
                    layer_id=layer_idx,
                ).to(device)
        
        # Trace for each context length
        for ctx_len in context_lengths:
            print(f"\n  Context length: {ctx_len}")
            
            # Trace with configured approximate total tokens
            num_tokens = args.num_tokens
            accessed_experts, expert_seq = trace_expert_routing(
                model, num_tokens, ctx_len
            )
            
            stats = compute_working_set_stats(accessed_experts, n_exp)
            
            print(f"    Working set size: {stats['working_set_size']}")
            print(f"    Utilization: {stats['utilization']:.1%}")
            print(f"    Expected (if random): {k_active:.1%} * {(num_tokens // ctx_len):.0f} batches = {min(n_exp, k_active * (num_tokens // ctx_len))}")
            
            results[f"n_experts={n_exp}"].append({
                'context_length': ctx_len,
                'working_set_size': stats['working_set_size'],
                'utilization': stats['utilization'],
            })
        
        # Analyze predictability
        print(f"\n  Predictability Analysis:")
        ws_sizes = [r['working_set_size'] for r in results[f"n_experts={n_exp}"]]
        ws_mean = np.mean(ws_sizes)
        ws_std = np.std(ws_sizes)
        ws_cv = ws_std / ws_mean if ws_mean > 0 else 0
        
        print(f"    Mean working set: {ws_mean:.1f}")
        print(f"    Std dev: {ws_std:.1f}")
        print(f"    Coefficient of variation: {ws_cv:.3f}")

        pred = classify_cv(float(ws_cv), high_threshold=0.2, moderate_threshold=0.5)
        per_config_predictability[f"n_experts={n_exp}"] = {
            "mean_working_set": float(ws_mean),
            "std_working_set": float(ws_std),
            "cv": float(ws_cv),
            "level": str(pred["level"]),
            "verdict": str(pred["verdict"]),
        }

        print(f"    [{pred['verdict']}] {pred['label']} ({'low' if pred['verdict']=='OK' else 'medium' if pred['verdict']=='WARN' else 'high'} variation)")
    
    # Plot
    rel_output = None
    if args.no_plot:
        print(f"\n{'='*70}")
        print("Generating working-set curve...")
        print("  Plot generation skipped (--no-plot)")
    else:
        print(f"\n{'='*70}")
        print("Generating working-set curve...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Working set size vs context length
        for n_exp_key in results:
            data = results[n_exp_key]
            ctx_lens = [r['context_length'] for r in data]
            ws_sizes = [r['working_set_size'] for r in data]
            axes[0].plot(ctx_lens, ws_sizes, marker='o', label=n_exp_key)

        axes[0].set_xlabel('Context Length (tokens)', fontsize=11)
        axes[0].set_ylabel('Working Set Size (# experts)', fontsize=11)
        axes[0].set_title('Expert Working Set vs Context Length', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')

        # Utilization % vs context length
        for n_exp_key in results:
            data = results[n_exp_key]
            ctx_lens = [r['context_length'] for r in data]
            utils = [r['utilization'] * 100 for r in data]
            axes[1].plot(ctx_lens, utils, marker='s', label=n_exp_key)

        axes[1].set_xlabel('Context Length (tokens)', fontsize=11)
        axes[1].set_ylabel('Utilization (%)', fontsize=11)
        axes[1].set_title('Expert Utilization vs Context Length', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')

        plt.tight_layout()
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(os.path.dirname(__file__), "..", output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=100)
        plt.close(fig)
        rel_output = os.path.relpath(output_path, os.path.join(os.path.dirname(__file__), ".."))
        print(f"  Saved: {rel_output}")
    
    # Summary
    print(f"\n{'='*70}")
    print("WORKING-SET HYPOTHESIS TEST")
    print(f"{'='*70}")
    
    all_cvs = []
    for n_exp_key in results:
        data = results[n_exp_key]
        ws_sizes = [r['working_set_size'] for r in data]
        ws_cv = np.std(ws_sizes) / np.mean(ws_sizes) if np.mean(ws_sizes) > 0 else 0
        all_cvs.append(ws_cv)
    
    overall_cv = np.mean(all_cvs)
    
    print(f"\nOverall coefficient of variation: {overall_cv:.3f}")
    
    overall_pred = classify_cv(float(overall_cv), high_threshold=0.15, moderate_threshold=0.3)

    if overall_pred["verdict"] == "OK":
        print("[OK] CONCLUSION: Working sets are HIGHLY PREDICTABLE")
        print("[OK] Use case: Compile expert dispatch, pre-fetch patterns")
    elif overall_pred["verdict"] == "WARN":
        print("[WARN] CONCLUSION: Working sets are MODERATELY PREDICTABLE")
        print("[WARN] Use case: Adaptive caching with occasional misses")
    else:
        print("[FAIL] CONCLUSION: Working sets are UNPREDICTABLE")
        print("[FAIL] Implication: Full cache coverage or no caching")
    
    print(f"\nDetails:")
    for n_exp_key in results:
        data = results[n_exp_key]
        ws_sizes = [r['working_set_size'] for r in data]
        print(f"  {n_exp_key}: CV={np.std(ws_sizes)/np.mean(ws_sizes):.3f}")

    if args.json_output:
        json_output_path = args.json_output
        if not os.path.isabs(json_output_path):
            json_output_path = os.path.join(os.path.dirname(__file__), "..", json_output_path)
        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

        config_results = []
        for n_exp_key in results:
            n_exp = int(n_exp_key.split("=")[-1])
            per_context = []
            for row in results[n_exp_key]:
                per_context.append(
                    {
                        "context_length": int(row["context_length"]),
                        "working_set_size": int(row["working_set_size"]),
                        "utilization": float(row["utilization"]),
                    }
                )

            config_results.append(
                {
                    "config_name": n_exp_key,
                    "n_experts": n_exp,
                    "k_active": int(k_active),
                    "num_tokens": int(args.num_tokens),
                    "per_context": per_context,
                    "predictability": per_config_predictability.get(n_exp_key),
                }
            )

        payload = {
            "metadata": {
                "script": "experiments/trace_driven_workset.py",
                "device": device,
                "smoke": bool(args.smoke),
                "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "plot_generated": not bool(args.no_plot),
                "plot_output": rel_output,
            },
            "run_config": {
                "context_lengths": [int(x) for x in context_lengths],
                "expert_configs": [int(x) for x in n_experts_config],
                "num_tokens": int(args.num_tokens),
                "k_active": int(k_active),
            },
            "summary": {
                "overall_cv": float(overall_cv),
                "level": str(overall_pred["level"]),
                "verdict": str(overall_pred["verdict"]),
            },
            "config_results": config_results,
        }

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        rel_json_output = os.path.relpath(json_output_path, os.path.join(os.path.dirname(__file__), ".."))
        print(f"  Wrote JSON summary: {rel_json_output}")


if __name__ == "__main__":
    main()
