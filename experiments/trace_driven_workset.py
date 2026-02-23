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

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Set
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from hgsel.layer import HGSELLayer
from experiments.baselines.dense_transformer import TransformerModel


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


def main():
    """Build working-set curve across context lengths."""
    print("=" * 70)
    print("Trace-Driven Expert Working-Set Modeling")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_experts = 64
    k_active = 2
    
    # Test across context lengths
    context_lengths = [32, 64, 128, 256, 512, 1024]
    max_ctx_len = max(context_lengths)
    n_experts_config = [32, 64, 128]
    
    results = defaultdict(list)
    
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
            
            # Trace with 512 total tokens
            num_tokens = 512
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
        
        if ws_cv < 0.2:
            print(f"    ✓ PREDICTABLE (low variation)")
        elif ws_cv < 0.5:
            print(f"    ~ MODERATE (medium variation)")
        else:
            print(f"    ✗ UNPREDICTABLE (high variation)")
    
    # Plot
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
    plt.savefig(os.path.join(os.path.dirname(__file__), '../results/workset_curve.png'), dpi=100)
    print(f"  Saved: results/workset_curve.png")
    
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
    
    if overall_cv < 0.15:
        print("✓ CONCLUSION: Working sets are HIGHLY PREDICTABLE")
        print("✓ Use case: Compile expert dispatch, pre-fetch patterns")
    elif overall_cv < 0.3:
        print("~ CONCLUSION: Working sets are MODERATELY PREDICTABLE")
        print("~ Use case: Adaptive caching with occasional misses")
    else:
        print("✗ CONCLUSION: Working sets are UNPREDICTABLE")
        print("✗ Implication: Full cache coverage or no caching")
    
    print(f"\nDetails:")
    for n_exp_key in results:
        data = results[n_exp_key]
        ws_sizes = [r['working_set_size'] for r in data]
        print(f"  {n_exp_key}: CV={np.std(ws_sizes)/np.mean(ws_sizes):.3f}")


if __name__ == "__main__":
    main()
