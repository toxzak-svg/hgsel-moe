#!/usr/bin/env python
"""
Quick validation that expert batching fix is working.

This script verifies the expert bank now executes in batched mode
instead of sequential token-by-token mode.

Expected results:
- CPU: ~50-150ms (batched Python loops)
- GPU: ~1-5ms (parallel GPU execution)
- Previous bug: ~890ms (sequential token loops)
"""

import time
import torch
import sys
from pathlib import Path

# Add parent directory to path
CURRENT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CURRENT_DIR))

from hgsel.expert import ExpertBank
from hgsel.routing import MultiHashRouter


def test_expert_batching(device_name='cuda'):
    """Test expert bank performance with batching fix."""
    
    device = torch.device(device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')
    print(f"Testing on: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print()
    
    # Create components
    n_experts = 64
    k_active = 2
    d_model = 256
    d_ff = 1024
    batch_size = 16
    seq_len = 128
    batch_tokens = batch_size * seq_len
    
    bank = ExpertBank(
        n_experts=n_experts,
        k_active=k_active,
        d_model=d_model,
        d_ff=d_ff,
        activation='gelu'
    ).to(device)
    
    router = MultiHashRouter(
        n_experts=n_experts,
        k_active=k_active,
        n_hashes=4,
        hidden_dim=d_model
    )
    
    # Create test input
    x = torch.randn(batch_tokens, d_model).to(device)
    
    # Route tokens
    selected_experts, expert_weights, expert_masks = router(x)
    selected_experts = selected_experts.to(device)
    expert_masks = expert_masks.to(device)
    
    # Warmup
    for _ in range(3):
        _, _ = bank(x, selected_experts, expert_masks)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Timed run
    t0 = time.perf_counter()
    
    for _ in range(10):
        expert_outputs, expert_loads = bank(x, selected_experts, expert_masks)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_call_ms = elapsed_ms / 10
    
    # Compute stats
    routing_entropy = -(expert_loads * torch.log(expert_loads + 1e-8)).sum() / torch.log(torch.tensor(n_experts, dtype=torch.float32))
    
    print(f"Configuration:")
    print(f"  Batch tokens: {batch_tokens}")
    print(f"  Experts: {n_experts}")
    print(f"  Active per token: {k_active}")
    print(f"  Model dim: {d_model}")
    print(f"  FFN dim: {d_ff}")
    print()
    print(f"Performance:")
    print(f"  Time per forward: {per_call_ms:.2f} ms")
    print(f"  Total for 10 runs: {elapsed_ms:.2f} ms")
    print()
    print(f"Correctness:")
    print(f"  Output shape: {expert_outputs.shape} (expected: [{batch_tokens}, {k_active}, {d_model}])")
    print(f"  Expert loads shape: {expert_loads.shape} (expected: [{n_experts}])")
    print(f"  Routing entropy: {routing_entropy:.4f} (ideal: 1.0)")
    print()
    
    # Verdict
    if device.type == 'cuda':
        if per_call_ms < 10:
            print("✅ PASS: GPU execution is fast (< 10ms)")
            print("   The batching fix is working correctly!")
        elif per_call_ms < 100:
            print("⚠️  MARGINAL: Slower than expected (10-100ms)")
            print("   May still have some overhead, but much better than 890ms bug")
        else:
            print("❌ FAIL: Still too slow (> 100ms)")
            print("   The sequential execution bug may still be present")
    else:
        if per_call_ms < 200:
            print("✅ PASS: CPU execution is reasonable")
            print("   Batching logic appears correct")
        else:
            print("⚠️  SLOW: CPU execution slower than expected")
            print("   Test on GPU for definitive results")
    
    return per_call_ms, routing_entropy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    
    test_expert_batching(args.device)
