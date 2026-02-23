#!/usr/bin/env python
"""Quick integration test for trace_driven_workset benchmark."""

import sys
import os
import torch

# Quick test: Create model and run trace_expert_routing
from hgsel.layer import HGSELLayer
from experiments.baselines.dense_transformer import TransformerModel
from experiments.trace_driven_workset import trace_expert_routing, compute_working_set_stats

device = 'cpu'  # Use CPU for speed
print('Creating test model...')

model = TransformerModel(
    vocab_size=256,
    d_model=128,
    d_ff=512,
    n_layers=1,  # Minimal for test
    n_heads=2,
    mlp_class=HGSELLayer,
).to(device)

print('Running trace_expert_routing test...')
accessed_experts, expert_seq = trace_expert_routing(model, num_tokens=64, context_length=32)

stats = compute_working_set_stats(accessed_experts, 32)
print('✓ Trace routing works!')
print('  Working set size: {}'.format(stats['working_set_size']))
print('  Utilization: {:.1%}'.format(stats['utilization']))
print('\n✓ Integration test passed!')
