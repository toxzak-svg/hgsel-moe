"""
HGSEL: Hash-based Gradient-guided Sparse Expert Layer

A deterministic, production-grade sparse mixture of experts architecture
designed for training and inference at 300M (baseline) and 1T+ (production) scales.

Key Components:
- hgsel.routing: Deterministic multi-hash routing engine
- hgsel.expert: Sparse expert bank and two-tier dispatch (300M + 1T)
- hgsel.layer: HGSEL layer replacing Transformer MLP blocks
- hgsel.training: Training utilities, load balancing, salt tuning
- hgsel.distributed: Multi-GPU communication and hierarchical dispatch
- hgsel.inference: Routing cache, block packing, prefetching
"""

__version__ = "0.1.0"
__author__ = "HGSEL Contributors"

from . import routing, expert, layer, training

__all__ = ["routing", "expert", "layer", "training"]
