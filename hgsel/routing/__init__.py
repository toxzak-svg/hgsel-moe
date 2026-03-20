"""Routing module: Hash-based deterministic routing with multi-candidate selection."""

from .hash_functions import MultiHashRouter
from .hash_functions_fast import MultiHashRouterFast, InvertedDispatchExpertBank

__all__ = ["MultiHashRouter", "MultiHashRouterFast", "InvertedDispatchExpertBank"]
