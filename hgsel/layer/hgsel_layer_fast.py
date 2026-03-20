"""
Optimized HGSEL Layer - Fast Version.

Uses:
1. Vectorized MultiHashRouterFast
2. InvertedDispatchExpertBank
3. Optional: mixed precision (bf16)
"""

import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from hgsel.routing.hash_functions_fast import MultiHashRouterFast, InvertedDispatchExpertBank
from .combine_weights import CombineFactory


class HGSELLayerFast(nn.Module):
    """
    Optimized HGSEL layer with vectorized routing and inverted dispatch.
    
    Args:
        d_model: Token embedding dimension
        d_ff: Expert intermediate dimension
        n_experts: Total number of experts
        k_active: Active experts per token
        n_hashes: Candidate hashes (try = k_active for speed)
        combine_mode: "uniform", "scalar", or "learned"
        layer_id: Layer index
        salt: Load balancing salt
        activation: "gelu" or "relu"
        use_bf16: Use bfloat16 for faster compute
    """

    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        n_experts: int = 64,
        k_active: int = 2,
        n_hashes: int = 4,
        combine_mode: str = "uniform",
        layer_id: int = 0,
        salt: float = 0.0,
        activation: str = "gelu",
        use_bf16: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.k_active = k_active
        self.n_hashes = n_hashes
        self.layer_id = layer_id
        self.combine_mode = combine_mode
        self.use_bf16 = use_bf16

        # Optimized router
        self.router = MultiHashRouterFast(
            n_experts=n_experts,
            k_active=k_active,
            n_hashes=n_hashes,
            hidden_dim=d_model,
            layer_id=layer_id,
            salt=salt,
        )

        # Optimized expert bank with inverted dispatch
        self.expert_bank = InvertedDispatchExpertBank(
            n_experts=n_experts,
            k_active=k_active,
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
        )

        # Combine weights
        self.combine = CombineFactory.create(
            mode=combine_mode,
            k_active=k_active,
            d_model=d_model,
        )

        # Load tracking
        self.register_buffer(
            "expert_load_ema",
            torch.ones(n_experts) / n_experts,
            persistent=True,
        )
        self.ema_decay = 0.99

        self.total_tokens_routed = 0
        self.routing_entropy = 0.0
        self.last_forward_trace: Optional[Dict[str, Any]] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_routing_info: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """Forward pass with optimized sparse expert routing."""
        
        # Handle input shape
        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            batch, seq_len, d_model = hidden_states.shape
            hidden_states_flat = hidden_states.view(batch * seq_len, d_model)
            should_reshape = True
        else:
            hidden_states_flat = hidden_states
            batch, seq_len = None, None
            should_reshape = False

        batch_seq_len = hidden_states_flat.shape[0]
        device = hidden_states_flat.device

        # Cast to bf16 if enabled
        original_dtype = hidden_states_flat.dtype
        if self.use_bf16 and hidden_states_flat.dtype == torch.float32:
            hidden_states_flat = hidden_states_flat.to(torch.bfloat16)

        # Route tokens
        t_router_start = time.perf_counter()
        selected_experts, expert_weights, expert_masks = self.router(hidden_states_flat)
        router_ms = (time.perf_counter() - t_router_start) * 1000.0

        # Expert compute
        t_expert_start = time.perf_counter()
        expert_outputs, expert_loads = self.expert_bank(
            hidden_states_flat,
            selected_experts,
            expert_masks,
        )
        expert_compute_ms = (time.perf_counter() - t_expert_start) * 1000.0

        # Combine
        t_combine_start = time.perf_counter()
        output = self.combine(expert_outputs)
        combine_ms = (time.perf_counter() - t_combine_start) * 1000.0

        # Cast back
        if self.use_bf16 and original_dtype == torch.float32:
            output = output.to(original_dtype)

        # Reshape
        if should_reshape:
            output = output.view(batch, seq_len, self.d_model)

        # Update EMA
        with torch.no_grad():
            self.expert_load_ema = (
                self.ema_decay * self.expert_load_ema
                + (1 - self.ema_decay) * expert_loads.to(self.expert_load_ema.dtype)
            )

            load_probs = (self.expert_load_ema / self.expert_load_ema.sum()).clamp(min=1e-8)
            entropy = -torch.sum(load_probs * torch.log(load_probs)) / torch.log(
                torch.tensor(self.n_experts, dtype=torch.float32)
            )
            self.routing_entropy = float(entropy)

        self.total_tokens_routed += batch_seq_len
        self.last_forward_trace = {
            "layer_id": int(self.layer_id),
            "batch_tokens": int(batch_seq_len),
            "router_ms": float(router_ms),
            "expert_compute_ms": float(expert_compute_ms),
            "combine_ms": float(combine_ms),
            "salt": float(self.router.salt),
            "k_active": int(self.k_active),
            "n_experts": int(self.n_experts),
            "expert_loads": expert_loads.detach().cpu(),
            "routing_entropy": float(self.routing_entropy),
        }

        if return_routing_info:
            routing_info = {
                "selected_experts": selected_experts.cpu().numpy(),
                "expert_loads": expert_loads.cpu().numpy(),
                "routing_entropy": self.routing_entropy,
                "ema_loads": self.expert_load_ema.cpu().numpy(),
                "batch_tokens": batch_seq_len,
            }
            return output, routing_info

        return output

    def set_salt(self, salt: float):
        self.router.set_salt(salt)

    def get_expert_load_stats(self) -> Dict:
        ema = self.expert_load_ema.cpu().numpy()
        return {
            "mean_load": float(ema.mean()),
            "std_load": float(ema.std()),
            "min_load": float(ema.min()),
            "max_load": float(ema.max()),
            "entropy": self.routing_entropy,
        }
