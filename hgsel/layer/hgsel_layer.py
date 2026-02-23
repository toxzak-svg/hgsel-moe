"""
HGSEL Layer: Main MLP replacement for Transformer blocks.

Integrates routing engine, expert bank, combining, and load monitoring.

Architecture:
  Input [batch * seq_len, d_model]
    ↓
  Routing (Multi-Hash) → selected_experts, expert_masks
    ↓
  Expert Bank (Sparse Dispatch) → expert_outputs, expert_loads
    ↓
  Combine Weights → final_output
    ↓
  Output [batch * seq_len, d_model]

Auxiliary outputs:
  - expert_loads: [n_experts] (load per expert for monitoring)
  - routing_info: Dict with diagnostics
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from hgsel.routing import MultiHashRouter
from hgsel.expert import ExpertBank
from hgsel.distributed.dispatch_pipeline import DispatchPipeline
from .combine_weights import CombineFactory


class HGSELLayer(nn.Module):
    """
    HGSEL layer: Hash-based sparse expert MLP.

    Drops into Transformer in place of standard MLP (d_model → d_ff → d_model).

    Args:
        d_model: Token embedding dimension (e.g., 512)
        d_ff: Expert intermediate dimension (e.g., 2048)
        n_experts: Total number of experts (e.g., 64)
        k_active: Active experts per token (e.g., 2)
        n_hashes: Candidate hashes (e.g., 4)
        combine_mode: "uniform" (default), "scalar", or "learned"
        layer_id: Layer index in Transformer (for routing stability)
        salt: Load balancing salt parameter (default: 0.0, tuned later)
        activation: Activation function ("gelu" or "relu")
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
        enable_dispatch_planning: bool = False,
        dispatch_shard_map: Optional[Dict[int, Tuple[int, int]]] = None,
        dispatch_rank: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.k_active = k_active
        self.n_hashes = n_hashes
        self.layer_id = layer_id
        self.combine_mode = combine_mode
        self.enable_dispatch_planning = enable_dispatch_planning
        self.dispatch_shard_map = dispatch_shard_map
        self.dispatch_rank = dispatch_rank

        # Routing engine
        self.router = MultiHashRouter(
            n_experts=n_experts,
            k_active=k_active,
            n_hashes=n_hashes,
            hidden_dim=d_model,
            layer_id=layer_id,
            salt=salt,
        )

        # Expert bank
        self.expert_bank = ExpertBank(
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

        # Load tracking (exponential moving average)
        self.register_buffer(
            "expert_load_ema",
            torch.ones(n_experts) / n_experts,
            persistent=True,
        )
        self.ema_decay = 0.99

        # Statistics for monitoring
        self.total_tokens_routed = 0
        self.routing_entropy = 0.0
        self.dispatch_pipeline = None

        if self.enable_dispatch_planning and self.dispatch_shard_map is not None:
            self.dispatch_pipeline = DispatchPipeline(
                shard_map=self.dispatch_shard_map,
                rank=self.dispatch_rank,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_routing_info: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        Forward pass with sparse expert routing.

        Args:
            hidden_states: [batch, seq_len, d_model] or [batch * seq_len, d_model]
            return_routing_info: If True, return (output, routing_info dict)

        Returns:
            output: Same shape as input
            routing_info: (optional) Dict with diagnostics
                - selected_experts: [batch * seq_len, k_active]
                - expert_loads: [n_experts]
                - routing_entropy: scalar (0 = concentrated, 1 = uniform)
        """
        # Handle both 2D and 3D inputs
        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            # [batch, seq_len, d_model] → [batch * seq_len, d_model]
            batch, seq_len, d_model = hidden_states.shape
            hidden_states_flat = hidden_states.view(batch * seq_len, d_model)
            should_reshape = True
        else:
            # Already [batch * seq_len, d_model]
            hidden_states_flat = hidden_states
            batch, seq_len = None, None
            should_reshape = False

        batch_seq_len = hidden_states_flat.shape[0]
        device = hidden_states_flat.device

        # Route tokens to experts
        selected_experts, expert_weights, expert_masks = self.router(hidden_states_flat)

        # Optionally build a dispatch plan (no communication yet).
        dispatch_summary = None
        if self.dispatch_pipeline is not None:
            dispatch_result = self.dispatch_pipeline.build(
                hidden_states=hidden_states_flat,
                expert_ids=selected_experts,
            )
            dispatch_summary = {
                "local_tokens": len(dispatch_result.local_batch.token_indices),
                "remote_ranks": sorted(dispatch_result.remote_requests.rank_to_token_indices.keys()),
            }

        # Execute sparse experts (returns [batch * seq_len, k_active, d_model])
        expert_outputs, expert_loads = self.expert_bank(
            hidden_states_flat,
            selected_experts,
            expert_masks,
        )

        # Combine expert outputs
        output = self.combine(expert_outputs)

        # Reshape back if needed
        if should_reshape:
            output = output.view(batch, seq_len, self.d_model)

        # Update load EMA
        with torch.no_grad():
            self.expert_load_ema = (
                self.ema_decay * self.expert_load_ema
                + (1 - self.ema_decay) * expert_loads.to(self.expert_load_ema.dtype)
            )

            # Compute routing entropy (monitor load balance)
            # Normalized loads → entropy
            load_probs = (self.expert_load_ema / self.expert_load_ema.sum()).clamp(min=1e-8)
            entropy = -torch.sum(load_probs * torch.log(load_probs)) / torch.log(
                torch.tensor(self.n_experts, dtype=torch.float32)
            )
            self.routing_entropy = float(entropy)

        # Statistics
        self.total_tokens_routed += batch_seq_len

        # Optionally return routing diagnostics
        if return_routing_info:
            routing_info = {
                "selected_experts": selected_experts.cpu().numpy(),
                "expert_loads": expert_loads.cpu().numpy(),
                "routing_entropy": self.routing_entropy,
                "ema_loads": self.expert_load_ema.cpu().numpy(),
                "batch_tokens": batch_seq_len,
                "dispatch_summary": dispatch_summary,
            }
            return output, routing_info

        return output

    def set_salt(self, salt: float):
        """Tune load-balancing salt parameter."""
        self.router.set_salt(salt)

    def get_expert_load_stats(self) -> Dict:
        """Get current expert load statistics."""
        ema = self.expert_load_ema.cpu().numpy()
        return {
            "mean_load": float(ema.mean()),
            "std_load": float(ema.std()),
            "min_load": float(ema.min()),
            "max_load": float(ema.max()),
            "entropy": self.routing_entropy,
        }

    def reset_statistics(self):
        """Reset routing statistics (call at start of epoch)."""
        self.total_tokens_routed = 0
        self.routing_entropy = 0.0
