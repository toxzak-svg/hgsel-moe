"""
Sparse expert bank with gather-based dispatch.

Efficient execution of k active experts per token from N total experts,
avoiding computation on unused experts.

Design:
  1. Store N experts as independent W1, W2 weight matrices
  2. For each token, gather columns from W1 for selected experts
  3. Execute expert FFN: activation(x @ W1) @ W2
  4. Combine outputs outside this module
"""

import torch
import torch.nn as nn
from typing import Tuple


class ExpertFFN(nn.Module):
    """
    Single expert: 2-layer MLP with intermediate activation.

    Args:
        d_model: Input/output dimension
        d_ff: Intermediate (hidden) dimension
        activation: Activation function (default: GELU)
    """

    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        activation: str = "gelu",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # W1: d_model → d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias=False, dtype=torch.float32)
        # W2: d_ff → d_model
        self.w2 = nn.Linear(d_ff, d_model, bias=False, dtype=torch.float32)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

        # Initialize weights
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch * seq_len, d_model]

        Returns:
            [batch * seq_len, d_model]
        """
        return self.w2(self.activation(self.w1(x)))


class ExpertBank(nn.Module):
    """
    Sparse expert bank with gather-based dispatch.

    Stores N experts and routes tokens to k active experts per token.

    Args:
        n_experts: Number of total experts
        k_active: Number of active experts per token
        d_model: Token embedding dimension
        d_ff: Expert FFN intermediate dimension
        activation: Activation function
    """

    def __init__(
        self,
        n_experts: int = 64,
        k_active: int = 2,
        d_model: int = 512,
        d_ff: int = 2048,
        activation: str = "gelu",
    ):
        super().__init__()
        self.n_experts = n_experts
        self.k_active = k_active
        self.d_model = d_model
        self.d_ff = d_ff

        # Create N expert modules
        self.experts = nn.ModuleList(
            [ExpertFFN(d_model, d_ff, activation) for _ in range(n_experts)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        expert_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to selected experts, gather outputs.

        Args:
            hidden_states: [batch * seq_len, d_model]
            selected_experts: [batch * seq_len, k_active] in [0, n_experts)
            expert_masks: [batch * seq_len, n_experts] with 1/k_active on selected, 0 elsewhere

        Returns:
            expert_outputs: [batch * seq_len, k_active, d_model] (per-expert outputs before combining)
            expert_loads: [n_experts] (# active tokens per expert for load tracking)
        """
        batch_seq_len, d_model = hidden_states.shape
        device = hidden_states.device

        # Initialize output: [batch * seq_len, k_active, d_model]
        expert_outputs = torch.zeros(
            batch_seq_len, self.k_active, d_model, dtype=hidden_states.dtype, device=device
        )

        # Track expert utilization
        expert_loads = torch.zeros(self.n_experts, dtype=torch.float32, device=device)

        # Flatten selected_experts to get all (token, k_idx, expert_id) routing decisions
        # selected_experts: [batch * seq_len, k_active]
        # We need to process each expert once with all tokens routed to it
        
        # Build dispatch lists for each expert
        for expert_id in range(self.n_experts):
            # Find all (token_idx, k_idx) pairs that route to this expert
            token_indices = []
            k_indices = []
            
            for k_idx in range(self.k_active):
                # Get mask for tokens where k_idx-th expert is expert_id
                mask = (selected_experts[:, k_idx] == expert_id)
                token_idx = torch.nonzero(mask, as_tuple=True)[0]
                
                if len(token_idx) > 0:
                    token_indices.append(token_idx)
                    k_indices.extend([k_idx] * len(token_idx))
            
            # Skip if no tokens route to this expert
            if len(token_indices) == 0:
                continue
            
            # Concatenate all token indices for this expert
            token_indices = torch.cat(token_indices)
            k_indices = torch.tensor(k_indices, dtype=torch.long, device=device)
            
            # Track load
            expert_loads[expert_id] = len(token_indices) / batch_seq_len
            
            # Gather inputs for this expert (batch processing)
            expert_inputs = hidden_states[token_indices]  # [num_routed_tokens, d_model]
            
            # Execute expert on all assigned tokens at once
            expert = self.experts[expert_id]
            expert_result = expert(expert_inputs)  # [num_routed_tokens, d_model]
            
            # Scatter results back to output tensor
            expert_outputs[token_indices, k_indices] = expert_result

        return expert_outputs, expert_loads

    def get_expert(self, expert_id: int) -> ExpertFFN:
        """Get individual expert module (for analysis/monitoring)."""
        return self.experts[expert_id]

    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and per-expert parameters.

        Returns:
            (total_params, per_expert_params)
        """
        total_params = sum(p.numel() for p in self.parameters())
        per_expert_params = self.experts[0].w1.weight.numel() + self.experts[0].w2.weight.numel()

        return total_params, per_expert_params
