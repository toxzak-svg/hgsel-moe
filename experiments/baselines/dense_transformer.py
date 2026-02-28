"""
Dense baseline Transformer blocks for comparison.

Used to establish quality/speed benchmarks before HGSEL optimization.

Provides:
- TransformerBlock: Standard attention + MLP
- DenseMLPBlock: Standard 2-layer FFN (baseline MLP)
- TransformerModel: Small Transformer model for testing
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class DenseMLPBlock(nn.Module):
    """
    Standard dense MLP: d_model → d_ff → d_model

    This is the baseline that HGSEL replaces.

    Args:
        d_model: Embedding dimension
        d_ff: Hidden dimension
        activation: Activation function
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=True)
        self.w2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model]
        """
        # Reshape to 2D for linear ops
        batch, seq_len, d_model = x.shape
        x_flat = x.view(batch * seq_len, d_model)

        # MLP
        out = self.w1(x_flat)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.w2(out)
        out = self.dropout(out)

        # Reshape back
        return out.view(batch, seq_len, d_model)


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention.

    Args:
        d_model: Embedding dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [seq_len, seq_len] (optional causal mask)

        Returns:
            [batch, seq_len, d_model]
        """
        attn_out, _ = self.mha(x, x, x, attn_mask=mask)
        return self.dropout(attn_out)


class TransformerBlock(nn.Module):
    """
    Single Transformer block: LayerNorm → Attention → LayerNorm → MLP/HGSEL

    Args:
        d_model: Embedding dimension
        d_ff: MLP hidden dimension
        n_heads: Number of attention heads
        mlp_class: MLP module class (DenseMLPBlock or HGSELLayer)
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        n_heads: int = 8,
        mlp_class=None,
        mlp_kwargs: Optional[Dict[str, Any]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if mlp_class is None:
            mlp_class = DenseMLPBlock

        self.attn = AttentionBlock(d_model, n_heads, dropout)

        # Instantiate MLP based on class signature
        mlp_kwargs = dict(mlp_kwargs or {})

        if mlp_class.__name__ == "HGSELLayer":
            # HGSELLayer uses different argument names
            mlp_kwargs.setdefault("d_model", d_model)
            mlp_kwargs.setdefault("d_ff", d_ff)
            self.mlp = mlp_class(
                **mlp_kwargs,
            )
        else:
            # DenseMLPBlock and other standard MLPs
            mlp_kwargs.setdefault("d_model", d_model)
            mlp_kwargs.setdefault("d_ff", d_ff)
            mlp_kwargs.setdefault("dropout", dropout)
            self.mlp = mlp_class(
                **mlp_kwargs,
            )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model]
        """
        # Attention + residual
        attn_out = self.attn(self.ln1(x))
        x = x + self.dropout(attn_out)

        # MLP + residual
        mlp_out = self.mlp(self.ln2(x))
        x = x + self.dropout(mlp_out)

        return x


class TransformerModel(nn.Module):
    """
    Minimal Transformer model for benchmarking.

    Args:
        vocab_size: Vocabulary size
        d_model: Embedding dimension
        d_ff: MLP hidden dimension
        n_layers: Number of transformer blocks
        n_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        mlp_class: MLP module class (defaults to DenseMLPBlock)
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        d_ff: int = 2048,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 512,
        mlp_class=None,
        mlp_kwargs: Optional[Dict[str, Any]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        if mlp_class is None:
            mlp_class = DenseMLPBlock

        blocks = []
        for layer_idx in range(n_layers):
            layer_mlp_kwargs = dict(mlp_kwargs or {})
            if mlp_class.__name__ == "HGSELLayer":
                layer_mlp_kwargs.setdefault("layer_id", layer_idx)
            blocks.append(
                TransformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    mlp_class=mlp_class,
                    mlp_kwargs=layer_mlp_kwargs,
                    dropout=dropout,
                )
            )

        self.layers = nn.ModuleList(blocks)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch, seq_len = input_ids.shape

        # Embeddings
        x = self.embedding(input_ids)  # [batch, seq_len, d_model]

        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)  # [1, seq_len, d_model]
        x = x + pos_emb

        x = self.dropout(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output
        logits = self.output_proj(x)  # [batch, seq_len, vocab_size]

        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_routing_diagnostics(self) -> Dict[str, Any]:
        """Aggregate HGSEL routing diagnostics across layers if present."""
        entropies: List[float] = []
        expert_loads: List[torch.Tensor] = []

        for layer in self.layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            if hasattr(mlp, "get_expert_load_stats"):
                stats = mlp.get_expert_load_stats()
                if "entropy" in stats:
                    entropies.append(float(stats["entropy"]))
            if hasattr(mlp, "expert_load_ema"):
                expert_loads.append(mlp.expert_load_ema.detach())

        if not entropies and not expert_loads:
            return {}

        diagnostics: Dict[str, Any] = {}
        if entropies:
            diagnostics["entropy"] = sum(entropies) / len(entropies)
        if expert_loads:
            diagnostics["expert_load"] = torch.stack(expert_loads, dim=0).mean(dim=0)
        return diagnostics

    def get_phase4_routing_traces(self) -> List[Dict[str, Any]]:
        """Return recent HGSEL layer traces (one entry per layer)."""
        traces: List[Dict[str, Any]] = []
        for layer in self.layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is None or not hasattr(mlp, "get_last_forward_trace"):
                continue
            trace = mlp.get_last_forward_trace()
            if trace:
                traces.append(trace)
        return traces


if __name__ == "__main__":
    # Simple test
    print("Baseline Transformer Model Test")
    print("=" * 60)

    vocab_size = 5000
    d_model = 256
    d_ff = 1024
    n_layers = 4
    batch_size = 8
    seq_len = 128

    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_layers=n_layers,
        max_seq_len=seq_len,
    )

    # Print model size
    n_params = model.count_parameters()
    print(f"Dense Baseline Model")
    print(f"  Vocab: {vocab_size:,}")
    print(f"  d_model: {d_model}")
    print(f"  d_ff: {d_ff}")
    print(f"  Layers: {n_layers}")
    print(f"  Total params: {n_params:,}")
    print()

    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    print(f"Input shape: {input_ids.shape}")
    logits = model(input_ids)
    print(f"Output shape: {logits.shape}")
    print(f"Output dtype: {logits.dtype}")
    print()

    # Check for NaNs
    if torch.isnan(logits).any():
        print("⚠ WARNING: Output contains NaNs!")
    else:
        print("✓ No NaNs in output")

    print("\n✓ Baseline model test passed!")
