"""Unit tests for TokenExchange behavior and stats (no real process group required)."""

import sys
from pathlib import Path

import pytest
import torch

current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

import hgsel.distributed.token_exchange as token_exchange_mod  # noqa: E402
from hgsel.distributed.token_exchange import TokenExchange  # noqa: E402


def test_token_exchange_single_rank_fallback_captures_stats():
    exchange = TokenExchange()
    payload = torch.randn(3, 4)

    result = exchange.exchange({0: payload})

    assert list(result.keys()) == [0]
    assert torch.equal(result[0], payload)

    stats = exchange.last_exchange_stats
    assert stats is not None
    assert stats["distributed_enabled"] is False
    assert stats["world_size"] == 1
    assert stats["max_tokens_per_rank_padded"] == 3
    assert stats["per_rank_send_counts"] == {0: 3}
    assert stats["per_rank_send_bytes"][0] == payload.numel() * payload.element_size()
    assert stats["shape_signature"] == "exchange_send:1:3"


def test_token_exchange_non_initialized_multiworld_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(token_exchange_mod, "get_world_size", lambda: 4)
    monkeypatch.setattr(token_exchange_mod, "get_rank", lambda: 0)
    monkeypatch.setattr(token_exchange_mod, "is_dist_initialized", lambda: False)

    exchange = TokenExchange()
    payloads = {
        0: torch.randn(2, 8),
        2: torch.randn(5, 8),
    }

    result = exchange.exchange(payloads)

    # Fallback returns rank-0 local payload only when process group is not initialized.
    assert list(result.keys()) == [0]
    assert torch.equal(result[0], payloads[0])

    stats = exchange.last_exchange_stats
    assert stats is not None
    assert stats["distributed_enabled"] is False
    assert stats["world_size"] == 4
    assert stats["max_tokens_per_rank_padded"] == 5
    assert stats["per_rank_send_counts"] == {0: 2, 1: 0, 2: 5, 3: 0}
    assert stats["shape_signature"] == "exchange_send:4:2,0,5,0"


def test_token_exchange_simulated_distributed_padding_and_stats(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(token_exchange_mod, "get_world_size", lambda: 2)
    monkeypatch.setattr(token_exchange_mod, "get_rank", lambda: 0)
    monkeypatch.setattr(token_exchange_mod, "is_dist_initialized", lambda: True)

    def fake_all_to_all(send_tensor: torch.Tensor, recv_tensor: torch.Tensor, group=None) -> None:
        # Identity copy simulates a degenerate distributed exchange for local unit testing.
        recv_tensor.copy_(send_tensor)

    monkeypatch.setattr(token_exchange_mod, "all_to_all", fake_all_to_all)

    exchange = TokenExchange()
    payload_rank0 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    payload_rank1 = torch.tensor([[10.0, 20.0]])

    result = exchange.exchange({0: payload_rank0, 1: payload_rank1})

    # max_tokens = 3, so each received slice is padded to [3, 2]
    assert set(result.keys()) == {0, 1}
    assert result[0].shape == (3, 2)
    assert result[1].shape == (3, 2)
    assert torch.equal(result[0], payload_rank0)
    assert torch.equal(result[1][0], payload_rank1[0])
    assert torch.equal(result[1][1:], torch.zeros(2, 2))

    stats = exchange.last_exchange_stats
    assert stats is not None
    assert stats["distributed_enabled"] is True
    assert stats["world_size"] == 2
    assert stats["max_tokens_per_rank_padded"] == 3
    assert stats["per_rank_send_counts"] == {0: 3, 1: 1}
    assert stats["per_rank_send_bytes"][0] == payload_rank0.numel() * payload_rank0.element_size()
    assert stats["per_rank_send_bytes"][1] == payload_rank1.numel() * payload_rank1.element_size()
    assert stats["shape_signature"] == "exchange_send:2:3,1"


def test_token_exchange_distributed_requires_nonempty_payloads(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(token_exchange_mod, "get_world_size", lambda: 2)
    monkeypatch.setattr(token_exchange_mod, "get_rank", lambda: 0)
    monkeypatch.setattr(token_exchange_mod, "is_dist_initialized", lambda: True)

    exchange = TokenExchange()

    with pytest.raises(ValueError, match="payloads cannot be empty"):
        exchange.exchange({})
