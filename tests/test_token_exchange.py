"""Unit tests for token exchange stub."""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch

from hgsel.distributed.dispatch_api import RemoteDispatchRequests
from hgsel.distributed.token_exchange import TokenExchange


def test_exchange_requests_returns_empty_payloads():
    exchange = TokenExchange()
    requests = RemoteDispatchRequests(
        rank_to_token_indices={1: torch.tensor([0, 1])},
        rank_to_expert_ids={1: torch.tensor([2, 3])},
    )

    payloads = exchange.exchange_requests(
        requests=requests,
        payload_shape=(0, 4),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert 1 in payloads
    assert payloads[1].shape == (0, 4)
