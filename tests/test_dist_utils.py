"""Unit tests for distributed utils (no process group required)."""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from hgsel.distributed.dist_utils import resolve_dist_env


def test_resolve_dist_env_defaults(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("TORCH_DIST_BACKEND", raising=False)

    env = resolve_dist_env(default_backend="gloo")
    assert env.rank == 0
    assert env.world_size == 1
    assert env.local_rank == 0
    assert env.backend == "gloo"


def test_resolve_dist_env_overrides(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("TORCH_DIST_BACKEND", "gloo")

    env = resolve_dist_env(rank=2, world_size=4, local_rank=0, default_backend="nccl")
    assert env.rank == 2
    assert env.world_size == 4
    assert env.local_rank == 0
    assert env.backend == "gloo"
