"""Minimal distributed smoke test entrypoint.

Run with:
  python -m torch.distributed.run --nproc_per_node 2 experiments/dist_smoke.py
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch

from hgsel.distributed.dist_utils import init_distributed, resolve_dist_env


def main() -> None:
    env = resolve_dist_env(default_backend="gloo")
    init_distributed(env)

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    print(f"dist_smoke: rank={rank} world_size={world_size}")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
