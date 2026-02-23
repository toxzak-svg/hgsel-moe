"""torch.distributed initialization helpers and all-reduce operations.

Safe wrappers to use in tests and scripts without hard dependency.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DistEnv:
    """Resolved distributed environment values."""

    rank: int
    world_size: int
    local_rank: int
    backend: str


def is_dist_available() -> bool:
    """Check if torch.distributed is available in this build."""
    try:
        return torch.distributed.is_available()
    except (ImportError, AttributeError):
        return False


def is_dist_initialized() -> bool:
    """Check if the current process has torch.distributed initialized."""
    if not is_dist_available():
        return False
    return torch.distributed.is_initialized()


def resolve_dist_env(
    default_backend: str = "nccl",
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    local_rank: Optional[int] = None,
) -> DistEnv:
    """Resolve distributed environment values with sane defaults."""
    env_rank = int(os.environ.get("RANK", "0"))
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = os.environ.get("TORCH_DIST_BACKEND", default_backend)

    return DistEnv(
        rank=env_rank if rank is None else rank,
        world_size=env_world_size if world_size is None else world_size,
        local_rank=env_local_rank if local_rank is None else local_rank,
        backend=backend,
    )


def init_distributed(env: Optional[DistEnv] = None) -> DistEnv:
    """Initialize torch.distributed if needed; returns resolved env."""
    env = env or resolve_dist_env()

    if not is_dist_available():
        raise RuntimeError("torch.distributed not available in this build")

    if is_dist_initialized():
        return env

    torch.distributed.init_process_group(
        backend=env.backend,
        rank=env.rank,
        world_size=env.world_size,
    )

    if env.backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(env.local_rank)

    return env


def get_rank() -> int:
    """Get the rank of the current process.
    
    Returns 0 if torch.distributed is not initialized (single-GPU fallback).
    """
    if is_dist_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    """Get the total number of processes in the distributed group.
    
    Returns 1 if torch.distributed is not initialized (single-GPU fallback).
    """
    if is_dist_initialized():
        return torch.distributed.get_world_size()
    return 1


def get_backend() -> Optional[str]:
    """Get the backend used by torch.distributed.
    
    Returns None if not initialized.
    """
    if is_dist_initialized():
        return torch.distributed.get_backend()
    return None


def barrier(group: Optional[object] = None) -> None:
    """Synchronize all processes.
    
    No-op if torch.distributed is not initialized.
    """
    if is_dist_initialized():
        torch.distributed.barrier(group=group)


def all_reduce_sum(tensor: torch.Tensor, group: Optional[object] = None) -> torch.Tensor:
    """Sum a tensor across all ranks.
    
    Args:
        tensor: A tensor on the current device
        group: Process group (default: None)
    
    Returns:
        The summed tensor (modified in-place, but also returned for convenience)
    """
    if not is_dist_initialized():
        return tensor
    
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=group)
    return tensor


def all_reduce_mean(tensor: torch.Tensor, group: Optional[object] = None) -> torch.Tensor:
    """Average a tensor across all ranks.
    
    Args:
        tensor: A tensor on the current device
        group: Process group (default: None)
    
    Returns:
        The averaged tensor (modified in-place, but also returned for convenience)
    """
    if not is_dist_initialized():
        return tensor
    
    world_size = get_world_size()
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=group)
    tensor.div_(world_size)
    return tensor


def all_gather(tensor: torch.Tensor, group: Optional[object] = None) -> list:
    """Gather tensors from all ranks.
    
    Args:
        tensor: A tensor on the current device
        group: Process group (default: None)
    
    Returns:
        List of tensors from all ranks (on the same device as the input)
    """
    if not is_dist_initialized():
        return [tensor]
    
    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor, group=group)
    return tensor_list


def all_to_all(send_tensor: torch.Tensor, recv_tensor: torch.Tensor,
               group: Optional[object] = None) -> None:
    """All-to-all collective operation.
    
    Args:
        send_tensor: Tensor to send. May have shape [world_size * batch_size, ...]
        recv_tensor: Tensor to receive. Will have same shape as send_tensor
        group: Process group (default: None)
    """
    if not is_dist_initialized():
        recv_tensor.copy_(send_tensor)
        return
    
    torch.distributed.all_to_all_single(recv_tensor, send_tensor, group=group)


def reduce_scatter(output: torch.Tensor, input_list: list,
                  group: Optional[object] = None) -> None:
    """Reduce-scatter operation: sum across ranks then scatter.
    
    Args:
        output: Output tensor for this rank
        input_list: List of tensors from all ranks (one per rank)
        group: Process group (default: None)
    """
    if not is_dist_initialized():
        output.copy_(input_list[0])
        return
    
    torch.distributed.reduce_scatter(output, input_list, group=group)


def broadcast(tensor: torch.Tensor, src: int = 0, group: Optional[object] = None) -> torch.Tensor:
    """Broadcast tensor from source rank to all ranks.
    
    Args:
        tensor: The tensor to broadcast (must be on device for src rank)
        src: Source rank
        group: Process group (default: None)
    
    Returns:
        The tensor (modified in-place if not src rank)
    """
    if not is_dist_initialized():
        return tensor
    
    torch.distributed.broadcast(tensor, src=src, group=group)
    return tensor


@functools.lru_cache(maxsize=1)
def get_device() -> torch.device:
    """Get the device for the current rank.
    
    For NCCL backend: cuda:{rank % num_gpus}
    For Gloo backend: cpu
    Otherwise: cuda:0 if available, else cpu
    """
    if not is_dist_initialized():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    backend = get_backend()
    rank = get_rank()
    
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA")
        num_gpus = torch.cuda.device_count()
        device_id = rank % num_gpus
        return torch.device(f"cuda:{device_id}")
    elif backend == "gloo":
        return torch.device("cpu")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_distributed() -> None:
    """Clean up distributed training resources.
    
    Safe to call even if torch.distributed is not initialized.
    """
    if is_dist_initialized():
        torch.distributed.destroy_process_group()
