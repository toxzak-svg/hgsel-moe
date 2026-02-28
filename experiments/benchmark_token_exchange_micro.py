#!/usr/bin/env python
"""
Expert-parallel communication microbenchmark for HGSEL.

Purpose:
- Measure raw all-to-all token exchange time vs local expert compute time
- Stress test routing distributions (balanced -> skewed)
- Produce a communication-share gate before full distributed trainer integration

Usage:
    # Single process (CPU or GPU fallback)
    python experiments/benchmark_token_exchange_micro.py \
        --tokens-per-rank 1024,4096 --hidden-dims 256,512

    # Multi-process / multi-GPU
    torchrun --nproc_per_node=4 experiments/benchmark_token_exchange_micro.py \
        --tokens-per-rank 2048,4096 \
        --hidden-dims 512,1024 \
        --routing-modes balanced,moderate_skew,worst_skew
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

# Add parent directory to path
CURRENT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CURRENT_DIR))

from hgsel.distributed import dist_utils  # noqa: E402


@dataclass
class IterationMetrics:
    """Per-iteration timing and load metrics."""

    exchange_out_ms: float
    local_compute_ms: float
    exchange_back_ms: float
    comm_share: float
    received_tokens: int


@dataclass
class RankSummary:
    """Summary for a single rank/configuration."""

    rank: int
    world_size: int
    device: str
    dtype: str
    tokens_per_rank: int
    hidden_dim: int
    ff_mult: int
    routing_mode: str
    num_bench_iters: int
    received_tokens_mean: float
    received_tokens_min: int
    received_tokens_max: int
    exchange_out_p50_ms: float
    exchange_out_p95_ms: float
    local_compute_p50_ms: float
    local_compute_p95_ms: float
    exchange_back_p50_ms: float
    exchange_back_p95_ms: float
    comm_share_p50: float
    comm_share_p95: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HGSEL expert-parallel token exchange microbenchmark"
    )
    parser.add_argument(
        "--tokens-per-rank",
        type=str,
        default="2048,4096",
        help="Comma-separated token counts per rank to benchmark",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="512,1024",
        help="Comma-separated hidden dimensions to benchmark",
    )
    parser.add_argument(
        "--routing-modes",
        type=str,
        default="balanced,moderate_skew,worst_skew",
        help="Comma-separated routing distributions: balanced, moderate_skew, worst_skew",
    )
    parser.add_argument(
        "--ff-mult",
        type=int,
        default=4,
        help="FFN expansion ratio for local expert compute simulation",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Warmup iterations per configuration",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=50,
        help="Measured iterations per configuration",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Tensor dtype for payloads and local compute",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to benchmark on (auto prefers CUDA)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "nccl", "gloo"],
        help="Distributed backend when running with torchrun",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed base",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/token_exchange_micro/benchmark_token_exchange_micro.json",
        help="Path to write rank-aggregated JSON results (rank 0 only)",
    )
    parser.add_argument(
        "--threshold-warn",
        type=float,
        default=0.20,
        help="Communication share warning threshold",
    )
    parser.add_argument(
        "--threshold-stop",
        type=float,
        default=0.40,
        help="Communication share stop/redesign threshold",
    )
    return parser.parse_args()


def parse_int_csv(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_backend(args: argparse.Namespace) -> str:
    if args.backend != "auto":
        return args.backend
    if args.device == "cpu":
        return "gloo"
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        if dist_utils.is_dist_initialized():
            return dist_utils.get_device()
        return torch.device("cuda")

    # auto
    if dist_utils.is_dist_initialized():
        # gloo defaults to CPU in dist_utils.get_device(); nccl uses local cuda device
        return dist_utils.get_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = mapping[name]
    if device.type == "cpu" and dtype == torch.float16:
        # float16 CPU support exists but can be extremely slow / unsupported in ops
        return torch.float32
    return dtype


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    t = torch.tensor(list(values), dtype=torch.float64)
    return float(torch.quantile(t, q).item())


def sample_destinations(
    *,
    world_size: int,
    tokens_per_rank: int,
    routing_mode: str,
    rank: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate destination-rank assignments for synthetic routed tokens."""
    if world_size == 1:
        return torch.zeros(tokens_per_rank, device=device, dtype=torch.long)

    if routing_mode == "balanced":
        # Uniform random routing; close to well-mixed deterministic hashing in expectation.
        return torch.randint(
            low=0,
            high=world_size,
            size=(tokens_per_rank,),
            device=device,
            generator=generator,
        )

    if routing_mode == "moderate_skew":
        hotspot_rank = 0  # Global hotspot to stress load imbalance
        hotspot_prob = 0.60
    elif routing_mode == "worst_skew":
        hotspot_rank = 0
        hotspot_prob = 0.90
    else:
        raise ValueError(f"Unsupported routing mode: {routing_mode}")

    probs = torch.full((world_size,), (1.0 - hotspot_prob) / max(world_size - 1, 1), device=device)
    probs[hotspot_rank] = hotspot_prob
    # torch.multinomial on CUDA supports float probs.
    return torch.multinomial(probs, num_samples=tokens_per_rank, replacement=True, generator=generator)


def build_send_buffers(
    *,
    tokens_per_rank: int,
    hidden_dim: int,
    world_size: int,
    routing_mode: str,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build padded all-to-all buffers and per-destination counts.

    We use a fixed capacity of `tokens_per_rank` per destination rank to keep send/recv
    shapes identical across ranks, which is required by all_to_all_single().
    """
    destinations = sample_destinations(
        world_size=world_size,
        tokens_per_rank=tokens_per_rank,
        routing_mode=routing_mode,
        rank=rank,
        device=device,
        generator=generator,
    )

    counts = torch.bincount(destinations, minlength=world_size).to(torch.int64)

    # Synthetic hidden states for this rank's tokens.
    token_payloads = torch.randn(
        (tokens_per_rank, hidden_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )

    # [dst_rank, capacity=tokens_per_rank, hidden_dim]
    send_matrix = torch.zeros(
        (world_size, tokens_per_rank, hidden_dim),
        device=device,
        dtype=dtype,
    )

    # Pack rows by destination rank (not timed in this microbenchmark).
    for dst_rank in range(world_size):
        idx = torch.nonzero(destinations == dst_rank, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        send_matrix[dst_rank, : idx.numel()] = token_payloads.index_select(0, idx)

    send_buffer = send_matrix.reshape(world_size * tokens_per_rank, hidden_dim).contiguous()
    return send_buffer, counts


def all_to_all_counts(send_counts: torch.Tensor) -> torch.Tensor:
    """Exchange integer counts so each rank knows how many valid tokens it received."""
    recv_counts = torch.empty_like(send_counts)
    dist_utils.all_to_all(send_counts, recv_counts)
    return recv_counts


def simulate_local_expert_compute(
    *,
    num_tokens: int,
    hidden_dim: int,
    ff_mult: int,
    device: torch.device,
    dtype: torch.dtype,
    weights: Dict[str, torch.Tensor],
    scratch: torch.Tensor,
) -> None:
    """Run a simple FFN to approximate local expert compute cost."""
    if num_tokens <= 0:
        return

    x = scratch[:num_tokens]
    # Shape conventions for F.linear: weight is [out_features, in_features]
    h = F.linear(x, weights["w1"])
    h = F.gelu(h)
    _ = F.linear(h, weights["w2"])


def time_block(device: torch.device, fn) -> float:
    sync_device(device)
    start = time.perf_counter()
    fn()
    sync_device(device)
    return (time.perf_counter() - start) * 1000.0


def summarize_iterations(
    *,
    metrics: Sequence[IterationMetrics],
    rank: int,
    world_size: int,
    device: torch.device,
    dtype: torch.dtype,
    tokens_per_rank: int,
    hidden_dim: int,
    ff_mult: int,
    routing_mode: str,
) -> RankSummary:
    exchange_out = [m.exchange_out_ms for m in metrics]
    local_compute = [m.local_compute_ms for m in metrics]
    exchange_back = [m.exchange_back_ms for m in metrics]
    comm_share = [m.comm_share for m in metrics]
    recv_tokens = [m.received_tokens for m in metrics]

    return RankSummary(
        rank=rank,
        world_size=world_size,
        device=str(device),
        dtype=str(dtype).replace("torch.", ""),
        tokens_per_rank=tokens_per_rank,
        hidden_dim=hidden_dim,
        ff_mult=ff_mult,
        routing_mode=routing_mode,
        num_bench_iters=len(metrics),
        received_tokens_mean=float(sum(recv_tokens) / max(len(recv_tokens), 1)),
        received_tokens_min=int(min(recv_tokens) if recv_tokens else 0),
        received_tokens_max=int(max(recv_tokens) if recv_tokens else 0),
        exchange_out_p50_ms=percentile(exchange_out, 0.50),
        exchange_out_p95_ms=percentile(exchange_out, 0.95),
        local_compute_p50_ms=percentile(local_compute, 0.50),
        local_compute_p95_ms=percentile(local_compute, 0.95),
        exchange_back_p50_ms=percentile(exchange_back, 0.50),
        exchange_back_p95_ms=percentile(exchange_back, 0.95),
        comm_share_p50=percentile(comm_share, 0.50),
        comm_share_p95=percentile(comm_share, 0.95),
    )


def grade_comm_share(comm_share: float, warn: float, stop: float) -> str:
    if comm_share > stop:
        return "STOP_REDESIGN"
    if comm_share > warn:
        return "WARN_OPTIMIZE"
    return "PASS"


def aggregate_rank_summaries(
    rank_summaries: Sequence[Dict[str, object]],
    warn_threshold: float,
    stop_threshold: float,
) -> Dict[str, object]:
    if not rank_summaries:
        return {}

    recv_means = [float(rs["received_tokens_mean"]) for rs in rank_summaries]
    comm_share_p50s = [float(rs["comm_share_p50"]) for rs in rank_summaries]
    comm_share_p95s = [float(rs["comm_share_p95"]) for rs in rank_summaries]

    mean_recv = sum(recv_means) / len(recv_means)
    recv_imbalance_ratio = (max(recv_means) / mean_recv) if mean_recv > 0 else 0.0

    worst_rank_comm_p50 = max(comm_share_p50s)
    worst_rank_comm_p95 = max(comm_share_p95s)

    representative_comm_share = worst_rank_comm_p50
    decision = grade_comm_share(representative_comm_share, warn_threshold, stop_threshold)

    first = rank_summaries[0]
    return {
        "config": {
            "world_size": int(first["world_size"]),
            "tokens_per_rank": int(first["tokens_per_rank"]),
            "hidden_dim": int(first["hidden_dim"]),
            "ff_mult": int(first["ff_mult"]),
            "routing_mode": str(first["routing_mode"]),
            "device": str(first["device"]),
            "dtype": str(first["dtype"]),
            "num_bench_iters": int(first["num_bench_iters"]),
        },
        "aggregate": {
            "received_tokens_mean_across_ranks": mean_recv,
            "received_tokens_imbalance_ratio_max_over_mean": recv_imbalance_ratio,
            "comm_share_p50_mean_rank": sum(comm_share_p50s) / len(comm_share_p50s),
            "comm_share_p50_worst_rank": worst_rank_comm_p50,
            "comm_share_p95_worst_rank": worst_rank_comm_p95,
            "decision": decision,
            "threshold_warn": warn_threshold,
            "threshold_stop": stop_threshold,
        },
        "rank_summaries": list(rank_summaries),
    }


def gather_rank_summaries(summary: RankSummary) -> List[Dict[str, object]]:
    if not dist_utils.is_dist_initialized():
        return [asdict(summary)]

    world_size = dist_utils.get_world_size()
    gathered: List[Dict[str, object]] = [None for _ in range(world_size)]  # type: ignore[list-item]
    torch.distributed.all_gather_object(gathered, asdict(summary))
    return gathered


def benchmark_config(
    *,
    tokens_per_rank: int,
    hidden_dim: int,
    routing_mode: str,
    ff_mult: int,
    warmup_iters: int,
    bench_iters: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> RankSummary:
    rank = dist_utils.get_rank()
    world_size = dist_utils.get_world_size()

    if device.type == "cuda":
        torch.cuda.set_device(device)

    gen = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    gen.manual_seed(seed + rank * 100_000 + tokens_per_rank * 31 + hidden_dim * 7)

    ff_dim = hidden_dim * ff_mult
    weights = {
        "w1": torch.randn((ff_dim, hidden_dim), device=device, dtype=dtype, generator=gen) / math.sqrt(hidden_dim),
        "w2": torch.randn((hidden_dim, ff_dim), device=device, dtype=dtype, generator=gen) / math.sqrt(ff_dim),
    }
    # Max possible received tokens = all ranks send all tokens to this rank.
    scratch = torch.randn((world_size * tokens_per_rank, hidden_dim), device=device, dtype=dtype, generator=gen)

    recv_buffer = torch.empty((world_size * tokens_per_rank, hidden_dim), device=device, dtype=dtype)
    return_buffer = torch.empty_like(recv_buffer)

    bench_metrics: List[IterationMetrics] = []
    total_iters = warmup_iters + bench_iters

    for step_idx in range(total_iters):
        send_buffer, send_counts = build_send_buffers(
            tokens_per_rank=tokens_per_rank,
            hidden_dim=hidden_dim,
            world_size=world_size,
            routing_mode=routing_mode,
            rank=rank,
            device=device,
            dtype=dtype,
            generator=gen,
        )

        # Metadata exchange for valid token counts per source.
        recv_counts = all_to_all_counts(send_counts)
        received_tokens = int(recv_counts.sum().item())

        def _exchange_out() -> None:
            dist_utils.all_to_all(send_buffer, recv_buffer)

        def _local_compute() -> None:
            simulate_local_expert_compute(
                num_tokens=received_tokens,
                hidden_dim=hidden_dim,
                ff_mult=ff_mult,
                device=device,
                dtype=dtype,
                weights=weights,
                scratch=scratch,
            )

        def _exchange_back() -> None:
            # Simulate returning expert outputs to source ranks.
            dist_utils.all_to_all(recv_buffer, return_buffer)

        exchange_out_ms = time_block(device, _exchange_out)
        local_compute_ms = time_block(device, _local_compute)
        exchange_back_ms = time_block(device, _exchange_back)

        denom = exchange_out_ms + local_compute_ms + exchange_back_ms
        comm_share = ((exchange_out_ms + exchange_back_ms) / denom) if denom > 0 else 0.0

        if step_idx >= warmup_iters:
            bench_metrics.append(
                IterationMetrics(
                    exchange_out_ms=exchange_out_ms,
                    local_compute_ms=local_compute_ms,
                    exchange_back_ms=exchange_back_ms,
                    comm_share=comm_share,
                    received_tokens=received_tokens,
                )
            )

    return summarize_iterations(
        metrics=bench_metrics,
        rank=rank,
        world_size=world_size,
        device=device,
        dtype=dtype,
        tokens_per_rank=tokens_per_rank,
        hidden_dim=hidden_dim,
        ff_mult=ff_mult,
        routing_mode=routing_mode,
    )


def maybe_init_distributed(backend: str) -> None:
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size_env <= 1:
        return
    if dist_utils.is_dist_initialized():
        return
    env = dist_utils.resolve_dist_env(default_backend=backend)
    dist_utils.init_distributed(env)


def print_aggregate_result(result: Dict[str, object]) -> None:
    cfg = result["config"]
    agg = result["aggregate"]
    print(
        "[microbench] "
        f"world={cfg['world_size']} "
        f"tokens/rank={cfg['tokens_per_rank']} "
        f"h={cfg['hidden_dim']} "
        f"mode={cfg['routing_mode']} "
        f"comm_share_p50_worst={agg['comm_share_p50_worst_rank']:.3f} "
        f"comm_share_p95_worst={agg['comm_share_p95_worst_rank']:.3f} "
        f"recv_imbalance={agg['received_tokens_imbalance_ratio_max_over_mean']:.2f}x "
        f"decision={agg['decision']}"
    )


def main() -> None:
    args = parse_args()
    backend = resolve_backend(args)

    maybe_init_distributed(backend)
    rank = dist_utils.get_rank()
    world_size = dist_utils.get_world_size()

    device = resolve_device(args)
    dtype = resolve_dtype(args.dtype, device)
    if rank == 0 and args.dtype != str(dtype).replace("torch.", ""):
        print(
            f"[microbench] dtype downgraded from {args.dtype} to {str(dtype).replace('torch.', '')} "
            f"for device={device}"
        )

    if rank == 0:
        print("HGSEL Expert-Parallel Token Exchange Microbenchmark")
        print(f"world_size={world_size} backend={dist_utils.get_backend() or 'none'} device={device} dtype={dtype}")
        print(f"tokens_per_rank={parse_int_csv(args.tokens_per_rank)}")
        print(f"hidden_dims={parse_int_csv(args.hidden_dims)}")
        print(f"routing_modes={parse_str_csv(args.routing_modes)}")
        print()

    results: List[Dict[str, object]] = []

    try:
        for tokens_per_rank in parse_int_csv(args.tokens_per_rank):
            for hidden_dim in parse_int_csv(args.hidden_dims):
                for routing_mode in parse_str_csv(args.routing_modes):
                    # Keep ranks aligned across configs.
                    dist_utils.barrier()
                    summary = benchmark_config(
                        tokens_per_rank=tokens_per_rank,
                        hidden_dim=hidden_dim,
                        routing_mode=routing_mode,
                        ff_mult=args.ff_mult,
                        warmup_iters=args.warmup_iters,
                        bench_iters=args.bench_iters,
                        device=device,
                        dtype=dtype,
                        seed=args.seed,
                    )
                    gathered = gather_rank_summaries(summary)
                    if rank == 0:
                        result = aggregate_rank_summaries(
                            gathered,
                            warn_threshold=args.threshold_warn,
                            stop_threshold=args.threshold_stop,
                        )
                        results.append(result)
                        print_aggregate_result(result)
    finally:
        dist_utils.barrier()
        if rank == 0:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "metadata": {
                    "script": "experiments/benchmark_token_exchange_micro.py",
                    "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "world_size": world_size,
                    "backend": dist_utils.get_backend(),
                    "device": str(device),
                    "dtype": str(dtype).replace("torch.", ""),
                    "warmup_iters": args.warmup_iters,
                    "bench_iters": args.bench_iters,
                    "threshold_warn": args.threshold_warn,
                    "threshold_stop": args.threshold_stop,
                },
                "results": results,
            }
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"\n[microbench] wrote results -> {output_path}")

        # Leave process group cleanup to process exit in torchrun, but explicit cleanup is fine.
        dist_utils.cleanup_distributed()


if __name__ == "__main__":
    main()
