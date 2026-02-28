#!/usr/bin/env python
"""
Instrumented GPU baseline benchmark for Phase 4.

This script establishes the single-device control condition used to evaluate
distributed scaling later. It measures:

- tokens/sec (training step throughput)
- forward time
- backward time
- optimizer step time
- peak memory (allocated/reserved)
- HGSEL expert utilization histogram / entropy

Usage:
    # GPU benchmark (recommended)
    python experiments/train_gpu_baseline.py --device cuda

    # Quick CPU smoke test
    python experiments/train_gpu_baseline.py \
        --device cpu --models hgsel \
        --batch-size 2 --seq-length 8 \
        --d-model 16 --d-ff 64 --num-layers 1 --num-heads 2 --num-experts 8 \
        --warmup-steps 1 --bench-steps 2
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

# Add parent directory to path
CURRENT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CURRENT_DIR))

from experiments.baselines.dense_transformer import TransformerModel  # noqa: E402
from hgsel.distributed.memory_profiler import (  # noqa: E402
    MemoryProfiler,
    estimate_model_memory_requirements,
)
from hgsel.layer import HGSELLayer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instrumented GPU baseline benchmark")

    # Benchmark loop
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup training steps")
    parser.add_argument("--bench-steps", type=int, default=50, help="Measured training steps")
    parser.add_argument("--models", type=str, default="dense,hgsel", help="Models to benchmark: dense,hgsel")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # Data shape
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size")

    # Model shape
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--d-ff", type=int, default=1024, help="FFN dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout (0 recommended for stable timing)")

    # HGSEL config
    parser.add_argument("--num-experts", type=int, default=64, help="HGSEL experts")
    parser.add_argument("--k-active", type=int, default=2, help="HGSEL active experts per token")
    parser.add_argument("--num-hashes", type=int, default=4, help="HGSEL routing hashes")
    parser.add_argument("--combine-mode", type=str, default="uniform", help="HGSEL combine mode")

    # Execution
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="results/gpu_baseline/train_gpu_baseline.json",
        help="Path to write JSON results",
    )
    return parser.parse_args()


def parse_csv(value: str) -> List[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping[dtype_arg]
    if device.type == "cpu" and dtype != torch.float32:
        return torch.float32
    return dtype


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return float(torch.quantile(tensor, q).item())


def create_model(model_kind: str, args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> TransformerModel:
    if model_kind == "hgsel":
        mlp_class = HGSELLayer
        mlp_kwargs = {
            "n_experts": args.num_experts,
            "k_active": args.k_active,
            "n_hashes": args.num_hashes,
            "combine_mode": args.combine_mode,
        }
    elif model_kind == "dense":
        mlp_class = None
        mlp_kwargs = None
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")

    model = TransformerModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_layers=args.num_layers,
        n_heads=args.num_heads,
        max_seq_len=args.seq_length,
        mlp_class=mlp_class,
        mlp_kwargs=mlp_kwargs,
        dropout=args.dropout,
    )
    model = model.to(device=device, dtype=dtype if dtype != torch.float32 else None)
    model.train()
    return model


def make_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_length), device=device, dtype=torch.long)
    labels = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_length), device=device, dtype=torch.long)
    return input_ids, labels


def time_step_cuda(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    optimizer.zero_grad(set_to_none=True)

    e_fwd_start = torch.cuda.Event(enable_timing=True)
    e_fwd_end = torch.cuda.Event(enable_timing=True)
    e_bwd_start = torch.cuda.Event(enable_timing=True)
    e_bwd_end = torch.cuda.Event(enable_timing=True)
    e_opt_start = torch.cuda.Event(enable_timing=True)
    e_opt_end = torch.cuda.Event(enable_timing=True)

    e_fwd_start.record()
    logits = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    e_fwd_end.record()

    e_bwd_start.record()
    loss.backward()
    e_bwd_end.record()

    e_opt_start.record()
    optimizer.step()
    e_opt_end.record()

    e_opt_end.synchronize()

    return {
        "loss": float(loss.item()),
        "forward_ms": float(e_fwd_start.elapsed_time(e_fwd_end)),
        "backward_ms": float(e_bwd_start.elapsed_time(e_bwd_end)),
        "optimizer_ms": float(e_opt_start.elapsed_time(e_opt_end)),
        "total_ms": float(e_fwd_start.elapsed_time(e_opt_end)),
    }


def time_step_cpu(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    optimizer.zero_grad(set_to_none=True)

    t0 = time.perf_counter()
    logits = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    t1 = time.perf_counter()

    loss.backward()
    t2 = time.perf_counter()

    optimizer.step()
    t3 = time.perf_counter()

    return {
        "loss": float(loss.item()),
        "forward_ms": (t1 - t0) * 1000.0,
        "backward_ms": (t2 - t1) * 1000.0,
        "optimizer_ms": (t3 - t2) * 1000.0,
        "total_ms": (t3 - t0) * 1000.0,
    }


def collect_hgsel_utilization(model: TransformerModel) -> Dict[str, Any]:
    per_layer: List[Dict[str, Any]] = []

    for layer_idx, block in enumerate(model.layers):
        mlp = getattr(block, "mlp", None)
        if mlp is None or not hasattr(mlp, "expert_load_ema"):
            continue

        ema = mlp.expert_load_ema.detach().float().cpu()
        stats = {}
        if hasattr(mlp, "get_expert_load_stats"):
            stats = mlp.get_expert_load_stats()

        per_layer.append(
            {
                "layer_index": layer_idx,
                "n_experts": int(ema.numel()),
                "expert_load_histogram": [float(x) for x in ema.tolist()],
                "mean_load": float(ema.mean().item()) if ema.numel() else 0.0,
                "std_load": float(ema.std(unbiased=False).item()) if ema.numel() else 0.0,
                "min_load": float(ema.min().item()) if ema.numel() else 0.0,
                "max_load": float(ema.max().item()) if ema.numel() else 0.0,
                "entropy": float(stats.get("entropy", 0.0)),
            }
        )

    diagnostics = model.get_routing_diagnostics() if hasattr(model, "get_routing_diagnostics") else {}
    aggregate_expert_load = diagnostics.get("expert_load")
    if isinstance(aggregate_expert_load, torch.Tensor):
        aggregate_hist = [float(x) for x in aggregate_expert_load.detach().float().cpu().tolist()]
    else:
        aggregate_hist = []

    traces = model.get_phase4_routing_traces() if hasattr(model, "get_phase4_routing_traces") else []
    serializable_traces = []
    for trace in traces:
        trace_copy: Dict[str, Any] = {}
        for k, v in trace.items():
            if isinstance(v, torch.Tensor):
                if v.numel() <= 128:
                    trace_copy[k] = v.detach().cpu().tolist()
                else:
                    trace_copy[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
            else:
                trace_copy[k] = v
        serializable_traces.append(trace_copy)

    return {
        "aggregate_entropy": float(diagnostics.get("entropy", 0.0)) if diagnostics else 0.0,
        "aggregate_expert_load_histogram": aggregate_hist,
        "per_layer": per_layer,
        "last_forward_traces": serializable_traces,
    }


def benchmark_model(
    model_kind: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    model = create_model(model_kind, args, device, dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    memory_profiler = MemoryProfiler(model, optimizer)

    input_ids, labels = make_batch(args, device)
    step_fn = time_step_cuda if device.type == "cuda" else time_step_cpu

    # Warmup
    for _ in range(args.warmup_steps):
        _ = step_fn(model, optimizer, input_ids, labels)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cuda_sync(device)

    memory_profiler.reset()
    pre_snapshot = memory_profiler.take_snapshot("pre_bench")

    timings: List[Dict[str, float]] = []
    losses: List[float] = []

    t_wall_start = time.perf_counter()
    for _ in range(args.bench_steps):
        metrics = step_fn(model, optimizer, input_ids, labels)
        timings.append(metrics)
        losses.append(metrics["loss"])
    cuda_sync(device)
    t_wall_end = time.perf_counter()

    post_snapshot = memory_profiler.take_snapshot("post_bench")

    wall_sec = max(t_wall_end - t_wall_start, 1e-9)
    tokens_total = args.bench_steps * args.batch_size * args.seq_length
    tokens_per_sec = tokens_total / wall_sec

    forward_vals = [m["forward_ms"] for m in timings]
    backward_vals = [m["backward_ms"] for m in timings]
    optimizer_vals = [m["optimizer_ms"] for m in timings]
    total_vals = [m["total_ms"] for m in timings]

    if device.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    else:
        peak_allocated_mb = 0.0
        peak_reserved_mb = 0.0

    result: Dict[str, Any] = {
        "model_kind": model_kind,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "parameters": int(model.count_parameters() if hasattr(model, "count_parameters") else sum(p.numel() for p in model.parameters())),
        "benchmark_config": {
            "warmup_steps": args.warmup_steps,
            "bench_steps": args.bench_steps,
            "batch_size": args.batch_size,
            "seq_length": args.seq_length,
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "d_ff": args.d_ff,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
        },
        "throughput": {
            "tokens_total": int(tokens_total),
            "wall_time_sec": wall_sec,
            "tokens_per_sec": tokens_per_sec,
        },
        "timings_ms": {
            "forward_mean": float(sum(forward_vals) / max(len(forward_vals), 1)),
            "forward_p50": percentile(forward_vals, 0.50),
            "forward_p95": percentile(forward_vals, 0.95),
            "backward_mean": float(sum(backward_vals) / max(len(backward_vals), 1)),
            "backward_p50": percentile(backward_vals, 0.50),
            "backward_p95": percentile(backward_vals, 0.95),
            "optimizer_mean": float(sum(optimizer_vals) / max(len(optimizer_vals), 1)),
            "optimizer_p50": percentile(optimizer_vals, 0.50),
            "optimizer_p95": percentile(optimizer_vals, 0.95),
            "step_total_mean": float(sum(total_vals) / max(len(total_vals), 1)),
            "step_total_p50": percentile(total_vals, 0.50),
            "step_total_p95": percentile(total_vals, 0.95),
        },
        "memory_mb": {
            "peak_allocated": float(peak_allocated_mb),
            "peak_reserved": float(peak_reserved_mb),
            "pre_bench_snapshot": pre_snapshot.__dict__,
            "post_bench_snapshot": post_snapshot.__dict__,
            "estimate": estimate_model_memory_requirements(model),
        },
        "loss": {
            "start": float(losses[0]) if losses else None,
            "end": float(losses[-1]) if losses else None,
            "mean": float(sum(losses) / max(len(losses), 1)) if losses else None,
        },
    }

    if model_kind == "hgsel":
        result["expert_utilization"] = collect_hgsel_utilization(model)

    # Cleanup to avoid contaminating subsequent runs.
    del optimizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def print_summary(results: List[Dict[str, Any]]) -> None:
    print("Phase 4 GPU Baseline (Control Condition)")
    print("=" * 72)
    for result in results:
        timings = result["timings_ms"]
        throughput = result["throughput"]
        memory = result["memory_mb"]
        print(
            f"{result['model_kind']:>5} | "
            f"tokens/sec={throughput['tokens_per_sec']:.1f} | "
            f"fwd={timings['forward_mean']:.2f}ms | "
            f"bwd={timings['backward_mean']:.2f}ms | "
            f"step={timings['step_total_mean']:.2f}ms | "
            f"peak_mem={memory['peak_allocated']:.1f}MB"
        )
        expert = result.get("expert_utilization")
        if expert:
            print(
                f"      HGSEL routing entropy={expert.get('aggregate_entropy', 0.0):.4f} "
                f"(layers={len(expert.get('per_layer', []))})"
            )
    print("=" * 72)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if args.dtype != str(dtype).replace("torch.", ""):
        print(
            f"[baseline] dtype downgraded from {args.dtype} to {str(dtype).replace('torch.', '')} "
            f"for device={device}"
        )

    set_seed(args.seed)

    model_kinds = parse_csv(args.models)
    supported = {"dense", "hgsel"}
    invalid = [m for m in model_kinds if m not in supported]
    if invalid:
        raise ValueError(f"Unsupported --models entries: {invalid}. Supported: {sorted(supported)}")

    print(f"[baseline] device={device} dtype={dtype} models={model_kinds}")
    print(
        f"[baseline] batch={args.batch_size} seq={args.seq_length} "
        f"steps={args.bench_steps} warmup={args.warmup_steps}"
    )

    results: List[Dict[str, Any]] = []
    for model_kind in model_kinds:
        print(f"[baseline] benchmarking {model_kind} ...")
        results.append(benchmark_model(model_kind, args, device, dtype))

    print_summary(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "script": "experiments/train_gpu_baseline.py",
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "seed": args.seed,
        },
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[baseline] wrote results -> {output_path}")


if __name__ == "__main__":
    main()
