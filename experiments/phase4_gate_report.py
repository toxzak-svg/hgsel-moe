#!/usr/bin/env python
"""
Phase 4 gate report aggregator.

Combines outputs from:
- experiments/train_gpu_baseline.py
- experiments/train_distributed_300m.py (DDP parity run)
- experiments/benchmark_token_exchange_micro.py

and emits a single go/warn/stop report.

Example:
    python experiments/phase4_gate_report.py \
        --baseline-json results/gpu_baseline/train_gpu_baseline.json \
        --parity-json results/phase4/ddp_parity.json \
        --microbench-json results/token_exchange_micro/benchmark_token_exchange_micro.json \
        --output results/phase4/phase4_gate_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


STATUS_ORDER = {"go": 0, "warn": 1, "stop": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase 4 gate JSON outputs into a single report")
    parser.add_argument("--baseline-json", type=str, required=True, help="train_gpu_baseline.py JSON output")
    parser.add_argument("--parity-json", type=str, required=True, help="train_distributed_300m.py JSON output")
    parser.add_argument("--microbench-json", type=str, required=True, help="benchmark_token_exchange_micro.py JSON output")
    parser.add_argument(
        "--parity-reference-json",
        type=str,
        default="",
        help="Optional Phase 3 / single-GPU reference JSON for DDP parity comparison",
    )
    parser.add_argument(
        "--parity-reference-label",
        type=str,
        default="phase3_reference",
        help="Label used in the parity reference comparison section",
    )
    parser.add_argument(
        "--parity-reference-metric",
        type=str,
        default="final_val_loss",
        choices=["final_train_loss", "final_val_loss", "final_val_perplexity"],
        help="Metric to compare against the optional parity reference",
    )
    parser.add_argument(
        "--parity-relative-warn",
        type=float,
        default=0.10,
        help="Warn if selected parity metric degrades by more than this relative amount vs reference",
    )
    parser.add_argument(
        "--parity-relative-stop",
        type=float,
        default=0.25,
        help="Stop if selected parity metric degrades by more than this relative amount vs reference",
    )
    parser.add_argument(
        "--parity-reference-final-train-loss",
        type=float,
        default=None,
        help="Optional manual override for reference final_train_loss",
    )
    parser.add_argument(
        "--parity-reference-final-val-loss",
        type=float,
        default=None,
        help="Optional manual override for reference final_val_loss",
    )
    parser.add_argument(
        "--parity-reference-final-val-perplexity",
        type=float,
        default=None,
        help="Optional manual override for reference final_val_perplexity",
    )
    parser.add_argument(
        "--strict-phase4",
        action="store_true",
        help=(
            "Promote Phase 4 representativeness warnings to hard stops "
            "(e.g., baseline CUDA evidence, parity/microbench CUDA+NCCL+multi-rank requirements)"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/phase4/phase4_gate_report.json",
        help="Path to write aggregated gate report JSON",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")
    return payload


def is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def worst_status(*statuses: str) -> str:
    valid = [s for s in statuses if s in STATUS_ORDER]
    if not valid:
        return "stop"
    return max(valid, key=lambda s: STATUS_ORDER[s])


def make_gate(
    *,
    name: str,
    status: str,
    summary: str,
    checks: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "summary": summary,
        "checks": checks,
        "metrics": metrics or {},
        "notes": notes or [],
    }


def last_finite_in_list(value: Any) -> Optional[float]:
    if not isinstance(value, list) or not value:
        return None
    for item in reversed(value):
        if is_finite_number(item):
            return float(item)
    return None


def extract_final_training_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract final training metrics from common payload shapes.

    Supports:
    - {"results": {"train_loss": [...], "val_loss": [...], "val_perplexity": [...]}}
    - {"train_loss": [...], ...}
    - {"summary": {"final_val_loss": ...}} or top-level final_* keys
    - {"best_val_loss": ...} (fallback for final_val_loss only)
    """
    metric_keys = {
        "train_loss": "final_train_loss",
        "val_loss": "final_val_loss",
        "val_perplexity": "final_val_perplexity",
    }
    finals: Dict[str, Optional[float]] = {
        "final_train_loss": None,
        "final_val_loss": None,
        "final_val_perplexity": None,
    }
    sources: Dict[str, str] = {}

    candidates: List[tuple[str, Any]] = [
        ("results", payload.get("results")),
        ("top_level", payload),
        ("summary", payload.get("summary")),
    ]

    for candidate_name, candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        for list_key, final_key in metric_keys.items():
            if finals[final_key] is not None:
                continue

            explicit_final = candidate.get(final_key)
            if is_finite_number(explicit_final):
                finals[final_key] = float(explicit_final)
                sources[final_key] = f"{candidate_name}.{final_key}"
                continue

            last_val = last_finite_in_list(candidate.get(list_key))
            if last_val is not None:
                finals[final_key] = last_val
                sources[final_key] = f"{candidate_name}.{list_key}[-1]"

        # Backward-compatible fallback often found in experiment summaries.
        if finals["final_val_loss"] is None and is_finite_number(candidate.get("best_val_loss")):
            finals["final_val_loss"] = float(candidate["best_val_loss"])
            sources["final_val_loss"] = f"{candidate_name}.best_val_loss"

    return {
        "metrics": finals,
        "sources": sources,
    }


def build_parity_reference_config(
    args: argparse.Namespace,
    reference_payload: Optional[Dict[str, Any]],
    reference_json_path: Optional[Path],
) -> Optional[Dict[str, Any]]:
    manual_overrides = {
        "final_train_loss": args.parity_reference_final_train_loss,
        "final_val_loss": args.parity_reference_final_val_loss,
        "final_val_perplexity": args.parity_reference_final_val_perplexity,
    }
    any_manual = any(v is not None for v in manual_overrides.values())
    if reference_payload is None and not any_manual:
        return None

    extracted = extract_final_training_metrics(reference_payload or {})
    ref_metrics = dict(extracted["metrics"])
    ref_sources = dict(extracted["sources"])

    for key, value in manual_overrides.items():
        if value is None:
            continue
        ref_metrics[key] = float(value)
        ref_sources[key] = "cli_override"

    return {
        "label": args.parity_reference_label,
        "metric": args.parity_reference_metric,
        "relative_warn": float(args.parity_relative_warn),
        "relative_stop": float(args.parity_relative_stop),
        "metrics": ref_metrics,
        "sources": ref_sources,
        "reference_json": str(reference_json_path) if reference_json_path else None,
    }


def analyze_baseline(payload: Dict[str, Any], strict_phase4: bool = False) -> Dict[str, Any]:
    results = payload.get("results")
    metadata = payload.get("metadata", {})

    if not isinstance(results, list) or not results:
        return make_gate(
            name="baseline",
            status="stop",
            summary="Baseline JSON is missing a non-empty results list",
            checks={"results_present": False},
        )

    model_kinds: List[str] = []
    throughput_ok = True
    timing_ok = True
    memory_ok = True
    hgsel_util_ok = True
    tokens_per_sec: Dict[str, float] = {}
    devices_seen: List[str] = []
    dtypes_seen: List[str] = []

    for result in results:
        if not isinstance(result, dict):
            continue
        kind = str(result.get("model_kind", "unknown"))
        model_kinds.append(kind)
        devices_seen.append(str(result.get("device", "unknown")))
        dtypes_seen.append(str(result.get("dtype", "unknown")))

        throughput = result.get("throughput", {})
        timings = result.get("timings_ms", {})
        memory = result.get("memory_mb", {})

        tps = throughput.get("tokens_per_sec")
        step_ms = timings.get("step_total_mean")

        if not is_finite_number(tps) or float(tps) <= 0:
            throughput_ok = False
        else:
            tokens_per_sec[kind] = float(tps)

        if not is_finite_number(step_ms) or float(step_ms) <= 0:
            timing_ok = False

        if not isinstance(memory, dict) or "estimate" not in memory:
            memory_ok = False

        if kind == "hgsel":
            expert_util = result.get("expert_utilization")
            if not isinstance(expert_util, dict):
                hgsel_util_ok = False
            else:
                entropy = expert_util.get("aggregate_entropy")
                per_layer = expert_util.get("per_layer")
                if not is_finite_number(entropy):
                    hgsel_util_ok = False
                if not isinstance(per_layer, list):
                    hgsel_util_ok = False

    unique_model_kinds = sorted(set(model_kinds))
    model_set = set(unique_model_kinds)

    has_dense = "dense" in model_set
    has_hgsel = "hgsel" in model_set
    all_cuda = all(device.startswith("cuda") for device in devices_seen) if devices_seen else False

    status = "go"
    notes: List[str] = []
    strict_rep_failure = False

    if not throughput_ok or not timing_ok or not memory_ok:
        status = "stop"
        notes.append("Baseline metrics are incomplete or invalid")

    if has_hgsel and not hgsel_util_ok:
        status = "stop"
        notes.append("HGSEL expert utilization diagnostics missing or invalid")

    if not has_dense or not has_hgsel:
        status = worst_status(status, "warn")
        notes.append("Baseline is missing dense or hgsel result; control comparison is incomplete")

    if not all_cuda:
        strict_rep_failure = strict_rep_failure or strict_phase4
        if strict_phase4:
            status = worst_status(status, "stop")
            notes.append("Baseline device is not CUDA (strict Phase 4 requires a CUDA baseline run)")
        else:
            status = worst_status(status, "warn")
            notes.append("Baseline device is not CUDA; useful for smoke checks but not Phase 4 scaling evidence")

    bench_steps = None
    try:
        # Inspect the first result only; baseline script keeps configs aligned across model kinds.
        bench_steps = int(results[0].get("benchmark_config", {}).get("bench_steps"))
        if bench_steps < 5:
            strict_rep_failure = strict_rep_failure or strict_phase4
            if strict_phase4:
                status = worst_status(status, "stop")
                notes.append("Baseline used a very small bench_steps value (strict Phase 4 rejects smoke runs)")
            else:
                status = worst_status(status, "warn")
                notes.append("Baseline used a very small bench_steps value (likely smoke run)")
    except Exception:
        pass

    if status == "go":
        summary = "GPU baseline instrumentation looks complete and representative"
    elif status == "warn":
        summary = "Baseline JSON is valid, but it looks like a smoke/non-representative run"
    else:
        if strict_rep_failure and throughput_ok and timing_ok and memory_ok and (not has_hgsel or hgsel_util_ok):
            summary = "Baseline run is valid, but it failed strict Phase 4 representativeness requirements"
        else:
            summary = "Baseline gate failed due to missing/invalid instrumentation fields"

    return make_gate(
        name="baseline",
        status=status,
        summary=summary,
        checks={
            "results_present": True,
            "has_dense": has_dense,
            "has_hgsel": has_hgsel,
            "throughput_metrics_valid": throughput_ok,
            "timing_metrics_valid": timing_ok,
            "memory_metrics_present": memory_ok,
            "hgsel_expert_utilization_present": (has_hgsel and hgsel_util_ok),
            "all_cuda": all_cuda,
            "strict_phase4": strict_phase4,
            "strict_representative_baseline": all_cuda and (bench_steps is not None and bench_steps >= 5),
        },
        metrics={
            "models_found": unique_model_kinds,
            "devices_seen": sorted(set(devices_seen)),
            "dtypes_seen": sorted(set(dtypes_seen)),
            "tokens_per_sec_by_model": tokens_per_sec,
            "bench_steps": bench_steps,
            "metadata_device": metadata.get("device"),
            "metadata_dtype": metadata.get("dtype"),
        },
        notes=notes,
    )


def analyze_parity(
    payload: Dict[str, Any],
    parity_reference: Optional[Dict[str, Any]] = None,
    strict_phase4: bool = False,
) -> Dict[str, Any]:
    metadata = payload.get("metadata", {})
    results = payload.get("results", {})

    if not isinstance(metadata, dict) or not isinstance(results, dict):
        return make_gate(
            name="ddp_parity",
            status="stop",
            summary="Parity JSON is missing metadata/results objects",
            checks={"metadata_present": isinstance(metadata, dict), "results_present": isinstance(results, dict)},
        )

    train_loss = results.get("train_loss")
    val_loss = results.get("val_loss")
    val_ppl = results.get("val_perplexity")

    train_loss_valid = isinstance(train_loss, list) and len(train_loss) > 0 and all(is_finite_number(x) for x in train_loss)
    val_loss_valid = isinstance(val_loss, list) and len(val_loss) > 0 and all(is_finite_number(x) for x in val_loss)
    val_ppl_valid = isinstance(val_ppl, list) and len(val_ppl) > 0 and all(is_finite_number(x) for x in val_ppl)

    world_size = int(metadata.get("world_size", 1) or 1)
    global_batch = metadata.get("global_batch_size")
    per_rank_batch = metadata.get("per_rank_batch_size")
    backend = metadata.get("backend")
    device = str(metadata.get("device", "unknown"))

    batch_consistent = False
    if is_finite_number(global_batch) and is_finite_number(per_rank_batch):
        batch_consistent = int(global_batch) == int(per_rank_batch) * world_size

    current_metric_extract = extract_final_training_metrics(payload)
    current_finals = current_metric_extract["metrics"]

    status = "go"
    notes: List[str] = []
    strict_exec_failure = False

    if not train_loss_valid:
        status = "stop"
        notes.append("Missing or invalid train_loss trace")

    if not batch_consistent:
        status = worst_status(status, "stop")
        notes.append("global_batch_size != per_rank_batch_size * world_size")

    if not val_loss_valid:
        status = worst_status(status, "warn")
        notes.append("val_loss trace missing or invalid; convergence parity evidence is weak")

    if world_size <= 1:
        strict_exec_failure = strict_exec_failure or strict_phase4
        if strict_phase4:
            status = worst_status(status, "stop")
            notes.append("world_size <= 1 (strict Phase 4 requires multi-rank DDP parity)")
        else:
            status = worst_status(status, "warn")
            notes.append("world_size <= 1 (single-rank smoke); multi-GPU parity not demonstrated")

    if world_size > 1 and backend != "nccl":
        strict_exec_failure = strict_exec_failure or strict_phase4
        if strict_phase4:
            status = worst_status(status, "stop")
            notes.append("Multi-rank parity was not run with NCCL (strict Phase 4 requires NCCL)")
        else:
            status = worst_status(status, "warn")
            notes.append("Multi-rank parity was not run with NCCL")

    if "cuda" not in device:
        strict_exec_failure = strict_exec_failure or strict_phase4
        if strict_phase4:
            status = worst_status(status, "stop")
            notes.append("Parity run device is not CUDA (strict Phase 4 requires CUDA)")
        else:
            status = worst_status(status, "warn")
            notes.append("Parity run device is not CUDA")

    reference_requested = parity_reference is not None
    reference_metric_available = None
    reference_within_warn = None
    reference_within_stop = None
    parity_reference_comparison: Optional[Dict[str, Any]] = None

    if parity_reference is not None:
        selected_metric = str(parity_reference.get("metric", "final_val_loss"))
        ref_metrics = parity_reference.get("metrics", {})
        ref_sources = parity_reference.get("sources", {})
        ref_value = ref_metrics.get(selected_metric) if isinstance(ref_metrics, dict) else None
        cur_value = current_finals.get(selected_metric)

        if is_finite_number(cur_value) and is_finite_number(ref_value):
            cur_value_f = float(cur_value)
            ref_value_f = float(ref_value)
            delta = cur_value_f - ref_value_f
            rel_delta = delta / max(abs(ref_value_f), 1e-12)
            # Loss/perplexity metrics are "lower is better"; only positive deltas are degradations.
            degraded_rel = max(rel_delta, 0.0)

            warn_threshold = float(parity_reference.get("relative_warn", 0.10))
            stop_threshold = float(parity_reference.get("relative_stop", 0.25))

            reference_metric_available = True
            reference_within_warn = degraded_rel <= warn_threshold
            reference_within_stop = degraded_rel <= stop_threshold

            verdict = "PASS"
            if degraded_rel > stop_threshold:
                verdict = "STOP"
                status = worst_status(status, "stop")
                notes.append(
                    f"Parity regression vs {parity_reference.get('label', 'reference')}: "
                    f"{selected_metric} degraded by {degraded_rel:.1%} (> {stop_threshold:.1%} stop threshold)"
                )
            elif degraded_rel > warn_threshold:
                verdict = "WARN"
                status = worst_status(status, "warn")
                notes.append(
                    f"Parity regression vs {parity_reference.get('label', 'reference')}: "
                    f"{selected_metric} degraded by {degraded_rel:.1%} (> {warn_threshold:.1%} warn threshold)"
                )

            parity_reference_comparison = {
                "label": parity_reference.get("label"),
                "metric": selected_metric,
                "current_value": cur_value_f,
                "reference_value": ref_value_f,
                "current_metric_source": current_metric_extract.get("sources", {}).get(selected_metric),
                "reference_metric_source": ref_sources.get(selected_metric) if isinstance(ref_sources, dict) else None,
                "delta": delta,
                "relative_delta": rel_delta,
                "degraded_relative_delta": degraded_rel,
                "relative_warn_threshold": warn_threshold,
                "relative_stop_threshold": stop_threshold,
                "verdict": verdict,
            }
        else:
            reference_metric_available = False
            status = worst_status(status, "warn")
            notes.append(
                "Parity reference comparison requested but selected metric is missing in current or reference data"
            )
            parity_reference_comparison = {
                "label": parity_reference.get("label"),
                "metric": parity_reference.get("metric"),
                "current_value": cur_value,
                "reference_value": ref_value,
                "current_metric_source": current_metric_extract.get("sources", {}).get(selected_metric),
                "reference_metric_source": ref_sources.get(selected_metric) if isinstance(ref_sources, dict) else None,
                "relative_warn_threshold": parity_reference.get("relative_warn"),
                "relative_stop_threshold": parity_reference.get("relative_stop"),
                "verdict": "MISSING_METRIC",
            }

    reference_verdict = None
    if isinstance(parity_reference_comparison, dict):
        reference_verdict = parity_reference_comparison.get("verdict")

    if status == "go":
        summary = "DDP parity run looks valid for multi-GPU plumbing/convergence checks"
    elif status == "warn":
        if reference_verdict == "WARN":
            summary = "Parity run is healthy, but it exceeded the configured reference warn threshold"
        else:
            summary = "Parity JSON is valid, but the run appears to be smoke/non-representative"
    else:
        if not train_loss_valid or not batch_consistent:
            summary = "Parity gate failed due to missing/invalid training metrics or batch metadata"
        elif reference_verdict == "STOP":
            summary = "Parity run is healthy, but it failed the configured reference stop threshold"
        elif strict_exec_failure:
            summary = "Parity run is healthy, but it failed strict Phase 4 execution requirements (CUDA + NCCL + multi-rank)"
        else:
            summary = "Parity gate failed"

    strict_execution_requirements_met = (world_size > 1) and (backend == "nccl") and ("cuda" in device)

    return make_gate(
        name="ddp_parity",
        status=status,
        summary=summary,
        checks={
            "metadata_present": True,
            "results_present": True,
            "train_loss_valid": train_loss_valid,
            "val_loss_valid": val_loss_valid,
            "val_perplexity_valid": val_ppl_valid,
            "batch_size_consistent": batch_consistent,
            "multi_rank": world_size > 1,
            "backend_nccl": backend == "nccl",
            "cuda_device": "cuda" in device,
            "strict_phase4": strict_phase4,
            "strict_execution_requirements_met": strict_execution_requirements_met,
            "reference_comparison_requested": reference_requested,
            "reference_metric_available": reference_metric_available,
            "reference_metric_within_warn": reference_within_warn,
            "reference_metric_within_stop": reference_within_stop,
        },
        metrics={
            "world_size": world_size,
            "backend": backend,
            "device": device,
            "use_hgsel": bool(metadata.get("use_hgsel")),
            "global_batch_size": global_batch,
            "per_rank_batch_size": per_rank_batch,
            "strict_phase4": strict_phase4,
            "final_train_loss": current_finals.get("final_train_loss"),
            "final_val_loss": current_finals.get("final_val_loss"),
            "final_val_perplexity": current_finals.get("final_val_perplexity"),
            "final_metric_sources": current_metric_extract.get("sources", {}),
            "parity_reference_comparison": parity_reference_comparison,
        },
        notes=notes,
    )


def micro_decision_to_status(decision: str) -> str:
    if decision == "PASS":
        return "go"
    if decision == "WARN_OPTIMIZE":
        return "warn"
    if decision == "STOP_REDESIGN":
        return "stop"
    return "warn"


def analyze_microbench(payload: Dict[str, Any], strict_phase4: bool = False) -> Dict[str, Any]:
    metadata = payload.get("metadata", {})
    results = payload.get("results")

    if not isinstance(results, list) or not results:
        return make_gate(
            name="microbenchmark",
            status="stop",
            summary="Microbenchmark JSON is missing a non-empty results list",
            checks={"results_present": False},
        )

    decision_counts = {"PASS": 0, "WARN_OPTIMIZE": 0, "STOP_REDESIGN": 0, "UNKNOWN": 0}
    worst_decision_status = "go"
    worst_comm_share_p50 = -1.0
    worst_comm_share_p95 = -1.0
    worst_config: Dict[str, Any] = {}

    valid_entries = 0
    for entry in results:
        if not isinstance(entry, dict):
            continue
        agg = entry.get("aggregate", {})
        cfg = entry.get("config", {})
        if not isinstance(agg, dict) or not isinstance(cfg, dict):
            continue
        valid_entries += 1

        decision = str(agg.get("decision", "UNKNOWN"))
        if decision in decision_counts:
            decision_counts[decision] += 1
        else:
            decision_counts["UNKNOWN"] += 1

        worst_decision_status = worst_status(worst_decision_status, micro_decision_to_status(decision))

        p50 = agg.get("comm_share_p50_worst_rank")
        p95 = agg.get("comm_share_p95_worst_rank")
        if is_finite_number(p50) and float(p50) > worst_comm_share_p50:
            worst_comm_share_p50 = float(p50)
            worst_config = {
                "tokens_per_rank": cfg.get("tokens_per_rank"),
                "hidden_dim": cfg.get("hidden_dim"),
                "routing_mode": cfg.get("routing_mode"),
                "world_size": cfg.get("world_size"),
                "device": cfg.get("device"),
                "dtype": cfg.get("dtype"),
            }
        if is_finite_number(p95) and float(p95) > worst_comm_share_p95:
            worst_comm_share_p95 = float(p95)

    if valid_entries == 0:
        return make_gate(
            name="microbenchmark",
            status="stop",
            summary="Microbenchmark results list contains no valid aggregate entries",
            checks={"results_present": True, "valid_entries": 0},
        )

    world_size = int(metadata.get("world_size", 1) or 1) if isinstance(metadata, dict) else 1
    backend = metadata.get("backend") if isinstance(metadata, dict) else None
    device = str(metadata.get("device", "unknown")) if isinstance(metadata, dict) else "unknown"
    threshold_warn = metadata.get("threshold_warn") if isinstance(metadata, dict) else None
    threshold_stop = metadata.get("threshold_stop") if isinstance(metadata, dict) else None

    status = worst_decision_status
    notes: List[str] = []
    strict_exec_failure = False

    if world_size <= 1:
        strict_exec_failure = strict_exec_failure or strict_phase4
        if strict_phase4:
            status = worst_status(status, "stop")
            notes.append("Microbenchmark ran with world_size <= 1 (strict Phase 4 requires multi-rank)")
        else:
            status = worst_status(status, "warn")
            notes.append("Microbenchmark ran with world_size <= 1; no distributed token exchange measured")

    if world_size > 1 and backend != "nccl":
        strict_exec_failure = strict_exec_failure or strict_phase4
        if strict_phase4:
            status = worst_status(status, "stop")
            notes.append("Microbenchmark was multi-rank but not NCCL (strict Phase 4 requires NCCL)")
        else:
            status = worst_status(status, "warn")
            notes.append("Microbenchmark was multi-rank but not NCCL")

    if "cuda" not in device:
        strict_exec_failure = strict_exec_failure or strict_phase4
        if strict_phase4:
            status = worst_status(status, "stop")
            notes.append("Microbenchmark device is not CUDA (strict Phase 4 requires CUDA)")
        else:
            status = worst_status(status, "warn")
            notes.append("Microbenchmark device is not CUDA")

    if decision_counts["UNKNOWN"] > 0:
        status = worst_status(status, "warn")
        notes.append("Some microbenchmark entries used unknown decision values")

    if status == "go":
        summary = "Microbenchmark gate passed under configured communication-share thresholds"
    elif status == "warn":
        summary = "Microbenchmark is valid but indicates optimization needed or non-representative hardware/run mode"
    else:
        if decision_counts["STOP_REDESIGN"] > 0:
            summary = "Microbenchmark indicates communication overhead is too high (redesign gate)"
        elif strict_exec_failure:
            summary = "Microbenchmark is valid, but it failed strict Phase 4 execution requirements (CUDA + NCCL + multi-rank)"
        else:
            summary = "Microbenchmark gate failed"

    strict_execution_requirements_met = (world_size > 1) and (backend == "nccl") and ("cuda" in device)

    return make_gate(
        name="microbenchmark",
        status=status,
        summary=summary,
        checks={
            "results_present": True,
            "valid_entries": valid_entries,
            "multi_rank": world_size > 1,
            "backend_nccl": backend == "nccl",
            "cuda_device": "cuda" in device,
            "strict_phase4": strict_phase4,
            "strict_execution_requirements_met": strict_execution_requirements_met,
            "no_stop_decisions": decision_counts["STOP_REDESIGN"] == 0,
        },
        metrics={
            "world_size": world_size,
            "backend": backend,
            "device": device,
            "strict_phase4": strict_phase4,
            "threshold_warn": threshold_warn,
            "threshold_stop": threshold_stop,
            "decision_counts": decision_counts,
            "worst_comm_share_p50_worst_rank": (worst_comm_share_p50 if worst_comm_share_p50 >= 0 else None),
            "worst_comm_share_p95_worst_rank": (worst_comm_share_p95 if worst_comm_share_p95 >= 0 else None),
            "worst_config_by_comm_share_p50": worst_config,
        },
        notes=notes,
    )


def summarize_overall(gates: List[Dict[str, Any]]) -> Dict[str, Any]:
    statuses = [str(g.get("status", "stop")) for g in gates]
    overall_status = worst_status(*statuses)
    blocking = [g["name"] for g in gates if g.get("status") == "stop"]
    warnings = [g["name"] for g in gates if g.get("status") == "warn"]

    if overall_status == "go":
        summary = "Phase 4 gates look green based on the provided JSON outputs"
    elif overall_status == "warn":
        summary = "Phase 4 gates are partially satisfied; at least one input looks smoke/non-representative or needs optimization"
    else:
        summary = "Phase 4 gates are blocked by at least one failed gate"

    return {
        "status": overall_status,
        "summary": summary,
        "blocking_gates": blocking,
        "warning_gates": warnings,
    }


def print_report(report: Dict[str, Any]) -> None:
    overall = report["overall"]
    print("Phase 4 Gate Report")
    print("=" * 72)
    if report.get("metadata", {}).get("strict_phase4"):
        print("preset: STRICT_PHASE4")
    print(f"overall: {overall['status'].upper()}  |  {overall['summary']}")
    if overall.get("blocking_gates"):
        print(f"blocking_gates: {overall['blocking_gates']}")
    if overall.get("warning_gates"):
        print(f"warning_gates: {overall['warning_gates']}")
    print("-" * 72)

    for gate in report["gates"]:
        print(f"{gate['name']}: {str(gate['status']).upper()}  |  {gate['summary']}")
        metrics = gate.get("metrics", {})

        if gate["name"] == "baseline":
            tps = metrics.get("tokens_per_sec_by_model", {})
            if tps:
                print(f"  tokens/sec: {tps}")
            print(f"  devices: {metrics.get('devices_seen')}  bench_steps: {metrics.get('bench_steps')}")

        elif gate["name"] == "ddp_parity":
            print(
                "  "
                f"world_size={metrics.get('world_size')} backend={metrics.get('backend')} "
                f"device={metrics.get('device')} final_train_loss={metrics.get('final_train_loss')} "
                f"final_val_loss={metrics.get('final_val_loss')}"
            )
            ref_cmp = metrics.get("parity_reference_comparison")
            if isinstance(ref_cmp, dict):
                print(
                    "  "
                    f"reference[{ref_cmp.get('label')}]: metric={ref_cmp.get('metric')} "
                    f"current={ref_cmp.get('current_value')} ref={ref_cmp.get('reference_value')} "
                    f"verdict={ref_cmp.get('verdict')} rel_delta={ref_cmp.get('relative_delta')}"
                )

        elif gate["name"] == "microbenchmark":
            print(
                "  "
                f"world_size={metrics.get('world_size')} backend={metrics.get('backend')} "
                f"device={metrics.get('device')} worst_comm_share_p50={metrics.get('worst_comm_share_p50_worst_rank')} "
                f"decision_counts={metrics.get('decision_counts')}"
            )

        notes = gate.get("notes") or []
        for note in notes:
            print(f"  note: {note}")

    print("=" * 72)


def main() -> int:
    args = parse_args()

    baseline_path = Path(args.baseline_json)
    parity_path = Path(args.parity_json)
    micro_path = Path(args.microbench_json)
    parity_reference_path = Path(args.parity_reference_json) if args.parity_reference_json else None

    if args.parity_relative_warn < 0 or args.parity_relative_stop < 0:
        print("Error: parity reference thresholds must be >= 0")
        return 1
    if args.parity_relative_stop < args.parity_relative_warn:
        print("Error: --parity-relative-stop must be >= --parity-relative-warn")
        return 1

    for path in (baseline_path, parity_path, micro_path):
        if not path.exists():
            print(f"Error: file not found: {path}")
            return 1
    if parity_reference_path and not parity_reference_path.exists():
        print(f"Error: file not found: {parity_reference_path}")
        return 1

    try:
        baseline_payload = load_json(baseline_path)
        parity_payload = load_json(parity_path)
        micro_payload = load_json(micro_path)
        parity_reference_payload = load_json(parity_reference_path) if parity_reference_path else None
    except Exception as exc:
        print(f"Error loading JSON inputs: {exc}")
        return 1

    parity_reference = build_parity_reference_config(args, parity_reference_payload, parity_reference_path)

    gates = [
        analyze_baseline(baseline_payload, strict_phase4=bool(args.strict_phase4)),
        analyze_parity(
            parity_payload,
            parity_reference=parity_reference,
            strict_phase4=bool(args.strict_phase4),
        ),
        analyze_microbench(micro_payload, strict_phase4=bool(args.strict_phase4)),
    ]
    overall = summarize_overall(gates)

    report = {
        "metadata": {
            "script": "experiments/phase4_gate_report.py",
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "strict_phase4": bool(args.strict_phase4),
        },
        "inputs": {
            "baseline_json": str(baseline_path),
            "parity_json": str(parity_path),
            "microbench_json": str(micro_path),
            "parity_reference_json": str(parity_reference_path) if parity_reference_path else None,
        },
        "parity_reference": parity_reference,
        "overall": overall,
        "gates": gates,
    }

    print_report(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[phase4_gate_report] wrote results -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
