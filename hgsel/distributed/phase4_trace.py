"""Phase 4 tracing utilities for distributed systems measurements.

Provides:
- lightweight span timing helpers
- JSONL trace writer with shape reuse tracking
- small metric helpers (CV, shape signatures)
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import torch


def maybe_cuda_sync(device: Optional[torch.device] = None) -> None:
    """Synchronize CUDA device for accurate wall timing when needed."""
    if not torch.cuda.is_available():
        return

    if device is not None and getattr(device, "type", None) != "cuda":
        return

    torch.cuda.synchronize(device=device)


class SpanRecorder:
    """Accumulate named span timings in milliseconds."""

    def __init__(self, device: Optional[torch.device] = None, sync_cuda: bool = True) -> None:
        self.device = device
        self.sync_cuda = sync_cuda
        self.times_ms: Dict[str, float] = {}

    def _sync(self) -> None:
        if self.sync_cuda:
            maybe_cuda_sync(self.device)

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.times_ms[name] = self.times_ms.get(name, 0.0) + elapsed_ms

    def record(self, name: str, value_ms: float) -> None:
        self.times_ms[name] = self.times_ms.get(name, 0.0) + float(value_ms)

    def get(self, name: str, default: float = 0.0) -> float:
        return float(self.times_ms.get(name, default))

    def to_dict(self) -> Dict[str, float]:
        return dict(self.times_ms)


def coefficient_of_variation(values: Optional[Iterable[float]]) -> Optional[float]:
    """Return coefficient of variation (std / mean), or None when undefined."""
    if values is None:
        return None

    vals = [float(v) for v in values]
    if not vals:
        return None

    mean = sum(vals) / len(vals)
    if abs(mean) < 1e-12:
        return None

    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    return (variance ** 0.5) / mean


def per_rank_shape_signature(
    per_rank_counts: Dict[int, int],
    world_size: Optional[int] = None,
    prefix: str = "counts",
) -> str:
    """Build a stable shape signature from per-rank counts."""
    if world_size is None:
        max_rank = max(per_rank_counts.keys(), default=-1)
        world_size = max_rank + 1 if max_rank >= 0 else 0

    counts = [int(per_rank_counts.get(rank, 0)) for rank in range(world_size)]
    counts_csv = ",".join(str(c) for c in counts)
    return f"{prefix}:{world_size}:{counts_csv}"


class Phase4TraceWriter:
    """Write Phase 4 per-step traces as JSONL and track shape reuse."""

    def __init__(
        self,
        output_dir: str | Path,
        run_id: str,
        rank: int = 0,
        static_fields: Optional[Dict[str, Any]] = None,
        flush_every: int = 1,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.rank = rank
        self.static_fields = dict(static_fields or {})
        self.flush_every = max(1, int(flush_every))

        self.path = self.output_dir / f"{run_id}.rank{rank}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")
        self._records_written = 0
        self._shape_seen: Dict[str, int] = {}
        self._shape_total = 0
        self._shape_reused = 0

    @property
    def shape_reuse_rate(self) -> float:
        if self._shape_total == 0:
            return 0.0
        return self._shape_reused / self._shape_total

    def _json_default(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                return value.item()
            return value.detach().cpu().tolist()
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def _annotate_shape_reuse(self, record: Dict[str, Any]) -> None:
        routing = record.setdefault("routing", {})
        signature = routing.get("shape_signature")
        if not signature:
            routing.setdefault("shape_plan_reused", False)
            return

        seen_count = self._shape_seen.get(signature, 0)
        reused = seen_count > 0
        self._shape_seen[signature] = seen_count + 1
        self._shape_total += 1
        if reused:
            self._shape_reused += 1

        routing["shape_plan_reused"] = reused

        metrics = record.setdefault("metrics", {})
        metrics.setdefault("dispatch_shape_reuse_rate_running", self.shape_reuse_rate)

    def write_step(self, record: Dict[str, Any]) -> None:
        payload = dict(self.static_fields)
        payload.update(record)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("rank", self.rank)

        self._annotate_shape_reuse(payload)

        line = json.dumps(payload, default=self._json_default, sort_keys=False)
        self._fh.write(line + "\n")
        self._records_written += 1

        if self._records_written % self.flush_every == 0:
            self._fh.flush()

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()

    def __enter__(self) -> "Phase4TraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
