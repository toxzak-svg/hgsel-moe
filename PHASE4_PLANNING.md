# Phase 4 Planning: Multi-GPU Distribution

**Current Status:** Phase 3 ✓ Complete  
**Target Duration:** Days 8–10  
**Goal:** Prove multi-GPU HGSEL scaling with a systems-first workflow: instrumented GPU baseline, DDP parity, expert-parallel token exchange microbenchmarks, then full expert sharding integration.

---

## Overview

Phase 4 shifts from single-GPU validation to **multi-GPU scaling**, but the priority is not "feature completeness." The priority is isolating the bottleneck that will decide whether HGSEL scales in practice: **expert-parallel token exchange bandwidth + synchronization**.

This phase should be executed as a sequence of controlled experiments, not as one large distributed trainer integration.

### What Exists
- ✓ Token dispatcher ([token_dispatcher.py](../hgsel/distributed/token_dispatcher.py)): Routes tokens to experts
- ✓ Expert sharding ([expert_sharding.py](../hgsel/distributed/expert_sharding.py)): Distributes experts across ranks
- ✓ Token exchange ([token_exchange.py](../hgsel/distributed/token_exchange.py)): Distributed token exchange primitives (to validate/benchmark for all-to-all)
- ✓ Dispatch API ([dispatch_api.py](../hgsel/distributed/dispatch_api.py)): Public routing interface
- ✓ Dispatch pipeline ([dispatch_pipeline.py](../hgsel/distributed/dispatch_pipeline.py)): Orchestration layer

### What Phase 4 Adds
- Instrumented GPU baseline (control condition)
- DDP-only distributed training parity (no expert sharding yet)
- Expert-parallel communication microbenchmark harness (all-to-all vs local expert compute)
- Expert sharding + token exchange integration after communication viability is proven
- Multi-GPU scaling validation, checkpoint restore testing, and communication budget reporting

---

## Execution Principle (Phase 4)

**Do not implement everything at once.** Phase 4 should isolate failure modes in this order:

1. GPU baseline with full instrumentation
2. DDP-only (data parallel) convergence parity
3. Expert-parallel communication microbenchmark
4. Expert sharding + all-to-all integration
5. Full multi-GPU training + scaling plot

If the communication microbenchmark shows expert token exchange already dominates forward time, redesign the dispatch path before investing in trainer integration.

---

## Detailed Phase 4 Roadmap

### 4.1 GPU Baseline (Control Condition)

**Objective:** Establish a trustworthy single-GPU control with instrumentation detailed enough to measure distributed scaling efficiency later.

**Implementation:**
```python
# experiments/train_gpu_baseline.py
"""
Run Phase 3 validation on GPU to measure throughput.
Compare:
- Dense baseline (GPU)
- HGSEL (GPU)
- HGSEL vs Phase 3 CPU (speedup)
"""

import torch
from hgsel.layer import HGSELLayer
from hgsel.training.hgsel_trainer import HGSELTrainer
from hgsel.training.trainer import TrainingConfig
from experiments.baselines.dense_transformer import TransformerModel

# Create config for GPU
config = TrainingConfig(
    batch_size=32,
    val_batch_size=64,
    num_epochs=5,
    device="cuda",
)

# Train HGSEL + Dense on GPU
# Measure control metrics used for distributed comparisons
```

**Metrics to Track:**
- Throughput (tokens/sec)
- Forward pass time (ms/step)
- Backward pass time (ms/step)
- GPU peak memory / reserved memory
- Training time (wall-clock)
- Convergence speed (epochs to target loss)
- Expert utilization histogram (per layer, per expert token counts)
- Expert entropy / load-balance summary

**Non-Negotiable Requirement:** These metrics must be logged before any multi-GPU scaling claims are made.

**Deliverable:** Baseline benchmark script + results report (used as Phase 4 control condition)

### 4.2 Distributed Data Loading

**Objective:** Ensure data consistency across GPUs.

**Implementation (Use `DistributedSampler`, avoid manual dataset slicing):**
```python
# hgsel/training/dist_data.py
"""
Distributed data loading with proper sampling.
"""

def create_distributed_loaders(
    data_dir: str,
    batch_size: int,
    rank: int,
    world_size: int,
) -> tuple:
    """Create train/val loaders for distributed training."""
    # Create identical datasets on each rank
    train_dataset, val_dataset = load_datasets(data_dir)
    
    # DistributedSampler handles partitioning + epoch shuffling
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )
    
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler
    )
    
    return train_loader, val_loader

def set_epoch(train_loader: DataLoader, epoch: int) -> None:
    """Must be called once per epoch for deterministic cross-rank shuffling."""
    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)
```

**Validation Checklist (must verify, not assume):**
- `global_batch_size == per_rank_batch_size * world_size`
- `DistributedSampler.set_epoch(epoch)` is called every epoch
- Train shuffle is deterministic per epoch across ranks (seeded)
- Validation sampler uses `shuffle=False`

**Deliverable:** Distributed loader helpers + sampler correctness test

### 4.3 DDP-Only Trainer Path (No Expert Sharding Yet)

**Objective:** Confirm multi-GPU training plumbing and convergence parity before introducing expert-parallel communication.

**Implementation:**
```python
# hgsel/training/dist_hgsel_trainer.py
"""
Distributed HGSEL trainer with DDP gradient sync first.
"""

class DistributedHGSELTrainer(HGSELTrainer):
    """
    Extends HGSELTrainer for multi-GPU training.
    
    Phase 4 order:
    - Start with DistributedDataParallel for gradient synchronization only
    - Keep expert dispatch local / disabled for parity validation
    - Add expert sharding + all-to-all only after microbenchmark gate passes
    - Synchronize salt optimization after expert sharding is enabled
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        rank: int,
        world_size: int,
        backend: str = "nccl",
        aux_loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__(model, config, aux_loss_fn)
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        
        # Initialize distributed backend
        if world_size > 1:
            dist.init_process_group(backend=backend)
        
        # Wrap model with DistributedDataParallel
        self.model = DistributedDataParallel(
            model,
            device_ids=[rank],
        )
        
        # Phase 4 step 1: DDP-only path (no expert sharding here)
    
    def train_step(self, batch):
        """DDP-only training step (Phase 4 parity gate)."""
        # ... same model math as single GPU
        # DDP handles gradient all-reduce
        pass
    
    def optimize_salt(self):
        """Synchronized salt optimization across all ranks."""
        if self.rank == 0:
            # Rank 0 computes best salt
            salt = super().optimize_salt()
        else:
            salt = None
        
        # Broadcast salt to all ranks
        if self.world_size > 1:
            salt_list = [salt]
            dist.broadcast_object_list(salt_list, src=0)
            salt = salt_list[0]
        
        # Apply to all models
        for name, module in self.model.named_modules():
            if isinstance(module, HGSELLayer):
                module.set_salt(salt)
        
        return salt
```

**Phase Gate:** DDP-only training must match Phase 3 convergence behavior (within expected noise) before expert sharding is added.

**Deliverable:** DDP-capable trainer path + convergence parity report

### 4.4 Expert-Parallel Communication Microbenchmark (Critical Gate)

**Objective:** Measure whether all-to-all token exchange is cheap enough relative to local expert compute to justify full integration.

**Why this comes before full expert sharding integration:** The main scaling risk is expert token exchange, not DDP gradient synchronization. If all-to-all is already too expensive in isolation, the full trainer will not fix that.

**Implementation:**
```python
# experiments/benchmark_token_exchange_micro.py
"""
Synthetic expert-parallel communication microbenchmark.

Measures:
- all_to_all time (token payload exchange)
- local expert compute time
- communication share of forward (%)
- scaling with tokens, hidden_dim, world_size, routing skew
"""

def run_microbenchmark():
    # 1. Generate synthetic token batches (B, T, H)
    # 2. Simulate deterministic HGSEL routing distribution (balanced + skewed)
    # 3. Pack tokens by owning rank (expert shard ownership)
    # 4. Measure torch.distributed.all_to_all_single() exchange time
    # 5. Measure local expert compute time on same payload sizes
    # 6. Report comm/compute ratio and p50/p95 latency
    pass
```

**Required Sweeps:**
- `world_size`: 1, 2, 4 (and 8 if available)
- `tokens_per_rank`: small/medium/large batch regimes
- `hidden_dim`: expected production values
- `routing_distribution`: balanced, moderate skew, worst-case skew

**Decision Thresholds (engineering gates):**
- Target: communication overhead `< 20%` of forward in representative batch regimes
- Warning: `20-40%` means likely optimization/refactor needed before scaling claims
- Stop and redesign: `> 40%` in representative regimes

**Deliverable:** Microbenchmark harness + results table/plots + go/no-go decision for expert-parallel integration

### 4.5 Expert Sharding + All-to-All Integration

**Objective:** Introduce expert sharding and token exchange only after communication viability is established.

**Implementation:**
```python
# hgsel/training/dist_hgsel_trainer.py (expert-parallel path)
"""
Add expert sharding + all-to-all token exchange to DDP trainer.
"""

# 1) Deterministically assign experts to owning ranks
# 2) Precompute token -> expert -> owner rank for current batch
# 3) Pack/send token payloads via all-to-all
# 4) Execute local experts
# 5) Return/gather outputs and restore token order
# 6) Log per-rank and per-expert utilization stats
```

**Implementation Notes:**
- Prefer `torch.distributed` + DDP for gradient synchronization (do not reinvent all-reduce)
- Use **all-to-all** for expert-parallel token exchange (not all-gather)
- Log communication time separately from local expert compute time
- Log per-rank expert token counts to validate global routing/load assumptions

**Deliverable:** Expert-sharded distributed trainer path with all-to-all instrumentation

### 4.6 Distributed Training Script & Scaling Benchmarks

**Objective:** Produce credible scaling evidence after expert-parallel integration.

**Implementation:**
```python
# experiments/train_distributed_300m.py
"""
Multi-GPU training launcher using torchrun.
Supports:
- DDP-only mode (parity validation)
- Expert-parallel mode (after microbenchmark gate)
"""

# experiments/benchmark_distributed_300m.py
"""
Benchmark distributed HGSEL training.
Measures:
- Throughput (tokens/sec)
- Forward/backward split
- All-to-all latency
- Expert dispatch overhead
- Load balance across ranks and experts
- Peak memory
- Scaling efficiency
"""

def benchmark_distributed():
    # Profile all-to-all time as function of expert batch size
    # Measure expert dispatch routing time
    # Compare single-GPU vs multi-GPU speedup
    # Track load imbalance (max vs min load per GPU)
    
    results = {
        "throughput": tokens_per_sec,
        "forward_time_ms": ms,
        "backward_time_ms": ms,
        "all_to_all_time_ms": ms,
        "dispatch_overhead": percent,
        "load_imbalance": percent,
        "scaling_efficiency": percent,  # Speedup / num_gpus
    }
```

**Metrics:**
- Throughput: tokens/sec
- Forward time: ms/step
- Backward time: ms/step
- All-to-all latency: ms per call
- Dispatch overhead: % of forward time
- Load imbalance: (max_load - min_load) / mean_load
- Expert utilization histogram (per-rank + global)
- Peak memory (allocated/reserved)
- Scaling efficiency: `throughput_N / (N * throughput_1)`

**Required Plot (credibility plot):**
- `1 GPU -> 100% baseline`
- `2 GPUs -> speedup + efficiency`
- `4 GPUs -> speedup + efficiency`

**Deliverable:** Benchmark script + scaling plot + communication breakdown report

### 4.7 Distributed Validation, Checkpointing, and Restore

**Objective:** Validate correctness and eliminate distributed failure modes early (especially checkpoint restore).

**Implementation:**
```python
# tests/test_dist_training.py
"""
Tests for distributed HGSEL training.
"""

def test_distributed_trainer_creation():
    """Can instantiate DistributedHGSELTrainer."""
    trainer = DistributedHGSELTrainer(
        model=model,
        config=config,
        rank=0,
        world_size=1,
    )
    assert trainer.rank == 0
    assert trainer.world_size == 1

def test_distributed_data_loading():
    """Data loaders partition data correctly."""
    for rank in range(world_size):
        train_l, val_l = create_distributed_loaders(
            batch_size=16,
            rank=rank,
            world_size=world_size,
        )
        # Verify sampler semantics and global batch math
        assert global_batch_size == per_rank_batch_size * world_size

def test_expert_sharding():
    """Experts distributed across ranks."""
    # Rank 0 owns experts 0-15, Rank 1 owns 16-31, etc.
    for module in model.modules():
        if isinstance(module, HGSELLayer):
            shard = module.expert_shard
            assert shard.rank == rank
            assert shard.n_experts_local == n_experts // world_size

def test_synchronized_salt():
    """Salt tuning synchronized across ranks."""
    # Rank 0 optimizes salt
    # All ranks apply same salt
    # Verify same routing keys across ranks
    pass

def test_distributed_checkpoint_roundtrip():
    """Save/load model + optimizer + RNG states across ranks."""
    # 1. Run a few steps
    # 2. Save checkpoint (model, optimizer, RNG)
    # 3. Restart ranks and restore
    # 4. Verify next-step loss matches expected tolerance
    pass
```

**Checkpoint Requirements (test early, not at end of phase):**
- Model state
- Optimizer state
- RNG states (CPU + CUDA)
- Resume consistency across ranks

**Deliverable:** Distributed tests + checkpoint roundtrip validation

### 4.8 Documentation Updates

**Objective:** Document Phase 4 achievements and Phase 5 roadmap.

**Files:**
- Update [README.md](../README.md): Add Phase 4 completion, distributed training instructions
- Create [PHASE4_COMPLETION.md](../PHASE4_COMPLETION.md): Summary of distributed implementation
- Update [HGSEL_BUILD_PLAN.md](../HGSEL_BUILD_PLAN.md): Add scaling results from Phase 4

---

## Success Criteria

Phase 4 is complete when:

1. ✓ Single-GPU GPU baseline is recorded with full instrumentation (tokens/sec, forward/backward time, peak memory, expert utilization histogram)
2. ✓ DDP-only multi-GPU training (no expert sharding) reaches convergence parity with Phase 3 / single-GPU baseline
3. ✓ Expert-parallel communication microbenchmark is completed and documented (all-to-all vs local expert compute)
4. ✓ In representative batch regimes, expert-parallel communication overhead is within budget (target `<20%` of forward; otherwise redesign/optimize before claiming scale)
5. ✓ Expert-sharded multi-GPU training (2–4 GPUs) reaches similar convergence as Phase 3
6. ✓ Scaling plot produced with reported speedup + efficiency (1, 2, 4 GPUs) and batch-size context
7. ✓ Expert load balance maintained across ranks and experts (entropy / token-count variance tracked)
8. ✓ Distributed validation tests all passing, including checkpoint save/restore with optimizer + RNG state
9. ✓ Documentation complete

---

## Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| 8 | GPU baseline instrumentation + distributed loaders | train_gpu_baseline.py, dist_data.py, baseline metrics report |
| 8 | DDP-only parity run | DDP trainer path + parity comparison |
| 9 | Expert-parallel communication microbenchmark | microbenchmark harness + comm/compute results |
| 9 | Checkpoint roundtrip test (distributed) | checkpoint save/restore validation |
| 10 | Expert sharding + all-to-all integration | dist_hgsel_trainer.py expert-parallel path |
| 10 | Full training + scaling benchmarks + docs | scaling plot, benchmark report, PHASE4_COMPLETION.md |

---

## Phase 5 Preview (Inference & Benchmarking)

Once Phase 4 is complete:

**Phase 5 Goals:**
- 300M model inference optimization
- Routing compiler & cache-hot kernels
- Token block packing & prefetch scheduling
- End-to-end throughput benchmark vs dense

**Phase 5 Deliverables:**
- Inference module with compiled routes
- Block packing algorithm
- Prefetch scheduler
- 300M benchmark results (throughput, latency, quality)

---

## Risk Mitigation

### Risk: Communication Overhead Dominates
**Mitigation:** 
- Run a dedicated all-to-all microbenchmark before full trainer integration
- Measure comm vs local expert compute time directly (same payload sizes)
- Optimize payload packing / batching before architectural changes
- Implement overlap (computation during communication) only after baseline measurements

### Risk: Load Imbalance Across GPUs
**Mitigation:**
- Synchronize salt globally (avoid per-rank hidden assumptions)
- Log per-rank expert token counts and global variance
- Validate routing distribution in microbenchmark and full training
- Consider remapping only if measured imbalance persists

### Risk: Distributed Bugs Hard to Debug
**Mitigation:**
- DDP-only parity path before expert-parallel path
- Unit tests for each component
- Comprehensive logging with rank IDs
- Single-GPU fallback mode in trainer
- Use `torchrun` for standard deployment
- Add timeouts / barriers only for debugging deadlocks

### Risk: Checkpoints Corrupt in Multi-GPU
**Mitigation:**
- Test checkpoint roundtrip during Phase 4 (not at the end)
- Save model + optimizer + RNG states
- Rank 0 saves globally consistent checkpoint metadata
- Validate resume determinism across ranks for at least one step

---

## References

- [torch.distributed docs](https://pytorch.org/docs/stable/distributed.html)
- [torchrun / torch.distributed.run](https://pytorch.org/docs/stable/elastic/run.html)
- [DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)
- [torch.distributed.all_to_all_single](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single)

---

## Conclusion

Phase 4 transforms HGSEL from single-GPU research to a multi-GPU systems experiment. The key question is not whether DDP works, but whether expert-parallel all-to-all token exchange stays within a viable communication budget.

Key focus areas:
- Instrumented GPU baseline (control condition)
- DDP-only parity (plumbing validation)
- All-to-all communication microbenchmark (scaling viability gate)
- Expert-parallel scaling efficiency + distributed checkpoint reliability

Next: Phase 5 (inference optimization) and Phase 6–9 (1T scale architecture).
