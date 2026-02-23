# Phase 4 Completion Summary

**Status:** ✓ IN PROGRESS

**Duration:** Days 6-7 (estimated)

**Objective:** Build multi-GPU distributed training infrastructure for HGSEL, enabling expert sharding and synchronized gradient updates across nodes.

---

## What Phase 4 Achieves

Phase 4 transforms HGSEL from single-GPU to multi-GPU capable, implementing:
- **All-to-all token exchange** for expert-parallel dispatch
- **All-reduce gradient synchronization** for distributed training
- **Expert sharding** across GPU ranks
- **Distributed training wrapper** for easy multi-GPU training
- **End-to-end validation** with synchronization tests

### Key Components

#### 1. Token Exchange: `hgsel/distributed/token_exchange.py`

**Status:** ○ Not Yet Implemented

Implements actual all-to-all communication for routing tokens to remote experts:

```python
class TokenExchange:
    """All-to-all token exchange for expert-parallel dispatch."""
    
    def exchange(self, payloads: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Exchange token payloads across all ranks.
        
        - payloads[rank] = tokens destined for that rank
        - returns: tokens received from all other ranks
        """
```

**Implementation Plan:**
- Use `torch.distributed.all_to_all_single()` for payload exchange
- Handle variable-length token batches (requires padding/unpadding)
- Support fallback to single-GPU mode (no-op when rank=0, world_size=1)

#### 2. All-Reduce for Loss & Metrics: `hgsel/distributed/dist_utils.py`

**Status:** ○ Not Yet Implemented

Synchronization helpers for distributed training:

```python
def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a tensor across all ranks."""
    
def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Sum a tensor across all ranks."""
    
def get_world_size() -> int:
    """Get total number of ranks."""
    
def get_rank() -> int:
    """Get current rank."""
    
def barrier() -> None:
    """Synchronize all ranks."""
```

#### 3. Distributed Trainer: `hgsel/training/distributed_trainer.py`

**Status:** ○ Not Yet Implemented

Wrapper around base Trainer for multi-GPU orchestration:

```python
class DistributedTrainer(Trainer):
    """Multi-GPU training wrapper using torch.distributed."""
    
    def setup_distributed(self, rank: int, world_size: int, 
                         backend: str = "nccl") -> None:
        """Initialize torch.distributed."""
    
    def train_step(self, batch) -> dict:
        """Forward, loss, backward with all-reduce of loss."""
    
    def cleanup(self) -> None:
        """Cleanup torch.distributed resources."""
```

#### 4. Expert Sharding: `hgsel/distributed/expert_sharding.py`

**Status:** ✓ PARTIAL (skeleton exists)

Deterministic expert partitioning across ranks is already implemented.

**Enhancements needed:**
- Build sharded expert banks per rank
- Verify expert ID → rank mapping
- Handle load balancing via salt across shards

#### 5. Distributed Dispatch Pipeline: `hgsel/distributed/dispatch_pipeline.py`

**Status:** ○ Not Yet Implemented

Complete dispatch flow: local experts + remote all-to-all:

```python
class DistributedDispatchPipeline:
    """End-to-end dispatch: route → local + remote → exchange → combine."""
    
    def forward(self, hidden_states: torch.Tensor, 
                expert_ids: torch.Tensor) -> torch.Tensor:
        """
        1. Split expert IDs into local vs remote
        2. Dispatch local experts
        3. All-to-all exchange for remote
        4. Combine all outputs
        """
```

---

## Implementation Steps

### Step 1: Implement dist_utils

**File:** `hgsel/distributed/dist_utils.py`

Core distributed utilities:
- Rank detection (single-GPU fallback)
- World size, backend detection
- All-reduce operations (mean, sum)
- Barrier synchronization

### Step 2: Implement Token Exchange

**File:** `hgsel/distributed/token_exchange.py`

All-to-all communication:
- Payload packing per destination rank
- `torch.distributed.all_to_all_single()` call
- Variable-length handling with padding
- Fallback to no-op for single-GPU

### Step 3: Create DistributedTrainer

**File:** `hgsel/training/distributed_trainer.py`

Multi-GPU training orchestration:
- Extend base Trainer
- Setup/cleanup torch.distributed
- Wrap loss computation with all-reduce
- Gradient averaging across ranks
- Checkpoint only on rank 0

### Step 4: Implement DispatchPipeline

**File:** `hgsel/distributed/dispatch_pipeline.py`

Complete dispatch system:
- Local expert dispatch (via existing ExpertDispatchController)
- Remote token exchange
- Output gathering and combining
- Handle empty batches (no local tokens, no remote)

### Step 5: Add Tests

**File:** `tests/test_dist_*.py`

Test coverage:
- `test_dist_utils.py`: All-reduce, rank detection
- `test_token_exchange.py`: Simulated all-to-all, edge cases
- `test_dispatch_pipeline.py`: Full dispatch loop with sharding
- `test_dist_smoke.py`: End-to-end training step with 2+ simulated ranks

### Step 6: Create End-to-End Script

**File:** `experiments/train_distributed_300m.py`

Distributed training entry point:
- Torchrun / torch.distributed.launch integration
- Master node training loop
- Periodic validation on rank 0
- Aggregated metrics logging

---

## Expected Deliverables

| File | Purpose | Status |
|------|---------|--------|
| `hgsel/distributed/dist_utils.py` | All-reduce, rank utilities | ○ |
| `hgsel/distributed/token_exchange.py` | All-to-all communication | ○ |
| `hgsel/training/distributed_trainer.py` | Multi-GPU trainer wrapper | ○ |
| `hgsel/distributed/dispatch_pipeline.py` | Complete distribute dispatch | ○ |
| `experiments/train_distributed_300m.py` | Distributed training script | ○ |
| `tests/test_dist_utils.py` | Utilities tests | ○ |
| `tests/test_token_exchange.py` | Exchange simulation tests | ○ |
| `tests/test_dispatch_api.py` | API integration tests | ○ |
| `tests/test_dist_smoke.py` | End-to-end smoke tests | ○ |
| `PHASE4_COMPLETION.md` | This document | ✓ |

---

## Testing Strategy

### Unit Tests
- **dist_utils**: Rank getters, all-reduce simulation (single-GPU mode)
- **token_exchange**: Payload packing, roundtrip correctness
- **dispatch_pipeline**: Local vs remote split, output gathering

### Integration Tests
- **Distributed smoke test**: 2-4 simulated ranks, single mini-batch
- **Convergence**: Training loss decreases over 10 steps on 2+ ranks
- **Load balancing**: Verify expert distribution across shards

### Simulation (Without Actual Multi-GPU)
- Mock `torch.distributed` in tests
- Simulate all-to-all exchanges
- Verify dispatch plan correctness without GPU-to-GPU comms

---

## Design Decisions

### 1. All-to-All via `torch.distributed.all_to_all_single()`
- **Why:** Native PyTorch, efficient, reliable
- **Trade-off:** Requires padding for variable-length exchanges
- **Fallback:** Single-GPU mode returns local batch only

### 2. Payload Packing Strategy
- **Row-wise:** Iterate tokens, append to `rank_to_payload[owner_rank]`
- **Batching:** All tokens for a rank concatenated
- **Padding:** Pad to max length + record actual sizes for unpacking

### 3. Distributed Trainer as Wrapper
- **Why:** Allows reusing single-GPU Trainer logic
- **Trade-off:** Adds layer of indirection but cleaner separation

### 4. Rank-0 Checkpointing
- **Why:** Avoid N copies of checkpoints on distributed file systems
- **Fallback:** All ranks can save if needed for fault tolerance

### 5. No Gradient Bucket Fusion (Phase 4)
- **Defer to Phase 5:** Overlapping comms + compute is Phase 5 optimization
- **Phase 4 Focus:** Just get correctness + basic performance

---

## How to Run (After Implementation)

### Multi-GPU Training with 4 GPUs
```bash
cd hgsel-moe
torchrun --nproc_per_node=4 \
    experiments/train_distributed_300m.py \
    --batch-size 32 \
    --num-epochs 5 \
    --use-hgsel \
    --num-experts 64 \
    --learning-rate 0.001
```

### Single-GPU Fallback (No torch.distributed)
```bash
python experiments/train_distributed_300m.py \
    --batch-size 32 \
    --num-epochs 5 \
    --use-hgsel
```

### Smoke Test (2 Simulated Ranks)
```bash
pytest tests/test_dist_smoke.py -v
```

---

## Success Criteria

✓ **Phase 4 Complete When:**
1. All-to-all token exchange works in single-GPU mode (fallback correct)
2. All-reduce for loss, perplexity, entropy across simulated ranks
3. DistributedTrainer sets up/cleans up torch.distributed correctly
4. DispatchPipeline routes local + remote experts and combines outputs
5. Distributed training script runs on 1+ GPUs with lower per-rank loss
6. Tests pass: unit + integration + smoke tests
7. Documentation complete with usage examples

---

## Known Limitations & Future Work

### Phase 4 Constraints
1. **No overlapping comms + compute:** All-to-all blocks until tokens arrive
   - Phase 5: Prefetch remote tokens, overlap with local compute
2. **No load-aware routing:** Expert selection deterministic, not adaptive
   - Phase 6: Scoring network + load-aware candidates
3. **No fault tolerance:** All ranks must stay synchronized
   - Phase 6+: Checkpoint aggregation, rebalancing
4. **No heterogeneous sharding:** Equal expert split per rank
   - Phase 6: Weighted sharding for unequal device capacity

### Debugging Tips
- **Device mismatch:** Ensure all tensors on same device (cuda:0, etc)
- **Rank misalignment:** Print rank, world_size in setup to debug
- **Deadlock:** Add timeouts to all-to-all, use torch.distributed.barrier() for debug
- **NCS (Never Complete Sync):** Check gather/scatter padding/unpacking logic

---

## References

- **Torch Distributed:** https://pytorch.org/docs/stable/distributed.html
- **All-to-All:** `torch.distributed.all_to_all_single()` docs
- **DDP vs Expert Parallel:** https://pytorch.org/docs/stable/notes/ddp.html
- **GShard (Google):** All-to-all expert dispatch pioneer

---

## Checklist

- [ ] dist_utils.py implemented and tested
- [ ] TokenExchange with all_to_all_single() works
- [ ] DistributedTrainer extends Trainer, handles all-reduce
- [ ] DispatchPipeline routes local + remote correctly
- [ ] test_dist_utils.py passes
- [ ] test_token_exchange.py passes
- [ ] test_dispatch_api.py passes
- [ ] test_dist_smoke.py passes (2+ simulated ranks)
- [ ] train_distributed_300m.py runs on multi-GPU
- [ ] Documentation updated with Phase 4 examples

---

**Next Phase:** Phase 5 (Benchmarking & Optimizations) — Measure throughput, memory, latency on distributed setup; overlap comms with compute.

