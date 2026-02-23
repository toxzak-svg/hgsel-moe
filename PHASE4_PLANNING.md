# Phase 4 Planning: Multi-GPU Distribution

**Current Status:** Phase 3 ✓ Complete  
**Target Duration:** Days 8–10  
**Goal:** Enable distributed HGSEL training across multiple GPUs with expert sharding and all-gather communication.

---

## Overview

Phase 4 shifts from single-GPU validation to **multi-GPU scaling**, integrating the distributed components already scaffolded in [hgsel/distributed/](../hgsel/distributed/).

### What Exists
- ✓ Token dispatcher ([token_dispatcher.py](../hgsel/distributed/token_dispatcher.py)): Routes tokens to experts
- ✓ Expert sharding ([expert_sharding.py](../hgsel/distributed/expert_sharding.py)): Distributes experts across ranks
- ✓ Token exchange ([token_exchange.py](../hgsel/distributed/token_exchange.py)): All-gather primitives
- ✓ Dispatch API ([dispatch_api.py](../hgsel/distributed/dispatch_api.py)): Public routing interface
- ✓ Dispatch pipeline ([dispatch_pipeline.py](../hgsel/distributed/dispatch_pipeline.py)): Orchestration layer

### What Phase 4 Adds
- Integration with HGSELTrainer for distributed training
- Multi-GPU validation on actual distributed cluster
- Throughput & communication overhead measurement
- Expert load balancing across GPUs
- Checkpoint save/restore for distributed state

---

## Detailed Phase 4 Roadmap

### 4.1 GPU Training Harness

**Objective:** Measure throughput on GPU and validate Phase 3 findings.

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
# Measure tokens/sec, wall-time, memory usage
```

**Metrics to Track:**
- Throughput (tokens/sec)
- GPU memory peak/reserved
- Training time (wall-clock)
- Convergence speed (epochs to target loss)

**Deliverable:** Benchmark script & results report

### 4.2 Distributed Data Loading

**Objective:** Ensure data consistency across GPUs.

**Implementation:**
```python
# hgsel/training/dist_data.py
"""
Distributed data loading with proper sampling.
"""

class DistributedLanguageModelDataset(LanguageModelDataset):
    """
    Sub-sample data per rank to avoid data duplication.
    Synchronized sampling ensures same tokens per step across ranks.
    """
    def __init__(self, data, rank=0, world_size=1, seed=42):
        super().__init__(data)
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self._setup_sampler()
    
    def _setup_sampler(self):
        """Distribute indices across ranks."""
        total_indices = len(self.data)
        per_rank = total_indices // self.world_size
        start_idx = self.rank * per_rank
        end_idx = start_idx + per_rank if self.rank < self.world_size - 1 else total_indices
        self.indices = list(range(start_idx, end_idx))

def create_distributed_loaders(
    data_dir: str,
    batch_size: int,
    rank: int,
    world_size: int,
) -> tuple:
    """Create train/val loaders for distributed training."""
    # Load data once (broadcast to all ranks)
    if rank == 0:
        data = load_data(data_dir)
    else:
        data = None
    
    # Broadcast data to all ranks (small string data)
    # Or use torch.distributed.broadcast_object_list
    
    # Create per-rank datasets
    train_dataset = DistributedLanguageModelDataset(
        data, rank, world_size, seed=42
    )
    val_dataset = DistributedLanguageModelDataset(
        data, rank, world_size, seed=99
    )
    
    # Create loaders with DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
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
```

**Deliverable:** DistributedLanguageModelDataset + loaders

### 4.3 Distributed HGSEL Trainer

**Objective:** Connect trainer to distributed dispatch.

**Implementation:**
```python
# hgsel/training/dist_hgsel_trainer.py
"""
Distributed HGSEL trainer with all-reduce and expert sharding.
"""

class DistributedHGSELTrainer(HGSELTrainer):
    """
    Extends HGSELTrainer for multi-GPU training.
    
    Integrates:
    - DistributedDataParallel for gradient synchronization
    - Expert sharding across ranks
    - All-gather for expert dispatch
    - Synchronized salt optimization
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
        
        # Setup expert sharding
        self._setup_expert_sharding()
    
    def _setup_expert_sharding(self):
        """Distribute experts across ranks."""
        for name, module in self.model.named_modules():
            if isinstance(module, HGSELLayer):
                # Shard expert bank across ranks
                shard = ExpertSharding(
                    num_experts=module.n_experts,
                    rank=self.rank,
                    world_size=self.world_size,
                )
                module.set_expert_sharding(shard)
    
    def train_step(self, batch):
        """Multi-GPU training step with all-gather."""
        # ... (similar to HGSELTrainer)
        # But with AllGather for expert dispatch
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

**Deliverable:** DistributedHGSELTrainer with sharding integration

### 4.4 Distributed Training Script

**Objective:** Launch multi-GPU training.

**Implementation:**
```python
# experiments/train_distributed.py
"""
Multi-GPU training launcher using torch.distributed.launch.

Usage:
    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        experiments/train_distributed.py \
        --epochs 5 \
        --batch-size 32
"""

import torch
import torch.distributed as dist
from hgsel.layer import HGSELLayer
from hgsel.training.dist_hgsel_trainer import DistributedHGSELTrainer
from hgsel.training.trainer import TrainingConfig
from hgsel.training.dist_data import create_distributed_loaders
from experiments.baselines.dense_transformer import TransformerModel

def main():
    # Get distributed info
    rank = dist.get_rank() if dist.is_available() else 0
    world_size = dist.get_world_size() if dist.is_available() else 1
    
    # Create model
    model = TransformerModel(
        vocab_size=256,
        d_model=512,
        d_ff=2048,
        n_layers=4,
        n_heads=8,
        mlp_class=HGSELLayer,
    )
    
    # Create config
    config = TrainingConfig(
        batch_size=32,
        val_batch_size=64,
        num_epochs=args.epochs,
        learning_rate=0.001,
        device="cuda",
    )
    
    # Create distributed loaders
    train_loader, val_loader = create_distributed_loaders(
        data_dir="./data",
        batch_size=config.batch_size,
        rank=rank,
        world_size=world_size,
    )
    
    # Create distributed trainer
    trainer = DistributedHGSELTrainer(
        model=model,
        config=config,
        rank=rank,
        world_size=world_size,
        backend="nccl",
    )
    
    # Train
    if rank == 0:
        print(f"Training on {world_size} GPUs")
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# Single GPU (4 processes on 1 node)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    experiments/train_distributed.py \
    --epochs 5

# Multi-node (2 nodes, 4 GPUs each)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_NODE \
    --master_port=29500 \
    experiments/train_distributed.py
```

**Deliverable:** Distributed training script with torch.distributed integration

### 4.5 Distributed Benchmarking

**Objective:** Measure communication overhead and scaling efficiency.

**Implementation:**
```python
# experiments/benchmark_distributed.py
"""
Benchmark distributed HGSEL training.
Measures:
- Throughput (tokens/sec)
- All-gather latency
- Expert dispatch overhead
- Load balance across ranks
"""

def benchmark_distributed():
    # Profile all-gather time as function of expert batch size
    # Measure expert dispatch routing time
    # Compare single-GPU vs multi-GPU speedup
    # Track load imbalance (max vs min load per GPU)
    
    results = {
        "throughput": tokens_per_sec,
        "allgather_time": ms,
        "dispatch_overhead": percent,
        "load_imbalance": percent,
        "scaling_efficiency": percent,  # Speedup / num_gpus
    }
```

**Metrics:**
- Throughput: tokens/sec
- All-gather latency: ms per call
- Dispatch overhead: % of forward time
- Load imbalance: (max_load - min_load) / mean_load
- Scaling efficiency: (T_1GPU * N_GPUs) / T_N_GPUs

**Deliverable:** Benchmark script + results table

### 4.6 Distributed Validation & Testing

**Objective:** Unit tests for distributed components.

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
        # Verify no data leakage across ranks
        assert len(train_l) == expected_batches

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
```

**Deliverable:** Distributed training unit tests

### 4.7 Documentation Updates

**Objective:** Document Phase 4 achievements and Phase 5 roadmap.

**Files:**
- Update [README.md](../README.md): Add Phase 4 completion, distributed training instructions
- Create [PHASE4_COMPLETION.md](../PHASE4_COMPLETION.md): Summary of distributed implementation
- Update [HGSEL_BUILD_PLAN.md](../HGSEL_BUILD_PLAN.md): Add scaling results from Phase 4

---

## Success Criteria

Phase 4 is complete when:

1. ✓ GPU training script runs and produces convergence curves
2. ✓ Multi-GPU training (2–4 GPUs) reaches similar convergence as Phase 3
3. ✓ Distributed benchmarks show reasonable scaling efficiency (>70%)
4. ✓ All-gather communication overhead < 20% of forward time
5. ✓ Expert load balance maintained across GPUs (entropy ~1.0)
6. ✓ Distributed validation tests all passing
7. ✓ Checkpoints save/restore correctly in distributed setting
8. ✓ Documentation complete

---

## Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| 8 | GPU baseline, distributed loaders | train_gpu_baseline.py, dist_data.py |
| 9 | Distributed trainer, sharding | dist_hgsel_trainer.py, expert_sharding integration |
| 10 | Distributed training script, benchmarks | train_distributed.py, benchmark results |
| 10 | Tests, documentation | test_dist_training.py, PHASE4_COMPLETION.md |

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
- Profile all-gather calls early
- Implement overlap (computation during communication)
- Consider Ring AllReduce for large clusters

### Risk: Load Imbalance Across GPUs
**Mitigation:**
- Salt optimization per GPU (local tuning)
- Dynamic expert remapping if skew detected
- Real-time load monitoring & rebalancing

### Risk: Distributed Bugs Hard to Debug
**Mitigation:**
- Unit tests for each component
- Comprehensive logging with rank IDs
- Single-GPU fallback mode in trainer
- Use torch.distributed.launch for standard deployment

### Risk: Checkpoints Corrupt in Multi-GPU
**Mitigation:**
- Rank 0 saves globally consistent checkpoint
- All ranks verify checkpoint consistency
- Test checkpoint load in distributed setting

---

## References

- [torch.distributed docs](https://pytorch.org/docs/stable/distributed.html)
- [torch.distributed.launch](https://pytorch.org/docs/stable/distributed.launch.html)
- [DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)
- [torch.distributed.AllGather](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather)

---

## Conclusion

Phase 4 transforms HGSEL from single-GPU research to multi-GPU production system. By integrating the existing distributed components and adding a distributed trainer, we enable training of larger models with expert sharding and synchronized load balancing.

Key focus areas:
- GPU throughput validation
- Multi-GPU scaling efficiency
- All-gather communication choreography
- Distributed checkpoint management

Next: Phase 5 (inference optimization) and Phase 6–9 (1T scale architecture).

