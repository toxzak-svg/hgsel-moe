"""
Distributed training wrapper for multi-GPU HGSEL training.

Extends the base Trainer with distributed synchronization,
all-reduce gradient averaging, and rank-0 checkpointing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hgsel.distributed import dist_utils
from hgsel.training.trainer import Trainer, TrainingConfig


class DistributedTrainer(Trainer):
    """Multi-GPU training wrapper using torch.distributed.
    
    Extends base Trainer with:
    - All-reduce gradient averaging
    - Loss synchronization across ranks
    - Rank-0 checkpointing
    - Distributed barrier synchronization
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        config: TrainingConfig,
        device: torch.device = None,
    ):
        """Initialize distributed trainer.
        
        Args:
            model: The model to train
            optimizer: Optimizer instance
            config: Training configuration
            device: Device to use (default: auto-detect from rank)
        """
        if device is None:
            device = dist_utils.get_device()
        
        super().__init__(model, optimizer, config, device)
        
        self.rank = dist_utils.get_rank()
        self.world_size = dist_utils.get_world_size()
        self.is_master = self.rank == 0
        
        if self.is_master:
            print(f"[Rank {self.rank}] DistributedTrainer initialized (world_size={self.world_size})")

    def setup_distributed(
        self,
        rank: int,
        world_size: int,
        backend: str = "nccl",
        master_addr: str = "localhost",
        master_port: str = "12355",
    ) -> None:
        """Initialize torch.distributed for this process.
        
        Args:
            rank: Process rank
            world_size: Total number of processes
            backend: Distributed backend (nccl, gloo, mpi)
            master_addr: Master node address
            master_port: Master node port
        """
        if dist_utils.is_dist_initialized():
            return
        
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % torch.cuda.device_count())
        
        torch.distributed.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
        
        if self.is_master:
            print(f"[Rank {self.rank}] torch.distributed initialized with {backend} backend")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training step with distributed gradient averaging.
        
        Args:
            batch: Input batch dict with 'input_ids', 'labels', optionally 'input_ids'
        
        Returns:
            Dict with loss, learning_rate, and other metrics
        """
        self.model.train()
        
        # Move batch to device
        if isinstance(batch, dict):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        else:
            batch = batch.to(self.device)
        
        # Forward pass
        if isinstance(batch, dict):
            outputs = self.model(batch['input_ids'])
        else:
            outputs = self.model(batch)
        
        # Loss computation
        if isinstance(batch, dict) and 'labels' in batch:
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch['labels'].view(-1)
            )
        else:
            # Assume outputs contain loss
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        
        # Add auxiliary loss if configured
        auxiliary_loss = self._compute_auxiliary_loss(batch)
        if auxiliary_loss is not None:
            loss = loss + self.config.auxiliary_loss_weight * auxiliary_loss
        
        # Distributed: all-reduce loss for monitoring
        loss_for_logging = loss.detach().clone()
        dist_utils.all_reduce_mean(loss_for_logging, group=None)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (applied locally per rank)
        if self.config.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
        
        # All-reduce gradients for synchronization
        self._all_reduce_gradients()
        
        # Update
        self.optimizer.step()
        
        # Metrics
        metrics = {
            "loss": loss_for_logging.item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        
        # Add HGSEL-specific metrics if available
        if hasattr(self.model, "get_routing_diagnostics"):
            diag = self.model.get_routing_diagnostics()
            if diag:
                # Average entropy across ranks
                if "entropy" in diag:
                    entropy = torch.tensor(diag["entropy"], device=self.device)
                    dist_utils.all_reduce_mean(entropy, group=None)
                    metrics["entropy"] = entropy.item()
                
                if "expert_load" in diag:
                    metrics["max_expert_load"] = diag["expert_load"].max().item()
        
        return metrics

    def validation_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute validation loss for a batch.
        
        Args:
            batch: Validation batch dict
        
        Returns:
            Scalar validation loss (aggregated across ranks)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            else:
                batch = batch.to(self.device)
            
            # Forward pass
            if isinstance(batch, dict):
                outputs = self.model(batch['input_ids'])
            else:
                outputs = self.model(batch)
            
            # Loss computation
            if isinstance(batch, dict) and 'labels' in batch:
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    batch['labels'].view(-1)
                )
            else:
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            
            # Distributed: all-reduce for aggregation
            dist_utils.all_reduce_mean(loss, group=None)
        
        return loss

    def _all_reduce_gradients(self) -> None:
        """Average gradients across all ranks.
        
        Only called when world_size > 1 and distributed is initialized.
        Single-GPU mode: no-op.
        """
        if self.world_size == 1 or not dist_utils.is_dist_initialized():
            return
        
        for param in self.model.parameters():
            if param.grad is not None:
                dist_utils.all_reduce_mean(param.grad, group=None)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Run distributed training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        
        Returns:
            Training results dict
        """
        results = {
            "train_loss": [],
            "val_loss": [],
            "val_perplexity": [],
            "learning_rates": [],
        }
        
        for epoch in range(self.config.num_epochs):
            # Synchronize all ranks at epoch start
            dist_utils.barrier(group=None)
            
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                metrics = self.train_step(batch)
                epoch_loss += metrics["loss"]
                num_batches += 1
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Log (only on master rank)
                if self.is_master and batch_idx % self.config.log_interval == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    print(
                        f"Epoch {epoch}, Batch {batch_idx}, "
                        f"Loss: {metrics['loss']:.4f}, Avg: {avg_loss:.4f}, "
                        f"LR: {metrics['learning_rate']:.6f}"
                    )
            
            avg_epoch_loss = epoch_loss / num_batches
            results["train_loss"].append(avg_epoch_loss)
            
            # Validation (on master only, then broadcast if needed)
            if val_loader is not None and epoch % self.config.val_interval == 0:
                val_loss = self._validate(val_loader)
                results["val_loss"].append(val_loss.item())
                results["val_perplexity"].append((val_loss.exp()).item())
                
                if self.is_master:
                    print(
                        f"Epoch {epoch}, Validation - "
                        f"Loss: {val_loss.item():.4f}, "
                        f"Perplexity: {(val_loss.exp()).item():.2f}"
                    )
                    
                    # Checkpoint on master rank
                    if isinstance(self.checkpoint_dir, str) or isinstance(self.checkpoint_dir, Path):
                        checkpoint_path = Path(self.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
                        self.save_checkpoint(checkpoint_path)
            
            # Synchronize at epoch end
            dist_utils.barrier(group=None)
        
        # Final cleanup
        dist_utils.cleanup_distributed()
        
        if self.is_master:
            print("Training complete!")
        
        return results

    def _validate(self, val_loader: DataLoader) -> torch.Tensor:
        """Compute aggregate validation loss across all ranks.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Aggregated validation loss
        """
        val_loss = 0.0
        num_batches = 0
        
        for batch in val_loader:
            loss = self.validation_step(batch)
            val_loss += loss.item()
            num_batches += 1
        
        avg_loss = torch.tensor(val_loss / num_batches, device=self.device)
        dist_utils.all_reduce_mean(avg_loss, group=None)
        
        return avg_loss

    def save_checkpoint(self, path: str | Path) -> None:
        """Save checkpoint (only on rank 0).
        
        Args:
            path: Path to save checkpoint
        """
        if not self.is_master:
            return
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "rank": self.rank,
            "world_size": self.world_size,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load checkpoint (on all ranks).
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.is_master:
            print(f"Loaded checkpoint from {path}")

    def cleanup(self) -> None:
        """Clean up distributed training resources."""
        dist_utils.cleanup_distributed()
        if self.is_master:
            print("Cleanup complete!")
