"""
Training harness for HGSEL models.

Provides:
- Training loop with gradual warmup
- Validation pipeline
- Checkpoint management
- Metrics tracking
- W&B integration
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Tuple, Optional
from pathlib import Path
import json
import time

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class TrainingConfig:
    """Training hyperparameters and configuration."""

    def __init__(self, **kwargs):
        self.batch_size = kwargs.get("batch_size", 32)
        self.val_batch_size = kwargs.get("val_batch_size", 64)
        self.num_epochs = kwargs.get("num_epochs", 10)
        self.learning_rate = kwargs.get("learning_rate", 1e-3)
        self.weight_decay = kwargs.get("weight_decay", 1e-4)
        self.warmup_steps = kwargs.get("warmup_steps", 1000)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
        self.clip_grad_norm = kwargs.get("clip_grad_norm", None)
        
        # Load balancing (Phase 1-2)
        self.aux_loss_weight = kwargs.get("aux_loss_weight", 0.05)
        self.aux_loss_schedule = kwargs.get("aux_loss_schedule", "constant")  # "constant" or "decay"
        self.salt_tuning_interval = kwargs.get("salt_tuning_interval", 100)  # batches
        
        # Device
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Checkpointing
        self.checkpoint_dir = Path(kwargs.get("checkpoint_dir", "./checkpoints"))
        self.save_interval = kwargs.get("save_interval", 500)  # batches
        self.num_keep_checkpoints = kwargs.get("num_keep_checkpoints", 3)
        
        # Logging
        self.log_interval = kwargs.get("log_interval", 10)  # batches
        self.val_interval = kwargs.get("val_interval", 500)  # batches
        self.use_wandb = kwargs.get("use_wandb", True)
        self.wandb_project = kwargs.get("wandb_project", "hgsel")
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert to dict for logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and k not in ["checkpoint_dir"]}


class Trainer:
    """Main training harness."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        aux_loss_fn: Optional[nn.Module] = None,
    ):
        self.model = model
        self.config = config
        self.aux_loss_fn = aux_loss_fn
        self.device = torch.device(config.device)
        
        self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler (warmup + cosine annealing)
        self.lr_scheduler = None

        # Checkpointing
        self.checkpoint_dir = config.checkpoint_dir
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.global_epoch = 0

        # Metrics
        self.train_loss_sum = 0.0
        self.train_steps = 0
        self.aux_loss_sum = 0.0

        # W&B
        if config.use_wandb:
            if HAS_WANDB:
                wandb.init(
                    project=config.wandb_project,
                    name="hgsel-phase2",
                    config=config.to_dict(),
                )
            else:
                print("⚠ W&B requested but not installed. Skipping wandb logging.")

    def setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler with warmup."""
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - self.config.warmup_steps,
        )

    def get_aux_loss_weight(self) -> float:
        """Get auxiliary loss weight based on training schedule."""
        if self.config.aux_loss_schedule == "constant":
            return self.config.aux_loss_weight

        elif self.config.aux_loss_schedule == "decay":
            # Decay from 0.05 to 0.01 over first half of training
            progress = self.global_step / (self.config.num_epochs * 10000)  # rough estimate
            decay_schedule = max(0.01, 0.05 * (1 - progress))
            return decay_schedule

        return self.config.aux_loss_weight

    def _apply_warmup(self):
        """Apply learning rate warmup for first N steps."""
        if self.global_step < self.config.warmup_steps:
            warmup_progress = self.global_step / self.config.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.config.learning_rate * warmup_progress

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        input_ids, labels = batch
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        # Forward
        logits = self.model(input_ids)
        
        # Main loss
        batch_size, seq_len, vocab_size = logits.shape
        loss = nn.functional.cross_entropy(
            logits.view(batch_size * seq_len, vocab_size),
            labels.view(batch_size * seq_len),
        )

        # Auxiliary loss (load balancing)
        aux_loss = 0.0
        if self.aux_loss_fn is not None:
            # Try to get expert loads from HGSEL layers
            # This is handled differently in Phase 2 - we'll track separately
            pass

        total_loss = loss  # + aux_loss (to be integrated with HGSEL hooks)

        # Backward
        total_loss.backward()

        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Learning rate warmup
            self._apply_warmup()

            # LR scheduler step (after warmup)
            if self.lr_scheduler and self.global_step >= self.config.warmup_steps:
                self.lr_scheduler.step()

            self.global_step += 1

        # Track metrics
        self.train_loss_sum += loss.item()
        self.train_steps += 1

        return {
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def validate(self, val_loader) -> float:
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(input_ids)
                batch_size, seq_len, vocab_size = logits.shape
                loss = nn.functional.cross_entropy(
                    logits.view(batch_size * seq_len, vocab_size),
                    labels.view(batch_size * seq_len),
                )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.model.train()

        return avg_loss

    def save_checkpoint(self, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "global_epoch": self.global_epoch,
            "val_loss": val_loss,
        }

        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Keep only last N checkpoints
        checkpoint_files = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoint_files) > self.config.num_keep_checkpoints:
            for old_checkpoint in checkpoint_files[: -self.config.num_keep_checkpoints]:
                old_checkpoint.unlink()

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)

    def train(self, train_loader, val_loader=None):
        """Full training loop."""
        self.model.train()

        # Setup scheduler
        total_steps = len(train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        self.setup_scheduler(total_steps)

        for epoch in range(self.config.num_epochs):
            self.global_epoch = epoch

            for batch_idx, batch in enumerate(train_loader):
                metrics = self.train_step(batch)

                # Logging
                if batch_idx % self.config.log_interval == 0:
                    avg_loss = self.train_loss_sum / max(self.train_steps, 1)
                    print(
                        f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                        f"Batch {batch_idx}/{len(train_loader)} | "
                        f"Loss: {avg_loss:.4f} | LR: {metrics['lr']:.2e}"
                    )

                    if self.config.use_wandb and HAS_WANDB:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/lr": metrics["lr"],
                            "train/step": self.global_step,
                        })

                # Validation
                if val_loader and batch_idx % self.config.val_interval == 0:
                    val_loss = self.validate(val_loader)
                    print(f"  Validation loss: {val_loss:.4f}")

                    if self.config.use_wandb and HAS_WANDB:
                        wandb.log({
                            "val/loss": val_loss,
                            "val/step": self.global_step,
                        })

                    self.save_checkpoint(val_loss)

                # Periodic checkpoint
                if batch_idx % self.config.save_interval == 0 and batch_idx > 0:
                    self.save_checkpoint(float("inf"))

            self.train_loss_sum = 0.0
            self.train_steps = 0

        print("✓ Training complete!")


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer=None):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
