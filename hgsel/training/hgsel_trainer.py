"""
Enhanced trainer with HGSEL-specific load balancing and metrics.

Features:
- Auxiliary loss integration for expert load balancing
- Salt parameter optimization during training
- Per-epoch expert utilization analysis
- Convergence tracking (loss, perplexity, entropy)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from hgsel.layer import HGSELLayer
from hgsel.training.trainer import Trainer, TrainingConfig
from hgsel.training.salt_optimizer import SaltOptimizer, UtilizationMonitor
from hgsel.training.losses import LoadBalancingLoss


class HGSELTrainer(Trainer):
    """
    Extended trainer for HGSEL models with load balancing.
    
    Adds:
    - Auxiliary loss computation from HGSEL layers
    - Salt optimization during training
    - Expert utilization monitoring
    - Per-layer routing analysis
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        aux_loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__(model, config, aux_loss_fn)
        
        # HGSEL-specific monitoring
        self.salt_optimizer = None
        self.utilization_monitors = {}  # One per HGSEL layer
        self.layer_stats_history = []  # Track per-layer stats over time
        
        # Initialize monitors for each HGSEL layer
        self._setup_hgsel_monitors()

    def _setup_hgsel_monitors(self):
        """Setup monitors for each HGSEL layer in the model."""
        for name, module in self.model.named_modules():
            if isinstance(module, HGSELLayer):
                monitor = UtilizationMonitor(
                    n_experts=module.n_experts,
                    ema_decay=0.99,
                )
                self.utilization_monitors[name] = monitor
                print(f"  HGSEL layer monitor setup: {name}")

        # Setup salt optimizer for first HGSEL layer (global tuning)
        for name, module in self.model.named_modules():
            if isinstance(module, HGSELLayer):
                self.salt_optimizer = SaltOptimizer(
                    n_experts=module.n_experts,
                    initial_salt=module.router.salt,
                    target_entropy=torch.log(torch.tensor(module.n_experts, dtype=torch.float32)),
                )
                print(f"  Salt optimizer initialized for layer: {name}")
                break

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Training step with auxiliary loss computation."""
        input_ids, labels = batch
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        # Forward
        logits = self.model(input_ids)
        
        # Main loss
        batch_size, seq_len, vocab_size = logits.shape
        main_loss = nn.functional.cross_entropy(
            logits.view(batch_size * seq_len, vocab_size),
            labels.view(batch_size * seq_len),
        )

        # Auxiliary loss (load balancing from HGSEL layers)
        aux_loss = 0.0
        aux_loss_weight = self.get_aux_loss_weight()

        if aux_loss_weight > 0 and self.aux_loss_fn is not None:
            # Collect expert loads from all HGSEL layers
            total_layer_aux_loss = 0.0
            num_hgsel_layers = 0

            for name, module in self.model.named_modules():
                if isinstance(module, HGSELLayer):
                    # Get expert loads from layer's EMA
                    layer_aux_loss = self.aux_loss_fn(module.expert_load_ema)
                    total_layer_aux_loss += layer_aux_loss
                    num_hgsel_layers += 1

            if num_hgsel_layers > 0:
                aux_loss = total_layer_aux_loss / num_hgsel_layers * aux_loss_weight

        total_loss = main_loss + aux_loss

        # Backward
        total_loss.backward()

        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Learning rate warmup
            self._apply_warmup()

            # LR scheduler step
            if self.lr_scheduler and self.global_step >= self.config.warmup_steps:
                self.lr_scheduler.step()

            self.global_step += 1

        # Track metrics
        self.train_loss_sum += main_loss.item()
        self.train_steps += 1
        self.aux_loss_sum += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss

        return {
            "loss": main_loss.item(),
            "aux_loss": aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
            "total_loss": total_loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def optimize_salt(self):
        """Optimize salt parameter based on current expert loads."""
        if self.salt_optimizer is None:
            return

        for name, module in self.model.named_modules():
            if isinstance(module, HGSELLayer):
                # Get current expert loads
                expert_loads = module.expert_load_ema.clone().detach()

                # Optimize salt
                new_salt, entropy = self.salt_optimizer.optimize(expert_loads)

                # Update router
                module.set_salt(new_salt)

                return {
                    "salt": new_salt,
                    "entropy": float(entropy),
                }

        return None

    def collect_layer_statistics(self) -> Dict:
        """Collect expert utilization stats from all HGSEL layers."""
        stats = {}

        for name, module in self.model.named_modules():
            if isinstance(module, HGSELLayer):
                layer_stats = module.get_expert_load_stats()
                stats[name] = layer_stats

                # Update monitor
                if name in self.utilization_monitors:
                    monitor = self.utilization_monitors[name]
                    monitor.update(module.expert_load_ema)

        return stats

    def train(self, train_loader, val_loader=None):
        """Full training loop with HGSEL-specific monitoring."""
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
                    avg_aux_loss = self.aux_loss_sum / max(self.train_steps, 1)

                    print(
                        f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                        f"Batch {batch_idx}/{len(train_loader)} | "
                        f"Loss: {avg_loss:.4f} | Aux: {avg_aux_loss:.4f} | LR: {metrics['lr']:.2e}"
                    )

                    if self.config.use_wandb:
                        try:
                            import wandb
                            if wandb.run is not None:
                                wandb.log({
                                    "train/loss": avg_loss,
                                    "train/aux_loss": avg_aux_loss,
                                    "train/lr": metrics["lr"],
                                    "train/step": self.global_step,
                                })
                        except ImportError:
                            pass

                # Salt optimization every N batches
                if (
                    batch_idx % self.config.salt_tuning_interval == 0
                    and batch_idx > 0
                ):
                    salt_stats = self.optimize_salt()
                    if salt_stats:
                        print(f"  Salt tuned: {salt_stats['salt']:.4f}, entropy: {salt_stats['entropy']:.4f}")

                # Validation
                if val_loader and batch_idx % self.config.val_interval == 0:
                    val_loss = self.validate(val_loader)
                    print(f"  Validation loss: {val_loss:.4f}")

                    # Collect layer stats
                    layer_stats = self.collect_layer_statistics()
                    for layer_name, stats in layer_stats.items():
                        print(f"    {layer_name}: entropy={stats['entropy']:.4f}, mean_load={stats['mean_load']:.4f}")

                    if self.config.use_wandb:
                        try:
                            import wandb
                            if wandb.run is not None:
                                wandb.log({
                                    "val/loss": val_loss,
                                    "val/step": self.global_step,
                                })
                        except ImportError:
                            pass

                    self.save_checkpoint(val_loss)

                # Periodic checkpoint
                if batch_idx % self.config.save_interval == 0 and batch_idx > 0:
                    self.save_checkpoint(float("inf"))

            # End-of-epoch statistics
            print(f"\nEpoch {epoch + 1} Summary:")
            layer_stats = self.collect_layer_statistics()
            for layer_name, stats in layer_stats.items():
                print(f"  {layer_name}:")
                print(f"    Mean expert load: {stats['mean_load']:.4f}")
                print(f"    Entropy (normalized): {stats['entropy']:.4f}")
                print(f"    Collapsed experts: {self.utilization_monitors[layer_name].get_summary()['ema_loads_min']:.4f}")

            self.train_loss_sum = 0.0
            self.train_steps = 0
            self.aux_loss_sum = 0.0

        print("\n✓ Training complete!")


def load_hgsel_checkpoint(checkpoint_path: str, model: nn.Module, optimizer=None):
    """Load checkpoint (wrapper for compatibility)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
