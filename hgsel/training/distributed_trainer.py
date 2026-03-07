"""
Distributed training wrapper for Phase 4 DDP-only parity validation.

Focus:
- Reliable DDP setup / teardown
- Correct loss aggregation for logging
- Rank-0 checkpointing with RNG state capture
- Support tuple batches from existing training data loaders

This module intentionally keeps moving parts minimal for Phase 4:
DDP for gradient sync first, expert-parallel sharding later.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from hgsel.distributed import dist_utils
from hgsel.training.trainer import Trainer, TrainingConfig


def _default_backend_for_device(device: torch.device) -> str:
    if device.type == "cuda":
        return "nccl"
    return "gloo"


class DistributedTrainer(Trainer):
    """DDP-first distributed training wrapper.

    This extends the base Trainer and adds:
    - process-group setup helpers
    - DDP wrapping (optional when world_size > 1)
    - distributed loss aggregation for logging
    - rank-0 checkpointing with RNG state capture
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        optimizer: Optional[optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        aux_loss_fn: Optional[nn.Module] = None,
        auto_init_from_env: bool = True,
    ) -> None:
        # Resolve / initialize distributed from torchrun env early so device selection is correct.
        self.rank = 0
        self.world_size = 1
        self.is_master = True
        self._ddp_wrapped = False

        if auto_init_from_env:
            env = dist_utils.resolve_dist_env()
            if env.world_size > 1 and not dist_utils.is_dist_initialized():
                dist_utils.init_distributed(env)

        if device is None:
            device = dist_utils.get_device()

        # Avoid multi-rank wandb initialization by default unless explicitly enabled on rank 0.
        if dist_utils.get_world_size() > 1 and dist_utils.get_rank() != 0:
            config.use_wandb = False

        super().__init__(model=model, config=config, aux_loss_fn=aux_loss_fn)

        # Override device chosen by base trainer if needed and ensure model is on the intended device.
        self.device = torch.device(device)
        self.model.to(self.device)

        # Optionally replace the base optimizer with the caller-provided one.
        if optimizer is not None:
            self.optimizer = optimizer

        self.rank = dist_utils.get_rank()
        self.world_size = dist_utils.get_world_size()
        self.is_master = self.rank == 0

        self._wrap_with_ddp_if_needed()

        if self.is_master:
            print(
                f"[Rank {self.rank}] DistributedTrainer ready "
                f"(world_size={self.world_size}, device={self.device}, ddp={self._ddp_wrapped})"
            )

    def _unwrap_model(self) -> nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _wrap_with_ddp_if_needed(self) -> None:
        if self._ddp_wrapped:
            return
        if self.world_size <= 1 or not dist_utils.is_dist_initialized():
            return

        if isinstance(self.model, DDP):
            self._ddp_wrapped = True
            return

        if self.device.type == "cuda":
            self.model = DDP(self.model, device_ids=[self.device.index])
        else:
            self.model = DDP(self.model)
        self._ddp_wrapped = True

    def setup_distributed(
        self,
        rank: int,
        world_size: int,
        backend: Optional[str] = None,
        master_addr: str = "localhost",
        master_port: str = "12355",
    ) -> None:
        """Initialize process group and wrap model for DDP.

        Safe to call even when already initialized (e.g., auto-init path).
        """
        if backend is None:
            backend = _default_backend_for_device(self.device)

        os.environ.setdefault("MASTER_ADDR", master_addr)
        os.environ.setdefault("MASTER_PORT", master_port)
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", os.environ.get("LOCAL_RANK", str(rank)))

        if not dist_utils.is_dist_initialized() and world_size > 1:
            env = dist_utils.resolve_dist_env(
                default_backend=backend,
                rank=rank,
                world_size=world_size,
                local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            )
            dist_utils.init_distributed(env)

        self.rank = dist_utils.get_rank()
        self.world_size = dist_utils.get_world_size()
        self.is_master = self.rank == 0
        self.device = dist_utils.get_device()
        self._unwrap_model().to(self.device)
        self._wrap_with_ddp_if_needed()

        if self.is_master:
            print(
                f"[Rank {self.rank}] Process group initialized "
                f"(world_size={self.world_size}, backend={dist_utils.get_backend()}, device={self.device})"
            )

    def _batch_to_device(self, batch: Any) -> Any:
        if isinstance(batch, dict):
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        if isinstance(batch, (tuple, list)):
            return tuple(
                x.to(self.device) if isinstance(x, torch.Tensor) else x
                for x in batch
            )
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        raise TypeError(f"Unsupported batch type: {type(batch)!r}")

    def _compute_loss(self, batch: Any) -> torch.Tensor:
        batch = self._batch_to_device(batch)

        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            labels = batch.get("labels")
        elif isinstance(batch, tuple):
            if len(batch) != 2:
                raise ValueError(f"Expected (input_ids, labels) tuple, got length={len(batch)}")
            input_ids, labels = batch
        else:
            input_ids = batch
            labels = None

        outputs = self.model(input_ids)

        if labels is None:
            if hasattr(outputs, "loss"):
                return outputs.loss
            if torch.is_tensor(outputs) and outputs.ndim == 0:
                return outputs
            raise ValueError("Labels missing and model output does not contain scalar loss")

        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
        vocab_size = logits.size(-1)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
        )

    def _compute_auxiliary_loss(self) -> tuple[torch.Tensor, int]:
        """Compute HGSEL auxiliary load-balancing loss across layers.

        Returns:
            (aux_loss, num_layers_used)
        """
        zero = torch.tensor(0.0, device=self.device)
        if self.aux_loss_fn is None:
            return zero, 0

        aux_weight = float(self.get_aux_loss_weight())
        if aux_weight <= 0.0:
            return zero, 0

        layer_losses = []
        base_model = self._unwrap_model()
        for module in base_model.modules():
            expert_load_ema = getattr(module, "expert_load_ema", None)
            if not isinstance(expert_load_ema, torch.Tensor):
                continue
            layer_loss = self.aux_loss_fn(expert_load_ema)
            if not torch.is_tensor(layer_loss):
                layer_loss = torch.tensor(float(layer_loss), device=self.device)
            layer_losses.append(layer_loss.to(device=self.device, dtype=torch.float32))

        if not layer_losses:
            return zero, 0

        aux_loss = torch.stack(layer_losses).mean() * aux_weight
        return aux_loss, len(layer_losses)

    def train_step(self, batch: Any) -> Dict[str, float]:
        """Execute one training step (DDP handles gradient sync when wrapped)."""
        self.model.train()

        self.optimizer.zero_grad(set_to_none=True)
        main_loss = self._compute_loss(batch)
        aux_loss, aux_layers = self._compute_auxiliary_loss()
        total_loss = main_loss + aux_loss
        total_loss.backward()

        if self.config.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._unwrap_model().parameters(), self.config.clip_grad_norm)

        # Manual all-reduce only if not using DDP but dist is initialized (fallback mode).
        if dist_utils.is_dist_initialized() and self.world_size > 1 and not self._ddp_wrapped:
            self._all_reduce_gradients()

        self.optimizer.step()

        # Distributed mean loss for logging/comparison.
        main_for_logging = main_loss.detach().clone()
        aux_for_logging = aux_loss.detach().clone()
        total_for_logging = total_loss.detach().clone()
        dist_utils.all_reduce_mean(main_for_logging)
        dist_utils.all_reduce_mean(aux_for_logging)
        dist_utils.all_reduce_mean(total_for_logging)

        metrics = {
            "loss": float(main_for_logging.item()),
            "aux_loss": float(aux_for_logging.item()),
            "total_loss": float(total_for_logging.item()),
            "aux_loss_layers": int(aux_layers),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }

        base_model = self._unwrap_model()
        if hasattr(base_model, "get_routing_diagnostics"):
            diag = base_model.get_routing_diagnostics()
            if diag and "entropy" in diag:
                entropy = torch.tensor(float(diag["entropy"]), device=self.device)
                dist_utils.all_reduce_mean(entropy)
                metrics["entropy"] = float(entropy.item())
            if diag and "expert_load" in diag and torch.is_tensor(diag["expert_load"]):
                expert_load = diag["expert_load"].detach().to(self.device).float()
                dist_utils.all_reduce_mean(expert_load)
                metrics["max_expert_load"] = float(expert_load.max().item())

        return metrics

    def validation_step(self, batch: Any) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            loss = self._compute_loss(batch)
            dist_utils.all_reduce_mean(loss)
        return loss

    def _all_reduce_gradients(self) -> None:
        if self.world_size == 1 or not dist_utils.is_dist_initialized():
            return
        for param in self._unwrap_model().parameters():
            if param.grad is not None:
                dist_utils.all_reduce_mean(param.grad)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "val_perplexity": [],
            "learning_rates": [],
        }

        total_steps = max(
            1,
            len(train_loader) * self.config.num_epochs // max(self.config.gradient_accumulation_steps, 1),
        )
        if total_steps > self.config.warmup_steps:
            self.setup_scheduler(total_steps)
        else:
            self.lr_scheduler = None

        for epoch in range(self.config.num_epochs):
            self.global_epoch = epoch
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            dist_utils.barrier()

            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                metrics = self.train_step(batch)
                epoch_loss += metrics["loss"]
                num_batches += 1
                results["learning_rates"].append(metrics["learning_rate"])

                self.global_step += 1
                self._apply_warmup()
                if self.lr_scheduler is not None and self.global_step >= self.config.warmup_steps:
                    self.lr_scheduler.step()

                if self.is_master and batch_idx % self.config.log_interval == 0:
                    avg_loss = epoch_loss / max(num_batches, 1)
                    msg = (
                        f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                        f"Batch {batch_idx}/{len(train_loader)} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Aux: {metrics.get('aux_loss', 0.0):.4f} | "
                        f"Total: {metrics.get('total_loss', metrics['loss']):.4f} | "
                        f"Avg: {avg_loss:.4f} | "
                        f"LR: {metrics['learning_rate']:.6f}"
                    )
                    if "entropy" in metrics:
                        msg += f" | Entropy: {metrics['entropy']:.4f}"
                    print(msg)

            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            results["train_loss"].append(avg_epoch_loss)

            # Validation cadence: treat val_interval as epochs for this wrapper.
            if val_loader is not None and (epoch % max(self.config.val_interval, 1) == 0):
                val_loss = self._validate(val_loader)
                val_loss_value = float(val_loss.item())
                results["val_loss"].append(val_loss_value)
                results["val_perplexity"].append(float(val_loss.exp().item()))

                if self.is_master:
                    print(
                        f"Epoch {epoch + 1} Validation | "
                        f"Loss: {val_loss_value:.4f} | Perplexity: {val_loss.exp().item():.2f}"
                    )
                    checkpoint_path = Path(self.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
                    self.save_checkpoint(checkpoint_path)

            dist_utils.barrier()

        if self.is_master:
            print("Training complete!")
        return results

    def _validate(self, val_loader: DataLoader) -> torch.Tensor:
        total_loss = 0.0
        num_batches = 0
        for batch in val_loader:
            loss = self.validation_step(batch)
            total_loss += float(loss.item())
            num_batches += 1

        avg_loss = torch.tensor(total_loss / max(num_batches, 1), device=self.device)
        dist_utils.all_reduce_mean(avg_loss)
        return avg_loss

    def _rng_state_dict(self) -> Dict[str, Any]:
        rng: Dict[str, Any] = {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                rng["torch_cuda_all"] = torch.cuda.get_rng_state_all()
            except RuntimeError:
                pass
        return rng

    def _load_rng_state_dict(self, rng: Dict[str, Any]) -> None:
        if not rng:
            return
        if "python" in rng:
            random.setstate(rng["python"])
        if "torch_cpu" in rng:
            torch.set_rng_state(rng["torch_cpu"])
        if "torch_cuda_all" in rng and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(rng["torch_cuda_all"])
            except RuntimeError:
                pass

    def save_checkpoint(self, path: str | Path) -> None:
        """Save checkpoint on rank 0 with model/optimizer/RNG state."""
        if not self.is_master:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self._unwrap_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else vars(self.config),
            "rank": self.rank,
            "world_size": self.world_size,
            "global_step": self.global_step,
            "global_epoch": self.global_epoch,
            "rng_state": self._rng_state_dict(),
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self._unwrap_model().load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        self.global_step = int(checkpoint.get("global_step", self.global_step))
        self.global_epoch = int(checkpoint.get("global_epoch", self.global_epoch))
        self._load_rng_state_dict(checkpoint.get("rng_state", {}))

        if self.is_master:
            print(f"Loaded checkpoint from {path}")

    def cleanup(self) -> None:
        dist_utils.cleanup_distributed()
        if self.is_master:
            print("Cleanup complete!")
