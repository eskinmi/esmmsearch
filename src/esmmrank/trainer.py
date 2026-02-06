import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import TrainingConfig
from .loss import ESMMLoss, IPSWeighter
from .metric import MetricsCalculator
from .model import TwoTowerESMM


class EarlyStopping:
    """
    Early stopping to halt training when validation metric stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs to wait before stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    mode : str
        'max' for metrics where higher is better, 'min' for loss.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """
    Trainer for the Two-Tower ESMM model.

    Handles training loop, validation, early stopping, LR scheduling,
    gradient clipping, and metric computation.

    Parameters
    ----------
    model : TwoTowerESMM
        Model to train.
    config : TrainingConfig
        Training configuration.
    """

    def __init__(self, model: TwoTowerESMM, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        ips_weighter = IPSWeighter(eta=config.ipsw_eta) if config.ipsw_enabled else None

        self.loss_fn = ESMMLoss(
            gamma=config.focal_gamma,
            alpha=config.focal_alpha,
            ctr_weight=config.ctr_loss_weight,
            ctcvr_weight=config.ctcvr_loss_weight,
            ips_weighter=ips_weighter,
        )

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
            min_lr=config.lr_scheduler_min_lr,
        )

        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode="max",
        )

        self.metrics = MetricsCalculator(
            ctr_loss_weight=config.ctr_loss_weight,
            ctcvr_loss_weight=config.ctcvr_loss_weight
        )
        self.best_model_state: dict | None = None

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_ctr_loss = 0.0
        total_ctcvr_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = self._move_to_device(batch)

            outputs = self.model(
                user_categorical=batch["user_categorical"],
                user_numerical=batch["user_numerical"],
                hotel_categorical=batch["hotel_categorical"],
                hotel_numerical=batch["hotel_numerical"],
                context_categorical=batch["context_categorical"],
                context_numerical=batch["context_numerical"],
            )

            losses = self.loss_fn(
                p_ctr=outputs["p_ctr"],
                p_ctcvr=outputs["p_ctcvr"],
                click_labels=batch["click_labels"],
                conversion_labels=batch["conversion_labels"],
                positions=batch.get("positions"),
            )

            self.optimizer.zero_grad()
            losses["total"].backward()

            if self.config.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )

            self.optimizer.step()

            total_loss += losses["total"].item()
            total_ctr_loss += losses["ctr"].item()
            total_ctcvr_loss += losses["ctcvr"].item()
            n_batches += 1

        return {
            "train_loss": total_loss / n_batches,
            "train_ctr_loss": total_ctr_loss / n_batches,
            "train_ctcvr_loss": total_ctcvr_loss / n_batches,
        }

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, prefix: str = "val") -> dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        self.metrics.reset()

        total_loss = 0.0
        n_batches = 0

        for batch in data_loader:
            batch = self._move_to_device(batch)

            outputs = self.model(
                user_categorical=batch["user_categorical"],
                user_numerical=batch["user_numerical"],
                hotel_categorical=batch["hotel_categorical"],
                hotel_numerical=batch["hotel_numerical"],
                context_categorical=batch["context_categorical"],
                context_numerical=batch["context_numerical"],
            )

            losses = self.loss_fn(
                p_ctr=outputs["p_ctr"],
                p_ctcvr=outputs["p_ctcvr"],
                click_labels=batch["click_labels"],
                conversion_labels=batch["conversion_labels"],
                positions=batch.get("positions"),
            )

            total_loss += losses["total"].item()
            n_batches += 1

            self.metrics.update(
                p_ctr=outputs["p_ctr"],
                p_ctcvr=outputs["p_ctcvr"],
                click_labels=batch["click_labels"],
                conversion_labels=batch["conversion_labels"],
                group_ids=batch["session_ids"],
            )

        metrics = self.metrics.compute(k_values=[10, 38])
        metrics["loss"] = total_loss / n_batches

        return {f"{prefix}_{k}": v for k, v in metrics.items()}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
        log_path: str | Path | None = None,
        tensorboard_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """
        Full training loop with early stopping and LR scheduling.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader
            Validation data loader.
        verbose : bool
            Whether to print progress.
        log_path : str | Path | None
            Path to save training metrics as JSON.
        tensorboard_dir : str | Path | None
            Directory for TensorBoard logs.

        Returns
        -------
        dict[str, list[float]]
            Training history with metrics per epoch.
        """
        writer = SummaryWriter(tensorboard_dir) if tensorboard_dir else None

        history: dict[str, list[float]] = {
            "epoch": [],
            "lr": [],
            "train_loss": [],
            "train_ctr_loss": [],
            "train_ctcvr_loss": [],
            "val_loss": [],
            "val_ctr_ndcg@10": [],
            "val_ctr_ndcg@38": [],
            "val_ctr_prauc": [],
            "val_ctcvr_ndcg@10": [],
            "val_ctcvr_ndcg@38": [],
            "val_ctcvr_prauc": [],
        }

        for epoch in range(self.config.num_epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader, prefix="val")

            current_lr = self.optimizer.param_groups[0]["lr"]
            val_ndcg38 = val_metrics["val_ctcvr_ndcg@38"]

            history["epoch"].append(epoch + 1)
            history["lr"].append(current_lr)
            history["train_loss"].append(train_metrics["train_loss"])
            history["train_ctr_loss"].append(train_metrics["train_ctr_loss"])
            history["train_ctcvr_loss"].append(train_metrics["train_ctcvr_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_ctr_ndcg@10"].append(val_metrics["val_ctr_ndcg@10"])
            history["val_ctr_ndcg@38"].append(val_metrics["val_ctr_ndcg@38"])
            history["val_ctr_prauc"].append(val_metrics["val_ctr_prauc"])
            history["val_ctcvr_ndcg@10"].append(val_metrics["val_ctcvr_ndcg@10"])
            history["val_ctcvr_ndcg@38"].append(val_metrics["val_ctcvr_ndcg@38"])
            history["val_ctcvr_prauc"].append(val_metrics["val_ctcvr_prauc"])

            if writer:
                writer.add_scalars("Loss", {
                    "train": train_metrics["train_loss"],
                    "val": val_metrics["val_loss"],
                }, epoch + 1)
                writer.add_scalars("Loss/Train", {
                    "total": train_metrics["train_loss"],
                    "ctr": train_metrics["train_ctr_loss"],
                    "ctcvr": train_metrics["train_ctcvr_loss"],
                }, epoch + 1)
                writer.add_scalars("NDCG@38", {
                    "ctr": val_metrics["val_ctr_ndcg@38"],
                    "ctcvr": val_metrics["val_ctcvr_ndcg@38"],
                }, epoch + 1)
                writer.add_scalars("NDCG@10", {
                    "ctr": val_metrics["val_ctr_ndcg@10"],
                    "ctcvr": val_metrics["val_ctcvr_ndcg@10"],
                }, epoch + 1)
                writer.add_scalars("PR-AUC", {
                    "ctr": val_metrics["val_ctr_prauc"],
                    "ctcvr": val_metrics["val_ctcvr_prauc"],
                }, epoch + 1)
                writer.add_scalar("LearningRate", current_lr, epoch + 1)
                writer.flush()

            self.scheduler.step(val_ndcg38)

            if self.early_stopping.best_score is None or val_ndcg38 > self.early_stopping.best_score:
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            if verbose:
                lr_str = f"lr={current_lr:.2e}" if current_lr < self.config.learning_rate else ""
                print(
                    f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} "
                    f"(CTR: {train_metrics['train_ctr_loss']:.4f}, CTCVR: {train_metrics['train_ctcvr_loss']:.4f}) | "
                    f"Val NDCG@38: {val_ndcg38:.4f}, NDCG@10: {val_metrics['val_ctcvr_ndcg@10']:.4f}, "
                    f"PR-AUC: {val_metrics['val_ctcvr_prauc']:.4f}"
                    + (f" | {lr_str}" if lr_str else "")
                )

            if log_path:
                self._save_history(history, log_path)

            if self.early_stopping(val_ndcg38, epoch):
                if verbose:
                    print(
                        f"\nEarly stopping at epoch {epoch + 1}. "
                        f"Best NDCG@38: {self.early_stopping.best_score:.4f} "
                        f"at epoch {self.early_stopping.best_epoch + 1}"
                    )
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"Restored best model from epoch {self.early_stopping.best_epoch + 1}")

        if log_path:
            self._save_history(history, log_path)
            if verbose:
                print(f"Training log saved to {log_path}")

        if writer:
            writer.close()
            if verbose:
                print(f"TensorBoard logs saved to {tensorboard_dir}")

        return history

    @staticmethod
    def _save_history(history: dict[str, list[float]], path: str | Path) -> None:
        """Save training history to JSON file."""
        with open(path, "w") as f:
            json.dump(history, f, indent=2)

    def _move_to_device(self, batch: dict) -> dict:
        """Move batch tensors to the configured device."""

        def move(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device)
            elif isinstance(x, dict):
                return {k: move(v) for k, v in x.items()}
            return x

        return {k: move(v) for k, v in batch.items()}

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
