"""
Federated client for FedLEASE.

Responsibilities
----------------
Initialisation phase:
  1. Train a WarmupFinBERT (single LoRA) for E_warmup epochs on local data.
  2. Return B matrices to the server for clustering.

Iterative training phase (per round):
  1. Receive M expert states + cluster assignment from server.
  2. Update FedLEASEFinBERT with received expert states.
  3. Set assigned expert and freeze others.
  4. Train assigned expert + router for E_local epochs.
  5. Return (assigned_expert_state, router_state) to server.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from models.finbert_lora_moe import FedLEASEFinBERT, WarmupFinBERT
from utils.logging_utils import get_logger

logger = get_logger("fedlease.client")


class FederatedClient:
    """
    A single federated learning client.

    Parameters
    ----------
    client_id : int
    model_name : str        Pre-trained model identifier.
    train_loader : DataLoader
    val_loader : DataLoader
    device : torch.device
    config : dict-like      Must contain lora, training, federated sub-configs.
    """

    def __init__(
        self,
        client_id: int,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Any,
    ) -> None:
        self.client_id = client_id
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        self.lora_rank = config.lora.rank
        self.lora_alpha = config.lora.alpha
        self.lora_dropout = config.lora.dropout
        self.num_labels = config.model.num_labels
        self.lr = float(config.training.learning_rate)
        self.max_grad_norm = float(config.training.max_grad_norm)
        self.mixed_precision = config.training.mixed_precision

        # Dataset size for weighted aggregation
        self.dataset_size = len(train_loader.dataset)

        self.warmup_model: Optional[WarmupFinBERT] = None
        self.fedlease_model: Optional[FedLEASEFinBERT] = None
        self.assigned_expert_idx: int = 0

        # Training statistics
        self.train_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Initialisation phase
    # ------------------------------------------------------------------

    def warmup_train(self, n_epochs: int) -> None:
        """
        Train a single-LoRA WarmupFinBERT for E warmup epochs.
        Result is stored in self.warmup_model.
        """
        logger.info(f"[Client {self.client_id}] Warmup training ({n_epochs} epochs)")

        self.warmup_model = WarmupFinBERT(
            model_name=self.model_name,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            num_labels=self.num_labels,
        ).to(self.device)

        self._train_model(self.warmup_model, n_epochs, phase="warmup")

    def get_b_matrices(self) -> List[torch.Tensor]:
        """Return B matrices from warmup LoRA (for server clustering)."""
        if self.warmup_model is None:
            raise RuntimeError("warmup_train() must be called before get_b_matrices()")
        return [b.cpu() for b in self.warmup_model.get_b_matrices()]

    def get_warmup_ab_matrices(self):
        """Return (A, B) tuples per layer for expert initialisation."""
        if self.warmup_model is None:
            raise RuntimeError("warmup_train() must be called first")
        return self.warmup_model.get_ab_matrices_per_layer()

    # ------------------------------------------------------------------
    # Iterative training phase
    # ------------------------------------------------------------------

    def setup_fedlease_model(
        self,
        n_experts: int,
        expert_states: List[Dict],
        router_state: Optional[Dict],
        assigned_expert_idx: int,
    ) -> None:
        """
        Initialise (or update) the FedLEASEFinBERT model with server-sent parameters.

        Called at the beginning of each communication round.
        """
        self.assigned_expert_idx = assigned_expert_idx

        if self.fedlease_model is None:
            self.fedlease_model = FedLEASEFinBERT(
                model_name=self.model_name,
                n_experts=n_experts,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                num_labels=self.num_labels,
                assigned_expert_idx=assigned_expert_idx,
            ).to(self.device)
        else:
            self.fedlease_model.set_trainable_expert(assigned_expert_idx)

        # Load all M expert states from server
        for e, st in enumerate(expert_states):
            if st is not None:
                self.fedlease_model.set_expert_state(e, {
                    k: v.to(self.device) for k, v in st.items()
                })

        # Load router state from server (if available)
        if router_state is not None:
            self.fedlease_model.set_router_state({
                k: v.to(self.device) for k, v in router_state.items()
            })

        # Ensure only assigned expert + router are trainable
        self.fedlease_model.set_trainable_expert(assigned_expert_idx)

    def local_train(self, n_epochs: int) -> Dict:
        """
        Train the FedLEASE model for n_epochs on local data.

        Returns a stats dict with train/val metrics.
        """
        if self.fedlease_model is None:
            raise RuntimeError("setup_fedlease_model() must be called first")

        logger.info(
            f"[Client {self.client_id}] Local training "
            f"(expert={self.assigned_expert_idx}, epochs={n_epochs})"
        )
        stats = self._train_model(self.fedlease_model, n_epochs, phase="fedlease")
        self.train_history.append(stats)
        return stats

    def get_upload_payload(self) -> Tuple[int, Dict, Dict]:
        """
        Return (client_id, expert_state, router_state) for upload to server.
        Only the assigned expert is uploaded.
        """
        expert_state = {
            k: v.cpu()
            for k, v in self.fedlease_model.get_expert_state(self.assigned_expert_idx).items()
        }
        router_state = {
            k: v.cpu()
            for k, v in self.fedlease_model.get_router_state().items()
        }
        return self.client_id, expert_state, router_state

    # ------------------------------------------------------------------
    # Internal training loop
    # ------------------------------------------------------------------

    def _collect_load_balance_loss(self, model: nn.Module) -> torch.Tensor:
        """Sum load-balancing auxiliary loss across any AdaptiveTopMRouter
        modules in the model. Returns 0 tensor if no router is present
        (e.g. during warmup when there's only a single LoRA expert).
        """
        try:
            from models.adaptive_router import AdaptiveTopMRouter
        except Exception:
            return torch.tensor(0.0, device=self.device)
        total = torch.tensor(0.0, device=self.device)
        count = 0
        for m in model.modules():
            if isinstance(m, AdaptiveTopMRouter):
                total = total + m.load_balance_loss()
                count += 1
        return total / max(count, 1)

    def _compute_class_weights(self, n_classes: int = 3) -> torch.Tensor:
        """Softened inverse-frequency class weights.

        Earlier we used full inverse-frequency (w_c ∝ 1/n_c) which on
        PhraseBank's 60/25/15 distribution gave weights {0.55, 1.33, 2.22}
        — too aggressive. It pulled the classifier into a balanced-prediction
        degeneracy (every client predicting ~33/33/33 → flat accuracy ≈ 76%).

        Softer choice: weights ∝ 1/sqrt(n_c), normalised so the mean weight
        is 1.0 and the max is capped at 1.5. This corrects mild imbalance
        without flattening the dominant class.
        """
        from collections import Counter
        import math

        counts: Counter = Counter()
        for batch in self.train_loader:
            counts.update(batch["labels"].tolist())

        # Raw inverse-sqrt-frequency
        raw = [1.0 / math.sqrt(max(counts.get(c, 0), 1)) for c in range(n_classes)]
        # Normalise so mean = 1.0
        mean_raw = sum(raw) / n_classes
        normed = [r / mean_raw for r in raw]
        # Cap at 1.5 so the minority class doesn't dominate
        capped = [min(w, 1.5) for w in normed]
        return torch.tensor(capped, dtype=torch.float32, device=self.device)

    def _train_model(
        self,
        model: nn.Module,
        n_epochs: int,
        phase: str = "train",
    ) -> Dict:
        model.train()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.lr, weight_decay=0.01)

        total_steps = n_epochs * len(self.train_loader)
        warmup_steps = max(1, int(total_steps * 0.1))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        scaler = GradScaler() if self.mixed_precision and self.device.type == "cuda" else None

        # Class-weighted CE + mild label smoothing.
        # smoothing=0.1 was too aggressive on PhraseBank (60% neutral skew):
        # it floored the loss above zero and prevented confident neutral
        # predictions, collapsing per-client variance. 0.03 is the gentler
        # standard (cf. Müller et al. 2019, "When does label smoothing help?").
        class_weights = self._compute_class_weights(n_classes=3)
        loss_fct = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)

        epoch_stats = []
        for epoch in range(n_epochs):
            total_loss = 0.0
            n_correct = 0
            n_total = 0

            for batch in self.train_loader:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                if scaler is not None:
                    with autocast():
                        out = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels,
                        )
                        logits = out["logits"]
                        # Override the model's internal CE loss with our
                        # class-weighted + smoothed loss
                        ce_loss = loss_fct(logits.float(), labels)
                        # Add load-balancing auxiliary loss if router exposes it
                        aux = self._collect_load_balance_loss(model)
                        loss = ce_loss + 0.01 * aux
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels,
                    )
                    logits = out["logits"]
                    ce_loss = loss_fct(logits, labels)
                    aux = self._collect_load_balance_loss(model)
                    loss = ce_loss + 0.01 * aux
                    loss.backward()
                    nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                total_loss += loss.item()

                preds = logits.argmax(dim=-1)
                n_correct += (preds == labels).sum().item()
                n_total += labels.size(0)

            train_acc = n_correct / n_total
            avg_loss = total_loss / len(self.train_loader)
            epoch_stats.append({"epoch": epoch, "loss": avg_loss, "accuracy": train_acc})

        val_acc = self._evaluate(model)
        return {
            "phase": phase,
            "client_id": self.client_id,
            "epochs": n_epochs,
            "epoch_stats": epoch_stats,
            "val_accuracy": val_acc,
            "dataset_size": self.dataset_size,
        }

    def _evaluate(self, model: nn.Module) -> float:
        model.eval()
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                preds = out["logits"].argmax(dim=-1)
                n_correct += (preds == labels).sum().item()
                n_total += labels.size(0)
        model.train()
        return n_correct / n_total if n_total > 0 else 0.0

    def evaluate_on_loader(self, loader: DataLoader) -> Dict:
        """Full evaluation on an arbitrary loader; returns accuracy + predictions."""
        if self.fedlease_model is None:
            raise RuntimeError("Model not initialised")
        self.fedlease_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"]
                out = self.fedlease_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                preds = out["logits"].argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())
        self.fedlease_model.train()
        from utils.metrics import compute_metrics
        return compute_metrics(all_preds, all_labels)
