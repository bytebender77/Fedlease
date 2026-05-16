"""Global test-set evaluator for FedLEASE."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from models.finbert_lora_moe import FedLEASEFinBERT
from utils.metrics import compute_metrics
from utils.logging_utils import get_logger

logger = get_logger("fedlease.evaluator")

LABEL_NAMES = ["negative", "neutral", "positive"]


class Evaluator:
    """
    Evaluate FedLEASE model on a held-out test set.

    For each client's model, uses the assigned expert and router to compute
    predictions on the test loader, then reports per-class and aggregate metrics.
    """

    def __init__(
        self,
        test_loader: DataLoader,
        device: torch.device,
        label_names: Optional[List[str]] = None,
    ) -> None:
        self.test_loader = test_loader
        self.device = device
        self.label_names = label_names or LABEL_NAMES

    def evaluate_model(
        self,
        model: FedLEASEFinBERT,
        round_idx: Optional[int] = None,
    ) -> Dict:
        """Run model on test_loader and compute full metrics."""
        model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"].tolist()

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                preds = out["logits"].argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels)

        model.train()
        metrics = compute_metrics(all_preds, all_labels, self.label_names)
        metrics["round"] = round_idx if round_idx is not None else -1

        prefix = f"[Round {round_idx}] " if round_idx is not None else ""
        logger.info(
            f"{prefix}Test — acc={metrics['accuracy']:.4f}  "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
        return metrics

    def evaluate_all_clients(
        self,
        client_models: Dict[int, FedLEASEFinBERT],
        round_idx: Optional[int] = None,
    ) -> Dict[str, Dict]:
        """Evaluate each client's personalised model and return results."""
        results: Dict[str, Dict] = {}
        for cid, model in client_models.items():
            m = self.evaluate_model(model, round_idx)
            m["client_id"] = cid
            results[f"client_{cid}"] = m
        return results

    def aggregate_metrics(self, per_client_metrics: Dict[str, Dict]) -> Dict:
        """Compute mean metrics across all clients."""
        keys = [k for k in next(iter(per_client_metrics.values())).keys()
                if isinstance(next(iter(per_client_metrics.values()))[k], float)]
        agg: Dict = {}
        for key in keys:
            vals = [per_client_metrics[c][key] for c in per_client_metrics]
            agg[f"mean_{key}"] = float(sum(vals) / len(vals))
            agg[f"min_{key}"] = float(min(vals))
            agg[f"max_{key}"] = float(max(vals))
        return agg
