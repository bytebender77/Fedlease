from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Sequence
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from collections import defaultdict


def compute_metrics(
    predictions: Sequence,
    labels: Sequence,
    label_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute accuracy, macro-F1, per-class F1 and confusion matrix."""
    preds = np.array(predictions)
    lbls = np.array(labels)

    acc = accuracy_score(lbls, preds)
    macro_f1 = f1_score(lbls, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(lbls, preds, average="weighted", zero_division=0)
    macro_prec = precision_score(lbls, preds, average="macro", zero_division=0)
    macro_rec = recall_score(lbls, preds, average="macro", zero_division=0)

    num_classes = len(np.unique(lbls))
    per_class_f1 = f1_score(lbls, preds, average=None, zero_division=0)

    metrics: Dict[str, float] = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
    }

    names = label_names or [str(i) for i in range(num_classes)]
    for i, name in enumerate(names):
        if i < len(per_class_f1):
            metrics[f"f1_{name}"] = float(per_class_f1[i])

    return metrics


class MetricTracker:
    """Accumulate and summarise per-round metrics across federated rounds."""

    def __init__(self) -> None:
        self._history: Dict[str, List[float]] = defaultdict(list)

    def update(self, metrics: Dict[str, float], prefix: str = "") -> None:
        for k, v in metrics.items():
            key = f"{prefix}/{k}" if prefix else k
            self._history[key].append(float(v))

    def get_history(self, key: str) -> List[float]:
        return self._history.get(key, [])

    def get_best(self, key: str, mode: str = "max") -> float:
        hist = self.get_history(key)
        if not hist:
            return float("-inf") if mode == "max" else float("inf")
        return max(hist) if mode == "max" else min(hist)

    def get_last(self, key: str) -> Optional[float]:
        hist = self.get_history(key)
        return hist[-1] if hist else None

    def summary(self) -> Dict[str, float]:
        return {k: v[-1] for k, v in self._history.items() if v}

    def all_keys(self) -> List[str]:
        return list(self._history.keys())
