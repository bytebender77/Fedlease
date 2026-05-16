"""Training curve and performance comparison visualisations."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = {
    "FedLEASE": "#E74C3C",
    "FedIT": "#3498DB",
    "FFA-LoRA": "#2ECC71",
    "FedDPA": "#F39C12",
    "FedSA": "#9B59B6",
    "IFCA+LoRA": "#1ABC9C",
}


def plot_training_curves(
    round_metrics: Dict[str, List[float]],
    output_path: str,
    title: str = "Training Curve",
    ylabel: str = "Validation Accuracy",
    xlabel: str = "Communication Round",
) -> None:
    """
    Plot per-round validation accuracy for one or multiple methods.

    round_metrics : {method_name: [acc per round]}
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for method, accs in round_metrics.items():
        rounds = list(range(1, len(accs) + 1))
        color = PALETTE.get(method, None)
        ax.plot(rounds, accs, marker="o", markersize=3, linewidth=2,
                label=method, color=color)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_client_accuracy_bars(
    per_client_accs: Dict[str, Dict[int, float]],
    output_path: str,
    title: str = "Per-Client Test Accuracy",
) -> None:
    """
    Grouped bar chart of per-client accuracy for multiple methods.

    per_client_accs : {method_name: {client_id: accuracy}}
    """
    methods = list(per_client_accs.keys())
    all_clients = sorted({cid for m in methods for cid in per_client_accs[m]})
    N = len(all_clients)
    M = len(methods)

    x = np.arange(N)
    bar_width = 0.8 / M
    fig, ax = plt.subplots(figsize=(max(8, N * 0.9), 5))
    cmap = plt.cm.get_cmap("Set1", M)

    for i, method in enumerate(methods):
        vals = [per_client_accs[method].get(cid, 0.0) * 100 for cid in all_clients]
        color = PALETTE.get(method, cmap(i))
        offset = (i - M / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=method, color=color, alpha=0.85)
        # Annotate top
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{cid}" for cid in all_clients])
    ax.set_xlabel("Client")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_communication_cost(
    bytes_per_round: List[float],
    output_path: str,
    title: str = "Communication Cost per Round",
) -> None:
    """Cumulative bytes transferred vs round."""
    cumulative = np.cumsum([b / 1e6 for b in bytes_per_round])
    rounds = list(range(1, len(bytes_per_round) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(rounds, [b / 1e6 for b in bytes_per_round], color="#3498DB", alpha=0.8)
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("MB Transferred")
    axes[0].set_title("Per-Round Upload")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].plot(rounds, cumulative, marker="o", markersize=3, color="#E74C3C", linewidth=2)
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Cumulative MB")
    axes[1].set_title("Cumulative Communication Cost")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(
    client_loss_histories: Dict[int, List[float]],
    output_path: str,
    title: str = "Client Training Loss",
) -> None:
    """Per-client training loss across communication rounds."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.get_cmap("tab10", len(client_loss_histories))

    for idx, (cid, losses) in enumerate(client_loss_histories.items()):
        rounds = list(range(1, len(losses) + 1))
        ax.plot(rounds, losses, marker=".", markersize=3, linewidth=1.5,
                label=f"Client {cid}", color=cmap(idx), alpha=0.85)

    ax.set_xlabel("Round")
    ax.set_ylabel("Training Loss")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Plot normalised confusion matrix."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm, ax=ax, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        cbar_kws={"label": "Fraction"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
