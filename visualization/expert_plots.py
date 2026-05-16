"""Expert utilisation and selection visualisations."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_expert_utilisation(
    utilisation_per_client: Dict[int, np.ndarray],
    output_path: str,
    title: str = "Expert Utilisation per Client",
    n_experts: Optional[int] = None,
) -> None:
    """
    Bar chart of expert utilisation ratios per client.

    utilisation_per_client : {client_id: np.ndarray [M] utilisation ratios}
    """
    client_ids = sorted(utilisation_per_client.keys())
    M = n_experts or len(next(iter(utilisation_per_client.values())))

    x = np.arange(len(client_ids))
    bar_width = 0.8 / M
    colors = plt.cm.get_cmap("tab10", M)

    fig, ax = plt.subplots(figsize=(max(8, len(client_ids) * 0.9), 5))

    for e in range(M):
        vals = [utilisation_per_client[cid][e] if e < len(utilisation_per_client[cid]) else 0.0
                for cid in client_ids]
        offset = (e - M / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, label=f"Expert {e}", color=colors(e), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{cid}" for cid in client_ids])
    ax.set_xlabel("Client")
    ax.set_ylabel("Utilisation Ratio")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_expert_selection_heatmap(
    selection_matrix: np.ndarray,
    output_path: str,
    client_labels: Optional[List[str]] = None,
    layer_labels: Optional[List[str]] = None,
    title: str = "Expert Selection Frequency (Client × Layer)",
) -> None:
    """
    Heatmap of which expert is most used per (client, layer).
    Reproduces Figure 6(c) from the FedLEASE paper.

    selection_matrix : [N_clients, N_layers]  values = avg active expert index
    """
    N, L = selection_matrix.shape
    c_labels = client_labels or [f"C{i}" for i in range(N)]
    l_labels = layer_labels or [str(i) for i in range(L)]

    fig, ax = plt.subplots(figsize=(max(8, L * 0.5), max(5, N * 0.5)))
    sns.heatmap(
        selection_matrix,
        ax=ax,
        cmap="YlGnBu",
        xticklabels=l_labels,
        yticklabels=c_labels,
        annot=(N * L <= 200),
        fmt=".1f",
        cbar_kws={"label": "Avg Expert Index Used"},
        linewidths=0.1,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Client")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_num_active_experts_distribution(
    n_active_per_layer: Dict[str, List[float]],
    output_path: str,
    title: str = "Average Number of Active Experts per Layer",
) -> None:
    """
    Bar chart of avg number of unique experts selected per layer per client group.
    """
    clients = list(n_active_per_layer.keys())
    n_layers = len(next(iter(n_active_per_layer.values())))

    fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.4), 5))
    cmap = plt.cm.get_cmap("Set2", len(clients))

    for idx, (cname, vals) in enumerate(n_active_per_layer.items()):
        ax.plot(range(n_layers), vals, marker="o", markersize=4,
                label=cname, color=cmap(idx), linewidth=1.5)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Avg # Active Experts")
    ax.set_title(title)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
