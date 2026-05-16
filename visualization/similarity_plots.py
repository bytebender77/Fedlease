"""Cosine similarity and distance heatmap visualisations."""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_similarity_heatmap(
    sim_matrix: np.ndarray,
    output_path: str,
    client_labels: Optional[List[str]] = None,
    title: str = "Client LoRA B-Matrix Cosine Similarity",
) -> None:
    """
    Visualise pairwise cosine similarity between clients as a heatmap.

    Paper Figure 7(b): distance matrix from cosine similarity of LoRA B matrices.
    """
    N = sim_matrix.shape[0]
    labels = client_labels or [f"C{i}" for i in range(N)]

    fig, ax = plt.subplots(figsize=(max(6, N * 0.7), max(5, N * 0.65)))
    mask = np.zeros_like(sim_matrix, dtype=bool)

    sns.heatmap(
        sim_matrix,
        ax=ax,
        annot=(N <= 16),
        fmt=".2f",
        cmap="YlOrRd",
        vmin=-1.0,
        vmax=1.0,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.3,
        cbar_kws={"label": "Cosine Similarity"},
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Client")
    ax.set_ylabel("Client")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distance_heatmap(
    dist_matrix: np.ndarray,
    output_path: str,
    client_labels: Optional[List[str]] = None,
    cluster_labels: Optional[np.ndarray] = None,
    title: str = "Client Distance Matrix (1 − Cosine Similarity)",
) -> None:
    """
    Visualise pairwise distance matrix, optionally sorted by cluster.

    Reproduces Figure 7(b) from the FedLEASE paper.
    """
    N = dist_matrix.shape[0]
    labels = client_labels or [f"C{i}" for i in range(N)]

    # Sort by cluster label if provided
    if cluster_labels is not None:
        order = np.argsort(cluster_labels)
        dist_matrix = dist_matrix[np.ix_(order, order)]
        labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(6, N * 0.7), max(5, N * 0.65)))
    sns.heatmap(
        dist_matrix,
        ax=ax,
        annot=(N <= 16),
        fmt=".2f",
        cmap="viridis_r",
        vmin=0.0,
        vmax=1.0,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.3,
        cbar_kws={"label": "Distance (1 − cos sim)"},
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_similarity(
    layer_sim: np.ndarray,
    output_path: str,
    client_pairs: Optional[List[str]] = None,
    title: str = "Per-Layer Cosine Similarity Between Client Pairs",
) -> None:
    """
    Plot average cosine similarity per layer for selected client pairs.
    Reproduces Figure 3(b) from the FedLEASE paper.

    layer_sim : [L, N, N]
    """
    L, N, _ = layer_sim.shape
    # Select a subset of pairs for readability
    pairs_to_show = []
    for i in range(min(N, 4)):
        for j in range(i + 1, min(N, 4)):
            pairs_to_show.append((i, j))

    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.cm.get_cmap("tab10", len(pairs_to_show))

    for idx, (i, j) in enumerate(pairs_to_show):
        sims = layer_sim[:, i, j]
        label = f"C{i} vs C{j}"
        ax.plot(range(L), sims, marker="o", markersize=3, label=label, color=cmap(idx))
        ax.axhline(np.mean(sims), color=cmap(idx), linestyle="--", alpha=0.5,
                   label=f"avg={np.mean(sims):.3f}")

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
