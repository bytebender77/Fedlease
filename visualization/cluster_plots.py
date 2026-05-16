"""Clustering visualisation: silhouette scores, dendrogram, 2-D projection."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram


CLUSTER_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#34495E",
]


def plot_silhouette_scores(
    silhouette_scores: Dict[int, float],
    optimal_k: int,
    output_path: str,
    title: str = "Silhouette Score vs Number of Clusters",
) -> None:
    """
    Line plot of silhouette scores for k = 2 … M_max with optimal k highlighted.
    Reproduces Figure 7(a) from the FedLEASE paper.
    """
    ks = sorted(silhouette_scores.keys())
    scores = [silhouette_scores[k] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, scores, marker="o", linewidth=2, color="#3498DB", label="Silhouette score")
    ax.axvline(optimal_k, color="#E74C3C", linestyle="--", linewidth=1.5,
               label=f"Optimal k={optimal_k}")
    ax.scatter([optimal_k], [silhouette_scores[optimal_k]], color="#E74C3C", s=100, zorder=5)
    ax.annotate(
        f"  S={silhouette_scores[optimal_k]:.4f}",
        (optimal_k, silhouette_scores[optimal_k]),
        fontsize=9, color="#E74C3C",
    )
    ax.set_xlabel("Number of Clusters k")
    ax.set_ylabel("Average Silhouette Score")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ks)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    output_path: str,
    client_labels: Optional[List[str]] = None,
    title: str = "Hierarchical Clustering Dendrogram",
    color_threshold: Optional[float] = None,
) -> None:
    """
    Plot hierarchical clustering dendrogram.
    Reproduces Figure 7(c) from the FedLEASE paper.
    """
    N = linkage_matrix.shape[0] + 1
    labels = client_labels or [f"C{i}" for i in range(N)]

    fig, ax = plt.subplots(figsize=(max(8, N * 0.8), 5))
    scipy_dendrogram(
        linkage_matrix,
        labels=labels,
        ax=ax,
        color_threshold=color_threshold,
        leaf_rotation=45,
        leaf_font_size=9,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Distance (1 − cos sim)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_2d(
    distance_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: str,
    client_labels: Optional[List[str]] = None,
    source_tags: Optional[List[str]] = None,
    title: str = "Client Clustering (MDS Projection)",
) -> None:
    """
    2-D MDS projection of clients coloured by cluster assignment.
    """
    from sklearn.manifold import MDS

    N = distance_matrix.shape[0]
    labels = client_labels or [f"C{i}" for i in range(N)]
    dist = np.clip(distance_matrix, 0.0, None)

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress=False)
    coords = mds.fit_transform(dist)

    fig, ax = plt.subplots(figsize=(7, 6))
    n_clusters = int(cluster_labels.max()) + 1

    for exp_idx in range(n_clusters):
        mask = (cluster_labels == exp_idx)
        color = CLUSTER_COLORS[exp_idx % len(CLUSTER_COLORS)]
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, s=120, edgecolors="black", linewidths=0.5,
            label=f"Expert {exp_idx}",
            zorder=3,
        )
        for i in np.where(mask)[0]:
            marker = "★" if (source_tags and "phrase" in source_tags[i]) else "●"
            ax.annotate(
                f"{labels[i]}{marker}",
                (coords[i, 0], coords[i, 1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("MDS Dim 1")
    ax.set_ylabel("MDS Dim 2")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_summary(
    silhouette_scores: Dict[int, float],
    optimal_k: int,
    dist_matrix: np.ndarray,
    linkage_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    output_dir: str,
    client_labels: Optional[List[str]] = None,
    source_tags: Optional[List[str]] = None,
) -> None:
    """Generate the full suite of cluster visualisations."""
    os.makedirs(output_dir, exist_ok=True)
    plot_silhouette_scores(
        silhouette_scores, optimal_k,
        os.path.join(output_dir, "silhouette_scores.png"),
    )
    plot_dendrogram(
        linkage_matrix,
        os.path.join(output_dir, "dendrogram.png"),
        client_labels=client_labels,
    )
    plot_distance_heatmap_wrapper(
        dist_matrix,
        os.path.join(output_dir, "distance_heatmap.png"),
        client_labels=client_labels,
        cluster_labels=cluster_labels,
    )
    plot_cluster_2d(
        dist_matrix, cluster_labels,
        os.path.join(output_dir, "cluster_2d_mds.png"),
        client_labels=client_labels,
        source_tags=source_tags,
    )


def plot_distance_heatmap_wrapper(dist_matrix, output_path, client_labels=None, cluster_labels=None):
    """Helper to avoid circular import."""
    from visualization.similarity_plots import plot_distance_heatmap
    plot_distance_heatmap(dist_matrix, output_path, client_labels, cluster_labels)
