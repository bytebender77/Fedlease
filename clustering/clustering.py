"""
Agglomerative Hierarchical Clustering of federated clients.

Operates on the precomputed distance matrix (1 − cosine similarity of B mats).
Uses sklearn with metric='precomputed' and linkage='average' as in the paper.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


def run_agglomerative_clustering(
    distance_matrix: np.ndarray,
    n_clusters: int,
    linkage: str = "average",
) -> np.ndarray:
    """
    Run agglomerative clustering for a fixed k.

    Parameters
    ----------
    distance_matrix : np.ndarray [N, N]
        Precomputed symmetric distance matrix.
    n_clusters : int
        Number of clusters k.
    linkage : str
        Linkage strategy for AgglomerativeClustering.

    Returns
    -------
    labels : np.ndarray [N]
        Cluster label for each client (0-indexed, 0 … k-1).
    """
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage=linkage,
    )
    labels = model.fit_predict(distance_matrix)
    return labels.astype(np.int32)


def compute_silhouette(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute silhouette coefficient for the given clustering.

    Returns the average silhouette score S(k) ∈ (−1, 1).
    Higher is better.
    """
    n_unique = len(np.unique(labels))
    n_samples = len(labels)

    # Silhouette is undefined for 1 cluster or N clusters
    if n_unique < 2 or n_unique >= n_samples:
        return -1.0

    return float(silhouette_score(distance_matrix, labels, metric="precomputed"))


def compute_per_sample_silhouette(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Return per-client silhouette scores s^k(i)."""
    from sklearn.metrics import silhouette_samples
    n_unique = len(np.unique(labels))
    if n_unique < 2:
        return np.zeros(len(labels))
    return silhouette_samples(distance_matrix, labels, metric="precomputed")


def build_linkage_matrix(distance_matrix: np.ndarray, method: str = "average") -> np.ndarray:
    """
    Build a scipy linkage matrix from a square distance matrix.
    Used for dendrogram visualisation.
    """
    from scipy.spatial.distance import squareform
    condensed = squareform(distance_matrix, checks=False)
    condensed = np.clip(condensed, 0.0, None)  # ensure non-negative
    return linkage(condensed, method=method)
