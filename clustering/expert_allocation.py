"""
Silhouette-score-based optimal expert allocation.

FedLEASE paper §4.1:
  M = argmax_{2 ≤ k ≤ M_max} S(k)

After selecting M, expert parameters are initialised by averaging
the LoRA A and B matrices of all clients in each cluster (Eq. 3):
  A_j^expert = (1/|C_j^M|) Σ_{i ∈ C_j^M} A_i
  B_j^expert = (1/|C_j^M|) Σ_{i ∈ C_j^M} B_i
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .clustering import (
    run_agglomerative_clustering,
    compute_silhouette,
    compute_per_sample_silhouette,
)


class ExpertAllocator:
    """
    Determines the optimal number of LoRA experts and initialises them.

    Parameters
    ----------
    min_clusters : int
        Minimum number of clusters to evaluate (default 2).
    max_clusters : int
        Maximum number of clusters M_max (default 8).
    linkage : str
        Agglomerative clustering linkage method.
    """

    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 8,
        linkage: str = "average",
    ) -> None:
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.linkage = linkage

        # Results populated after allocate()
        self.optimal_k: int = 2
        self.silhouette_scores: Dict[int, float] = {}
        self.cluster_labels: np.ndarray = np.array([])
        self.cluster_map: Dict[int, List[int]] = {}  # expert_idx → [client_ids]

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def allocate(
        self,
        distance_matrix: np.ndarray,
        client_ids: Optional[List[int]] = None,
    ) -> Tuple[int, np.ndarray, Dict[int, float]]:
        """
        Find optimal M and cluster clients.

        Parameters
        ----------
        distance_matrix : np.ndarray [N, N]
        client_ids : list of int, optional
            If provided, used in cluster_map keys instead of 0…N-1.

        Returns
        -------
        optimal_k : int
        labels : np.ndarray [N]  cluster assignment per client
        silhouette_scores : dict {k: score}
        """
        N = distance_matrix.shape[0]
        ids = client_ids if client_ids is not None else list(range(N))

        # ── Special case: a single global expert (FedAvg-LoRA baseline)
        if self.max_clusters <= 1:
            optimal_k = 1
            best_labels = np.zeros(N, dtype=int)
            scores = {1: 0.0}    # silhouette undefined for k=1
            cluster_map: Dict[int, List[int]] = {0: list(ids)}
            self.optimal_k = optimal_k
            self.silhouette_scores = scores
            self.cluster_labels = best_labels
            self.cluster_map = cluster_map
            return optimal_k, best_labels, scores

        # Clamp max_clusters
        k_max = min(self.max_clusters, N - 1)
        k_min = max(self.min_clusters, 2)
        k_max = max(k_max, k_min)

        scores: Dict[int, float] = {}
        all_labels: Dict[int, np.ndarray] = {}

        for k in range(k_min, k_max + 1):
            labels = run_agglomerative_clustering(distance_matrix, k, self.linkage)
            score = compute_silhouette(distance_matrix, labels)
            scores[k] = score
            all_labels[k] = labels

        # Select k with highest silhouette
        optimal_k = max(scores, key=scores.get)
        best_labels = all_labels[optimal_k]

        # Build cluster → client map
        cluster_map: Dict[int, List[int]] = {e: [] for e in range(optimal_k)}
        for client_pos, expert_idx in enumerate(best_labels):
            cluster_map[int(expert_idx)].append(ids[client_pos])

        self.optimal_k = optimal_k
        self.silhouette_scores = scores
        self.cluster_labels = best_labels
        self.cluster_map = cluster_map

        return optimal_k, best_labels, scores

    # ------------------------------------------------------------------
    # Expert initialisation (Eq. 3)
    # ------------------------------------------------------------------

    def initialise_experts(
        self,
        client_ab_lists: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]],
        cluster_labels: np.ndarray,
        n_experts: int,
    ) -> List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Average LoRA parameters within each cluster to initialise expert weights.

        Parameters
        ----------
        client_ab_lists : List (N clients) of List (L layers) of
                          (A_q, B_q, A_v, B_v) tuples — from WarmupFinBERT.
        cluster_labels : [N] assignment of each client to an expert index.
        n_experts : M (number of experts).

        Returns
        -------
        expert_params : List (M) of List (L) of (A_q, B_q, A_v, B_v)
            Averaged LoRA parameters per expert per layer.
        """
        N = len(client_ab_lists)
        L = len(client_ab_lists[0])

        expert_params: List[Optional[List]] = [None] * n_experts

        for exp_idx in range(n_experts):
            member_indices = [i for i, lbl in enumerate(cluster_labels) if lbl == exp_idx]
            if not member_indices:
                # No clients in this cluster → use zeros
                zero_layer = []
                for _ in range(L):
                    ref = client_ab_lists[0][0]
                    zero_layer.append((
                        torch.zeros_like(ref[0]),
                        torch.zeros_like(ref[1]),
                        torch.zeros_like(ref[2]),
                        torch.zeros_like(ref[3]),
                    ))
                expert_params[exp_idx] = zero_layer
                continue

            # Average A_q, B_q, A_v, B_v per layer
            averaged: List[Tuple] = []
            for l in range(L):
                sum_Aq = torch.zeros_like(client_ab_lists[member_indices[0]][l][0])
                sum_Bq = torch.zeros_like(client_ab_lists[member_indices[0]][l][1])
                sum_Av = torch.zeros_like(client_ab_lists[member_indices[0]][l][2])
                sum_Bv = torch.zeros_like(client_ab_lists[member_indices[0]][l][3])

                for client_idx in member_indices:
                    Aq, Bq, Av, Bv = client_ab_lists[client_idx][l]
                    sum_Aq = sum_Aq + Aq.float()
                    sum_Bq = sum_Bq + Bq.float()
                    sum_Av = sum_Av + Av.float()
                    sum_Bv = sum_Bv + Bv.float()

                n = len(member_indices)
                averaged.append((
                    sum_Aq / n,
                    sum_Bq / n,
                    sum_Av / n,
                    sum_Bv / n,
                ))

            expert_params[exp_idx] = averaged

        return expert_params  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Per-sample silhouette for monitoring
    # ------------------------------------------------------------------

    def per_client_silhouette(
        self,
        distance_matrix: np.ndarray,
    ) -> np.ndarray:
        return compute_per_sample_silhouette(distance_matrix, self.cluster_labels)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> str:
        lines = ["=== Expert Allocation Report ==="]
        lines.append(f"Optimal M = {self.optimal_k}")
        lines.append("Silhouette scores per k:")
        for k in sorted(self.silhouette_scores):
            mark = " ←" if k == self.optimal_k else ""
            lines.append(f"  k={k}: {self.silhouette_scores[k]:.4f}{mark}")
        lines.append("Cluster membership:")
        for exp_idx, members in self.cluster_map.items():
            lines.append(f"  Expert {exp_idx}: clients {members}")
        return "\n".join(lines)
