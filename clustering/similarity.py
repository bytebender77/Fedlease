"""
Inter-client cosine similarity computation on LoRA B matrices.

FedLEASE paper (Eq. 2):

  d(i, j) = (1 / |L|) Σ_{l ∈ L} (1 - (B_i^l · B_j^l) / (‖B_i^l‖ ‖B_j^l‖))

where L is the set of layers and B_i^l is the flattened B matrix of client i
at layer l (concatenation of query and value B matrices).

High cosine similarity → small distance → clients belong in the same cluster.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F


def compute_similarity_matrix(
    client_b_matrices: List[List[torch.Tensor]],
) -> np.ndarray:
    """
    Compute pairwise cosine similarity between clients.

    Parameters
    ----------
    client_b_matrices : List[List[Tensor]]
        Outer list: one entry per client (N clients).
        Inner list: one flattened B tensor per LoRA layer position.

    Returns
    -------
    sim_matrix : np.ndarray, shape [N, N]
        Symmetric matrix of average cosine similarities in [−1, 1].
    """
    N = len(client_b_matrices)
    sim_matrix = np.zeros((N, N), dtype=np.float64)

    # Normalise every layer vector for all clients (unit vectors)
    # client_norms[i][l] = normalised B_i^l   shape: [flat_dim]
    client_norms: List[List[torch.Tensor]] = []
    for b_list in client_b_matrices:
        normed = [F.normalize(b.float().cpu(), dim=0) for b in b_list]
        client_norms.append(normed)

    L = len(client_norms[0])  # number of layer positions

    for i in range(N):
        sim_matrix[i, i] = 1.0
        for j in range(i + 1, N):
            # Average cosine similarity across all layers
            cos_sum = 0.0
            for l in range(L):
                cos_l = torch.dot(client_norms[i][l], client_norms[j][l]).item()
                cos_sum += cos_l
            avg_cos = cos_sum / L
            sim_matrix[i, j] = avg_cos
            sim_matrix[j, i] = avg_cos

    return sim_matrix


def compute_distance_matrix(
    client_b_matrices: List[List[torch.Tensor]],
) -> np.ndarray:
    """
    Compute pairwise distance matrix (1 − cosine similarity).

    Returns
    -------
    dist_matrix : np.ndarray, shape [N, N]
        Symmetric distance matrix in [0, 2], with zeros on diagonal.
    """
    sim = compute_similarity_matrix(client_b_matrices)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    # Clamp numerical noise
    dist = np.clip(dist, 0.0, 2.0)
    return dist


def compute_distance_matrix_from_ab(
    client_ab_matrices: List[List],
) -> np.ndarray:
    """
    Cluster on the full LoRA update direction (B @ A) rather than B alone.

    Empirically this gives a much stronger silhouette signal — B alone can
    still be close to zero after short warmup, but B@A captures the input-
    conditioned update direction that the layer would apply, and is far
    more discriminative across register classes.

    Parameters
    ----------
    client_ab_matrices : List[List[Tuple[Aq, Bq, Av, Bv]]]
        Per client: list over layers of (A_query, B_query, A_value, B_value).

    Returns
    -------
    dist : np.ndarray [N, N]
    """
    N = len(client_ab_matrices)
    # Build the per-layer B@A flattened vectors for each client
    feats: List[List[torch.Tensor]] = []
    for ab_list in client_ab_matrices:
        client_layer_vecs: List[torch.Tensor] = []
        for entry in ab_list:
            # entry = (Aq, Bq, Av, Bv) for one layer
            Aq, Bq, Av, Bv = (t.float().cpu() for t in entry)
            # B @ A has shape [out, in]; flatten gives the full update direction
            ba_q = (Bq @ Aq).flatten()
            ba_v = (Bv @ Av).flatten()
            # Concatenate query + value update vectors
            client_layer_vecs.append(torch.cat([ba_q, ba_v], dim=0))
        feats.append(client_layer_vecs)

    # Normalise per layer per client
    feats_norm = [
        [F.normalize(v, dim=0) for v in layer_vecs]
        for layer_vecs in feats
    ]
    L = len(feats_norm[0])

    sim = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        sim[i, i] = 1.0
        for j in range(i + 1, N):
            cos_sum = 0.0
            for l in range(L):
                cos_sum += torch.dot(feats_norm[i][l], feats_norm[j][l]).item()
            avg_cos = cos_sum / L
            sim[i, j] = avg_cos
            sim[j, i] = avg_cos

    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return np.clip(dist, 0.0, 2.0)


def pairwise_layer_similarities(
    client_b_matrices: List[List[torch.Tensor]],
) -> np.ndarray:
    """
    Return per-layer cosine similarity matrix for visualisation.

    Returns
    -------
    layer_sim : np.ndarray, shape [L, N, N]
    """
    N = len(client_b_matrices)
    L = len(client_b_matrices[0])

    layer_sim = np.zeros((L, N, N), dtype=np.float64)
    client_norms = [
        [F.normalize(b.float().cpu(), dim=0) for b in b_list]
        for b_list in client_b_matrices
    ]

    for l in range(L):
        for i in range(N):
            layer_sim[l, i, i] = 1.0
            for j in range(i + 1, N):
                cos_ij = torch.dot(client_norms[i][l], client_norms[j][l]).item()
                layer_sim[l, i, j] = cos_ij
                layer_sim[l, j, i] = cos_ij

    return layer_sim
