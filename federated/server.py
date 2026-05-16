"""
Federated server for FedLEASE.

Server responsibilities
-----------------------
Initialisation phase:
  1. Collect B matrices from all clients.
  2. Compute pairwise cosine distance matrix.
  3. Find optimal M via silhouette maximisation.
  4. Cluster clients → M groups.
  5. Initialise M expert parameters by averaging within each cluster.
  6. Broadcast experts + cluster assignments to all clients.

Iterative training (each round t):
  1. Receive (expert_state, router_state) from every client.
  2. For each expert j: average uploaded expert_j states from cluster j.
  3. For each cluster j: average router states.
  4. Broadcast updated expert states and routers.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from clustering.similarity import compute_distance_matrix, compute_distance_matrix_from_ab
from clustering.expert_allocation import ExpertAllocator
from federated.aggregation import cluster_wise_aggregation
from utils.logging_utils import get_logger
from utils.checkpointing import save_checkpoint

logger = get_logger("fedlease.server")


class FederatedServer:
    """
    Central server coordinating the FedLEASE training process.

    Parameters
    ----------
    config : ConfigDict
    output_dir : str
    """

    def __init__(
        self,
        config: Any,
        output_dir: str = "outputs",
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Set during initialisation phase
        self.n_experts: int = 0
        self.cluster_labels: np.ndarray = np.array([])
        self.cluster_map: Dict[int, List[int]] = {}
        self.client_assignments: Dict[int, int] = {}  # client_id → expert_idx

        # Expert and router states (M each)
        self.expert_states: List[Optional[Dict]] = []
        self.router_states: List[Optional[Dict]] = []

        self.allocator = ExpertAllocator(
            min_clusters=config.clustering.min_clusters,
            max_clusters=config.clustering.max_clusters,
            linkage=config.clustering.linkage,
        )

        # Communication statistics
        self.comm_stats: Dict = {
            "init_bytes_uploaded": 0,
            "bytes_per_round": [],
            "rounds": 0,
        }

    # ------------------------------------------------------------------
    # Initialisation phase
    # ------------------------------------------------------------------

    def run_initialization(
        self,
        client_b_matrices: Dict[int, List[torch.Tensor]],
        client_ab_matrices: Dict[int, List],
        client_ids: List[int],
    ) -> Dict:
        """
        Execute the full initialisation phase.

        Parameters
        ----------
        client_b_matrices : {client_id: [b_tensor per layer]}
            B matrices from each client's warm-up LoRA.
        client_ab_matrices : {client_id: [(Aq, Bq, Av, Bv) per layer]}
            Full (A, B) matrices from each client for expert initialisation.
        client_ids : List[int]

        Returns
        -------
        info dict with n_experts, cluster assignments, silhouette scores.
        """
        logger.info("=== Server Initialisation Phase ===")
        N = len(client_ids)

        # --- 1. Build ordered list of B matrices ---
        ordered_b  = [client_b_matrices[cid]  for cid in client_ids]
        ordered_ab = [client_ab_matrices[cid] for cid in client_ids]

        # --- 2. Compute distance matrix ---
        # Use B@A rather than B alone — gives a much stronger silhouette signal
        # because the update direction is captured rather than just the output
        # projection (which stays near zero for short warmup).
        t0 = time.time()
        try:
            dist_matrix = compute_distance_matrix_from_ab(ordered_ab)
            logger.info(f"Distance matrix (B@A) computed in {time.time() - t0:.2f}s")
        except Exception as exc:
            logger.warning(f"B@A distance failed ({exc}); falling back to B-only")
            dist_matrix = compute_distance_matrix(ordered_b)
            logger.info(f"Distance matrix (B-only) computed in {time.time() - t0:.2f}s")

        # --- 3. Find optimal M via silhouette ---
        optimal_k, labels, scores = self.allocator.allocate(dist_matrix, client_ids)
        self.n_experts = optimal_k
        self.cluster_labels = labels
        self.cluster_map = self.allocator.cluster_map
        self.client_assignments = {cid: int(lbl) for cid, lbl in zip(client_ids, labels)}

        logger.info(self.allocator.report())

        # --- 4. Initialise M experts by averaging within clusters ---
        ordered_ab = [client_ab_matrices[cid] for cid in client_ids]
        expert_layer_params = self.allocator.initialise_experts(
            ordered_ab, labels, optimal_k
        )

        # Convert to per-expert state dicts (layer0.A, layer0.B etc.)
        self.expert_states = self._build_expert_states(expert_layer_params)
        self.router_states = [None] * optimal_k  # routers initialised on first round

        # Save distance matrix and cluster labels for visualisation
        np.save(os.path.join(self.output_dir, "distance_matrix.npy"), dist_matrix)
        np.save(os.path.join(self.output_dir, "cluster_labels.npy"), labels)
        np.save(
            os.path.join(self.output_dir, "silhouette_scores.npy"),
            np.array([[k, v] for k, v in scores.items()]),
        )

        return {
            "n_experts": optimal_k,
            "cluster_labels": labels,
            "client_assignments": self.client_assignments,
            "silhouette_scores": scores,
            "distance_matrix": dist_matrix,
        }

    # ------------------------------------------------------------------
    # Per-round operations
    # ------------------------------------------------------------------

    def broadcast(self, client_id: int) -> Dict:
        """
        Return the payload to send to a client at the start of a round.

        Returns
        -------
        {
          'expert_states': List[Dict],    # M expert states
          'router_state': Optional[Dict], # cluster-averaged router
          'assigned_expert_idx': int,
        }
        """
        assigned = self.client_assignments.get(client_id, 0)
        router_for_client = self.router_states[assigned] if self.router_states else None

        return {
            "expert_states": self.expert_states,
            "router_state": router_for_client,
            "assigned_expert_idx": assigned,
            "n_experts": self.n_experts,
        }

    def aggregate_round(
        self,
        client_uploads: List[Tuple[int, Dict, Dict]],
        dataset_sizes: Optional[Dict[int, int]] = None,
    ) -> None:
        """
        Aggregate client uploads for a single communication round.

        Parameters
        ----------
        client_uploads : List of (client_id, expert_state, router_state)
        dataset_sizes : optional {client_id: train_size}
        """
        client_expert = [(cid, es) for cid, es, _ in client_uploads]
        client_router = [(cid, rs) for cid, _, rs in client_uploads]

        new_exp, new_rot = cluster_wise_aggregation(
            client_expert,
            client_router,
            self.cluster_map,
            self.n_experts,
            dataset_sizes=dataset_sizes,
        )

        # Update only experts that received at least one upload
        for e in range(self.n_experts):
            if new_exp[e] is not None:
                self.expert_states[e] = new_exp[e]
            if new_rot[e] is not None:
                self.router_states[e] = new_rot[e]

        self.comm_stats["rounds"] += 1
        # Track bytes (rough estimate: count float32 parameters)
        bytes_this_round = sum(
            sum(v.numel() * 4 for v in es.values())
            for _, es, _ in client_uploads
        )
        self.comm_stats["bytes_per_round"].append(bytes_this_round)

    def save_state(self, round_idx: int) -> str:
        """Checkpoint all expert states to disk."""
        state = {
            "round": round_idx,
            "n_experts": self.n_experts,
            "expert_states": self.expert_states,
            "router_states": self.router_states,
            "cluster_map": self.cluster_map,
            "client_assignments": self.client_assignments,
        }
        path = save_checkpoint(
            state,
            self.checkpoint_dir,
            filename=f"server_round_{round_idx:03d}.pt",
        )
        return path

    def load_state(self, path: str) -> None:
        """Restore server state from checkpoint."""
        from utils.checkpointing import load_checkpoint
        state = load_checkpoint(path)
        self.n_experts = state["n_experts"]
        self.expert_states = state["expert_states"]
        self.router_states = state["router_states"]
        self.cluster_map = state["cluster_map"]
        self.client_assignments = state["client_assignments"]

    def communication_summary(self) -> Dict:
        """Return communication cost statistics."""
        rounds_done = self.comm_stats["rounds"]
        bytes_list = self.comm_stats["bytes_per_round"]
        return {
            "total_rounds": rounds_done,
            "total_bytes_MB": sum(bytes_list) / 1e6,
            "avg_bytes_per_round_MB": (sum(bytes_list) / len(bytes_list) / 1e6)
            if bytes_list else 0.0,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_expert_states(
        self,
        expert_layer_params: List[List],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert expert_layer_params to list of state dicts.

        expert_layer_params : List[M] of List[L] of (Aq, Bq, Av, Bv)

        LoRA-MoE layers are ordered: layer0_query, layer0_value,
        layer1_query, layer1_value, …
        Total: 2 * num_transformer_layers entries.
        """
        M = len(expert_layer_params)
        expert_state_list: List[Dict[str, torch.Tensor]] = []

        for exp_idx in range(M):
            state: Dict[str, torch.Tensor] = {}
            layer_params = expert_layer_params[exp_idx]  # List[L] of (Aq, Bq, Av, Bv)
            lid = 0
            for l, (Aq, Bq, Av, Bv) in enumerate(layer_params):
                # Query layer
                state[f"layer{lid}.A"] = Aq.clone()
                state[f"layer{lid}.B"] = Bq.clone()
                lid += 1
                # Value layer
                state[f"layer{lid}.A"] = Av.clone()
                state[f"layer{lid}.B"] = Bv.clone()
                lid += 1
            expert_state_list.append(state)

        return expert_state_list
