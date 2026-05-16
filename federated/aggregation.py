"""
Federated aggregation utilities.

Group-wise expert aggregation (per cluster j):
  A_j^expert ← (1/|C_j|) Σ_{i ∈ C_j} A_j^i
  B_j^expert ← (1/|C_j|) Σ_{i ∈ C_j} B_j^i

Router aggregation within each cluster:
  G_j^expert ← (1/|C_j|) Σ_{i ∈ C_j} G_i
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


ExpertState = Dict[str, torch.Tensor]
RouterState = Dict[str, torch.Tensor]


def aggregate_expert_states(
    expert_states: List[ExpertState],
    weights: Optional[List[float]] = None,
) -> ExpertState:
    """
    Weighted average of expert state dicts.

    Parameters
    ----------
    expert_states : List[ExpertState]
        Each dict has keys like 'layer0.A', 'layer0.B', …
    weights : list of float, optional
        Aggregation weights (e.g. proportional to local dataset size).
        If None, uniform averaging is used.

    Returns
    -------
    ExpertState — averaged expert parameters.
    """
    if not expert_states:
        return {}

    if weights is None:
        weights = [1.0 / len(expert_states)] * len(expert_states)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    agg: ExpertState = {}
    for key in expert_states[0]:
        stacked = torch.stack([es[key].float() for es in expert_states], dim=0)
        w_tensor = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
        # Weighted sum over the client dimension
        agg[key] = (stacked * w_tensor.view(-1, *([1] * (stacked.dim() - 1)))).sum(0)

    return agg


def aggregate_router_states(
    router_states: List[RouterState],
    weights: Optional[List[float]] = None,
) -> RouterState:
    """Weighted average of router gate weight dicts."""
    return aggregate_expert_states(router_states, weights)


def cluster_wise_aggregation(
    client_expert_states: List[Tuple[int, ExpertState]],
    client_router_states: List[Tuple[int, RouterState]],
    cluster_map: Dict[int, List[int]],
    n_experts: int,
    dataset_sizes: Optional[Dict[int, int]] = None,
) -> Tuple[List[ExpertState], List[RouterState]]:
    """
    Full server aggregation step.

    Parameters
    ----------
    client_expert_states : List of (client_id, expert_state)
        Each client uploads only its assigned expert's state.
    client_router_states : List of (client_id, router_state)
    cluster_map : Dict expert_idx → [client_id, …]
    n_experts : M
    dataset_sizes : optional per-client training sizes for weighted averaging.

    Returns
    -------
    new_expert_states : List[ExpertState]  length M
    new_router_states : List[RouterState]  length M
    """
    expert_dict: Dict[int, ExpertState] = {cid: es for cid, es in client_expert_states}
    router_dict: Dict[int, RouterState] = {cid: rs for cid, rs in client_router_states}

    new_expert_states: List[Optional[ExpertState]] = [None] * n_experts
    new_router_states: List[Optional[RouterState]] = [None] * n_experts

    for exp_idx in range(n_experts):
        members = cluster_map.get(exp_idx, [])
        # Only keep members that uploaded updates this round
        active = [cid for cid in members if cid in expert_dict]

        if not active:
            continue

        if dataset_sizes is not None:
            weights = [float(dataset_sizes.get(cid, 1)) for cid in active]
        else:
            weights = None

        exp_states = [expert_dict[cid] for cid in active]
        rot_states = [router_dict[cid] for cid in active if cid in router_dict]

        new_expert_states[exp_idx] = aggregate_expert_states(exp_states, weights)
        if rot_states:
            new_router_states[exp_idx] = aggregate_router_states(rot_states, weights)

    return new_expert_states, new_router_states  # type: ignore[return-value]
