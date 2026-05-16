"""
Adaptive Top-M Router for FedLEASE.

Paper (Eq. 5):
  ω̂ = softmax(G_i · x)  ∈  R^{2M-1}

  y = W_0 x + Σ_{p ∈ TopK(ω̂, M)} ω̂_p ·
      { B_j  A_j  x,         if p < M
      { B_{q} A_{q} x,       if p ≥ M   (q = other_experts[p - M])

Router output layout for assigned expert j, M total experts:
  positions [0 .. M-1]    → M "slots" all routed to assigned expert E_j
  positions [M .. 2M-2]   → one slot each for every expert k ≠ j

This design gives E_j M chances to be selected (bias toward local expert)
while still allowing full cross-expert utilisation.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveTopMRouter(nn.Module):
    """
    Adaptive top-M routing mechanism.

    Parameters
    ----------
    hidden_dim : int
        Input dimension d (e.g. 768 for BERT-base).
    n_experts : int
        Total number of LoRA experts M.
    assigned_expert_idx : int
        Index j of the client's assigned expert (0-indexed).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_experts: int,
        assigned_expert_idx: int,
        use_gumbel: bool = True,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        self.n_experts = n_experts
        self.assigned_expert_idx = assigned_expert_idx
        self.router_dim = 2 * n_experts - 1  # (2M - 1)

        # Differentiable Gumbel-softmax option (vs plain softmax + argmax).
        # Annealed temperature (high → low) gives smooth exploration early
        # then sharp specialisation late. Set use_gumbel=False to fall back
        # to the original paper-faithful softmax routing.
        self.use_gumbel = use_gumbel
        self.temperature = temperature

        # Trainable router: G ∈ R^{(2M-1) × d}
        self.gate = nn.Linear(hidden_dim, self.router_dim, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)

        # Pre-compute mapping: router output index → actual expert index
        # Cached as a list; updated when assigned_expert changes.
        self._other_experts: List[int] = [
            k for k in range(n_experts) if k != assigned_expert_idx
        ]

        # Last-computed full routing distribution, for the load-balancing
        # auxiliary loss in the model wrapper (set in forward()).
        self._last_omega: torch.Tensor = None  # type: ignore

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing weights and expert assignments.

        Parameters
        ----------
        hidden : Tensor [batch, hidden_dim]
            Pooled representation used for routing (e.g. [CLS] embedding).

        Returns
        -------
        routing_weights : Tensor [batch, M]
            Raw softmax probabilities for the M selected positions.
        expert_indices : Tensor [batch, M]
            Actual expert indices (0 … M-1) for each selected position.
        """
        M = self.n_experts

        # logits ∈ R^{batch × 2M-1}
        logits = self.gate(hidden)

        if self.use_gumbel and self.training:
            # Sample Gumbel(0,1) noise and apply temperature
            # → differentiable end-to-end (paper: Jang et al. 2017)
            g = -torch.empty_like(logits).exponential_().log()
            omega = F.softmax((logits + g) / max(self.temperature, 1e-6), dim=-1)
        else:
            omega = F.softmax(logits, dim=-1)

        # Cache full distribution for the load-balancing auxiliary loss
        self._last_omega = omega

        # Top-M selection
        top_values, top_indices = torch.topk(omega, M, dim=-1)
        # top_indices ∈ {0, …, 2M-2}  shape: [batch, M]

        expert_indices = self._map_to_expert_indices(top_indices)

        return top_values, expert_indices

    # ------------------------------------------------------------------
    # Load-balancing auxiliary loss
    # ------------------------------------------------------------------

    def load_balance_loss(self) -> torch.Tensor:
        """Shazeer 2017 "importance" coefficient of variation squared.

        Penalises imbalanced expert use (some experts always picked, others
        never). Add a small multiple (e.g. 0.01) to the main CE loss.

        Returns 0 if the router has not produced a distribution yet.
        """
        if self._last_omega is None:
            return torch.tensor(0.0, device=self.gate.weight.device)
        # Importance: mass into each routing slot, summed over batch
        importance = self._last_omega.sum(dim=0)              # [2M-1]
        mean = importance.mean()
        # Coefficient of variation squared = (std/mean)^2
        cv2 = (importance.float().var(unbiased=False) / (mean.float() ** 2 + 1e-9))
        return cv2

    def anneal_temperature(self, new_temp: float) -> None:
        """Reduce temperature for sharper specialisation as training advances."""
        self.temperature = max(new_temp, 1e-3)

    # ------------------------------------------------------------------
    # Routing-index → expert-index mapping
    # ------------------------------------------------------------------

    def _map_to_expert_indices(self, router_indices: torch.Tensor) -> torch.Tensor:
        """
        Map router output indices to actual expert indices.

        router_indices : [batch, M],  values in {0, …, 2M-2}

        For p < M  → assigned expert j
        For p ≥ M  → other_experts[p - M]
        """
        M = self.n_experts
        j = self.assigned_expert_idx
        device = router_indices.device

        expert_indices = torch.full_like(router_indices, j)  # default = assigned

        other_mask = router_indices >= M
        if other_mask.any():
            # other_experts list: all experts except j, in ascending order
            other_tensor = torch.tensor(self._other_experts, device=device, dtype=torch.long)
            # Relative index into other_tensor
            rel = (router_indices[other_mask] - M).clamp(0, len(self._other_experts) - 1)
            expert_indices[other_mask] = other_tensor[rel]

        return expert_indices

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def update_assigned_expert(self, new_idx: int) -> None:
        """Update assigned expert (called by server when redistributing)."""
        self.assigned_expert_idx = new_idx
        self._other_experts = [k for k in range(self.n_experts) if k != new_idx]

    def expert_utilisation(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute fraction of samples routing to each expert.

        Returns Tensor [M] with utilisation ratios summing to ≤ M
        (a sample may route to the same expert multiple times via
        the M-slot mechanism).
        """
        with torch.no_grad():
            _, exp_idx = self.forward(hidden)  # [batch, M]
        util = torch.zeros(self.n_experts, device=hidden.device)
        for e in range(self.n_experts):
            util[e] = (exp_idx == e).float().sum() / exp_idx.numel() * self.n_experts
        return util

    def extra_repr(self) -> str:
        return (
            f"n_experts={self.n_experts}, "
            f"assigned={self.assigned_expert_idx}, "
            f"router_dim={self.router_dim}"
        )
