"""
LoRA Mixture-of-Experts Linear Layer.

Implements the per-layer LoRA-MoE computation used in FedLEASE:

  y = W_0 x + Σ_{p ∈ TopK(ω̂, M)} ω̂_p · B_{exp(p)} A_{exp(p)} x

where the expert mapping exp(p) is:
  p < M  →  assigned expert j
  p ≥ M  →  other experts (in canonical order excluding j)
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAMoELinear(nn.Module):
    """
    Drop-in replacement for nn.Linear supporting M LoRA experts.

    Parameters
    ----------
    base_layer : nn.Linear
        The frozen pre-trained linear layer (W_0).
    n_experts : int
        Total number of LoRA experts M.
    lora_rank : int
        Low-rank dimension r ≪ min(in, out).
    lora_alpha : float
        Scaling factor α; effective scale = α / r.
    lora_dropout : float
        Dropout applied to input before LoRA computation.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        n_experts: int,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.n_experts = n_experts
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank

        # Freeze the base layer
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        # M expert LoRA matrices
        # A_e ∈ R^{rank × in},  B_e ∈ R^{out × rank}
        self.lora_A: nn.ParameterList = nn.ParameterList([
            nn.Parameter(torch.empty(lora_rank, self.in_features))
            for _ in range(n_experts)
        ])
        self.lora_B: nn.ParameterList = nn.ParameterList([
            nn.Parameter(torch.zeros(self.out_features, lora_rank))
            for _ in range(n_experts)
        ])

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Initialise A with Kaiming uniform (B stays at zero)
        for A_e in self.lora_A:
            nn.init.kaiming_uniform_(A_e, a=math.sqrt(5))

        # Routing state injected by the model before each forward pass
        # Shape: [batch, M] and [batch, M] respectively
        self._routing_weights: Optional[torch.Tensor] = None
        self._routing_expert_indices: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [..., in_features]
        Returns [..., out_features]
        """
        base_out = self.base_layer(x)

        if self._routing_weights is None:
            return base_out

        rw = self._routing_weights       # [B, M]
        re = self._routing_expert_indices  # [B, M]

        batch = x.size(0)
        M_selected = rw.size(1)
        lora_out = torch.zeros_like(base_out)

        for m in range(M_selected):
            exp_idx_vec = re[:, m]   # [B] — expert index per sample
            w_vec = rw[:, m]         # [B] — weight per sample

            for e in range(self.n_experts):
                mask = (exp_idx_vec == e)   # [B] bool
                if not mask.any():
                    continue

                x_e = x[mask]                           # [n_e, ..., in]
                x_lora = self.lora_dropout(x_e)
                # B_e A_e x_e
                h = F.linear(x_lora, self.lora_A[e])   # [n_e, ..., rank]
                h = F.linear(h, self.lora_B[e])         # [n_e, ..., out]
                h = h * self.scaling

                # Broadcast weight across spatial / sequence dimensions
                w_e = w_vec[mask]                       # [n_e]
                for _ in range(x_e.dim() - 1):
                    w_e = w_e.unsqueeze(-1)             # [n_e, 1, ...]

                # Cast to lora_out dtype to avoid Half/Float mismatch under
                # autocast (routing weights may be fp32 while lora_out is fp16)
                contrib = (w_e * h).to(lora_out.dtype)
                lora_out[mask] = lora_out[mask] + contrib

        return base_out + lora_out

    # ------------------------------------------------------------------
    # Expert parameter management
    # ------------------------------------------------------------------

    def freeze_expert(self, expert_idx: int) -> None:
        self.lora_A[expert_idx].requires_grad_(False)
        self.lora_B[expert_idx].requires_grad_(False)

    def unfreeze_expert(self, expert_idx: int) -> None:
        self.lora_A[expert_idx].requires_grad_(True)
        self.lora_B[expert_idx].requires_grad_(True)

    def freeze_all_experts(self) -> None:
        for e in range(self.n_experts):
            self.freeze_expert(e)

    def set_trainable_expert(self, expert_idx: int) -> None:
        """Freeze all experts except expert_idx."""
        self.freeze_all_experts()
        self.unfreeze_expert(expert_idx)

    def get_expert_A(self, expert_idx: int) -> torch.Tensor:
        return self.lora_A[expert_idx].data.clone()

    def get_expert_B(self, expert_idx: int) -> torch.Tensor:
        return self.lora_B[expert_idx].data.clone()

    def set_expert_A(self, expert_idx: int, A: torch.Tensor) -> None:
        with torch.no_grad():
            self.lora_A[expert_idx].copy_(A)

    def set_expert_B(self, expert_idx: int, B: torch.Tensor) -> None:
        with torch.no_grad():
            self.lora_B[expert_idx].copy_(B)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.lora_rank}, experts={self.n_experts}, "
            f"alpha={self.lora_alpha}"
        )
