"""
FedLEASE FinBERT model variants.

WarmupFinBERT     – FinBERT + single LoRA (used during init phase)
FedLEASEFinBERT   – FinBERT + LoRA-MoE + adaptive top-M router (main training)
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertForSequenceClassification,
)

from .lora_layers import LoRAMoELinear
from .adaptive_router import AdaptiveTopMRouter


# ---------------------------------------------------------------------------
# Warmup model: single LoRA per layer
# ---------------------------------------------------------------------------

class SingleLoRALinear(nn.Module):
    """Standard single-expert LoRA linear layer for the warmup phase."""

    def __init__(
        self,
        base_layer: nn.Linear,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.lora_rank = lora_rank
        self.scaling = lora_alpha / lora_rank

        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(lora_rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, lora_rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        x_drop = self.lora_dropout(x)
        lora_out = F.linear(F.linear(x_drop, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


class WarmupFinBERT(nn.Module):
    """
    FinBERT + single LoRA for the FedLEASE initialisation phase.

    After E warm-up epochs, the server collects each client's B matrices
    to compute inter-client cosine similarity for clustering.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.1,
        num_labels: int = 3,
    ) -> None:
        super().__init__()

        self.lora_rank = lora_rank
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.bert = BertForSequenceClassification.from_pretrained(model_name, config=config)

        for p in self.bert.parameters():
            p.requires_grad_(False)

        self._inject_single_lora(lora_rank, lora_alpha, lora_dropout)
        self.num_layers = config.num_hidden_layers

    def _inject_single_lora(
        self,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
    ) -> None:
        for layer in self.bert.bert.encoder.layer:
            attn = layer.attention.self
            attn.query = SingleLoRALinear(attn.query, lora_rank, lora_alpha, lora_dropout)
            attn.value = SingleLoRALinear(attn.value, lora_rank, lora_alpha, lora_dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        return {"loss": out.loss, "logits": out.logits}

    def get_b_matrices(self) -> List[torch.Tensor]:
        """
        Collect all B matrices from LoRA layers for clustering.

        Returns flat list of tensors, one per (layer, proj):
        [layer0_query_B, layer0_value_B, layer1_query_B, layer1_value_B, …]
        """
        b_mats: List[torch.Tensor] = []
        for layer in self.bert.bert.encoder.layer:
            attn = layer.attention.self
            b_mats.append(attn.query.lora_B.data.flatten())
            b_mats.append(attn.value.lora_B.data.flatten())
        return b_mats

    def get_ab_matrices_per_layer(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Return [(A_q, B_q), (A_v, B_v)] per layer for expert initialisation."""
        result = []
        for layer in self.bert.bert.encoder.layer:
            attn = layer.attention.self
            result.append((
                attn.query.lora_A.data.clone(),
                attn.query.lora_B.data.clone(),
                attn.value.lora_A.data.clone(),
                attn.value.lora_B.data.clone(),
            ))
        return result

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Main model: FedLEASE FinBERT with LoRA-MoE + adaptive router
# ---------------------------------------------------------------------------

class FedLEASEFinBERT(nn.Module):
    """
    FinBERT enhanced with M LoRA experts and an adaptive top-M router.

    Architecture
    ------------
    - Frozen FinBERT backbone (BertForSequenceClassification).
    - Q and V projections in every attention layer replaced by LoRAMoELinear.
    - A single global AdaptiveTopMRouter G ∈ R^{(2M-1) × hidden_dim} that
      routes based on the [CLS] token embedding (from the embedding layer).
    - During local training only the assigned expert j and the router are
      updated; all other expert parameters are frozen.

    Forward pass
    ------------
    1. Compute full token embeddings (word + position + token_type).
    2. Route from the [CLS] embedding → routing_weights, expert_indices.
    3. Inject routing state into every LoRAMoELinear layer.
    4. Run BERT encoder + pooler using inputs_embeds (avoids re-computing
       embeddings) → logits.
    5. Clear routing state.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        n_experts: int = 4,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.1,
        num_labels: int = 3,
        assigned_expert_idx: int = 0,
        use_per_cluster_heads: bool = True,
    ) -> None:
        super().__init__()

        self.n_experts = n_experts
        self.lora_rank = lora_rank
        self.assigned_expert_idx = assigned_expert_idx
        self.num_labels = num_labels
        self.use_per_cluster_heads = use_per_cluster_heads

        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.bert = BertForSequenceClassification.from_pretrained(model_name, config=config)

        for p in self.bert.parameters():
            p.requires_grad_(False)

        self.num_layers = config.num_hidden_layers
        self._inject_lora_moe(n_experts, lora_rank, lora_alpha, lora_dropout)

        hidden_size = config.hidden_size
        self.router = AdaptiveTopMRouter(hidden_size, n_experts, assigned_expert_idx)

        # Per-cluster classifier heads.
        # Each head is initialised from the pretrained FinBERT classifier so
        # all heads start identical. Only the assigned cluster's head trains
        # locally; the server aggregates heads cluster-wise (same as experts).
        if self.use_per_cluster_heads:
            self.heads = nn.ModuleList([
                nn.Linear(hidden_size, num_labels)
                for _ in range(n_experts)
            ])
            with torch.no_grad():
                src_w = self.bert.classifier.weight.data.clone()
                src_b = self.bert.classifier.bias.data.clone()
                for h in self.heads:
                    h.weight.copy_(src_w)
                    h.bias.copy_(src_b)
            # The original BERT classifier is no longer used in forward,
            # but we keep it (frozen) so .pooler_output handling stays unchanged.
            for p in self.bert.classifier.parameters():
                p.requires_grad_(False)

        self._lora_layers: List[LoRAMoELinear] = [
            m for m in self.bert.modules() if isinstance(m, LoRAMoELinear)
        ]

        # Start with only assigned expert trainable
        self.set_trainable_expert(assigned_expert_idx)

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    def _inject_lora_moe(
        self,
        n_experts: int,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
    ) -> None:
        for layer in self.bert.bert.encoder.layer:
            attn = layer.attention.self
            attn.query = LoRAMoELinear(attn.query, n_experts, lora_rank, lora_alpha, lora_dropout)
            attn.value = LoRAMoELinear(attn.value, n_experts, lora_rank, lora_alpha, lora_dropout)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        # 1. Compute embeddings once (reused for encoder)
        emb_out = self.bert.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )  # [B, seq, hidden]

        # 2. Route from [CLS] embedding
        cls_emb = emb_out[:, 0, :]  # [B, hidden]
        rw, ri = self.router(cls_emb)  # [B, M], [B, M]

        # 3. Inject routing into LoRA-MoE layers
        self._set_routing(rw, ri)

        try:
            # 4. Run BERT with precomputed embeddings
            bert_out = self.bert.bert(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=emb_out,
            )
            pooled = self.bert.dropout(bert_out.pooler_output)

            # Per-cluster heads: use the assigned cluster's head.
            # This eliminates cross-cluster gradient conflict at the read-out
            # layer (PhraseBank head learns 60/25/15, Twitter head learns its
            # own distribution — no FedAvg blending).
            if self.use_per_cluster_heads:
                logits = self.heads[self.assigned_expert_idx](pooled)
            else:
                logits = self.bert.classifier(pooled)  # [B, num_labels]
        finally:
            self._clear_routing()

        result: Dict = {"logits": logits}
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            result["loss"] = loss

        return result

    # ------------------------------------------------------------------
    # Routing state management
    # ------------------------------------------------------------------

    def _set_routing(self, rw: torch.Tensor, ri: torch.Tensor) -> None:
        for layer in self._lora_layers:
            layer._routing_weights = rw
            layer._routing_expert_indices = ri

    def _clear_routing(self) -> None:
        for layer in self._lora_layers:
            layer._routing_weights = None
            layer._routing_expert_indices = None

    # ------------------------------------------------------------------
    # Expert parameter access (for federated aggregation)
    # ------------------------------------------------------------------

    def set_trainable_expert(self, expert_idx: int) -> None:
        """Freeze all experts except expert_idx; keep router + assigned head trainable."""
        self.assigned_expert_idx = expert_idx
        self.router.update_assigned_expert(expert_idx)
        for layer in self._lora_layers:
            layer.freeze_all_experts()
            layer.unfreeze_expert(expert_idx)
        # Per-cluster heads: freeze all except the assigned cluster's head
        if self.use_per_cluster_heads:
            for i, h in enumerate(self.heads):
                for p in h.parameters():
                    p.requires_grad_(i == expert_idx)

    def get_expert_state(self, expert_idx: int) -> Dict[str, torch.Tensor]:
        """Return all A/B tensors for a given expert across all LoRA layers."""
        state: Dict[str, torch.Tensor] = {}
        for lid, layer in enumerate(self._lora_layers):
            state[f"layer{lid}.A"] = layer.get_expert_A(expert_idx)
            state[f"layer{lid}.B"] = layer.get_expert_B(expert_idx)
        return state

    def set_expert_state(self, expert_idx: int, state: Dict[str, torch.Tensor]) -> None:
        for lid, layer in enumerate(self._lora_layers):
            A_key = f"layer{lid}.A"
            B_key = f"layer{lid}.B"
            if A_key in state:
                layer.set_expert_A(expert_idx, state[A_key])
            if B_key in state:
                layer.set_expert_B(expert_idx, state[B_key])

    def get_all_expert_states(self) -> List[Dict[str, torch.Tensor]]:
        return [self.get_expert_state(e) for e in range(self.n_experts)]

    def set_all_expert_states(self, states: List[Dict[str, torch.Tensor]]) -> None:
        for e, st in enumerate(states):
            self.set_expert_state(e, st)

    def get_router_state(self) -> Dict[str, torch.Tensor]:
        return {"gate.weight": self.router.gate.weight.data.clone()}

    def set_router_state(self, state: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self.router.gate.weight.copy_(state["gate.weight"])

    # ------------------------------------------------------------------
    # Per-cluster classifier-head access
    # ------------------------------------------------------------------

    def get_head_state(self, head_idx: int) -> Dict[str, torch.Tensor]:
        """Return weight+bias for one cluster's classifier head."""
        if not self.use_per_cluster_heads:
            return {}
        h = self.heads[head_idx]
        return {
            "weight": h.weight.data.clone(),
            "bias":   h.bias.data.clone(),
        }

    def set_head_state(self, head_idx: int, state: Dict[str, torch.Tensor]) -> None:
        if not self.use_per_cluster_heads or not state:
            return
        h = self.heads[head_idx]
        with torch.no_grad():
            if "weight" in state:
                h.weight.copy_(state["weight"])
            if "bias" in state:
                h.bias.copy_(state["bias"])

    def get_all_head_states(self) -> List[Dict[str, torch.Tensor]]:
        if not self.use_per_cluster_heads:
            return []
        return [self.get_head_state(i) for i in range(self.n_experts)]

    def set_all_head_states(self, states: List[Dict[str, torch.Tensor]]) -> None:
        if not self.use_per_cluster_heads or not states:
            return
        for i, st in enumerate(states):
            if st:
                self.set_head_state(i, st)

    # ------------------------------------------------------------------
    # B-matrix extraction (for FedLEASE similarity computation in main phase)
    # ------------------------------------------------------------------

    def get_b_matrices(self) -> List[torch.Tensor]:
        """
        Return flattened B matrices of the assigned expert for all layers.
        Used by the server for intra-round similarity monitoring.
        """
        j = self.assigned_expert_idx
        b_mats: List[torch.Tensor] = []
        for layer in self._lora_layers:
            b_mats.append(layer.get_expert_B(j).flatten())
        return b_mats

    # ------------------------------------------------------------------
    # Warm-up state transfer
    # ------------------------------------------------------------------

    def load_from_warmup(
        self,
        warmup_model: WarmupFinBERT,
        target_expert_idx: int,
    ) -> None:
        """
        Initialise expert target_expert_idx from a warm-up SingleLoRA model.
        Used during server-side expert construction.
        """
        warmup_layers = list(warmup_model.bert.bert.encoder.layer)
        for lid, (fedlease_layer, warmup_bert_layer) in enumerate(
            zip(self._lora_layers, [l for wl in warmup_layers for l in [
                wl.attention.self.query, wl.attention.self.value]])
        ):
            src_A = warmup_bert_layer.lora_A.data.clone()
            src_B = warmup_bert_layer.lora_B.data.clone()
            fedlease_layer.set_expert_A(target_expert_idx, src_A)
            fedlease_layer.set_expert_B(target_expert_idx, src_B)

    # ------------------------------------------------------------------
    # Parameter counts
    # ------------------------------------------------------------------

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_efficiency(self) -> float:
        return self.count_trainable() / self.count_total() * 100
