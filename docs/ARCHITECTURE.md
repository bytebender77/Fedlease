# System Architecture and Federated Workflow

This document covers sections **4 (Overall Architecture)** and
**5 (Federated Learning Workflow)** of the exposition. It describes how
the codebase is decomposed into modules, how the modules communicate at
runtime, and the precise round-by-round protocol that the FedLEASE
pipeline executes.

The intent is that a reader who has internalised this document can,
unaided, sketch the system on a whiteboard, identify which file each
component lives in, and trace any tensor from a raw input string through
to a gradient.

---

## 1. Component graph

```
                                ┌─────────────────────────────────────────┐
                                │  experiments/run_fedlease.py            │
                                │  (CLI entry — argparse, config build,   │
                                │   logger, plot orchestration)           │
                                └────────────┬────────────────────────────┘
                                             │
                                             ▼
       ┌────────────────────┐    ┌───────────────────────────────────────────┐
       │ configs/*.yaml     │───▶│ utils/config.py  : ConfigDict, merge      │
       └────────────────────┘    │ utils/seed.py    : set_seed, seed_worker  │
                                 │ utils/logging_utils.py                    │
                                 └───────────────────────────────────────────┘
                                             │
                                             ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ data/                                                                    │
   │   dataset_loader.py    → FinancialDatasetLoader  (PhraseBank + Twitter,  │
   │                          unified 3-class label space)                    │
   │   data_partitioner.py  → FederatedDataPartitioner                        │
   │                          (heterogeneous | iid | non_iid Dirichlet)       │
   │   preprocessing.py     → FinancialPreprocessor   (FinBERT tokeniser,     │
   │                          DataLoader builder)                             │
   └──────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ training/fedlease_pipeline.py   :  FedLEASEPipeline                      │
   │                                                                          │
   │   ┌─────────────── Phase A (warmup) ─────────────┐                       │
   │   │                                              │                       │
   │   │   for each client i:                         │                       │
   │   │     local_train( WarmupFinBERT )             │                       │
   │   │     upload (A_i^ℓ, B_i^ℓ) for all ℓ          │                       │
   │   └──────────────────────────────────────────────┘                       │
   │                                                                          │
   │   ┌─────────────── Clustering ───────────────────┐                       │
   │   │   clustering/similarity.py      : D_ij       │                       │
   │   │   clustering/clustering.py      : linkage    │                       │
   │   │   clustering/expert_allocation. : M*, init   │                       │
   │   └──────────────────────────────────────────────┘                       │
   │                                                                          │
   │   ┌─────────────── Phase B (clustered training) ─────────────────────┐   │
   │   │                                                                  │   │
   │   │   for round t = 1..T:                                            │   │
   │   │     server.broadcast(global expert states, router state)         │   │
   │   │     for each client i:                                           │   │
   │   │        FedLEASEFinBERT.local_train()                             │   │
   │   │        upload (assigned-expert ΔA, ΔB, router weights)           │   │
   │   │     server.aggregate( cluster-wise weighted average )            │   │
   │   │                                                                  │   │
   │   └──────────────────────────────────────────────────────────────────┘   │
   │                                                                          │
   │   evaluate on held-out PhraseBank-test and Twitter-test                  │
   └──────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ models/                                                                  │
   │   lora_layers.py        : LoRAMoELinear      (multi-expert low-rank)     │
   │   adaptive_router.py    : AdaptiveTopMRouter (2M-1 gate, top-M select)   │
   │   finbert_lora_moe.py   : WarmupFinBERT, FedLEASEFinBERT                 │
   │                                                                          │
   │ federated/                                                               │
   │   client.py             : FederatedClient   (local SGD, payload pack)    │
   │   server.py             : FederatedServer   (broadcast, comm stats)      │
   │   aggregation.py        : cluster_wise_aggregation                       │
   │                                                                          │
   │ visualization/                                                           │
   │   similarity_plots, cluster_plots, training_plots, expert_plots          │
   └──────────────────────────────────────────────────────────────────────────┘
```

The arrows in the diagram are *runtime data flow*. Import dependencies
flow the opposite direction (the pipeline imports models, models import
torch).

---

## 2. Execution lifecycle

The end-to-end lifecycle, in eleven steps, is:

| # | Step | File / function | Owner |
|---|------|-----------------|-------|
| 1 | Parse CLI args, load YAML, merge overrides | `experiments/run_fedlease.py: parse_args, build_config` | driver |
| 2 | Seed all RNGs (Python, NumPy, Torch, CUDA) | `utils/seed.py: set_seed` | driver |
| 3 | Load both datasets, normalise labels to {0,1,2} | `data/dataset_loader.py: FinancialDatasetLoader` | driver |
| 4 | Partition across N clients | `data/data_partitioner.py: FederatedDataPartitioner.partition` | driver |
| 5 | Build per-client DataLoaders (FinBERT tokenisation) | `data/preprocessing.py: FinancialPreprocessor` | pipeline |
| 6 | Build a `WarmupFinBERT` per client and run Phase A | `training/fedlease_pipeline.py: FedLEASEPipeline._warmup` | pipeline |
| 7 | Collect each client's `{B_i^ℓ}` and form distance matrix | `clustering/similarity.py: compute_distance_matrix` | pipeline |
| 8 | Sweep k, pick M* by silhouette, run agglomerative clustering | `clustering/expert_allocation.py: ExpertAllocator.allocate` | pipeline |
| 9 | Initialise the global expert bank by intra-cluster averaging (Eq. 3) | `clustering/expert_allocation.py: initialise_experts` | pipeline |
| 10 | Build a `FedLEASEFinBERT` per client and run Phase B for T rounds | `training/fedlease_pipeline.py: FedLEASEPipeline._train` | pipeline |
| 11 | Evaluate on held-out test sets; dump metrics, plots, checkpoints | `training/evaluator.py`, `visualization/*` | pipeline |

---

## 3. Data flow: from text to gradient

### 3.1 Stage 1 — text → token IDs

`FinancialPreprocessor.make_dataloader(texts, labels)` uses the FinBERT
WordPiece tokenizer to produce, for each example:

- `input_ids ∈ Z^{L}`,  L = `max_length` = 128
- `attention_mask ∈ {0, 1}^{L}`
- `token_type_ids ∈ {0}^{L}` (single-segment)
- `labels ∈ {0, 1, 2}`

Batches of size 32 are emitted by a `torch.utils.data.DataLoader` with
`seed_worker` for deterministic shuffling.

### 3.2 Stage 2 — token IDs → contextual embeddings (frozen)

`FedLEASEFinBERT.forward` first calls

```python
emb_out = self.bert.bert.embeddings(input_ids, token_type_ids)   # [B, L, 768]
```

i.e. it manually runs FinBERT's embedding sublayer to obtain
position+token+segment embeddings. The `[CLS]` token's embedding is then
extracted:

```python
cls_emb = emb_out[:, 0, :]                                       # [B, 768]
```

This `cls_emb` is the **input to the router**. We compute embeddings
explicitly (rather than letting BERT do it inside its forward) because
the router needs them *before* the transformer layers run, and we do not
want to embed twice.

### 3.3 Stage 3 — routing decision

```python
top_values, expert_indices = self.router(cls_emb)
#   top_values     : [B, M*]  — softmax-weighted top-M routing probs
#   expert_indices : [B, M*]  — long tensor of expert IDs in [0, N_experts)
```

The router lives in `models/adaptive_router.py`. Its internal
`gate ∈ R^{(2M*-1) × 768}` projects the CLS embedding to a `2M*-1`-vector
of logits, softmaxes, takes the top-M values, and **maps the resulting
indices to expert IDs via a deterministic-plus-adaptive scheme**
(see `METHODOLOGY.md §4` for the precise mapping).

### 3.4 Stage 4 — routing state injection

The pipeline now needs every `LoRAMoELinear` inside FinBERT's encoder to
see `(top_values, expert_indices)` so it can compute the
expert-weighted LoRA update. We do this by *setting attributes on each
LoRA layer* before the encoder runs:

```python
for lora_layer in self.lora_layers:
    lora_layer._routing_weights        = top_values        # [B, M*]
    lora_layer._routing_expert_indices = expert_indices    # [B, M*]
```

This is the cleanest way to thread per-batch routing into the existing
HuggingFace BERT forward signature without reimplementing the encoder.
After the forward pass, the attributes are cleared in a `finally`
block (see §3.6).

### 3.5 Stage 5 — transformer encoder with LoRA-MoE adapters

The encoder is invoked with `inputs_embeds` to avoid re-embedding:

```python
bert_out = self.bert.bert(
    input_ids=None,
    attention_mask=attention_mask,
    inputs_embeds=emb_out,
)
```

Inside the encoder, every transformer layer has two `nn.Linear`
projections — `query` and `value` — that have been replaced by
`LoRAMoELinear` (see `models/lora_layers.py`). Each `LoRAMoELinear`
holds:

- a frozen `base_layer ∈ R^{768 × 768}` (FinBERT's pre-trained weights),
- `n_experts` independent LoRA pairs
  `(A_e ∈ R^{r × 768}, B_e ∈ R^{768 × r})` for `e ∈ [0, n_experts)`.

The forward computes:

```
out_base = base_layer(x)                              # [B, L, 768]
out_lora = zeros_like(out_base)
for m in range(M*):
    w_m   = top_values[:, m]      # [B]
    eid_m = expert_indices[:, m]  # [B]
    for e in range(n_experts):
        mask = (eid_m == e)
        if mask.any():
            x_sub  = x[mask]                          # [B', L, 768]
            lora_e = (B_e @ (A_e @ x_sub))            # [B', L, 768] (with scaling α/r)
            out_lora[mask] += w_m[mask, None, None] * lora_e
return out_base + out_lora
```

The mask loop ensures that an example only sees its M* assigned experts;
no example pays the cost of all `n_experts` experts.

### 3.6 Stage 6 — classifier and loss

After the encoder:

```python
pooled = self.bert.dropout(bert_out.pooler_output)    # [B, 768]
logits = self.bert.classifier(pooled)                 # [B, 3]
loss   = F.cross_entropy(logits, labels)
```

The pooler and classifier weights are *also* fine-tuned (they are the
sentiment-specific head FinBERT shipped with), but they are tiny relative
to LoRA parameters and are aggregated globally, not per-cluster.

The forward returns `(loss, logits)`. The `finally` block clears
`_routing_weights` and `_routing_expert_indices` on every LoRA layer,
ensuring no stale state leaks into the next batch.

### 3.7 Stage 7 — backward and optimiser step

Gradients flow through the LoRA expert combination back into both the
selected `(A_e, B_e)` pairs *and* the router's gate matrix. The frozen
`base_layer` does not receive gradients. The optimiser
(`torch.optim.AdamW`, LR = 3e-4) updates only the trainable parameters,
which is a small subset of the total model state.

---

## 4. The two phases of training

FedLEASE training is *not* a single loop. It is two distinct phases,
separated by a clustering step that runs on the server.

### 4.1 Phase A — initialisation warmup

**Goal:** produce a *signature* of each client (the LoRA B matrices)
that we can cluster on.

**Model:** `WarmupFinBERT` — FinBERT with a *single* LoRA pair per
target module (query, value). No router, no MoE. Identical to a
standard FedAvg-LoRA local model.

**Procedure (per client, in parallel-conceptually but sequential in code):**

1. Server initialises the warmup model with identical A, B across all
   clients (A ~ Kaiming-uniform, B = 0 — the standard LoRA init).
2. Each client `i` trains *locally* for `warmup_epochs` epochs (default 3)
   on its own data D_i. No communication during these epochs.
3. After warmup, each client uploads its trained
   `{(A_i^ℓ, B_i^ℓ)}_{ℓ ∈ adapter_layers}` to the server.

**Communication for Phase A:** 1 upload per client at the end of the
warmup. Size ≈ |LoRA| ≈ 590 KB per client.

**What changes during Phase A:** every LoRA matrix has had a few epochs
of supervised gradient signal pushed into it. Crucially, the LoRA B
matrices have moved from zero toward the top singular directions of
the client's task gradient (see `PROBLEM.md §9.2`). The B matrices are
now informative about the client's task and domain.

### 4.2 Clustering (interlude, runs on the server only)

See `METHODOLOGY.md §3` for the math. In code:

```python
allocator = ExpertAllocator(min_k, max_k, linkage='average')
result    = allocator.allocate(client_lora_states)
#   result.M               : optimal number of experts
#   result.cluster_labels  : [N] array of integers in [0, M)
#   result.distance_matrix : [N, N]
#   result.silhouette      : dict {k: score}
```

The allocator returns the cluster labels and the chosen M. It then
*initialises* the global expert bank by averaging the (A, B) of all
clients in each cluster (Eq. 3 of the paper, `initialise_experts`).

### 4.3 Phase B — clustered iterative training

**Model:** `FedLEASEFinBERT` — FinBERT with `LoRAMoELinear` (M experts)
on every target module, plus an `AdaptiveTopMRouter` whose `assigned_expert_idx`
is set to the client's cluster ID.

**Procedure (per round, repeated T times):**

1. **Server → Client broadcast.** The server sends each client `i`:
   - The current state of all M experts at every adapter layer.
   - The router state. (Note: routers can either be per-cluster
     aggregated or per-client. In this implementation each client owns
     its own router, and only router *initialisations* are shared at
     allocation time. The router payload is small enough that
     per-client routers cost ~negligible communication.)
2. **Local training.** Each client:
   - Sets `assigned_expert_idx = cluster(i)` on every router instance.
   - Runs `local_epochs` epochs of supervised SGD on D_i with
     `FedLEASEFinBERT`. Mixed precision is enabled if CUDA is
     available (`autocast` + `GradScaler`).
   - Only the *assigned* expert receives gradient most of the time
     (because the top-M routing guarantees that expert participates
     in every forward; other experts participate only when the router
     selects them).
3. **Client → Server upload.** Each client uploads:
   - Its **assigned expert** `(A_{cluster(i)}, B_{cluster(i)})` at every
     layer.
   - Its **router state** (the 2M-1-row gate matrix).
   - Its dataset size n_i (for weighted averaging).
4. **Cluster-wise aggregation.** The server, for each cluster c:
   - Averages the assigned expert across all clients in cluster c,
     weighted by n_i (`federated/aggregation.py: cluster_wise_aggregation`).
   - Writes the result back as the new global expert c.

**What is *not* communicated:**

- The FinBERT backbone (frozen, identical everywhere).
- The router gates of *other* clusters (a client never sees other
  clusters' router state).
- Any raw text or embedding.

**Communication for Phase B per round per client:**

|  Component          | Size (rank=4, FinBERT-base, 12 layers, 2 target modules) |
|---|---|
| Assigned expert ΔA  | 12 × 2 × 4 × 768   = ~73 K floats |
| Assigned expert ΔB  | 12 × 2 × 768 × 4   = ~73 K floats |
| Router gate         | (2M-1) × 768 × M_layers ≈ 1–3 K floats |
| **Total**           | **≈ 600 KB / round / client at fp32** |

For T = 25 rounds, N = 10 clients: total federation traffic ≈ 150 MB
per direction, well below any commercial-finance VPN budget.

---

## 5. What is frozen, what is trainable, what moves

This is the single most common source of confusion in LoRA-MoE federated
codebases. The exact state of every parameter is:

| Parameter group | Trainable? | Communicated? | Aggregation scope |
|---|:---:|:---:|---|
| FinBERT embeddings | ❌ frozen | ❌ | — |
| FinBERT encoder W_0 (every weight matrix) | ❌ frozen | ❌ | — |
| LoRA A_e for assigned expert e of cluster(i) | ✅ | ✅ | within cluster (weighted by n_i) |
| LoRA B_e for assigned expert e of cluster(i) | ✅ | ✅ | within cluster |
| LoRA A_e / B_e for *other* experts | ✅ (when sampled by router) | ❌ | — (each client's copy may drift) |
| Router gate G ∈ R^{(2M-1) × d} | ✅ | ✅ | per-client (own gate) |
| FinBERT classifier head | ✅ (small) | ✅ | global FedAvg, weighted by n_i |

A subtle implementation point: in this codebase, *other-cluster experts*
do receive gradient when sampled by a client's router, but those gradient
updates are **not uploaded**. They are effectively a client-local
augmentation of the broadcast expert state. After the next round's
broadcast, those drifts are overwritten by the freshly-aggregated global
expert. This is a deliberate design choice — uploading every expert
multiplies communication by M and yields negligible accuracy gain.

---

## 6. Tensor-flow summary

For a single training batch on client i with cluster(i) = c, local
batch size B:

```
text [B] ── tokeniser ──▶ input_ids [B, L]
                          attention_mask [B, L]

input_ids ── frozen embeddings ──▶ emb [B, L, 768]
                                       │
                              cls_emb = emb[:, 0, :]  [B, 768]
                                       │
            ┌──────────── router (2M-1 logits → softmax → top-M) ────────────┐
            │   gate(cls_emb)  ─▶ logits [B, 2M-1]                           │
            │                  ─▶ omega  [B, 2M-1]   (softmax)               │
            │                  ─▶ top_v  [B, M],  top_i [B, M]               │
            │                  ─▶ expert_ids = map(top_i, c) ∈ [0, M)        │
            └─────────────────────────────────────────────────────────────────┘
                                       │
            inject (top_v, expert_ids) onto every LoRAMoELinear
                                       │
            BERT encoder (12 layers, frozen attn-out / FF; LoRA on q,v)
                                       │
            ─ for each LoRAMoELinear:                                        ─
              out_base   = base_layer(x)                                     │
              out_lora   = Σ_m top_v[:, m] · (B_{e_m} · A_{e_m} · x)         │
              return out_base + (α/r) · out_lora                             │
            ─────────────────────────────────────────────────────────────────┘
                                       │
            pooler_output [B, 768]
                                       │
            classifier ──▶ logits [B, 3]
                                       │
            cross-entropy(logits, labels)
                                       │
            backward → update {A_e, B_e for sampled e}, router gate, classifier
```

This is the canonical mental model. Every subsystem in this codebase
serves exactly one job in this diagram. If your debugging question is
"where does X come from?", locate X above and follow the arrows back.

---

## 7. Synchronisation and asynchrony

This implementation is **synchronous**: round t+1 cannot start until
every client has uploaded its round-t payload. The server waits, then
aggregates, then broadcasts.

This is a deliberate research-experiment choice; nothing in the
algorithm forbids asynchronous variants, but synchronous training keeps
the science clean (round t's aggregate is a deterministic function of
the round-t local updates).

Within a round, clients are simulated sequentially on a single device
for reproducibility. In a production deployment they would run in
parallel on separate hosts.

---

## 8. Failure modes and how they are caught

A few classes of bug are easy to introduce in this kind of codebase and
the implementation guards against them:

1. **Stale routing state.** If `_routing_weights` is left set on a
   LoRAMoELinear from a previous batch, the next batch will silently
   compute LoRA outputs with the wrong shape. Guarded by the
   `try / finally` in `FedLEASEFinBERT.forward`.
2. **NumPy int indexing into HuggingFace columns.** Indexing
   `dataset["text"]` with `np.int64` raises (the column does not accept
   it). Fixed in `data/data_partitioner.py` by `list(...)`-wrapping
   every column before indexing.
3. **Expert bank desynchronisation.** If `cluster_labels` are not
   stable between phases A and B, the assigned-expert guarantee fails.
   Cluster labels are computed once at the end of phase A and frozen
   for the entire training.
4. **Imbalanced cluster sizes.** The silhouette criterion does not
   forbid singleton clusters. If one occurs, that client's "cluster
   average" is just itself, and the round becomes locally personalised
   — which is the correct behaviour, but it is worth being aware of.

Continue to `METHODOLOGY.md` for the mathematics of LoRA, MoE,
clustering, routing, and aggregation, or to `EXPERIMENTS.md` for the
financial-NLP-specific setup.
