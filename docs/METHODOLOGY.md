# Methodology: LoRA, MoE, Clustering, Routing, Aggregation

This document covers sections **6 through 10** of the exposition — the
mathematical core of FedLEASE-FinBERT. It is intentionally formula-heavy
and is the right reference if you are reproducing the method from
scratch.

Notation conventions used throughout:

| Symbol | Meaning |
|---|---|
| `N` | Number of federated clients |
| `i, j` | Client indices, i, j ∈ [N] |
| `M` | Number of experts (decided by clustering) |
| `r` | LoRA rank (default 4) |
| `α` | LoRA scaling factor (default 8) |
| `L` | Set of adapter-equipped layers; `\|L\|` is its size |
| `ℓ` | Layer index, ℓ ∈ L |
| `d_in, d_out` | Input/output dimension of a target projection (768 for FinBERT) |
| `D_i` | Client i's training set; `n_i = \|D_i\|` |
| `c(i)` | Cluster ID of client i, c(i) ∈ [M] |

---

## 1. LoRA: low-rank parameter-efficient adaptation

### 1.1 The base equation

A LoRA-augmented linear layer replaces a frozen `W_0 ∈ R^{d_out × d_in}`
with

> `y = W_0 x + ΔW x`,   `ΔW = (α / r) · B A`,
>
> `A ∈ R^{r × d_in}`,   `B ∈ R^{d_out × r}`.

Only `(A, B)` are trainable; `W_0` is frozen at its pre-trained values.
The factor `(α / r)` is a scaling constant that makes the
effective magnitude of `ΔW` invariant to the choice of `r`.

### 1.2 Why low-rank works at all

The empirical observation behind LoRA (Hu et al., 2022) is that the
update needed to specialise a pre-trained model to a downstream task
has **low intrinsic rank**. Fine-tuning a model of d × d ≈ 600 000
parameters typically pushes those parameters along a subspace of
effective dimension ≪ d. Formally, the optimal task update Δ* has fast
singular-value decay, and a rank-r approximation `B A` with r ≪ d
captures most of the relevant directions.

In our LoRA-MoE setting r = 4 captures the task signal cleanly for
financial sentiment because the task involves a small number of
"axes of meaning" (polarity, hedge, certainty), not a wholesale
re-specialisation of the backbone.

### 1.3 Parameter savings

For one target projection `W_0 ∈ R^{768 × 768}`:

| Quantity | Count |
|---|---|
| Full fine-tune | 768 × 768 = **589 824** |
| LoRA r = 4 | 4·768 + 768·4 = **6 144** |
| Savings | **≈ 96×** |

FinBERT has 12 transformer layers and we apply LoRA to two projections
per layer (query, value), so the *total* trainable parameter count for
the warmup model is:

> 24 layers × 6 144 = **≈ 147 K trainable** vs **110 M frozen**.

This is the number that crosses the network per upload.

### 1.4 Why query and value, not key or output

The choice of target modules is consequential. We follow the standard
LoRA recipe of **query and value only**, for three reasons:

1. **Empirical:** Hu et al. and many follow-ups show that for
   classification-style tasks on BERT-family models, `{q, v}` LoRA
   matches `{q, k, v, o}` LoRA at half the parameter cost.
2. **Mechanistic:** the attention output is `softmax(QK^T/√d_k) V`.
   Adapting V changes *what attention writes*; adapting Q changes
   *what attention asks for*. K and the output projection contribute
   redundant degrees of freedom because the softmax is invariant to
   a joint rescaling of Q and K.
3. **Federated:** smaller LoRA = smaller upload.

### 1.5 Initialisation

`A ~ Kaiming-uniform`, `B ≡ 0`. This makes `ΔW = 0` at step 0, i.e.
the model is identical to the frozen pre-trained backbone before any
training. Crucially, *B starts at zero and accumulates the task signal*,
which is why B-similarity is the right clustering object (see §3.2).

---

## 2. LoRA-MoE: multiple experts per layer

### 2.1 The expert-augmented linear layer

We replace the single LoRA pair with `n_experts` independent pairs:

> `ΔW_e = (α / r) · B_e A_e`,    `e ∈ [n_experts]`.

The forward pass, given per-batch routing weights
`ω ∈ R^{B × n_experts}` (we'll derive ω in §4), is

> `y = W_0 x + Σ_{e=1}^{n_experts} ω_e · (α/r) · B_e A_e x`.

In our implementation `ω` is *sparse* — for each batch element only `M`
entries are non-zero (the top-M selected experts). We therefore only
compute the LoRA contribution for those `M` experts, which is
implemented as a masked grouped matmul (see
`models/lora_layers.py: LoRAMoELinear.forward`).

### 2.2 Why multiple experts

The motivation is in `PROBLEM.md §4`: a single LoRA across
heterogeneous clients suffers from gradient interference. Multiple
experts allow disjoint regions of the input distribution to be served
by disjoint update directions. The MoE is the *parameter-space
factorisation* of the data-space heterogeneity.

### 2.3 Expert specialisation through assigned-expert routing

Naive soft-routing MoE collapses (one expert dominates) or fragments
(experts converge to each other). FedLEASE prevents both:

- **Anti-collapse:** the assigned-expert guarantee (§4.5) forces a
  client's assigned expert to participate in every forward pass.
  This expert receives gradient on every batch, so it cannot become
  "the unused one".
- **Anti-fragmentation:** within a cluster, the assigned expert is
  *shared* across all member clients via the cluster-wise aggregation
  (§5). So even if a client's gradient pushes its assigned expert
  in an idiosyncratic direction, the averaging step pulls it back
  toward the cluster centroid.

### 2.4 Expert initialisation from clustering

Once the M clusters are known, expert `e` is initialised as

> `A_e^{(0)} = (1 / |C_e|) Σ_{i ∈ C_e} A_i^{warmup}`,
>
> `B_e^{(0)} = (1 / |C_e|) Σ_{i ∈ C_e} B_i^{warmup}`,   for each layer ℓ.

This is **Eq. 3** of the FedLEASE paper. The intuition: the warmup
already produced a per-client adapter; the cluster's "starting point"
expert is the mean of the warmup adapters of its members. Because the
clients in a cluster were already similar (by silhouette), this mean is
a good initialisation rather than a destructive one.

Implementation: `clustering/expert_allocation.py: initialise_experts`.

---

## 3. Similarity and clustering of clients

### 3.1 Distance between two clients

For clients i, j, define

> `d(i, j) = (1 / |L|) · Σ_{ℓ ∈ L} (1 − cos(B_i^ℓ, B_j^ℓ))`,
>
> where `cos(U, V) = ⟨vec(U), vec(V)⟩ / (‖vec(U)‖ · ‖vec(V)‖)`.

This is **Eq. 2** of the paper. We flatten each `B_i^ℓ ∈ R^{d_out × r}`
to a vector of length `d_out · r`, compute cosine similarity, subtract
from 1, and average over layers.

Computed in `clustering/similarity.py: compute_distance_matrix`. The
result is a symmetric matrix `D ∈ R^{N × N}` with zero diagonal.

### 3.2 Why B, why not A, why not both, why not gradients?

| Candidate | Pros | Cons |
|---|---|---|
| **B only** (chosen) | task-signature; starts at 0 and accumulates supervised signal; downstream of low-rank bottleneck so it's information-compressed | none significant for federated case |
| A only | reflects input-side adaptation | dominated by tokeniser / register artifacts that are not task-relevant |
| (A, B) concatenated | "more information" | A's contribution is mostly noise for clustering; empirically worse silhouette |
| Raw gradients | extremely informative | enormous in size; aggregation is harder; raises privacy concerns |
| Mean embeddings of data | natural | privacy-leaky and task-agnostic |

The information-bottleneck argument (`PROBLEM.md §9`) is the cleanest
reason: B is small (r columns), forced to compress the task signal, and
therefore the cosine of two B matrices is a low-noise estimator of
task similarity.

### 3.3 Cosine similarity on flattened matrices: subtleties

We flatten `B` matrices before cosine. This is exact when `r = 1` (then
B is a vector and cosine is the natural Grassmannian similarity). For
`r > 1`, flattened cosine ignores the within-column orthogonality
structure and treats B as a single long vector. Alternatives —
principal-angle distance, Grassmannian geodesic — are mathematically
cleaner but empirically yield very similar clusterings for small r
(r ≤ 8). We stick with flattened cosine because it is faster, has no
SVD step, and matches the FedLEASE paper's reported method.

### 3.4 Agglomerative clustering

Given `D`, we run **agglomerative (bottom-up) hierarchical clustering**
with **average linkage**:

> `d_avg(C_a, C_b) = (1 / (|C_a| · |C_b|)) · Σ_{i ∈ C_a, j ∈ C_b} D_{ij}`.

Implementation: `clustering/clustering.py: build_linkage_matrix`,
backed by `scipy.cluster.hierarchy.linkage`.

Why average linkage rather than Ward or single?

- **Single linkage** suffers from the chaining problem (long stragglers
  pull clusters together).
- **Ward linkage** assumes Euclidean distances; our `D` is cosine-
  derived and not Euclidean-embeddable in general.
- **Average linkage** is the right metric-agnostic compromise; it is
  the silhouette-paper's recommended default for non-Euclidean
  pairwise distances.

### 3.5 Choosing M by silhouette

For each candidate `k ∈ [M_min, M_max]` (default `[2, 8]`):

1. Cut the dendrogram to produce k clusters.
2. For every client i:
   - `a_i = mean intra-cluster distance` from i to other members of c(i).
   - `b_i = min over c ≠ c(i) of mean distance from i to cluster c`.
   - `s_i = (b_i − a_i) / max(a_i, b_i)`.
3. `S(k) = (1/N) Σ_i s_i`.

The chosen number of experts is

> `M* = argmax_{k ∈ [M_min, M_max]} S(k)`.

Implementation: `clustering/expert_allocation.py: ExpertAllocator.allocate`.

**Interpretation.** s_i ∈ [-1, 1]. s_i ≈ 1 means client i is closer to
its own cluster than to any other (good). s_i ≈ 0 means client i lies
on a cluster boundary (ambiguous). s_i < 0 means client i would be
better classified elsewhere (bad). Averaging gives a global
goodness-of-clustering metric that does **not** monotonically increase
with k — adding more clusters generally hurts cohesion past the
"right" k. Hence argmax is a meaningful model selection.

### 3.6 Limits and failure modes of silhouette

- **Boundary problem at k = N.** If we let `M_max → N`, every client
  is its own cluster and intra-cluster distance is undefined. We cap
  `M_max` well below N (default 8 for N = 10).
- **Silhouette is metric-sensitive.** Our distance is `1 − cos(B, B')`,
  not Euclidean, so silhouette values are not directly comparable
  to those in the clustering literature. They are still ordinally
  meaningful, which is all argmax needs.
- **No prior on M.** If the federation actually contains 1 cluster,
  silhouette will still pick k ≥ 2 (we never test k = 1). This is
  a deliberate choice — FedLEASE without experts reduces to
  FedAvg-LoRA and our pipeline does not optimise for that case.

---

## 4. Adaptive Top-M routing

This is the most novel component of the system and deserves the
most detailed treatment.

### 4.1 Router architecture

The router is a single linear projection with no bias:

> `G ∈ R^{(2M − 1) × d}`,    `d = d_hidden = 768`.

Given the CLS embedding `h ∈ R^d` from FinBERT's embedding sublayer,
the router computes

> `logits = G h ∈ R^{2M − 1}`,
>
> `ω̂ = softmax(logits) ∈ Δ^{2M − 1}`.

Implementation: `models/adaptive_router.py: AdaptiveTopMRouter`.

### 4.2 Why the output dimension is 2M − 1

This is the single most non-obvious design choice in the entire system,
and it is the key to the assigned-expert guarantee. The output
dimension is not 2M (which one might naïvely expect), and not M
(which would be a vanilla top-k router); it is **2M − 1**.

The reason is combinatorial. Label the M positions of the output:

```
   slot index p :   0   1   2   ...   M−1   |   M   M+1   ...   2M−2
                   ┌─────────────────────┐  ┌──────────────────────────┐
                   │   M  "internal"     │  │   M−1  "external"        │
                   │   slots that all    │  │   slots, each mapped     │
                   │   point to the      │  │   to a distinct          │
                   │   assigned expert   │  │   non-assigned expert    │
                   └─────────────────────┘  └──────────────────────────┘
```

**Reading the design.** When the router selects top-M positions out of
the 2M − 1 outputs, *at least one* of the selected positions must lie
in the first M-slot block, because the second block has only `M − 1`
slots. Concretely, the router *cannot* pick a top-M that avoids the
assigned-expert block — there are not enough non-assigned slots.

Hence the assigned expert is **deterministically present** in the
top-M result. But — and this is the elegant part — the *number* of
internal slots selected can vary from 1 (if the router strongly
prefers external experts) to M (if the router strongly prefers the
assigned expert). The router thus has a *continuous knob* between
"use only the assigned expert" and "use M − 1 external experts plus
one slot of the assigned expert".

If the output dimension were 2M, the router could select M external
positions and avoid the assigned expert entirely; the guarantee
would not hold. If the output dimension were M − 1 plus M, i.e.
2M − 1, but with the blocks reversed, the guarantee is symmetric.
Either way, 2M − 1 is the *minimal* output dimension for which the
guarantee holds at top-M.

### 4.3 Index mapping

After computing `top_values, top_indices = torch.topk(ω̂, M)`, we map
each slot index `p ∈ [0, 2M − 1)` to an expert ID `e ∈ [0, M)` via:

```python
def _map_to_expert_indices(self, router_indices):
    # p < M       → assigned expert  (= self.assigned_expert_idx)
    # p >= M      → other_experts[p - M]
    #   where other_experts is the sorted list of [0, M) \ {assigned_idx}
```

The other-experts list is a fixed permutation set at router
construction time. So if M = 4 and the client is in cluster 1, then:

```
slot p:        0   1   2   3   |   4   5   6
maps to:       1   1   1   1   |   0   2   3
                ^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^
                assigned expert     other experts
```

For a single example, if the router's top-M is `{2, 5, 1, 0}`, the
expert IDs are `{1, 2, 1, 1}` — i.e. expert 1 (assigned) appears
three times with three different weights, plus one slot of expert 2.
The masked grouped-matmul in `LoRAMoELinear.forward` then *sums* the
contributions of expert 1's three slots before applying the LoRA
update, so the assigned expert effectively gets weight
`ω̂_2 + ω̂_1 + ω̂_0` and expert 2 gets `ω̂_5`.

### 4.4 Routing example

Concrete example with M = 3 (so router output dim = 5),
assigned_expert = 0, other = [1, 2]:

```
input h          → logits ω̂ (softmax) over slots [0, 1, 2, 3, 4]
                                       ↓
                                       say ω̂ = [0.1, 0.4, 0.05, 0.1, 0.35]
                                       ↓
top-3 by value   → slots [1, 4, 0],  weights [0.4, 0.35, 0.1]
                                       ↓
map to experts   → [0, 2, 0]
                                       ↓
group by expert  → expert 0 with combined weight 0.5,
                   expert 2 with weight 0.35
                                       ↓
LoRA out         → 0.5 · B_0 A_0 x  +  0.35 · B_2 A_2 x
```

Note that the weights `[0.4, 0.35, 0.1]` *do not sum to 1*. They sum to
the mass of top-3 ≈ 0.85; the remaining 0.15 of probability is on the
non-selected slots and is dropped. The implementation does not
re-normalise the top-M weights, mirroring the standard hard-MoE
practice (Shazeer et al. 2017, Fedus et al. 2022). Renormalising
would change the LoRA magnitude.

### 4.5 The assigned-expert guarantee, formally

**Claim.** For any softmax distribution `ω̂` over 2M − 1 slots, the
top-M argmax-set `S = argtop_M(ω̂)` satisfies
`S ∩ {0, …, M − 1} ≠ ∅`.

**Proof.** `|S| = M`. `|{M, …, 2M − 2}| = M − 1 < M = |S|`. So at
least one element of `S` lies outside `{M, …, 2M − 2}`, i.e. in
`{0, …, M − 1}`, which is the assigned-expert block. ∎

This is why expert collapse cannot happen at the assigned expert in
FedLEASE. Every client *must* push gradient into its assigned expert
on every batch, regardless of router state.

### 4.6 Training behaviour vs inference behaviour

There is no training-time / inference-time gap. The routing computation
is the same in both modes:

- **Training.** Gradients flow through `ω̂` (softmax is differentiable),
  through the selected `B_e A_e x` terms, into both `(A_e, B_e)` and
  the gate matrix `G`. The top-M selection is *not* differentiable
  through the index — we use the straight-through identity that
  treats the indices as constants and only routes gradient through
  `top_values`. This is standard MoE practice and is sufficient because
  the values *do* carry the gradient signal that pushes the router
  toward better routings over time.
- **Inference.** Same forward pass, no gradient. Routing is
  deterministic given `G`.

### 4.7 Comparison with fixed top-k MoE

A standard top-k softmax MoE with k = M and N_experts = M experts is
parameter-equivalent to our router (one gate matrix of comparable
size). The differences are:

|  | Fixed top-k MoE | FedLEASE adaptive top-M |
|---|---|---|
| Gate output dim | M | 2M − 1 |
| Assigned expert guaranteed? | ❌ | ✅ |
| Can collapse to one expert? | ✅ (common failure) | ❌ at the assigned expert |
| Per-client structural prior? | ❌ | ✅ via assigned_expert_idx |
| Federated stability? | poor | good |

The 2M − 1 router is the *minimum-complexity* modification of soft
top-k MoE that buys the assigned-expert guarantee. There is no extra
parameter explosion.

---

## 5. Aggregation strategy

### 5.1 What gets aggregated

In Phase B, every client uploads:

- Its assigned expert's `(A_{c(i)}^ℓ, B_{c(i)}^ℓ)` for every adapter
  layer ℓ.
- Its router gate `G_i ∈ R^{(2M − 1) × d}`.
- Its dataset size n_i.

(See the per-parameter table in `ARCHITECTURE.md §5`.)

### 5.2 Cluster-wise weighted averaging

For each cluster c ∈ [M] and each adapter layer ℓ, the server computes

> `A_c^{ℓ, (t+1)} = Σ_{i ∈ C_c} (n_i / Σ_{j ∈ C_c} n_j) · A_{c(i)}^{ℓ, (t)}`,
>
> `B_c^{ℓ, (t+1)} = Σ_{i ∈ C_c} (n_i / Σ_{j ∈ C_c} n_j) · B_{c(i)}^{ℓ, (t)}`.

This is the standard FedAvg weighting *restricted to cluster c*.
Implementation: `federated/aggregation.py: cluster_wise_aggregation`.

### 5.3 Why aggregate within clusters, not globally

If we aggregated globally — i.e. averaged every client's assigned
expert into a single global LoRA — we would be exactly back to
FedAvg-LoRA. The whole point of clustering is to **limit the
averaging blast radius** to clients whose data is similar.

The negative-transfer cost of averaging two clients i, j scales
with their gradient dissimilarity Γ_{ij}². For within-cluster
pairs, Γ_{ij}² is small by construction (that is what the silhouette
optimum certifies). For cross-cluster pairs, Γ_{ij}² is large.
Cluster-wise aggregation suppresses the large-Γ² contributions
that destroy the FedAvg variance.

### 5.4 Router aggregation

Each client retains *its own* router. We do not aggregate router
gates across clients in this implementation. Two reasons:

1. **Personalisation:** the router gate encodes which experts a
   particular client tends to need. Even within a cluster, individual
   clients may have idiosyncratic mixtures of in-domain and
   out-of-domain inputs.
2. **Communication:** the router gate is small enough that the
   per-client routing cost is negligible. Globalising it would save
   little and lose personalisation.

This is a defensible choice in either direction; the original FedLEASE
paper aggregates routers per-cluster, and this implementation can be
extended to do so trivially by re-using `cluster_wise_aggregation`
on the gate parameter.

### 5.5 Classifier head aggregation

The pooler / classifier head *is* aggregated globally with FedAvg
weighting:

> `θ_head^{(t+1)} = Σ_i (n_i / Σ_j n_j) · θ_head,i^{(t)}`.

Justification: the head maps a final 768-dim pooled representation to
3 sentiment logits. The mapping from "representation" to "logit" is
not domain-specific in the way that the encoder updates are — every
client agrees on what "negative" and "positive" mean (we normalise
labels to the same space upstream). So global averaging is the right
thing here, and it gives us a single deliverable classifier at the
end of training.

### 5.6 How aggregation preserves personalisation

Personalisation in FedLEASE is *not* preserved by per-client
parameters (the cluster expert is shared within the cluster). It is
preserved by **per-client routing** plus **structural anchoring on
the assigned expert**. Two clients in the same cluster receive
identical expert weights after aggregation but will route inputs
through that expert with different weights, and will mix in
different out-of-cluster experts as their gates direct.

This is a *thinner* form of personalisation than per-client LoRA but
a *thicker* form than FedAvg's "everyone uses the same model". The
empirical question — does the resulting per-client risk beat both
endpoints? — is answered in `EXPERIMENTS.md`.

---

## 6. End-to-end mathematical summary

The FedLEASE-FinBERT loss at round t is

> `L^{(t)} = Σ_i (n_i / Σ_j n_j) · E_{(x, y) ∼ D_i} [ ℓ(f_i(x; θ^{(t)}), y) ]`,

where the per-client model `f_i` is

> `f_i(x; θ^{(t)}) = head( BERT_{W_0}( x ; { ΔW_e^ℓ }_{ℓ ∈ L, e ∈ [M]}, G_i ) )`,

and the per-target-module LoRA-MoE update is

> `ΔW^ℓ(x) = Σ_{m=1}^{M} ω̂_{σ(m)}(h_x) · (α/r) · B_{e(σ(m))}^ℓ A_{e(σ(m))}^ℓ`,
>
> `σ = argtop_M(ω̂(h_x))`,    `e(p) = c(i)` if `p < M` else `other_experts[p − M]`,
>
> `ω̂(h) = softmax(G_i · h) ∈ Δ^{2M − 1}`.

The training procedure alternates:

1. **Local SGD** on each client w.r.t.
   `{A_{c(i)}, B_{c(i)}, G_i, θ_head}` (and the broadcast copies of
   other-cluster experts, which receive gradient when sampled but
   are not uploaded).
2. **Cluster-wise aggregation** of `(A_c, B_c)` and global FedAvg
   of `θ_head`.

Initialisation:

- `(A, B)` of phase A: standard LoRA init (`A ~ Kaiming`, `B = 0`).
- Cluster assignments c(i): from silhouette-optimal agglomerative
  clustering of the warmup B matrices.
- `(A_c^{(0)}, B_c^{(0)})` of phase B: intra-cluster mean of warmup
  LoRA pairs (Eq. 3).
- Gate `G_i`: small random init; assigned-expert structural prior
  embedded in the index map, not in the gate values.

This is the entire algorithm.

Continue to `EXPERIMENTS.md` for the dataset and protocol details, or
to `DISCUSSION.md` for advantages, limitations, and contributions.
