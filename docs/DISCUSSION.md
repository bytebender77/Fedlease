# Advantages, Limitations, and Contribution Summary

This document covers sections **13 (Expected Advantages)**,
**14 (Limitations and Research Challenges)**, and
**15 (Research Contribution Summary)** of the exposition. It is the
"discussion section" you would write last in a paper.

---

## 1. Expected advantages

### 1.1 Better personalisation under heterogeneity

The orthodox personalisation/sharing tradeoff in federated learning is
single-axis: more sharing buys lower variance, more personalisation
buys lower bias. FedLEASE collapses the tradeoff into two axes —
intra-cluster sharing and cross-cluster non-sharing — and lets the
silhouette criterion pick the balance.

Concretely, two PhraseBank clients in the same cluster *share* the
assigned expert (averaged into a stable, low-variance estimate of the
domain's task-update direction); two clients in different clusters
*do not share* their assigned experts (so the formal-prose update
direction is never averaged with the tweet update direction).

The expected payoff in our headline experiment is several macro-F1
points over FedAvg-LoRA on both datasets simultaneously. Crucially,
neither dataset is sacrificed.

### 1.2 Reduced negative transfer

Negative transfer (the phenomenon described in `PROBLEM.md §3.1`) is
*structurally* eliminated for cross-cluster client pairs because their
gradients are never averaged together. It is *empirically* small for
within-cluster pairs because those clients are, by silhouette, close
in B-space — which is itself a proxy for "small gradient
dissimilarity". The cluster-wise FedAvg variance reduction can be
seen as decomposing

> `Var[FedAvg] = Var[FedAvg within] + Var[FedAvg across]`

and zeroing out the "across" term by construction.

### 1.3 Parameter efficiency

The total trainable parameter count of FedLEASE-FinBERT, with M = 2
experts and `[query, value]` LoRA on 12 layers:

| Group | Params |
|---|---|
| LoRA A matrices, all experts, all layers | M · |L| · r · d_in = 2 · 24 · 4 · 768 ≈ 147 K |
| LoRA B matrices, all experts, all layers | M · |L| · d_out · r = 2 · 24 · 768 · 4 ≈ 147 K |
| Router gates (per client, summed: ignored for "model size") | (2M − 1) · d_hidden ≈ 2.3 K each |
| Classifier head | 768 · 3 ≈ 2.3 K |
| **Total trainable** | **≈ 297 K** |
| FinBERT frozen backbone (not trainable) | 110 M |
| **Trainable fraction** | **≈ 0.27%** |

This is the source of the communication efficiency claim: per-round
upload is a few hundred KB per client at fp32, not hundreds of MB.

### 1.4 Communication efficiency

For T = 25 rounds, N = 10 clients, r = 4, M = 2:

- **Per-round per-client upload** ≈ 600 KB.
- **Total federation traffic** ≈ 25 × 10 × 600 KB ≈ 150 MB.

For comparison, full-model FedAvg of FinBERT would be 25 × 10 × 440 MB
≈ 110 GB, which is *730× more*. The communication cost is the
single most cited reason federated foundation-model fine-tuning is
considered impractical; LoRA-MoE makes it routine.

### 1.5 Adaptive specialisation

Because M is chosen by silhouette and experts are initialised from
the warmup geometry, the federation *finds the right amount of
specialisation* automatically. A federation of identical clients
collapses to M = 1 (or M = 2 with weakly-distinguishable clusters
and low silhouette); a federation of K distinct domains finds M = K.
The operator never picks M.

### 1.6 Scalability

Two senses of scalability matter:

- **Scalability in N (number of clients).** Communication is O(N)
  per round, aggregation is O(N) per cluster. The silhouette sweep
  is O(N²) but it only runs *once* after warmup. The system scales
  to hundreds of clients without changing the algorithm.
- **Scalability in M_max (number of experts).** Each forward pass
  costs O(M · r · d²) extra over the frozen backbone. With M ≤ 8 and
  r = 4 this is a < 5% overhead. The 2M − 1 router is O(M · d), also
  negligible.

### 1.7 Conceptual comparison to baselines

| Method | Personalisation | Information sharing | Negative transfer | Communication | Discovers structure? |
|---|---|---|---|---|---|
| Centralised fine-tune | none | maximal | n/a | n/a (no FL) | no |
| FedAvg (full model) | none | maximal | severe under non-IID | very high | no |
| FedAvg-LoRA | none | maximal (over LoRA) | severe over LoRA | low | no |
| Per-client LoRA (no aggregation) | maximal | none | none | none | no |
| Fixed top-k LoRA-MoE | partial | partial | partial | low | no |
| **FedLEASE** (this) | **per-cluster + per-client routing** | **within cluster only** | **eliminated cross-cluster, suppressed within-cluster** | **low** | **yes (silhouette)** |

---

## 2. Limitations and research challenges

The honest version. Each item is a real issue we have either accepted
or partially mitigated.

### 2.1 Routing instability

Soft-routing MoEs are famously hard to train stably. Our 2M − 1
construction prevents the *worst* failure mode (expert collapse at
the assigned expert) but does not eliminate routing instability
entirely:

- The *non*-assigned-expert mass can collapse, leaving the router
  effectively top-1 on the assigned expert. This is benign for
  in-domain inputs but loses cross-cluster generalisation.
- The straight-through gradient through the top-M index argmax is
  biased. Empirically this manifests as routers that take many
  rounds to commit to a non-trivial cross-cluster mixture.

Mitigations in the codebase: dropout on the LoRA path, gradient
clipping at 1.0, mixed-precision GradScaler. None of these are
silver bullets — better routers (gumbel-softmax, expert-choice
routing, soft top-k) are obvious follow-ups.

### 2.2 Expert collapse at the non-assigned slots

In a federation with very strong within-cluster homogeneity, the
non-assigned experts may receive almost no router mass, and the
gradient signal pushed into them is dominated by noise. They drift
slowly into approximately-zero ΔW. This is not a correctness bug
but it wastes parameters — a federation with two perfectly-separated
populations may be better served by M = 2 explicit experts and *no*
cross-expert routing at all (i.e. hard assignment).

A natural extension is to penalise dead experts with an auxiliary
load-balancing loss (cf. Shazeer 2017's "importance" auxiliary).

### 2.3 Convergence challenges in extreme non-IID

When the Dirichlet α is very small (say 0.1), some clients see only
one or two classes. The classifier head, which is FedAvg'd globally,
receives conflicting gradient signals — one client pushes "always
predict positive", another "always predict negative". The head
becomes a battleground. The LoRA experts are largely insulated from
this because they sit before the head, but the deliverable model is
still hurt.

Possible fixes: per-cluster classifier heads (a deeper personalisation
of the read-out), or a balanced sampler at each client that re-weights
minority classes during local SGD.

### 2.4 Communication overhead from warmup

The warmup phase costs one extra round of communication per client
plus the local epochs. For T = 25 rounds this is a 4% overhead, which
is acceptable. For very small T (T = 5, e.g. a privacy-preserving
quick adaptation), the warmup is a 20% tax. There is room for a
*streaming* version that interleaves warmup with the first few rounds
of clustered training.

### 2.5 Clustering sensitivity

Silhouette is sensitive to outliers — a single client whose B is
distant from everyone else's can produce a singleton cluster and
artificially inflate S(k) at large k. Two mitigations are possible:

1. Floor the cluster size at some `min_cluster_size` and merge
   singletons into their nearest neighbour cluster.
2. Use a robustified silhouette (median rather than mean).

Neither is implemented in the headline run; both are one-line changes
in `clustering/expert_allocation.py`.

There is also a subtler issue: silhouette is computed on a *single
snapshot* of the warmup B matrices. If the warmup were run for
longer (or shorter), the clustering could in principle differ. We
have observed that 3 warmup epochs are sufficient for cluster
stability in our datasets; this should be re-verified for each new
domain.

### 2.6 Non-IID label balance

The dataset partitioner is class-stratified for the heterogeneous
strategy but not for IID and not always for Dirichlet. This means
that in some configurations a client may have zero examples of one
class, which corrupts the per-client validation metrics. We report
*aggregate* metrics across clients to avoid this, but per-client
metrics should be interpreted with the dataset sizes in mind.

### 2.7 Privacy is not differentially private

FedLEASE provides **structural** privacy (raw text never leaves the
client) but not **provable** privacy. A determined adversary with
access to the server's broadcasts could in principle reconstruct
aspects of the client distributions from the uploaded LoRA updates
(Wang et al., 2020 show reconstruction attacks against shared
gradients). The natural extension is DP-FedLEASE: clip per-example
gradients and add calibrated Gaussian noise to the upload payload.
This is straightforward to bolt on but is left as future work.

---

## 3. Future improvements and extensions

### 3.1 Stronger routing mechanisms

- **Gumbel-softmax top-M** to give differentiable index selection.
- **Expert-choice routing** (Zhou et al., 2022) where experts pick
  inputs rather than inputs picking experts. Naturally balances load.
- **Sparsely-gated routing** (Shazeer et al., 2017) with an
  importance auxiliary loss.

### 3.2 Dynamic clustering

The current implementation clusters *once* after warmup and freezes
the assignment. As the experts evolve during Phase B, the clustering
might drift. A natural extension: re-cluster every K rounds based on
the current expert states, and warm-restart the experts.

### 3.3 Hierarchical experts

For very large federations, two-level hierarchies (a coarse cluster
of clusters) reduce the per-router output dimension and allow
specialisation at multiple scales. The 2M − 1 router generalises to
a tree of assigned-anchor routers; the math carries over cleanly.

### 3.4 Larger and longer backbones

The codebase is backbone-agnostic up to choice of target modules.
Substituting `FinBERT-LARGE`, `LLaMA-2-7B-Finance`, or
`mistral-7b-finetuned-finance` requires changes only in
`models/finbert_lora_moe.py` (the wrapper) and possibly in the
choice of target modules (LLMs typically target `q_proj, k_proj,
v_proj, o_proj, gate_proj, up_proj, down_proj`).

### 3.5 Beyond sentiment

Sentiment is one of the cleanest financial NLP tasks. Real
applications often involve:

- Named entity recognition for financial entities.
- Event extraction from earnings calls.
- Document-level multi-aspect sentiment.
- Time-conditional classification (the same news has different
  sentiment depending on the price action).

FedLEASE's heterogeneity-handling is *more* valuable on these
multi-faceted tasks, not less; the experiments here are an
existence proof, not the ceiling.

### 3.6 Privacy-preserving extensions

- **Differential privacy** at the per-client gradient level.
- **Secure aggregation** so the server only ever sees the aggregated
  payload, never per-client uploads.
- **Encrypted similarity computation** for the clustering step — the
  server should be able to cluster without seeing raw B matrices.
  Homomorphic-encryption-friendly cosine similarity is an active
  research area.

---

## 4. Research contribution summary

### 4.1 Core novelty

The combination of:

1. **B-matrix similarity** as the client-similarity signal. Avoids
   data-space embedding leakage; exploits the rank-r bottleneck as
   a task-information filter.
2. **Silhouette-optimised agglomerative clustering** for M selection.
   Removes the operator hyperparameter; makes the federation
   *self-organising*.
3. **Intra-cluster expert initialisation** from warmup LoRAs.
   Warm-starts Phase B from a position that is already
   register-specialised.
4. **2M − 1 adaptive top-M router**. Provably guarantees assigned-
   expert participation; eliminates the worst MoE failure mode at
   no extra parameter cost relative to top-M softmax.
5. **Cluster-wise FedAvg aggregation**. Decomposes the FedAvg
   variance into within- and across-cluster terms and eliminates
   the across-cluster term.

None of these five ideas is individually unprecedented, but their
*composition* into a single algorithm is the FedLEASE contribution,
and this codebase realises it end-to-end for a non-trivial financial
NLP testbed.

### 4.2 Methodological contribution

A complete, modular, well-tested implementation pipeline:

- Two-phase training orchestrator (`training/fedlease_pipeline.py`).
- Independent, reusable components: dataset loaders, partitioners,
  LoRA-MoE layers, adaptive router, similarity / clustering /
  allocation modules, federated client/server abstractions,
  cluster-wise aggregation utilities, evaluation, and visualisation.
- Configuration via YAML with CLI overrides, full seed control,
  output artefacts that capture every intermediate state needed for
  post-hoc analysis and reproducibility.

### 4.3 Engineering contribution

Several non-obvious engineering choices were necessary to make the
system work cleanly:

- **Routing state injection** as transient attributes on
  `LoRAMoELinear`, cleared in a `finally` block. Threads per-batch
  routing through the standard HuggingFace BERT forward without
  reimplementing the encoder.
- **`inputs_embeds` trick** to compute embeddings once for both the
  router and the encoder.
- **Lazy `__init__.py` imports** (`__getattr__` pattern) to keep
  module discovery cheap and avoid loading `transformers` until
  actually needed.
- **List-wrapping HuggingFace columns** to allow NumPy integer
  indexing during data partitioning.
- **Masked grouped matmul** in `LoRAMoELinear.forward` so that
  each example only pays the cost of its M experts, not all
  `n_experts`.

These are the kind of details that distinguish a working
implementation from a paper-style pseudocode listing.

### 4.4 Research significance

Federated learning literature has converged on two unsatisfying
endpoints: full sharing (FedAvg variants) and full personalisation
(local fine-tuning, per-client adapters). The space of *structured
intermediate* methods has been mostly unexplored at the level of
foundation models.

FedLEASE-FinBERT is an empirical existence proof that

- the latent structure of a federation can be *recovered* from
  parameter-space similarity,
- *recovered structure* is sufficient to allocate experts in a way
  that beats both endpoints,
- and the entire procedure costs the same per-round bandwidth as
  vanilla FedAvg-LoRA.

This positions the work as a step toward **structurally adaptive
federated foundation-model training** — an ambition that, until
LoRA made the communication tractable, was theoretical. The
domain choice (financial NLP) is not incidental; it is the rare
domain where *all three* of the federated-learning preconditions
(legal non-poolability, statistical heterogeneity, foundation-model
applicability) hold simultaneously.

### 4.5 Replicability and openness

The complete training pipeline, configuration files, all baseline
configurations, the partitioning strategies, the silhouette and
clustering code, the LoRA-MoE layer, and the adaptive router are
shipped in this repository. Reproducing the headline numbers is a
single CLI invocation:

```bash
python experiments/run_fedlease.py --config configs/finbert_fedlease.yaml
```

We consider this the minimum bar for a "research-grade"
implementation and the entry condition for the method to be useful
to others.

---

## 5. One-paragraph summary

> FedLEASE-FinBERT trains a single ProsusAI/FinBERT backbone across a
> federation of clients holding heterogeneous financial text — formal
> analyst prose and informal retail tweets — without ever centralising
> the raw text. The method is a two-phase clustered LoRA Mixture-of-
> Experts: a short warmup produces per-client LoRA adapters whose
> B-matrices are clustered by cosine similarity into a silhouette-
> optimal number of expert groups; cluster centroids initialise a
> shared expert bank; iterative federated rounds then train the
> expert bank under an adaptive top-M router whose 2M − 1 output
> dimension structurally guarantees that each client's assigned
> expert always participates, while still allowing the router to
> borrow weight from other-cluster experts on a per-input basis.
> Aggregation is cluster-wise FedAvg of the assigned expert, which
> eliminates the cross-cluster negative-transfer term that vanilla
> FedAvg-LoRA cannot avoid. The method is parameter-efficient
> (≈ 0.27% of FinBERT trainable), communication-efficient (≈ 150 MB
> total federation traffic for the 10-client, 25-round headline
> run), self-organising (M is chosen by silhouette, not by the
> operator), and structurally personalisation-preserving (per-client
> routers + assigned-expert anchoring). The implementation is
> modular, deterministic up to GPU non-determinism, fully configured
> via YAML, and reproduces from a single CLI invocation. It is
> intended as a reference implementation of FedLEASE for the
> financial NLP domain and as a substrate for follow-up research on
> stronger routing, dynamic clustering, hierarchical experts, and
> differentially-private federated foundation-model fine-tuning.
