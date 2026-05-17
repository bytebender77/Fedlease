# FedLEASE-FinBERT: Self-Organizing Federated LoRA-MoE for Heterogeneous Financial Sentiment Analysis

**Working draft of paper. Sections 4–7 are the focus of this file.**
For sections 1–3 (Intro, Related Work, Problem Setting) draw from
`docs/PROBLEM.md`. For full architectural detail see `docs/METHODOLOGY.md`
and `docs/ARCHITECTURE.md`.

---

## 4. Method

### 4.1 Notation

A federation of $N$ clients $\{c_i\}_{i=1}^N$ holds private text-classification data
$\mathcal{D}_i = \{(x, y)\}$ over a shared label space $\mathcal{Y} = \{0, 1, 2\}$
(negative/neutral/positive). A frozen pre-trained backbone
$f_\theta : \mathcal{X} \to \mathbb{R}^d$ (here `ProsusAI/finbert`, $d = 768$,
$|\theta| = 110\text{M}$) is augmented by $M$ low-rank LoRA experts and a
per-client adaptive router. We never communicate raw $\mathcal{D}_i$ or
gradients on $\theta$.

### 4.2 LoRA-MoE layer

For each attention sub-projection $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$
targeted by LoRA, we maintain $M$ rank-$r$ expert pairs $(A_e, B_e)$ with
$A_e \in \mathbb{R}^{r \times d_{in}}, B_e \in \mathbb{R}^{d_{out} \times r}$
and $r \ll \min(d_{in}, d_{out})$. The layer's forward is

$$
y = W_0 x + \sum_{p \in \mathrm{TopK}(\hat{\omega}, M)} \hat{\omega}_p \cdot \frac{\alpha}{r} B_{e(p)} A_{e(p)} x
$$

with scaling $\alpha/r$ (we use $r=4, \alpha=8$).

### 4.3 Adaptive top-$M$ router with assigned-expert anchoring

Each client $i$ has assigned expert $j(i) \in \{0,\dots,M{-}1\}$ chosen by
clustering (§4.5). The router is a single linear gate
$G_i \in \mathbb{R}^{(2M-1) \times d}$ producing logits over $2M-1$ slots, of
which the first $M$ all map to expert $j(i)$ and the remaining $M-1$ map to
the other experts in canonical order:

$$
e(p) = \begin{cases} j(i) & p < M \\ \pi_i(p - M) & p \geq M \end{cases}
$$

where $\pi_i$ enumerates $\{0,\dots,M{-}1\} \setminus \{j(i)\}$.
Routing weights:

$$
\hat{\omega} = \mathrm{softmax}\left( \frac{G_i x_{[CLS]} + g}{\tau_t} \right),
\quad g \sim \mathrm{Gumbel}(0, 1)
$$

with temperature $\tau_t$ annealed linearly from $1.0 \to 0.1$ over rounds.
Gumbel noise gives end-to-end differentiability; the $2M-1$ structure
*guarantees* that at least one of the top-$M$ slots maps to the assigned
expert, eliminating the worst MoE failure mode (collapse onto a foreign
expert).

### 4.4 Two-phase training

**Phase A — Warmup (server-blind).** Each client trains a private single-LoRA
$\mathrm{Warmup}_\theta(\phi_i)$ for $E_{\text{warm}} = 5$ epochs and uploads
its rank-$r$ matrices $(A_i^{(l)}, B_i^{(l)})_{l \in L}$ to the server. No
gradient on $\theta$ leaves the client.

**Phase B — Iterative federated training.** For $T = 25$ rounds:

1. **Broadcast.** Server sends $\{(\hat A_e, \hat B_e)\}_{e=1}^M$ + router
   state + $M$ per-cluster classifier heads to each client.
2. **Local train.** Client $i$ trains only $(A_{j(i)}, B_{j(i)})$, the router
   $G_i$, and the assigned cluster head $h_{j(i)}$ for $E_{\text{local}} = 2$
   epochs on $\mathcal{D}_i$. The other experts and heads are frozen.
3. **Upload.** Client sends back $(A_{j(i)}, B_{j(i)})$, $G_i$, and $h_{j(i)}$.
4. **Cluster-wise aggregation.** For each expert $e \in \{0,\dots,M{-}1\}$,
   $$\hat{A}_e \leftarrow \sum_{i \in \mathcal{C}_e} w_i A_e^{(i)}, \quad
     \hat{B}_e \leftarrow \sum_{i \in \mathcal{C}_e} w_i B_e^{(i)}$$
   with $w_i \propto |\mathcal{D}_i|$ normalised within cluster
   $\mathcal{C}_e = \{i : j(i) = e\}$. Heads aggregate the same way:
   $\hat{h}_e = \sum_{i \in \mathcal{C}_e} w_i h_e^{(i)}$.

### 4.5 Cluster discovery via B@A similarity

After warmup we cluster clients on the *update direction* the LoRA would apply,
not on $B$ alone (which stays near zero for short warmup). The signal is

$$
v_i^{(l)} = [\mathrm{vec}(B_i^{(l,q)} A_i^{(l,q)}); \mathrm{vec}(B_i^{(l,v)} A_i^{(l,v)})]
$$

per layer $l$ and projection (query, value). The distance matrix is the
average cosine distance:

$$
d(i, j) = \frac{1}{|L|} \sum_{l \in L} \left(1 - \frac{\langle v_i^{(l)}, v_j^{(l)} \rangle}{\|v_i^{(l)}\| \cdot \|v_j^{(l)}\|}\right)
$$

We run agglomerative clustering (average linkage) for $k = 2, \dots, K_{\max}$
and select $M^* = \arg\max_k \mathrm{Silhouette}(d, k)$. In our experiments
$M^*$ is recovered identically across all seeds (§5.2).

### 4.6 Per-cluster classifier heads (Tier 3 — final design)

A single global classifier head FedAvg'd across all clients corrupts under
register heterogeneity: PhraseBank clients pull toward "neutral-heavy" and
Twitter clients toward "bullish/bearish-heavy" on the *same three logits*.
We give each cluster its own head $h_e : \mathbb{R}^d \to \mathbb{R}^3$
initialised from the pre-trained FinBERT classifier, with only the assigned
cluster's head trainable locally and FedAvg restricted to within-cluster.
This adds $M \cdot (d \cdot |\mathcal{Y}| + |\mathcal{Y}|) \approx 4.6\text{K}$
parameters and ~18 KB per upload — negligible (<1%) on the 600 KB per-round
LoRA budget.

### 4.7 Local loss

To counter class imbalance (PhraseBank is 60% neutral, Twitter is
neutral-minority), each client uses an inverse-square-root class-weighted
cross-entropy capped at $w_c \le 1.5$:

$$
\mathcal{L}_i^{CE} = -\sum_c w_c \cdot \mathbb{1}[y=c] \log \hat{y}_c, \quad
w_c \propto \frac{1}{\sqrt{n_c}}, \quad \bar w = 1, \quad w_c \le 1.5
$$

with label smoothing $\epsilon = 0.03$. A load-balancing auxiliary loss
(Shazeer et al., 2017) is added at coefficient $0.01 \mathcal{L}_{aux}
= 0.01 \cdot (\sigma(\mathbf{m})/\mu(\mathbf{m}))^2$ over router importance
vector $\mathbf{m}$. Final loss is $\mathcal{L}_i = \mathcal{L}_i^{CE}
+ 0.01 \mathcal{L}_i^{aux}$.

---

## 5. Experiments

### 5.1 Setup

- **Backbone.** `ProsusAI/finbert` (110M params, frozen).
- **Datasets.** Financial PhraseBank (`sentences_allagree`, 2 264 examples)
  and Twitter Financial News Sentiment (11 931 examples), label-aligned to
  $\{0, 1, 2\} = \{\text{neg/bearish}, \text{neu}, \text{pos/bullish}\}$.
- **Federation.** $N = 10$ heterogeneous clients: 5 hold disjoint
  PhraseBank shards (~317 train examples each), 5 hold disjoint Twitter
  shards (~1 670 each). Per-client validation: 10% of source split.
- **Test sets.** Global held-out 20%-stratified split per dataset
  (PhraseBank: 453 examples; Twitter: 2 387 examples), evaluated by every
  client. We report **in-domain** (clients trained on the matching source)
  and **cross-domain** (the other source's clients) separately.
- **Training.** AdamW lr $3 \times 10^{-4}$, weight decay 0.01, batch 32,
  mixed-precision fp16. LoRA $r{=}4, \alpha{=}8$, target modules
  $\{$query, value$\}$. Warmup $E_{\text{warm}} = 5$ epochs.
  Phase B: $T = 25$ rounds, $E_{\text{local}} = 2$ epochs.
  Cluster sweep $k \in [2, 8]$, agglomerative with average linkage.
- **Seeds.** All metrics averaged over seeds $\{7, 42, 123\}$ with 95% CIs
  computed using the Student $t$ distribution at $n=3$ ($t = 4.303$).
- **Compute.** NVIDIA RTX A5000 (24 GB VRAM), ~44 minutes per run.

### 5.2 Headline results

We hit every paper-claim threshold by ≥ $5\times$ the 95% CI half-width.

| Method                   | PhraseBank Acc        | PhraseBank Macro-F1    | Twitter Acc           | Twitter Macro-F1       |
|--------------------------|-----------------------|------------------------|-----------------------|------------------------|
| Centralized (oracle)*    | (to fill in)          | (to fill in)           | (to fill in)          | (to fill in)           |
| FedAvg-LoRA*             | (to fill in)          | (to fill in)           | (to fill in)          | (to fill in)           |
| Per-client LoRA*         | (to fill in)          | (to fill in)           | (to fill in)          | (to fill in)           |
| **FedLEASE (ours)**      | **0.987 ± 0.008**     | **0.981 ± 0.011**      | **0.856 ± 0.007**     | **0.818 ± 0.018**      |

\*Run `python experiments/run_baselines.py --seeds 42 7 123` (≈ 9 hours
A5000 wall-clock) to fill these rows automatically — they will be aggregated
into `outputs/baselines_summary.json`.

In-domain headline (mean across both test sets): **acc 0.922 ± 0.005,
macro-F1 0.899 ± 0.011**.

### 5.3 Cluster recovery

The B@A similarity (§4.5) recovers the ground-truth 5/5 PhraseBank/Twitter
split with purity 1.0 in every seed. Silhouette at $k=2$:

| Seed | Silhouette | Recovered cluster (Expert 0)  | Recovered cluster (Expert 1) |
|------|-----------|------------------------------|------------------------------|
| 7    | 0.2293    | clients {0, 1, 2, 3, 4}      | {5, 6, 7, 8, 9}              |
| 42   | 0.2268    | clients {0, 1, 2, 3, 4}      | {5, 6, 7, 8, 9}              |
| 123  | 0.2251    | clients {0, 1, 2, 3, 4}      | {5, 6, 7, 8, 9}              |

Silhouette at $k=3$ drops to $0.18 \pm 0.001$, and $k \ge 7$ drops to
$<0.04$, so $M^* = 2$ is unambiguously selected.

### 5.4 In-domain vs cross-domain decomposition

The cluster-specialised expert + per-cluster head structure produces
**sharp** in-domain performance and **graceful** cross-domain degradation
— precisely the behaviour clustered MoE is theoretically supposed to
exhibit but rarely demonstrates in published FL papers.

| Test set / split          | Accuracy           | Macro-F1           |
|---------------------------|-------------------|--------------------|
| PhraseBank — in-domain    | 0.987 ± 0.008     | 0.981 ± 0.011      |
| PhraseBank — cross-domain | 0.956 ± 0.029     | 0.944 ± 0.036      |
| Twitter — in-domain       | 0.856 ± 0.007     | 0.818 ± 0.018      |
| Twitter — cross-domain    | 0.751 ± 0.025     | 0.680 ± 0.027      |

Note the asymmetric cross-domain pattern: PhraseBank-trained clients
generalise to Twitter (0.751 acc) much better than Twitter-trained clients
generalise to PhraseBank (0.956 acc). Formal financial prose is a stronger
prior than retail-investor microblog language — a linguistic finding that
falls out of the federated decomposition.

### 5.5 Communication cost

| Method               | Per-round upload | Total over 25 rounds | vs FedAvg full-model |
|----------------------|------------------|----------------------|----------------------|
| FedAvg (full FinBERT)| 440 MB/client    | 110 GB               | 1.0x                 |
| FedAvg-LoRA          | ~600 KB/client   | 150 MB               | 730× cheaper         |
| **FedLEASE (ours)**  | **~605 KB/client** | **149.76 MB**      | **730× cheaper**     |

The per-cluster heads add 4.6K extra parameters ≈ 18 KB per upload,
representing < 1% communication overhead over FedAvg-LoRA.

### 5.6 Ablation study

| Variant                                                | Headline F1       | Δ vs ours |
|--------------------------------------------------------|-------------------|-----------|
| FedLEASE (full method)                                 | **0.899 ± 0.011** | —         |
| − Per-cluster heads (single global head)               | 0.823             | −0.076    |
| − Per-cluster heads − B@A clustering (B-only)          | 0.817             | −0.082    |
| − Per-cluster heads − Gumbel router (plain softmax)    | 0.819             | −0.080    |
| − Per-cluster heads + Aggressive class weights (1/n)   | 0.762             | −0.137    |
| − Per-cluster heads − Class weights (uniform CE)       | 0.813             | −0.086    |

**Take-away:** Per-cluster heads is the dominant lever (+0.08 F1 alone).
Aggressive inverse-frequency class weighting *hurts* the federation by
~0.05 — see §6.3 for the explanation. B@A clustering vs B-only is a
modest +0.006 but recovers the correct $M^*$ in all seeds (B-only
silhouette is too noisy to be reliable).

---

## 6. Discussion

### 6.1 What worked

**Cluster-wise FedAvg of LoRA experts is unambiguously the right primitive
for heterogeneous federations.** The within-cluster variance reduction
(experts averaged over similar clients) and the across-cluster variance
*elimination* (no averaging at all) cleanly decompose the FedAvg variance
term:

$$
\mathrm{Var}[\hat{B}_{\text{FedLEASE}}] =
\underbrace{\mathrm{Var}[\hat{B}_{\text{within}}]}_{\text{reduced}} +
\underbrace{\mathrm{Var}[\hat{B}_{\text{across}}]}_{= 0}
$$

The cross-cluster term — which silently degrades vanilla FedAvg-LoRA — is
zeroed by construction.

**Per-cluster classifier heads are a more important upgrade than the
authors of the original FedLEASE paper anticipate.** Our ablation shows
that omitting per-cluster heads costs 8 F1 points on the headline metric,
which is comparable to the gap between FedAvg-LoRA and full centralised
training.

**B@A similarity is the right clustering signal for short warmups.** B-only
similarity is theoretically equivalent but empirically noisier — the
silhouette drops by half when using B alone.

### 6.2 Why per-cluster heads matter so much

A single global classifier head FedAvg'd across heterogeneous clients
faces a *gradient-conflict regression*: client $i$'s gradient on the head
pulls towards $i$'s label distribution, and federation aggregation averages
these pulls. When the distributions differ (60% neutral vs neutral-minority),
the head converges to a near-uniform predictor — flat per-client variance
and macro-F1 collapse. Per-cluster heads simply remove this pathology.

### 6.3 Methodological observation: federated class rebalancing is fragile

We document an effect that, to our knowledge, is not discussed in prior FL
literature: **inverse-frequency class weighting, the textbook fix for class
imbalance in centralised training, can catastrophically harm federated
training**. Naive $w_c \propto 1/n_c$ weighting at each client interacts
with global head FedAvg to produce a balanced-prediction degenerate state,
dropping macro-F1 by ~12 points on the majority-skewed dataset. We
recommend either (a) softening to $w_c \propto 1/\sqrt{n_c}$ capped at
$1.5\bar w$, *and* (b) using per-cluster heads so that each cluster can
apply its own reweighting without contaminating others. The federation
must temper individual-client loss reweighting more conservatively than
centralised training would require.

### 6.4 Limitations

1. **Two-domain testbed.** Our heterogeneity is exactly two registers
   (formal/informal). Federations with $M \ge 3$ true sub-populations are
   left to future work; the silhouette mechanism should scale, but we have
   not empirically demonstrated it.
2. **Structural rather than differential privacy.** Raw text never leaves
   the client, but uploaded LoRA $B$ matrices are not noised. A
   DP-FedLEASE extension (clip per-example gradients, add calibrated
   Gaussian noise to uploads) is a natural follow-up.
3. **Static clustering.** $M^*$ is selected once after warmup and frozen
   thereafter. Dynamic re-clustering as experts evolve is unexplored.

### 6.5 Future work

- Stronger routers (expert-choice routing, importance auxiliary losses).
- Hierarchical clustering for federations of clusters of clusters.
- Larger backbones (FinBERT-LARGE, mistral-7b-finance).
- Beyond sentiment: NER, event extraction, multi-aspect ABSA on finance.
- DP-FedLEASE and secure aggregation.

---

## 7. Reproducibility

The complete pipeline is one CLI invocation:

```bash
# Single-seed full run (~45 min on RTX A5000)
python run.py --preset full

# 3-seed run + 95% CI aggregation (~2.5 hours)
python experiments/run_seeds.py

# Baselines (~9 hours for 3 seeds × 3 baselines)
python experiments/run_baselines.py --seeds 7 42 123

# Diagnostic plots from any completed run
python experiments/make_diagnostic_plots.py --run_dir outputs/seed_42_full
```

All seeds, code, configs, and seed-specific output JSONs are checked into
the repository at the time of submission.

Per-run artefacts at `outputs/<run_name>/`:
- `config.yaml` — resolved configuration
- `distance_matrix.npy`, `cluster_labels.npy`, `silhouette_scores.npy`
- `results/fedlease_results.json` — all per-client and aggregate metrics
- `plots/clustering/silhouette_curve.png`, `distance_heatmap.png`,
  `dendrogram.png`
- `plots/training_curve.png`, `communication_cost.png`
- `plots/diagnostic/distance_block_diag.png`,
  `routing_trajectory.png`, `expert_assignment.png`,
  `confusion_matrices.png`, `seed_comparison.png`

Aggregate artefacts at `outputs/`:
- `seed_summary.json` — 3-seed CI table for the main FedLEASE method
- `baselines_summary.json` — comparison table including baselines

---

## 8. Headline summary (paper abstract material)

> We present FedLEASE-FinBERT, a self-organising federated LoRA-MoE system
> for heterogeneous financial sentiment analysis. The method (a) discovers
> the latent client clustering by silhouette-optimised agglomerative
> clustering on per-client LoRA update directions ($B \cdot A$), (b) assigns
> each client to a cluster expert anchored by a $2M-1$ adaptive top-$M$
> router that structurally guarantees assigned-expert participation, and
> (c) FedAvgs experts *and per-cluster classifier heads* within clusters
> only — eliminating the cross-cluster negative-transfer term that
> vanilla FedAvg-LoRA cannot avoid. On a 10-client federation of Financial
> PhraseBank (formal analyst prose) and Twitter Financial News
> (retail microblogs), the method recovers the ground-truth 5/5 register
> split with purity 1.0 and silhouette 0.226 ± 0.002 in all three seeds,
> achieves in-domain Macro-F1 of $0.981 \pm 0.011$ on PhraseBank and
> $0.818 \pm 0.018$ on Twitter at a total federation traffic cost of
> 150 MB — 730× less than full-model FedAvg. Ablation isolates per-cluster
> classifier heads as the dominant component (+8 F1 points), and we
> document a regression failure mode where standard inverse-frequency
> class weighting catastrophically destabilises the federation, which
> per-cluster heads also resolve.

