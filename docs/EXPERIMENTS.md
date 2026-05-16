# Financial NLP Adaptation and Experimental Design

This document covers sections **11 (Financial NLP Adaptation)** and
**12 (Experimental Design)** of the exposition. The first half explains
why this domain is the right testbed for FedLEASE; the second half
specifies the precise experimental protocol — what we run, what we
measure, and what we expect to see.

---

## 1. Why FinBERT

We use `ProsusAI/finbert` as the frozen backbone for three reasons:

1. **Vocabulary alignment.** FinBERT was pre-trained on a financial
   corpus (Reuters TRC2-financial + analyst reports) and its WordPiece
   vocabulary contains finance-specific tokens (`buyback`, `EBITDA`,
   `consensus`, `guidance`, cashtags like `$AAPL`) that vanilla BERT
   tokenises into multi-piece sequences. Pre-tokenised, FinBERT
   sequences are shorter and more semantically coherent, which means
   the LoRA adapter has less work to do per unit of sentiment signal.

2. **Task alignment.** FinBERT was already fine-tuned on Financial
   PhraseBank for sentiment classification. This means the
   pooler/classifier head ships with sentiment-relevant weights. Our
   federated procedure can begin from a calibrated starting point
   and only needs to *specialise* the head per-domain via the LoRA-MoE
   experts.

3. **Parameter count.** FinBERT-base has the same 110M-parameter
   architecture as bert-base-uncased, putting it in the
   "communicable-via-LoRA" regime. Going larger (e.g. FinBERT-LARGE,
   FinLLaMA-7B) is feasible with the same code and is the natural
   next experiment, but the headline run uses the base model so that
   the federated dynamics are not confounded by extreme model scale.

The backbone is fully frozen during training. Only LoRA experts, router
gates, and the classifier head receive gradient. Hence "FinBERT" in
this codebase is treated as a fixed feature extractor with adaptive
read-out heads, not as a trainable model.

---

## 2. Why financial sentiment is a hard testbed

Financial sentiment is not a generic classification task. It is harder
than e.g. movie-review sentiment for reasons that *amplify* the
FedLEASE thesis:

### 2.1 Polarity ambiguity

Many sentiment-bearing words in finance are register-conditional:

- "**aggressive** expansion" → bullish in research prose, bearish in
  retail tweets.
- "**volatile**" → positive for short-sellers, negative for long-only
  funds — sentiment depends on the implied reader.
- "**downgrade**" → mechanically negative, but a downgrade from "Hold"
  to "Sell" is much more negative than from "Strong Buy" to "Buy".

A single global model trained on a mixture cannot represent these
register-conditional polarities without learning a register classifier
implicitly. FedLEASE *explicitly* assigns separate experts per
register, lifting that burden from the backbone.

### 2.2 Compositional negation

Financial prose loves clauses like

> "*not* materially *below* consensus".

Two negations and a comparator. The compositional parse matters. A
LoRA adapter that has only seen tweets ("*not* paying dividends fr")
will mis-parse this. Experts trained on the right register handle
their own compositional patterns.

### 2.3 Domain-specific neutrality

In PhraseBank, "neutral" is the dominant class because most analyst
sentences are factual statements ("the company reported Q3 revenue
of €1.4B"). In Twitter, "neutral" is rare because tweet authors
self-select for opinionated content. The base rate of `y = 1` differs
sharply by source, and a model that doesn't condition on source
will systematically over- or under-predict neutral.

---

## 3. The two datasets

### 3.1 Financial PhraseBank (`financial_phrasebank / sentences_allagree`)

- **Origin.** Malo et al. (2014). Sentences from English-language
  financial news, annotated by ≥ 5 finance graduate students. The
  `sentences_allagree` configuration retains only sentences where
  *all* annotators agreed on a single label.
- **Size.** ~2 264 sentences total in the all-agree configuration.
- **Label space.** `{negative, neutral, positive}`, mapped to
  `{0, 1, 2}` in `data/dataset_loader.py:LABEL_MAP_PHRASEBANK`.
- **Class balance.** Skewed toward neutral (~60%), followed by
  positive (~25%), negative (~15%).
- **Linguistic register.** Formal analyst prose. Average length ~25
  WordPiece tokens after FinBERT tokenisation. Vocabulary contains
  multi-clause sentences, quantitative figures, and analyst jargon
  ("guidance", "consensus", "FY24").

After our 70/10/20 stratified split: ~1 584 train, ~227 val, ~453 test.

### 3.2 Twitter Financial News Sentiment (`zeroshot/twitter-financial-news-sentiment`)

- **Origin.** ~12 000 finance-related English tweets, annotated
  three-way as Bearish / Bullish / Neutral.
- **Label space.** `{Bearish, Neutral, Bullish}` natively, *remapped*
  to `{0, 1, 2}` to align with PhraseBank in
  `data/dataset_loader.py:LABEL_MAP_TWITTER` — specifically:

  ```python
  label_remap = {0: 0, 1: 2, 2: 1}   # Bearish→0, Bullish→2, Neutral→1
  ```

  The source uses `{Bearish: 0, Bullish: 1, Neutral: 2}`; we normalise
  to the polarity-monotone `{negative: 0, neutral: 1, positive: 2}`
  to keep label semantics consistent across both datasets and across
  evaluation metrics.

- **Size.** ~12 000 tweets; after a 70/10/20 stratified split:
  ~8 351 train, ~1 193 val, ~2 387 test (sizes reported during run).
- **Class balance.** Mostly Bullish + Bearish; Neutral is the minority.
- **Linguistic register.** Informal, abbreviated, emoji-rich, with
  cashtags (`$TSLA`), hashtags (`#crypto`), URLs, and frequent
  ungrammatical clauses. Average length ~15 WordPiece tokens.

### 3.3 Label normalisation details

The two datasets disagree on integer-label conventions in the
original sources. PhraseBank ships with integers already in the
`{neg=0, neu=1, pos=2}` convention; Twitter ships with
`{Bear=0, Bull=1, Neu=2}`. We deliberately compose the loaders to
return a unified `{0=negative/bearish, 1=neutral, 2=positive/bullish}`
schema. This is essential because the FedLEASE classifier head is
shared across all clients via global FedAvg — if labels meant
different things on different clients, the gradient signal into the
head would be self-cancelling.

---

## 4. Federated partitioning strategies

The partitioner exposes three strategies (`data/data_partitioner.py:
FederatedDataPartitioner.partition`):

### 4.1 Heterogeneous (default; corresponds to the headline experiment)

- 5 clients receive disjoint shards of PhraseBank (each ~317 train
  examples).
- 5 clients receive disjoint shards of Twitter (each ~1 670 train
  examples).
- Each client gets a proportional slice of the val split of its
  source dataset for local validation.
- Cluster ground truth: the partition into {phrasebank clients} and
  {twitter clients} is the *latent* clustering that we hope
  silhouette + B-similarity will recover.

This is the canonical FedLEASE testbed: domain heterogeneity, balanced
clients per domain, ground-truth cluster structure available for
post-hoc evaluation.

### 4.2 IID (control)

All examples from both datasets are pooled, shuffled, and split into
N equal shards. Each shard contains a near-uniform mixture of formal
prose and tweets. We expect FedLEASE to find M = 1 ideal (no
non-trivial clustering exists), or to fall back to weak clustering
with low silhouette. This is the **null condition** that tests
"does FedLEASE invent clusters when there are none?".

### 4.3 Dirichlet non-IID

A Dirichlet(α) distribution on the *label* space (not the dataset
source) is used to allocate per-class samples across clients. With
α small (≤ 0.5) clients are skewed toward one or two classes;
with α large (≥ 5) the allocation approaches IID. This stresses
the routing in a different axis: label skew rather than register
skew.

---

## 5. Training hyperparameters

All defaults from `configs/finbert_fedlease.yaml`:

| Group | Parameter | Value | Justification |
|---|---|---|---|
| Model | name | `ProsusAI/finbert` | see §1 |
| Model | num_labels | 3 | see §3.3 |
| Model | max_length | 128 | covers > 99% of both datasets without truncation |
| LoRA | rank | 4 | smallest rank where B captures task signal cleanly; matches Hu et al. recommendation for r ≪ d_min |
| LoRA | alpha | 8 | scaling = α/r = 2.0, standard choice |
| LoRA | dropout | 0.1 | mild regularisation on the LoRA path |
| LoRA | target_modules | `[query, value]` | see `METHODOLOGY.md §1.4` |
| Federated | num_clients | 10 | small enough for fast iteration, large enough to populate multiple clusters |
| Federated | num_rounds | 25 | empirically sufficient for convergence with LoRA rank 4 |
| Federated | local_epochs | 2 | balances local progress vs client drift |
| Federated | warmup_epochs | 3 | enough to push B out of zero and into a stable subspace |
| Federated | max_experts | 8 | upper bound on the silhouette sweep |
| Federated | min_experts | 2 | exclude the degenerate M = 1 case |
| Federated | batch_size | 32 | standard fine-tuning batch for FinBERT-base on a single GPU |
| Training | learning_rate | 3e-4 | AdamW; LoRA layers tolerate higher LR than full fine-tune |
| Training | weight_decay | 0.01 | standard AdamW decoupled WD |
| Training | warmup_ratio | 0.1 | linear LR warmup over first 10% of local steps |
| Training | max_grad_norm | 1.0 | gradient clipping for stability under mixed precision |
| Training | mixed_precision | true | autocast + GradScaler on CUDA |
| Training | seed | 42 | propagated to Python, NumPy, Torch, CUDA |
| Clustering | linkage | average | see `METHODOLOGY.md §3.4` |

These values are not tuned aggressively; they are the *first reasonable
choice* in each case, picked to make the experiment reproducible
rather than to win a leaderboard.

---

## 6. Evaluation metrics

### 6.1 Classification metrics

For every (client, dataset-test-split) pair, and aggregated globally:

- **Accuracy.** `(correct predictions) / total`.
- **Macro F1.** `(1/3) Σ_c F1_c`, averaged across the three
  sentiment classes. Robust to class imbalance, which matters because
  PhraseBank is neutral-heavy and Twitter is not.
- **Weighted F1.** F1 averaged with class-frequency weights.
  Reported for completeness; the headline number is macro F1.
- **Per-class precision, recall, F1.** For diagnostic purposes —
  in particular to detect "classifier always predicts neutral"
  failure modes.

Implementation: `utils/metrics.py: compute_metrics`.

### 6.2 Federated metrics

- **Total communication (MB).** Sum over all rounds, all clients,
  of upload payload size. Reported by `server.communication_summary()`.
- **Per-round communication (MB).** Time-series; used to detect any
  payload growth (there shouldn't be any).
- **Convergence rate.** Average validation accuracy across clients
  vs round number. We expect monotone-with-noise improvement; a
  characteristic "plateau then improve" pattern at round 5–10 is
  a sign that the silhouette-selected M is appropriate.

### 6.3 Expert / routing metrics

These are the FedLEASE-specific diagnostics:

- **Expert utilisation.** For each expert e, the fraction of all
  forward-pass slot selections that map to e. We expect:
  - The *assigned* expert of each cluster receives a baseline
    ≈ k/(2M − 1) of selections (the structural floor from the
    guarantee).
  - The remaining mass is distributed by the router. Empirically
    in financial sentiment, cross-cluster borrowing is small —
    a PhraseBank-cluster expert is selected ~70–90% of the time
    by PhraseBank clients.
- **Routing entropy.** `H(ω̂) = − Σ_p ω̂_p log ω̂_p` averaged across
  inputs. Low entropy = router is decisive; high entropy = router
  is uncertain. Trajectory of entropy over rounds tells us when the
  router has *settled*.
- **Cluster-purity heatmap.** Confusion matrix of cluster assignment
  vs ground-truth source. We expect strong diagonal in the
  heterogeneous partitioning.

Plots are produced by `visualization/cluster_plots.py`,
`visualization/training_plots.py`, and
`visualization/expert_plots.py`.

---

## 7. Expected behaviour

This section is the falsifiable-prediction half of the experimental
design.

### 7.1 Expected clustering behaviour

In the heterogeneous configuration we expect:

- Silhouette as a function of k peaks at **k = 2**, with a clear
  margin over k ∈ {3, 4, …, 8}.
- The two clusters correspond to PhraseBank-clients and
  Twitter-clients with **purity ≥ 0.95** (allowing for occasional
  swaps if a particular shard happens to look mixed).
- Distance matrix `D` shows a clear block-diagonal structure with
  small within-block distances (~0.1–0.3 after cosine subtraction)
  and large cross-block distances (~0.5–0.8).

In the IID configuration we expect:

- Silhouette is low across all k (≤ 0.15 typically) — there is no
  meaningful clustering to find.
- argmax silhouette may pick any small k essentially at random.
- FedLEASE in this regime should *not* outperform FedAvg-LoRA
  significantly. This is the negative-control prediction; if it
  *does* outperform, we should investigate.

In the Dirichlet non-IID configuration we expect:

- Silhouette peaks at k > 1 but the peak is shallower than in the
  heterogeneous case.
- Clusters correspond to *label-skew profiles* rather than dataset
  sources.

### 7.2 Expected convergence behaviour

- The warmup phase (3 epochs × 10 clients) takes ~5–10 minutes on a
  single A100 / RTX 4090.
- Phase B convergence: validation accuracy reaches > 80% on
  PhraseBank-test and > 75% on Twitter-test by round 10–15;
  full convergence by round 20–25.
- The training curve should *not* show oscillation; if it does, the
  router is unstable (see `DISCUSSION.md §2` on routing instability).

### 7.3 Expected routing behaviour

- The router for a PhraseBank-client should place ≥ 60% of its
  routing mass on slots 0..M-1 (the assigned-expert block) after
  convergence, occasionally borrowing from a Twitter-expert when
  encountering a casual-style phrase.
- The router for a Twitter-client should similarly anchor on the
  Twitter-cluster expert.
- The 2M − 1 router floor implies the **minimum** assigned-expert
  share is `1/M` per input (one slot at minimum). We expect the
  *realised* share to be considerably higher than this floor.

### 7.4 Quantitative target numbers

For the headline run we target, on held-out test sets:

| Dataset | Metric | Target |
|---|---|---|
| Financial PhraseBank test | Accuracy | ≥ 0.92 |
| Financial PhraseBank test | Macro F1 | ≥ 0.88 |
| Twitter Financial News test | Accuracy | ≥ 0.82 |
| Twitter Financial News test | Macro F1 | ≥ 0.80 |

with **total communication ≤ 200 MB** over all 25 rounds.

These targets reflect what a competently-implemented FedLEASE should
achieve; they are not loose bounds. Material under-performance against
any of them indicates either a tokenisation problem, a router
instability, or a clustering misallocation.

---

## 8. Reproducibility

The codebase is deterministic to the extent possible:

1. All seeds (Python, NumPy, Torch, CUDA) are set via
   `utils/seed.py: set_seed(seed)` in `experiments/run_fedlease.py`
   *before* any model or data is constructed.
2. DataLoaders use `seed_worker` and `torch.Generator(seed)` for
   shuffle reproducibility.
3. Mixed precision (when enabled) introduces minor non-determinism
   in matmul order on some GPUs; this is the residual source of
   round-to-round variation across re-runs.
4. The config used for a run is saved to `output_dir/config.yaml`
   alongside the run outputs.

To reproduce the headline numbers exactly, run with `--no_cuda` or
on the same GPU type, and set the same seed.

---

## 9. Output artefacts

A successful run produces, under `output_dir/`:

```
output_dir/
├── config.yaml                     # the resolved config for this run
├── distance_matrix.npy             # [N, N] B-similarity distances
├── cluster_labels.npy              # [N] integer cluster IDs
├── silhouette_scores.npy           # [(k, score)] for k swept
├── checkpoints/                    # saved model states
│   ├── warmup_client_*.pt
│   └── round_*.pt
├── results/
│   ├── final_metrics.json          # per-dataset, per-client metrics
│   ├── communication.json          # bytes per round, total bytes
│   └── training_curves.json        # acc/loss per round
├── logs/
│   └── fedlease.log                # full run log (INFO level)
└── plots/
    ├── dataset_overview.png
    ├── clustering/
    │   ├── silhouette_curve.png
    │   ├── distance_heatmap.png
    │   └── dendrogram.png
    ├── training_curve.png
    └── communication_cost.png
```

These artefacts are the input to the post-run analysis and to the
DISCUSSION.md narrative.

Continue to `DISCUSSION.md` for the advantages, limitations, future
work, and contribution summary.
