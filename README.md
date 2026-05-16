# FedLEASE-FinBERT

**Federated Adaptive LoRA Experts Allocation and Selection for Financial Sentiment Analysis.**

A research-grade implementation of the FedLEASE methodology (Adaptive LoRA Experts
Allocation and Selection for Federated Fine-Tuning, NeurIPS 2025) instantiated on
ProsusAI/FinBERT and applied to two heterogeneous financial-text corpora:
Financial PhraseBank (formal analyst prose) and Twitter Financial News Sentiment
(informal retail microblogs).

---

## 🚀 Quickstart — one command per platform

The code is fully self-contained. Cross-platform device detection (CUDA / Apple
MPS / CPU), dataset auto-download (via `huggingface_hub`, bypassing the
`datasets>=3.0` script ban), and config presets are all built in.

### macOS / Linux / WSL2

```bash
git clone https://github.com/<your-username>/fedlease.git
cd fedlease
chmod +x setup.sh && ./setup.sh
# Edit .env and paste your HF token: HF_TOKEN=hf_xxxxxxxx
source .venv/bin/activate
python run.py --preset full
```

### Windows (cmd.exe / PowerShell)

```cmd
git clone https://github.com/<your-username>/fedlease.git
cd fedlease
setup.bat
REM Edit .env and paste your HF token: HF_TOKEN=hf_xxxxxxxx
.venv\Scripts\activate
python run.py --preset full
```

That's it. `setup.sh` / `setup.bat` will:
1. Create a virtual environment in `.venv/`
2. Auto-detect NVIDIA GPU and install the right PyTorch wheel (CUDA 12.1 if present, CPU otherwise)
3. Install all dependencies from `requirements.txt`
4. Copy `.env.example` → `.env` for you to fill in

### Training presets

| Preset | Rounds × Local Epochs | Wall-clock (RTX A5000) | Use for |
|---|---|---|---|
| `debug` | 3 × 1 | ~3 min | Smoke test the pipeline |
| `quick` | 10 × 1 | ~10 min | Sanity check |
| `medium` | 15 × 2 | ~25 min | Quick iteration |
| **`full`** | **25 × 2** | **~40 min** | **Paper-quality run** |

```bash
python run.py --preset full           # paper run (default)
python run.py --preset quick          # 10-min smoke test
python run.py --preset full --no-cuda # force CPU
python run.py --preset full --seed 7  # different seed for confidence interval
python run.py --preset full --partition iid  # IID control instead of heterogeneous
```

### Common issues

| Symptom | Fix |
|---|---|
| `HF_TOKEN missing` warning during dataset download | Edit `.env` and add your token from https://huggingface.co/settings/tokens |
| Silent exit after PhraseBank loads | Already fixed in current code — but make sure `pip install pandas` ran |
| `RuntimeError: Index put requires source/destination dtype match` | Already fixed in current code (fp16/fp32 cast in `models/lora_layers.py`) |
| `nvidia-smi` not found on Windows | Driver not installed — install latest NVIDIA driver from nvidia.com |
| `CUDA out of memory` | Drop `batch_size`: edit `configs/finbert_fedlease.yaml` → `batch_size: 16` |

---

This is not a re-implementation of vanilla federated averaging. It is a complete
realisation of a **two-phase clustered LoRA-MoE federated training pipeline** in
which the *number of experts is discovered from data*, experts are *initialised
from the geometry of LoRA's B matrices*, and routing is performed by a
**deterministic-plus-adaptive top-M policy** that guarantees the client's
assigned expert is always activated.

---

## 1. Executive Overview

### 1.1 What the system does

The system fine-tunes a single 110M-parameter financial language model across a
simulated federation of clients. Each client holds private text in one of two
linguistic registers — analyst-grade financial prose or retail-investor tweets —
and *never shares it*. Only small low-rank update matrices and routing scores
cross the network. The result is a single shared backbone augmented by a
small, automatically-sized set of **specialist LoRA experts**, where each
client uses a *mixture* of experts but is *anchored* to one assigned expert
selected by data-driven clustering.

### 1.2 The core idea

Federated learning over heterogeneous clients faces an unresolved tension:

- **Sharing too much** (a single global adapter) causes *negative transfer* —
  the formal-prose update partially cancels the tweet-style update.
- **Sharing too little** (one adapter per client) discards the inductive bias
  that *makes federated learning useful in the first place*.

FedLEASE resolves this tension by introducing **a layer of latent structure
between "one model" and "N models"**: a small set of M experts, where M is
chosen by the data, not the operator. The novelty rests on three observations:

1. The **B matrix of a LoRA adapter** is a low-rank readout from the model's
   intermediate representations into the task-relevant update subspace.
   Two clients whose data live on similar manifolds produce B matrices whose
   *column spaces align*. Cosine similarity over B is therefore a proxy for
   task/domain similarity.
2. **Agglomerative clustering with silhouette-optimised k**, applied to the
   matrix of pairwise B-distances, recovers the latent client-domain structure
   *without supervision* and *without knowing the number of domains in advance*.
3. A **top-M router with output dimension 2M − 1** can be wired so that
   M of its outputs deterministically point to the client's assigned expert
   and M − 1 outputs are adaptively routed to other experts — guaranteeing
   the assigned-expert always participates while still allowing the router
   to *borrow* from other clusters when useful.

### 1.3 Research goal

To demonstrate that on a non-IID financial NLP federation, FedLEASE:

1. **Matches or exceeds** centralised FinBERT fine-tuning despite never moving
   the raw text;
2. Recovers the latent (PhraseBank vs Twitter) cluster structure from B-matrix
   geometry alone, with no oracle supervision;
3. Achieves better **per-client personalisation** than FedAvg-LoRA while using
   the **same per-round upload budget**;
4. Uses experts in an interpretable, non-collapsed way (the assigned-expert
   guarantee provably prevents winner-take-all collapse).

### 1.4 Why this matters in Financial NLP

Financial language is *bimodal-at-minimum*. Equity-research prose ("EBITDA
exceeded consensus") and retail tweets ("ngmi, this stock is cooked")
disagree on vocabulary, syntax, sentiment polarity conventions ("aggressive"
is bullish in equity research, bearish in tweets), and even label semantics
(neutral-vs-mixed). At the same time, financial data is subject to GDPR,
MiFID II, broker-dealer confidentiality, and proprietary-research contracts —
all of which forbid raw-text centralisation. A federated method that
*adapts* to register heterogeneity without leaking text is therefore not
incremental: it is the only operating regime that is simultaneously legal
and accurate.

---

## 2. Repository layout

```
fedlease/
├── configs/                        # YAML hyperparameter files
│   ├── base_config.yaml
│   └── finbert_fedlease.yaml
├── data/
│   ├── dataset_loader.py           # PhraseBank + Twitter loaders, label normalisation
│   ├── data_partitioner.py         # heterogeneous / IID / Dirichlet partitioning
│   ├── preprocessing.py            # FinBERT tokenisation, DataLoader construction
│   └── visualization.py            # dataset and partition plots
├── models/
│   ├── lora_layers.py              # LoRAMoELinear: multi-expert low-rank adapter
│   ├── adaptive_router.py          # 2M−1 router, top-M with assigned-expert guarantee
│   └── finbert_lora_moe.py         # WarmupFinBERT + FedLEASEFinBERT wrappers
├── clustering/
│   ├── similarity.py               # B-matrix cosine similarity / distance matrix
│   ├── clustering.py               # agglomerative linkage + cluster labels
│   └── expert_allocation.py        # silhouette-based M selection, expert init
├── federated/
│   ├── client.py                   # local SGD, mixed-precision, upload payload build
│   ├── server.py                   # broadcast, aggregation orchestration, comm stats
│   └── aggregation.py              # cluster-wise weighted averaging
├── training/
│   ├── fedlease_pipeline.py        # two-phase orchestration: warmup → cluster → train
│   └── evaluator.py                # per-client, per-dataset metric computation
├── utils/                          # seed, config, logging, metrics, checkpointing
├── visualization/                  # cluster, similarity, training, expert plots
├── experiments/
│   └── run_fedlease.py             # CLI entry point
└── docs/                           # the deep-dive documentation set
    ├── PROBLEM.md                  # §2–3: problem statement + research motivation
    ├── ARCHITECTURE.md             # §4–5: system architecture + federated workflow
    ├── METHODOLOGY.md              # §6–10: LoRA, MoE, clustering, routing, aggregation
    ├── EXPERIMENTS.md              # §11–12: financial NLP adaptation + experimental design
    └── DISCUSSION.md               # §13–15: advantages, limitations, contribution summary
```

---

## 3. Quickstart

### 3.1 Install

```bash
cd fedlease
python -m pip install -r requirements.txt
# or, for a development install with extras:
python -m pip install -e ".[dev,tracking]"
```

Requires Python ≥ 3.9, `torch ≥ 2.1`, `transformers ≥ 4.36`,
`datasets ≥ 2.15`, `scikit-learn ≥ 1.3`, `scipy ≥ 1.11`.

### 3.2 Reproduce the headline experiment

```bash
python experiments/run_fedlease.py \
    --config configs/finbert_fedlease.yaml \
    --num_clients 10 \
    --num_rounds 25 \
    --lora_rank 4 \
    --max_experts 8 \
    --partition heterogeneous \
    --output_dir outputs/fedlease_run
```

This runs the full pipeline:

1. **Load and normalise** Financial PhraseBank and Twitter Financial News
   Sentiment to the unified label space `{0: negative/bearish,
   1: neutral, 2: positive/bullish}`.
2. **Partition** the data across 10 clients (5 PhraseBank + 5 Twitter).
3. **Phase A — Warmup.** Each client locally fine-tunes a *single* LoRA on
   FinBERT for `warmup_epochs` epochs. Client `i` returns its trained LoRA's
   `(A_i^ℓ, B_i^ℓ)` for every adapter layer `ℓ`.
4. **Clustering.** The server computes a pairwise distance matrix over the
   `{B_i^ℓ}` matrices, runs agglomerative clustering, sweeps k ∈
   `[min_clusters, max_clusters]`, and selects M = argmax silhouette.
5. **Expert initialisation.** Each cluster's experts are initialised by
   averaging the warmup `(A, B)` matrices of its member clients (Eq. 3).
6. **Phase B — Iterative federated training.** For each of `num_rounds`
   rounds: broadcast → local LoRA-MoE training with adaptive top-M
   routing → cluster-wise aggregation of assigned experts and routers.
7. **Evaluation** on held-out test splits of both datasets, plus
   communication-cost accounting and clustering / routing / training plots.

Outputs (checkpoints, JSON metrics, distance matrix, cluster labels,
silhouette scores, plots) are written to `output_dir`.

### 3.3 Configuration knobs

All defaults are in `configs/finbert_fedlease.yaml`. The most important:

| Knob | Default | Meaning |
|---|---|---|
| `lora.rank` | 4 | LoRA bottleneck rank r |
| `lora.alpha` | 8 | LoRA scaling α (effective scaling = α/r = 2.0) |
| `lora.target_modules` | `[query, value]` | Which projections receive LoRA |
| `federated.num_clients` | 10 | N |
| `federated.num_rounds` | 25 | Communication rounds |
| `federated.local_epochs` | 2 | Local epochs per round |
| `federated.warmup_epochs` | 3 | Local epochs in the warmup phase |
| `federated.max_experts` | 8 | M_max — upper bound on number of experts |
| `federated.min_experts` | 2 | Lower bound on M |
| `clustering.linkage` | average | Agglomerative linkage criterion |
| `training.learning_rate` | 3e-4 | AdamW LR |
| `data.partition_strategy` | heterogeneous | also: `iid`, `non_iid` (Dirichlet) |
| `data.dirichlet_alpha` | 0.5 | Concentration for non-IID partitions |

---

## 4. Documentation map

| Topic | File |
|---|---|
| Why federated, why MoE, why LoRA B-matrices encode domain | [docs/PROBLEM.md](docs/PROBLEM.md) |
| Component graph, tensor flow, round-by-round protocol | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| LoRA math, MoE math, clustering math, routing math, aggregation math | [docs/METHODOLOGY.md](docs/METHODOLOGY.md) |
| Datasets, partitioning, metrics, expected behaviour | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) |
| Advantages, limitations, contributions, future work | [docs/DISCUSSION.md](docs/DISCUSSION.md) |

---

## 5. Citing / Acknowledgement

This implementation operationalises the FedLEASE methodology described in:

> *Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning.*
> NeurIPS 2025.

Backbone model: `ProsusAI/finbert` (Araci, 2019). Datasets: Financial
PhraseBank (Malo et al., 2014) and the Zeroshot Twitter Financial News
Sentiment dataset.
