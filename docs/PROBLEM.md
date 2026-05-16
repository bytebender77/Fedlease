# Problem Statement and Research Motivation

This document covers sections **2 (Problem Statement)** and **3 (Research
Motivation)** of the FedLEASE-FinBERT exposition. Its job is to make precise
the engineering and statistical problem we are solving, why prior art on
either the federated-learning side or the parameter-efficient-fine-tuning
side individually fails to solve it, and why the specific design choices
of FedLEASE (LoRA-MoE + B-matrix similarity clustering + adaptive top-M
routing) are the right joint answer.

---

## 1. The problem we solve

We are given a federation of N clients

> C = {C_1, …, C_N},   with private datasets D_i = {(x_{i,k}, y_{i,k})}_{k=1}^{n_i},
>      x_{i,k} ∈ X (financial text),   y_{i,k} ∈ {0, 1, 2}.

Crucially:

- The marginal P_i(X) over text differs across clients. Some clients hold
  formal analyst prose (Financial PhraseBank), some hold informal retail
  microblogs (Twitter Financial News Sentiment). These differ in vocabulary,
  syntax, sentiment-marker conventions, and average sequence length.
- The conditional P_i(Y | X) is *not* identical either: in equity-research
  prose "aggressive growth strategy" is bullish; in retail tweets the same
  phrase tends to be skeptical. Polarity-bearing adjectives shift sign with
  register.
- Raw text **cannot leave the client**. This is a hard constraint from GDPR /
  MiFID II / NDA, not a soft preference.
- The number of underlying client *populations* is unknown a priori. We do
  not get to assume the operator says "there are two domains."

We want a single global procedure that, after T rounds of communication,
produces a model f_i(·; θ_i) for each client such that:

1. The expected per-client risk
   `E_{(x,y)∼P_i} ℓ(f_i(x; θ_i), y)` is low for every i (no client is
   sacrificed).
2. The total per-round upload from any client is O(|θ_LoRA|) — i.e. tens of
   thousands of floats, not tens of millions.
3. The procedure adapts to heterogeneity automatically: if the federation
   actually contains K latent populations, the algorithm discovers K (up to
   silhouette-discrimination limits) without supervision.

Standard federated fine-tuning fails (1) by amplifying interference between
populations; standard personalised FL fails (2) by reverting to per-client
models that no longer share information; oracle-clustered FL fails (3) by
requiring K as input.

---

## 2. Why federated learning is needed at all

### 2.1 Centralisation is illegal and contractually impossible

Financial text is not generic web text. It is:

- **Personally identifiable.** A retail tweet bears a username, a timestamp,
  and a position-sized payload that — in aggregate — re-identifies traders
  with high accuracy. GDPR Art. 4(1) classifies this as personal data; Art. 9
  treats trading behaviour as inferentially sensitive.
- **Contractually restricted.** Equity-research prose, broker chatter,
  earnings-call transcripts, and proprietary newswire feeds carry
  redistribution clauses. Pooling them on a central server is breach of
  contract even when it is not breach of statute.
- **Jurisdictionally fragmented.** MiFID II (EU), Reg FD (US), and Hong Kong
  SFC rules disagree on what may be aggregated, by whom, and where the
  compute may physically run.

Federation is therefore not an optimisation choice. It is the only legal
configuration in which a single model can benefit from multiple
financial-text corpora simultaneously.

### 2.2 Centralisation is also statistically wasteful

Even if we *could* pool the data, naive pooling forces a single distribution
P̂(x, y) on the model. A model trained on this mixture must learn a
sentiment classifier that is correct on *both* "the company executed a
disciplined buyback programme" and "lmao this CEO is fr cooked" — which
requires either an enormous backbone that *implicitly* infers the register
of the input, or a *register-aware* model that explicitly conditions on it.
FedLEASE chooses the second route and makes the register-conditioning
*explicit* via the expert allocation.

### 2.3 Heterogeneity makes the federation interesting and hard

The interesting regime of federated learning is precisely the regime where
the clients *are not exchangeable*. In our setting:

- **Marginal heterogeneity (P_i(X)):** PhraseBank documents average ~25
  tokens of clause-rich prose; tweets average ~15 tokens with hashtags,
  cashtags, and emoji.
- **Conditional heterogeneity (P_i(Y | X)):** sentiment-bearing words have
  different polarity priors across registers (see §2.1 above).
- **Quantity heterogeneity:** PhraseBank `sentences_allagree` yields
  ~2 264 examples total; the Twitter dataset yields ~12 000. Naive uniform
  averaging therefore *under-weights* tweet evidence per gradient step but
  *over-represents* it per training token.

Each of these axes breaks a different optimisation guarantee of vanilla
federated averaging.

---

## 3. Why FedAvg fails

Let θ ∈ R^d be the shared model state. FedAvg performs

> θ^{t+1} = Σ_i (n_i / n) · θ_i^t,   where θ_i^t = θ^t − η ∑_b ∇ℓ_i,

i.e. an averaged local SGD step. Three properties of FedAvg fail in our
setting:

### 3.1 The client objectives are mutually destructive

Because P_i(X, Y) ≠ P_j(X, Y), the gradient fields ∇ℓ_i and ∇ℓ_j are
not aligned. When their average θ^{t+1} is computed in parameter space,
*the components that disagree cancel*, and the components that agree
survive. Empirically (and this is the textbook MoE motivation) the surviving
component is a *low-information consensus* that performs at the mean of
the populations on each — i.e. badly on both. This phenomenon is what we
will refer to throughout as **negative transfer**.

### 3.2 Convergence proofs require restrictive assumptions

The known convergence bounds for FedAvg (Li et al. 2020, Karimireddy et al.
2020) all require either bounded gradient dissimilarity Γ² or bounded
client drift. Both quantities are large in our problem: register-level
distribution shift produces gradient cosines that are routinely negative,
not merely small.

### 3.3 FedAvg has no mechanism to express "different parameters for
different clients"

There is exactly one parameter vector. Any per-client adaptation must come
from local fine-tuning *after* the federation ends — which destroys the
shared-information benefit we paid for. Personalisation has to be built
into the protocol, not bolted on afterwards.

---

## 4. Why a single LoRA adapter is not enough

LoRA (Hu et al., 2022) replaces a frozen weight `W_0 ∈ R^{d_out × d_in}` with

> W = W_0 + (α / r) · B A,    A ∈ R^{r × d_in}, B ∈ R^{d_out × r}.

For one adapter shared across all clients in a federation, the federated
update is exactly FedAvg restricted to `(A, B)`. The negative-transfer
argument of §3 therefore applies *unchanged*: the only thing that changes
is the size of the message that gets corrupted. The interference is
mathematically identical.

In our experiments (10 clients, two latent registers) the single-LoRA
configuration is a strict baseline for FedLEASE, and we expect it to lose
both on the formal-prose dataset *and* on the tweet dataset because the
averaged adapter is a compromise between two non-overlapping signals.

---

## 5. Why one LoRA per client is also not enough

The other extreme — give every client its own private LoRA, never
aggregate — solves negative transfer by elimination but suffers from three
defects:

1. **No information sharing across clients with similar data.** Two
   PhraseBank clients are essentially solving the same subproblem with
   half the data each. The federation buys them nothing.
2. **Poor generalisation under low data.** Each client trains on at most a
   few hundred examples. r-rank adapters trained from scratch on few
   hundred examples overfit visibly.
3. **No "global model" deliverable.** There is no single artefact the
   federation produces. New clients joining later have no warm start.

This is the textbook **personalisation-vs-sharing tradeoff**. FedLEASE is
positioned as a *third axis* on this tradeoff: it shares parameters within
discovered clusters and personalises *between* clusters, with the cluster
boundaries themselves learned from data.

---

## 6. Why LoRA and PEFT matter for federation

Two reasons. The first is communication, the second is regularisation.

### 6.1 Communication

A FinBERT-base parameter dump is ~110M floats ≈ 440 MB at fp32. Twenty-five
rounds × ten clients × full-model upload ≈ 110 GB of network traffic, which
is operationally untenable on either consumer broadband or commercial
finance VPNs. With rank-4 LoRA on query and value projections only, *each
LoRA pair* is

> |A| + |B| = r · d_in + d_out · r = 4 · 768 + 768 · 4 = 6 144 floats.

FinBERT has 12 transformer layers × 2 projections = 24 LoRA pairs ≈ 147 k
floats ≈ 590 KB at fp32. Per round, per client. The federation now fits
on a residential link.

### 6.2 Regularisation

The low-rank constraint is itself a regulariser. It forbids client updates
that are "wide" in directions the backbone has no reason to move. This is
especially valuable in federation because it prevents one client's
idiosyncratic noise from being broadcast to every other client.

---

## 7. Why expert allocation matters

Once we accept that we need *multiple* adapters in the federation, the
operator-facing question becomes: **how many?** The naive answers are
unsatisfactory:

- **Fixed M (e.g. M = 4) chosen by the operator.** Too few experts =
  residual negative transfer. Too many experts = expert collapse (some
  experts receive no traffic), wasted parameter budget, and identifiability
  problems during clustering.
- **One expert per client (M = N).** Reduces to per-client LoRA; loses
  sharing.

FedLEASE makes M a *learned quantity*. The mechanism is:

1. Run a short warmup where each client trains its own single LoRA.
2. Cluster the resulting B matrices.
3. Score every candidate M ∈ [M_min, M_max] by the silhouette of the
   resulting clustering.
4. Pick M = argmax silhouette.

The silhouette score measures how well-separated the clusters are
relative to within-cluster cohesion. Maximising it is a model-selection
criterion that is *agnostic to the downstream loss* — it asks only "does
the data, as represented by the B matrices, prefer this partition?"

This is the difference between a hyperparameter and a discovered structure.

---

## 8. Why client similarity should be measured in *parameter* space, not data space

A natural alternative is to compute similarity in **data space** — e.g.
embed each client's text with FinBERT, compute mean embeddings, compare
those. We reject this for three reasons:

1. **Privacy.** Mean-pooled embeddings of small datasets can be inverted
   to a degree sufficient to leak text. Parameter updates are much harder
   to invert (Wang et al. 2020 quantify this).
2. **Task alignment.** Mean embeddings are *task-agnostic*: two clients
   with overlapping vocabularies but opposite label conventions look
   identical. The B matrix, having been trained against the *labels*,
   reflects task-conditional similarity.
3. **Stability under data size.** Embedding means are noisy on small
   clients. B matrices, after a short warmup, converge to the
   leading singular directions of the gradient — much more stable.

So similarity in **B-space**, computed *after* a few epochs of supervised
warmup, is the right object.

---

## 9. The intuition: LoRA B matrices encode task/domain similarity

This is the linchpin claim of the entire method, so it deserves a careful
unpacking.

### 9.1 Conceptual reading

A LoRA update is `ΔW = (α/r) · B A`. The matrix `A : R^{d_in} → R^r`
*picks out* a low-dimensional view of the input representation; the
matrix `B : R^r → R^{d_out}` *writes back* into the output representation.
A is the "what to read", B is the "what to do with it".

Two clients that need to *do the same thing* with a possibly different
representation will tend to converge on similar B matrices, even if their
A matrices look slightly different (because A absorbs idiosyncratic input
preprocessing). Hence: **B is the task signature; A is the input
projection**.

### 9.2 Representation-learning reading

Hu et al.'s LoRA analysis (and follow-up work on the implicit bias of
low-rank fine-tuning) shows that, when r is small, B converges into the
**top singular directions of the task-gradient covariance**:

> B^* ≈ top-r singular vectors of  E_{(x,y)}[ ∂ℓ/∂W_0 ].

These directions are *task-conditional*: they answer "in which output
directions does this client want to push the frozen representation?"
Clients with the same task push in the same directions. Clients with
different tasks push in different directions.

### 9.3 Geometric reading

Treat each B^ℓ as a point on the Grassmannian Gr(r, d_out) — i.e. as a
*subspace* of R^{d_out}. Cosine similarity between flattened B matrices
is a tractable surrogate for principal-angle similarity between
subspaces (it is exact for r = 1 and a tight proxy for small r). Two
clients are "similar" if their B-subspaces have small principal angles,
i.e. they care about overlapping output directions.

### 9.4 Why B and not A

A is contaminated by *input-side* differences — tokenisation artifacts,
length distributions, register-specific embedding statistics. These have
little to do with the task. B, downstream of the rank-r bottleneck, has
been forced to express the task signal in r ≤ 8 dimensions. The
information bottleneck disambiguates.

This is also the empirical finding of the FedLEASE paper: B-similarity
recovers ground-truth domain clusters more reliably than A-similarity or
combined-similarity at every rank tested.

---

## 10. Why routing matters

Even after experts are allocated, *which* expert each input uses must be
decided. Three regimes are possible:

1. **Hard assignment by client.** Client i always uses expert assigned
   to cluster(i). Simple, but loses the ability to handle within-client
   heterogeneity (e.g. a tweet client that occasionally retweets formal
   commentary).
2. **Hard top-1 routing per input.** Each input picks one expert via
   argmax. Suffers from training instability (the argmax is
   non-differentiable; standard fixes are noisy) and from expert
   collapse.
3. **Soft top-M routing with weights.** Each input mixes M experts with
   learned softmax weights.

FedLEASE chooses (3) — but with a twist that is the actual research
contribution: it makes the **client's assigned expert deterministically
participate** in the top-M, while letting the router *adaptively choose*
the other M − 1 experts from the remaining N_experts − 1. This is the
**2M − 1 router** explained in `METHODOLOGY.md §4`.

The reason this matters: pure soft routing in federated MoE is notoriously
unstable, with experts collapsing to identity functions or to each other.
The assigned-expert guarantee is a *structural* prior that says "this
expert exists for you, you must use it" — eliminating the collapse mode
entirely while leaving room for cross-cluster borrowing.

---

## 11. Communication efficiency and the personalisation/sharing axis

We can locate every method on a 2-D plane whose axes are:

- **x: per-round upload size**, measured in fraction of model parameters.
- **y: per-client risk**, measured on the client's own test distribution.

The frontier looks like this:

```
   per-client risk
        ▲
        │
        │   FedAvg (full model)         ◀── high comm, high risk
        │
        │   FedAvg-LoRA  ●
        │
        │                                   FedLEASE  ★
        │   single-LoRA-MoE  ●                   ▲
        │                                        │
        │                                Local-only LoRA ●  ◀── low comm, high risk
        │
        └────────────────────────────────────────────────────▶  upload size
                                                              (smaller = better)
```

FedLEASE sits in the lower-left: same upload budget as FedAvg-LoRA
(plus a one-time silhouette computation and a tiny router upload), but
**lower per-client risk** because the negative-transfer term has been
removed by clustering and the personalisation term has been added by
adaptive routing.

This is the headline efficiency claim: *better risk at no extra
communication*.

---

## 12. Summary of the research motivation

In one paragraph:

> Federated fine-tuning of foundation models is only useful when (a) it
> preserves privacy, (b) it is communication-efficient, and (c) it
> tolerates non-IID client distributions. LoRA solves (a) and (b) but
> not (c). Existing personalised-FL methods solve (c) but break the
> shared-model abstraction. FedLEASE achieves all three by introducing a
> latent layer of clustered LoRA experts between the global backbone and
> the per-client adapter — and by making the *number* of experts a
> data-discovered quantity, the *initialisation* of experts derived from
> the geometry of LoRA's B matrices, and the *routing* a deterministic-
> plus-adaptive top-M policy that provably prevents expert collapse.
> Financial NLP — bimodal in register, legally non-poolable,
> commercially valuable — is the canonical application domain.

Continue to `ARCHITECTURE.md` for the system-level realisation of these
ideas, or to `METHODOLOGY.md` for the mathematical details.
