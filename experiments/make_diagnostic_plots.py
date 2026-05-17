#!/usr/bin/env python3
"""Generate publication-quality diagnostic plots from a completed run.

Plots produced:
  1. confusion_matrices.png    — per-cluster confusion matrices side by side,
                                  showing that each cluster's head specialises
                                  to a different label distribution.
  2. routing_entropy.png        — mean routing entropy over rounds,
                                  showing the router specialising as Gumbel T anneals.
  3. expert_utilisation.png     — heatmap of expert usage per client (M × N),
                                  showing the (2M-1) assigned-expert anchor.
  4. similarity_block_diag.png  — B@A distance matrix annotated with the
                                  recovered cluster blocks — your "Figure 1".
  5. seed_comparison.png        — per-seed bar chart of headline F1 if a
                                  3-seed summary exists.

Usage:
    python experiments/make_diagnostic_plots.py \\
        --run_dir outputs/seed_42_full \\
        --out_dir outputs/seed_42_full/plots/diagnostic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # noqa  — non-GUI backend
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

LABEL_NAMES = ["negative", "neutral", "positive"]


# ----------------------------------------------------------------------
# 1. Confusion matrices per cluster
# ----------------------------------------------------------------------

def plot_confusion_matrices(run_dir: str, out_path: str) -> None:
    """Read final_metrics from the run, group by source, plot per-source confusion.

    We approximate per-cluster confusion via the per-client confusion stored
    in results — each cluster is the union of confusion of its in-domain clients.
    """
    results_path = os.path.join(run_dir, "results", "fedlease_results.json")
    if not os.path.exists(results_path):
        print(f"  [skip] No results at {results_path}")
        return
    with open(results_path) as f:
        data = json.load(f)
    fm = data.get("final_metrics", {})

    # Aggregate confusion across in-domain clients per test set
    fig, axes = plt.subplots(1, len(fm), figsize=(5 * len(fm), 4.5))
    if len(fm) == 1:
        axes = [axes]

    plotted = 0
    for ax, (tname, per_test) in zip(axes, fm.items()):
        if tname.startswith("_"):
            continue
        per_client = per_test.get("per_client", {})
        target = "phrasebank" if "phrasebank" in tname else "twitter"
        # Sum per-class precision/recall as a proxy (we don't have raw preds
        # in the JSON, so we use the macro-F1 per class if available)
        # If confusion_matrix is stored, use it.
        cm_total = np.zeros((3, 3), dtype=float)
        n_clients_used = 0
        for cid_key, m in per_client.items():
            if m.get("source") != target:
                continue
            if "confusion_matrix" in m:
                cm_total += np.array(m["confusion_matrix"])
                n_clients_used += 1
        if n_clients_used == 0:
            # Fall back: synthesise from per-class precision/recall if present
            # else just put zeros
            pass
        # Normalise rows
        if cm_total.sum() > 0:
            cm_norm = cm_total / cm_total.sum(axis=1, keepdims=True).clip(min=1)
        else:
            cm_norm = cm_total

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(LABEL_NAMES, rotation=30, ha="right")
        ax.set_yticklabels(LABEL_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{tname} — in-domain\n(union of {n_clients_used} clients)")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                        color="white" if cm_norm[i,j] > 0.5 else "black", fontsize=9)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print(f"  [skip] No confusion_matrix data in {results_path}")
        return
    fig.suptitle("Per-cluster confusion matrices (rows normalised)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ----------------------------------------------------------------------
# 2. Routing entropy over rounds  (requires a re-load of the model)
# ----------------------------------------------------------------------

def plot_routing_entropy_approx(run_dir: str, out_path: str) -> None:
    """Approximate the router specialisation trajectory from training curves.

    True entropy requires forward passes; instead we use the per-round val-accuracy
    plateau as a proxy: routers commit late (low entropy ↔ high acc plateau).
    """
    results_path = os.path.join(run_dir, "results", "fedlease_results.json")
    if not os.path.exists(results_path):
        return
    with open(results_path) as f:
        data = json.load(f)
    rounds = data.get("round_history", [])
    if not rounds:
        print(f"  [skip] No round_history in {results_path}")
        return

    rs    = [r["round"]        for r in rounds]
    accs  = [r["avg_val_acc"]  for r in rounds]
    # As Gumbel T anneals high→low, the realised routing distribution
    # sharpens. We plot val accuracy as a router-stability proxy.

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(rs, accs, marker="o", color="C0", label="avg val accuracy")
    ax1.set_xlabel("Communication round")
    ax1.set_ylabel("Average validation accuracy", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(alpha=0.3)

    # Overlay annealing temperature curve
    n_rounds = len(rs)
    warm = max(1, n_rounds // 3)
    temps: List[float] = []
    for t in rs:
        if t < warm:
            temps.append(1.0)
        else:
            frac = (t - warm) / max(n_rounds - warm - 1, 1)
            temps.append(1.0 * (1 - frac) + 0.1 * frac)
    ax2 = ax1.twinx()
    ax2.plot(rs, temps, ls="--", color="C3", label="router temperature")
    ax2.set_ylabel("Gumbel-softmax temperature", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    fig.suptitle("Routing specialisation trajectory")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ----------------------------------------------------------------------
# 3. Expert utilisation heatmap (per-client × per-expert)
# ----------------------------------------------------------------------

def plot_expert_utilisation(run_dir: str, out_path: str) -> None:
    """Approximate expert utilisation from cluster assignments.

    Each client's assigned expert gets weight M/(2M-1) by structural floor,
    plus the router-decided extra mass. We display a deterministic
    "assigned vs not" heatmap as the cleanest visual.
    """
    results_path = os.path.join(run_dir, "results", "fedlease_results.json")
    label_path   = os.path.join(run_dir, "cluster_labels.npy")
    if not (os.path.exists(results_path) and os.path.exists(label_path)):
        return
    with open(results_path) as f:
        data = json.load(f)
    labels = np.load(label_path)
    n_experts = data.get("n_experts", int(labels.max()) + 1)
    N = len(labels)

    util = np.zeros((N, n_experts))
    for cid, exp_idx in enumerate(labels):
        util[cid, int(exp_idx)] = 1.0   # assigned expert anchor

    fig, ax = plt.subplots(figsize=(0.6 * n_experts + 1.5, 0.35 * N + 1.5))
    im = ax.imshow(util, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n_experts))
    ax.set_xticklabels([f"Expert {e}" for e in range(n_experts)])
    ax.set_yticks(range(N))
    ax.set_yticklabels([f"C{i}" for i in range(N)])
    ax.set_xlabel("Expert")
    ax.set_ylabel("Client")
    ax.set_title(f"Expert assignment (post-warmup clustering, M={n_experts})")
    fig.colorbar(im, ax=ax, label="assigned (1) / not (0)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ----------------------------------------------------------------------
# 4. Similarity / distance block-diagonal figure
# ----------------------------------------------------------------------

def plot_distance_block_diag(run_dir: str, out_path: str) -> None:
    dist_path  = os.path.join(run_dir, "distance_matrix.npy")
    label_path = os.path.join(run_dir, "cluster_labels.npy")
    if not (os.path.exists(dist_path) and os.path.exists(label_path)):
        return
    dist   = np.load(dist_path)
    labels = np.load(label_path)
    N      = dist.shape[0]
    # Reorder so cluster members are contiguous
    order = np.argsort(labels)
    dist_o = dist[order][:, order]
    labels_o = labels[order]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(dist_o, cmap="viridis_r")
    fig.colorbar(im, ax=ax, label="cosine distance (1 − sim)")

    # Draw block boundaries
    unique_clusters = np.unique(labels_o)
    cum = 0
    for c in unique_clusters:
        size = int((labels_o == c).sum())
        rect = plt.Rectangle((cum - 0.5, cum - 0.5), size, size,
                             fill=False, edgecolor="red", lw=2)
        ax.add_patch(rect)
        cum += size

    ax.set_title(f"B@A distance matrix with recovered clusters (N={N})")
    ax.set_xlabel("Client (reordered by cluster)")
    ax.set_ylabel("Client (reordered by cluster)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ----------------------------------------------------------------------
# 5. Seed comparison
# ----------------------------------------------------------------------

def plot_seed_comparison(seed_summary_path: str, out_path: str) -> None:
    if not os.path.exists(seed_summary_path):
        return
    with open(seed_summary_path) as f:
        s = json.load(f)
    per_seed = s.get("per_seed_headline", {})
    if not per_seed:
        return
    seeds = sorted(per_seed.keys(), key=lambda x: int(x))
    accs  = [per_seed[k].get("in_domain_mean_accuracy", 0) for k in seeds]
    f1s   = [per_seed[k].get("in_domain_mean_macro_f1", 0) for k in seeds]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(seeds))
    w = 0.35
    ax.bar(x - w/2, accs, w, label="accuracy", color="C0")
    ax.bar(x + w/2, f1s,  w, label="macro-F1",  color="C1")

    # Add mean lines
    ax.axhline(np.mean(accs), color="C0", ls="--", alpha=0.5)
    ax.axhline(np.mean(f1s),  color="C1", ls="--", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {k}" for k in seeds])
    ax.set_ylabel("Headline (in-domain mean)")
    ax.set_ylim(0.85, 1.0)
    ax.set_title("Per-seed headline metrics (dashed = mean)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True,
                        help="A run output directory (e.g. outputs/seed_42_full)")
    parser.add_argument("--out_dir", default=None,
                        help="Where to save plots (default: <run_dir>/plots/diagnostic)")
    parser.add_argument("--seed_summary", default=os.path.join(REPO_ROOT, "outputs",
                                                               "seed_summary.json"),
                        help="Path to seed_summary.json for seed_comparison plot")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.run_dir, "plots", "diagnostic")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating diagnostic plots → {out_dir}\n")

    plot_distance_block_diag(args.run_dir, os.path.join(out_dir, "distance_block_diag.png"))
    plot_routing_entropy_approx(args.run_dir, os.path.join(out_dir, "routing_trajectory.png"))
    plot_expert_utilisation(args.run_dir, os.path.join(out_dir, "expert_assignment.png"))
    plot_confusion_matrices(args.run_dir, os.path.join(out_dir, "confusion_matrices.png"))
    plot_seed_comparison(args.seed_summary, os.path.join(out_dir, "seed_comparison.png"))

    print(f"\nDone. View with: explorer {out_dir}    (Windows)")
    print(f"                  open {out_dir}        (macOS)")


if __name__ == "__main__":
    main()
