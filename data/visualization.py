"""Dataset statistics visualisation utilities."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

LABEL_NAMES = ["negative", "neutral", "positive"]
COLORS = ["#E74C3C", "#3498DB", "#2ECC71"]


def plot_class_distribution(
    datasets: Dict[str, Dict],
    output_path: str,
    title: str = "Class Distribution",
) -> None:
    """Bar chart of class distribution per dataset split."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]

    for ax, (name, stats) in zip(axes, datasets.items()):
        counts = [stats["class_counts"].get(lbl, 0) for lbl in LABEL_NAMES]
        bars = ax.bar(LABEL_NAMES, counts, color=COLORS, edgecolor="black", linewidth=0.5)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_ylim(0, max(counts) * 1.2)
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.02,
                str(count),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_client_distribution(
    client_stats: Dict[str, Dict],
    output_path: str,
    num_classes: int = 3,
) -> None:
    """Stacked bar chart showing per-client label distribution."""
    client_ids = sorted(client_stats.keys())
    n_clients = len(client_ids)

    matrix = np.zeros((n_clients, num_classes))
    sources = []
    for i, cid in enumerate(client_ids):
        dist = client_stats[cid]["label_distribution"]
        total = max(sum(dist.values()), 1)
        for lbl in range(num_classes):
            matrix[i, lbl] = dist.get(lbl, 0) / total
        sources.append(client_stats[cid]["source"])

    fig, ax = plt.subplots(figsize=(max(8, n_clients * 0.8), 5))
    bottom = np.zeros(n_clients)
    x = np.arange(n_clients)

    for cls_idx, (lbl_name, color) in enumerate(zip(LABEL_NAMES, COLORS)):
        ax.bar(x, matrix[:, cls_idx], bottom=bottom, label=lbl_name, color=color, alpha=0.85)
        bottom += matrix[:, cls_idx]

    # Annotate source
    for i, src in enumerate(sources):
        marker = "★" if "phrase" in src else "●"
        ax.text(i, 1.02, marker, ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("client_", "C") for c in client_ids], rotation=45)
    ax.set_ylabel("Class Proportion")
    ax.set_title("Per-Client Label Distribution (★=PhraseBank, ●=Twitter)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_token_length_distribution(
    length_stats_per_dataset: Dict[str, List[int]],
    output_path: str,
    max_length: int = 128,
) -> None:
    """Histogram of token lengths with truncation threshold."""
    n = len(length_stats_per_dataset)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, lengths) in zip(axes, length_stats_per_dataset.items()):
        arr = np.array(lengths)
        ax.hist(arr, bins=30, color="#5B8CFA", edgecolor="white", alpha=0.85)
        ax.axvline(max_length, color="red", linestyle="--", label=f"max_len={max_length}")
        ax.set_title(f"{name}\nMedian={np.median(arr):.0f}, Max={arr.max()}")
        ax.set_xlabel("Token Length")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle("Token Length Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_overview(
    phrasebank_stats: Dict,
    twitter_stats: Dict,
    client_stats: Dict,
    output_dir: str,
) -> None:
    """Generate all dataset visualisation plots."""
    os.makedirs(output_dir, exist_ok=True)
    plot_class_distribution(
        {
            "PhraseBank Train": phrasebank_stats.get("train", {}),
            "PhraseBank Val": phrasebank_stats.get("val", {}),
            "PhraseBank Test": phrasebank_stats.get("test", {}),
        },
        os.path.join(output_dir, "phrasebank_class_dist.png"),
        "Financial PhraseBank Class Distribution",
    )
    plot_class_distribution(
        {
            "Twitter Train": twitter_stats.get("train", {}),
            "Twitter Val": twitter_stats.get("val", {}),
            "Twitter Test": twitter_stats.get("test", {}),
        },
        os.path.join(output_dir, "twitter_class_dist.png"),
        "Twitter Financial News Class Distribution",
    )
    plot_client_distribution(
        client_stats,
        os.path.join(output_dir, "client_label_distribution.png"),
    )
