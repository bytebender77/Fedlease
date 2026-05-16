"""
FedLEASE main experiment runner.

Usage:
  python experiments/run_fedlease.py
  python experiments/run_fedlease.py --config configs/finbert_fedlease.yaml
  python experiments/run_fedlease.py --num_clients 10 --num_rounds 25

Run from the fedlease/ directory:
  cd /path/to/fedlease
  python experiments/run_fedlease.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.dataset_loader import FinancialDatasetLoader
from data.data_partitioner import FederatedDataPartitioner
from data.preprocessing import FinancialPreprocessor
from data.visualization import plot_dataset_overview
from training.fedlease_pipeline import FedLEASEPipeline
from utils.config import load_config, merge_configs, ConfigDict
from utils.logging_utils import setup_logger
from utils.seed import set_seed
from visualization.cluster_plots import plot_cluster_summary
from visualization.similarity_plots import plot_similarity_heatmap, plot_distance_heatmap
from visualization.training_plots import (
    plot_training_curves,
    plot_client_accuracy_bars,
    plot_communication_cost,
)
from clustering.clustering import build_linkage_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedLEASE Financial NLP Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "configs", "base_config.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument("--num_clients", type=int, default=None)
    parser.add_argument("--num_rounds", type=int, default=None)
    parser.add_argument("--local_epochs", type=int, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--max_experts", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument(
        "--partition",
        choices=["heterogeneous", "iid", "non_iid"],
        default=None,
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ConfigDict:
    """Load base config and apply CLI overrides."""
    cfg = load_config(args.config)

    overrides: dict = {}
    if args.num_clients is not None:
        overrides.setdefault("federated", {})["num_clients"] = args.num_clients
    if args.num_rounds is not None:
        overrides.setdefault("federated", {})["num_rounds"] = args.num_rounds
    if args.local_epochs is not None:
        overrides.setdefault("federated", {})["local_epochs"] = args.local_epochs
    if args.warmup_epochs is not None:
        overrides.setdefault("federated", {})["warmup_epochs"] = args.warmup_epochs
    if args.batch_size is not None:
        overrides.setdefault("federated", {})["batch_size"] = args.batch_size
    if args.max_experts is not None:
        overrides.setdefault("federated", {})["max_experts"] = args.max_experts
        overrides.setdefault("clustering", {})["max_clusters"] = args.max_experts
    if args.lora_rank is not None:
        overrides.setdefault("lora", {})["rank"] = args.lora_rank
    if args.lr is not None:
        overrides.setdefault("training", {})["learning_rate"] = args.lr
    if args.seed is not None:
        overrides.setdefault("training", {})["seed"] = args.seed
    if args.output_dir is not None:
        overrides.setdefault("paths", {})["output_dir"] = args.output_dir
    if args.partition is not None:
        overrides.setdefault("data", {})["partition_strategy"] = args.partition

    if overrides:
        cfg = merge_configs(cfg, overrides)

    return cfg


def load_and_partition_data(cfg: ConfigDict, logger):
    """Load both datasets and partition across clients."""
    logger.info("Loading datasets …")
    loader = FinancialDatasetLoader(
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.training.seed,
    )
    phrasebank_splits = loader.load_phrasebank()
    twitter_splits = loader.load_twitter()

    pb_stats = loader.get_statistics(phrasebank_splits)
    tw_stats = loader.get_statistics(twitter_splits)
    logger.info(f"PhraseBank: {pb_stats['train']['total']} train, "
                f"{pb_stats['val']['total']} val, {pb_stats['test']['total']} test")
    logger.info(f"Twitter:    {tw_stats['train']['total']} train, "
                f"{tw_stats['val']['total']} val, {tw_stats['test']['total']} test")

    partitioner = FederatedDataPartitioner(seed=cfg.training.seed)
    client_data_list = partitioner.partition(
        phrasebank_splits=phrasebank_splits,
        twitter_splits=twitter_splits,
        num_clients=cfg.federated.num_clients,
        strategy=cfg.data.partition_strategy,
        num_phrasebank_clients=cfg.data.num_phrasebank_clients,
        num_twitter_clients=cfg.data.num_twitter_clients,
        dirichlet_alpha=cfg.data.dirichlet_alpha,
    )

    partition_stats = partitioner.get_partition_stats(client_data_list)
    for cid_key, s in partition_stats.items():
        logger.info(
            f"  {cid_key} [{s['source']}]: "
            f"train={s['train_size']}, val={s['val_size']}, "
            f"dist={s['label_distribution']}"
        )

    # Plot dataset stats
    plot_dir = os.path.join(cfg.paths.output_dir, "plots")
    try:
        plot_dataset_overview(pb_stats, tw_stats, partition_stats, plot_dir)
        logger.info(f"Dataset plots saved to {plot_dir}")
    except Exception as e:
        logger.warning(f"Dataset visualisation failed: {e}")

    return phrasebank_splits, twitter_splits, client_data_list, partition_stats


def build_test_loaders(
    phrasebank_splits,
    twitter_splits,
    cfg: ConfigDict,
):
    """Build global test DataLoaders for held-out evaluation."""
    preprocessor = FinancialPreprocessor(
        model_name=cfg.model.name,
        max_length=cfg.model.max_length,
    )
    test_loaders = {}

    pb_test = phrasebank_splits["test"]
    test_loaders["phrasebank_test"] = preprocessor.make_dataloader(
        pb_test["text"], pb_test["label"],
        batch_size=cfg.federated.batch_size, shuffle=False,
    )

    tw_test = twitter_splits["test"]
    test_loaders["twitter_test"] = preprocessor.make_dataloader(
        tw_test["text"], tw_test["label"],
        batch_size=cfg.federated.batch_size, shuffle=False,
    )

    return test_loaders


def generate_post_run_plots(pipeline: FedLEASEPipeline, cfg: ConfigDict, logger) -> None:
    """Generate all visualisation plots after training completes."""
    plot_dir = os.path.join(cfg.paths.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    server = pipeline.server

    # ---- Clustering plots ----
    dist_path = os.path.join(cfg.paths.output_dir, "distance_matrix.npy")
    label_path = os.path.join(cfg.paths.output_dir, "cluster_labels.npy")
    sil_path = os.path.join(cfg.paths.output_dir, "silhouette_scores.npy")

    if os.path.exists(dist_path):
        dist_matrix = np.load(dist_path)
        cluster_labels = np.load(label_path)
        sil_data = np.load(sil_path)
        silhouette_scores = {int(k): float(v) for k, v in sil_data}

        try:
            linkage_mat = build_linkage_matrix(dist_matrix)
            source_tags = [c["source"] for c in pipeline.client_data_list]
            client_labels = [f"C{c['client_id']}" for c in pipeline.client_data_list]

            plot_cluster_summary(
                silhouette_scores=silhouette_scores,
                optimal_k=server.n_experts,
                dist_matrix=dist_matrix,
                linkage_matrix=linkage_mat,
                cluster_labels=cluster_labels,
                output_dir=os.path.join(plot_dir, "clustering"),
                client_labels=client_labels,
                source_tags=source_tags,
            )
            logger.info("Clustering plots generated.")
        except Exception as e:
            logger.warning(f"Clustering plot generation failed: {e}")

    # ---- Training curves ----
    try:
        curves = pipeline.get_training_curves()
        plot_training_curves(
            {"FedLEASE": curves["avg_val_accuracy"]},
            os.path.join(plot_dir, "training_curve.png"),
            title="FedLEASE Training Curve — Financial Sentiment",
        )
        logger.info("Training curve saved.")
    except Exception as e:
        logger.warning(f"Training curve plot failed: {e}")

    # ---- Communication cost ----
    try:
        comm = server.communication_summary()
        bytes_list = server.comm_stats.get("bytes_per_round", [])
        if bytes_list:
            plot_communication_cost(
                bytes_list,
                os.path.join(plot_dir, "communication_cost.png"),
            )
            logger.info("Communication cost plot saved.")
    except Exception as e:
        logger.warning(f"Communication plot failed: {e}")


def _detect_device(no_cuda: bool) -> torch.device:
    """Pick the best available device across CUDA / Apple MPS / CPU."""
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon (M1/M2/M3) — Metal Performance Shaders
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_env_file() -> None:
    """Load HF_TOKEN (etc.) from a .env file in the repo root, if present.

    Lightweight parser — no external dotenv dependency required.
    """
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    if not os.path.isfile(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'").strip('"')
                # Don't clobber values already in the env
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass  # .env is optional — silently ignore parse errors


def main() -> None:
    _load_env_file()                       # picks up HF_TOKEN if user has .env
    args = parse_args()
    cfg = build_config(args)

    # Setup
    set_seed(cfg.training.seed)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    logger = setup_logger(
        "fedlease",
        log_dir=cfg.paths.log_dir,
        level=cfg.logging.level,
    )

    device = _detect_device(args.no_cuda)

    # mixed_precision (fp16 autocast) only works cleanly on CUDA.
    # Force-disable it on MPS / CPU to avoid Half/Float mismatches.
    if device.type != "cuda" and cfg.training.mixed_precision:
        logger.warning(
            f"Disabling mixed_precision on device={device.type} "
            "(autocast is CUDA-only here)"
        )
        cfg.training.mixed_precision = False

    logger.info(f"Device: {device}")
    if device.type == "cuda":
        try:
            logger.info(f"GPU: {torch.cuda.get_device_name(0)} "
                        f"| VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        except Exception:
            pass
    logger.info(f"Config: clients={cfg.federated.num_clients}, "
                f"rounds={cfg.federated.num_rounds}, "
                f"lora_rank={cfg.lora.rank}, "
                f"max_experts={cfg.clustering.max_clusters}")

    # Data — wrap in try/except so silent crashes become loud failures
    try:
        phrasebank_splits, twitter_splits, client_data_list, partition_stats = \
            load_and_partition_data(cfg, logger)
    except Exception as exc:
        logger.error("=" * 60)
        logger.error(f"DATA LOADING FAILED: {type(exc).__name__}: {exc}")
        logger.error("Common causes:")
        logger.error("  1. No network access to huggingface.co")
        logger.error("  2. HF_TOKEN missing or invalid (set via env var or .env)")
        logger.error("  3. Stale HF cache — try: rm -rf ~/.cache/huggingface")
        logger.error("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    test_loaders = build_test_loaders(phrasebank_splits, twitter_splits, cfg)

    # Run FedLEASE
    logger.info("=== Starting FedLEASE Training ===")
    t_start = time.time()

    pipeline = FedLEASEPipeline(
        config=cfg,
        client_data_list=client_data_list,
        test_loaders=test_loaders,
        device=device,
    )

    results = pipeline.run()

    elapsed = time.time() - t_start
    logger.info(f"Training complete in {elapsed / 60:.1f} min")

    # Summarise results
    comm_stats = results["comm_stats"]
    logger.info("=== Results Summary ===")
    logger.info(f"Total rounds: {comm_stats['total_rounds']}")
    logger.info(f"Total communication: {comm_stats['total_bytes_MB']:.2f} MB")
    logger.info(f"Optimal experts (M): {pipeline.server.n_experts}")

    for dataset_name, metrics in results["final_metrics"].items():
        agg = metrics.get("aggregate", {})
        logger.info(
            f"[{dataset_name}] mean_acc={agg.get('mean_accuracy', 0):.4f}  "
            f"mean_macro_f1={agg.get('mean_macro_f1', 0):.4f}"
        )

    # Plots
    logger.info("Generating visualisation plots …")
    generate_post_run_plots(pipeline, cfg, logger)

    logger.info(f"All outputs saved to: {cfg.paths.output_dir}")


if __name__ == "__main__":
    main()
