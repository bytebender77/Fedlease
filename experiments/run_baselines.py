#!/usr/bin/env python3
"""Run the three comparison baselines and aggregate results into one table.

Baselines:
  1. fedavg_lora     — Single global LoRA expert, single global classifier head.
                        (M=1, no clustering, no per-cluster heads.) The standard
                        FedAvg-of-LoRA-adapters baseline.
  2. per_client_lora — Each client trains a private LoRA forever, no aggregation.
                        Personalization upper bound, no info sharing.
  3. centralized     — Pool both datasets, train one LoRA on the union.
                        Oracle upper bound on what FL could in principle achieve.

Usage:
    python experiments/run_baselines.py                       # all three
    python experiments/run_baselines.py --baselines fedavg_lora centralized
    python experiments/run_baselines.py --seed 42 --preset full

Outputs are saved to:
    outputs/baseline_{name}_{preset}_seed{seed}/results/fedlease_results.json
And a comparison summary at:
    outputs/baselines_summary.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import tempfile
import time
from typing import Dict, List

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


PRESETS = {
    "debug":  dict(num_rounds=3,  local_epochs=1, warmup_epochs=1, batch_size=16, max_experts=4),
    "quick":  dict(num_rounds=10, local_epochs=1, warmup_epochs=3, batch_size=32, max_experts=6),
    "medium": dict(num_rounds=15, local_epochs=2, warmup_epochs=4, batch_size=32, max_experts=8),
    "full":   dict(num_rounds=25, local_epochs=2, warmup_epochs=5, batch_size=32, max_experts=8),
}

BASELINES = ["fedavg_lora", "per_client_lora", "centralized"]


def _format_metric(mean: float, std: float, n: int) -> str:
    if n <= 1:
        return f"{mean:.4f}"
    # 95% CI t-multiplier
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)
    ci = t * std / math.sqrt(n)
    return f"{mean:.4f} ± {ci:.4f}"


# ----------------------------------------------------------------------
# Baseline 1: FedAvg-LoRA — re-uses run.py with M=1 and global head
# ----------------------------------------------------------------------

def run_fedavg_lora(seed: int, preset: str, output_dir: str, base_config: str) -> None:
    """Patch config to M=1 + no per-cluster heads, then call run_fedlease."""
    import subprocess

    # Write a patched config
    with open(base_config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["clustering"]["min_clusters"] = 1
    cfg["clustering"]["max_clusters"] = 1
    cfg["federated"]["max_experts"]   = 1
    cfg["federated"]["min_experts"]   = 1
    cfg["model"]["use_per_cluster_heads"] = False  # single global head

    patched_path = os.path.join(tempfile.gettempdir(),
                                f"fedavg_lora_{seed}.yaml")
    with open(patched_path, "w") as f:
        yaml.safe_dump(cfg, f)

    P = PRESETS[preset]
    cmd = [
        sys.executable, os.path.join(REPO_ROOT, "experiments", "run_fedlease.py"),
        "--config", patched_path,
        "--output_dir", output_dir,
        "--seed", str(seed),
        "--num_rounds", str(P["num_rounds"]),
        "--local_epochs", str(P["local_epochs"]),
        "--warmup_epochs", str(P["warmup_epochs"]),
        "--batch_size", str(P["batch_size"]),
        "--max_experts", "1",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


# ----------------------------------------------------------------------
# Baseline 2: Per-client LoRA — each client trains a private warmup model
# ----------------------------------------------------------------------

def run_per_client_lora(seed: int, preset: str, output_dir: str, base_config: str) -> None:
    """Each client trains its own LoRA for the FULL local budget, no aggregation.

    Total local epochs per client = warmup_epochs + num_rounds * local_epochs.
    This is the "no federation at all" personalization upper bound.
    """
    import torch
    import numpy as np
    from utils.config import load_config
    from utils.logging_utils import setup_logger
    from utils.seed import set_seed
    from data.dataset_loader import FinancialDatasetLoader
    from data.data_partitioner import FederatedDataPartitioner
    from data.preprocessing import FinancialPreprocessor
    from federated.client import FederatedClient
    from training.evaluator import Evaluator
    from models.finbert_lora_moe import WarmupFinBERT
    from utils.metrics import compute_metrics

    cfg = load_config(base_config)
    P = PRESETS[preset]
    # Apply preset overrides
    cfg.training.seed = seed
    cfg.federated.num_rounds   = P["num_rounds"]
    cfg.federated.local_epochs = P["local_epochs"]
    cfg.federated.warmup_epochs = P["warmup_epochs"]
    cfg.federated.batch_size   = P["batch_size"]
    cfg.paths.output_dir = output_dir

    # Setup
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    log = setup_logger("baseline_pclora", log_dir=os.path.join(output_dir, "logs"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"PER-CLIENT-LoRA baseline | seed={seed} | device={device}")

    # Data
    loader = FinancialDatasetLoader(
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=seed,
    )
    pb = loader.load_phrasebank()
    tw = loader.load_twitter()

    partitioner = FederatedDataPartitioner(seed=seed)
    client_data_list = partitioner.partition(
        phrasebank_splits=pb, twitter_splits=tw,
        num_clients=cfg.federated.num_clients,
        strategy=cfg.data.partition_strategy,
        num_phrasebank_clients=cfg.data.num_phrasebank_clients,
        num_twitter_clients=cfg.data.num_twitter_clients,
    )

    preprocessor = FinancialPreprocessor(
        model_name=cfg.model.name,
        max_length=cfg.model.max_length,
    )
    test_loaders = {
        "phrasebank_test": preprocessor.make_dataloader(
            pb["test"]["text"], pb["test"]["label"],
            batch_size=cfg.federated.batch_size, shuffle=False,
        ),
        "twitter_test": preprocessor.make_dataloader(
            tw["test"]["text"], tw["test"]["label"],
            batch_size=cfg.federated.batch_size, shuffle=False,
        ),
    }

    client_source = {c["client_id"]: c["source"] for c in client_data_list}

    # Per-client training (no aggregation)
    total_epochs = (
        cfg.federated.warmup_epochs
        + cfg.federated.num_rounds * cfg.federated.local_epochs
    )
    log.info(f"Each client trains for {total_epochs} local epochs (no sharing)")

    per_client_metrics: Dict[str, Dict] = {
        "phrasebank_test": {}, "twitter_test": {},
    }
    for cdata in client_data_list:
        cid = cdata["client_id"]
        loaders = preprocessor.prepare_client_loaders(
            cdata, batch_size=cfg.federated.batch_size
        )
        client = FederatedClient(
            client_id=cid, model_name=cfg.model.name,
            train_loader=loaders["train"], val_loader=loaders["val"],
            device=device, config=cfg,
        )
        client.warmup_train(total_epochs)
        # Evaluate on both test sets
        for tname, tloader in test_loaders.items():
            ev = Evaluator(tloader, device)
            m = ev.evaluate_model(client.warmup_model)
            m["source"] = cdata["source"]
            per_client_metrics[tname][f"client_{cid}"] = m
        del client
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Build the same in/cross/all aggregates as the main pipeline
    def _agg(d):
        if not d:
            return {}
        keys = ["accuracy", "macro_f1"]
        out = {}
        for k in keys:
            vals = [v[k] for v in d.values()]
            out[f"mean_{k}"] = float(sum(vals) / len(vals))
        return out

    final_metrics: Dict = {}
    in_dom_accs, in_dom_f1s = [], []
    for tname, per in per_client_metrics.items():
        target_src = "phrasebank" if "phrasebank" in tname else "twitter"
        in_d  = {k: v for k, v in per.items() if v["source"] == target_src}
        cross = {k: v for k, v in per.items() if v["source"] != target_src}
        agg_in  = _agg(in_d)
        agg_x   = _agg(cross)
        agg_all = _agg(per)
        final_metrics[tname] = {
            "per_client":   per,
            "aggregate":    agg_all,
            "in_domain":    agg_in,
            "cross_domain": agg_x,
        }
        if agg_in:
            in_dom_accs.append(agg_in["mean_accuracy"])
            in_dom_f1s.append(agg_in["mean_macro_f1"])
        log.info(f"[{tname}] in-domain acc={agg_in.get('mean_accuracy', 0):.4f}  "
                 f"F1={agg_in.get('mean_macro_f1', 0):.4f}")

    if in_dom_accs:
        final_metrics["_headline"] = {
            "in_domain_mean_accuracy": float(np.mean(in_dom_accs)),
            "in_domain_mean_macro_f1": float(np.mean(in_dom_f1s)),
        }
        log.info(f"HEADLINE: acc={final_metrics['_headline']['in_domain_mean_accuracy']:.4f}  "
                 f"F1={final_metrics['_headline']['in_domain_mean_macro_f1']:.4f}")

    out_path = os.path.join(output_dir, "results", "fedlease_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"final_metrics": final_metrics,
                   "baseline": "per_client_lora",
                   "seed": seed,
                   "total_local_epochs": total_epochs}, f, indent=2)
    log.info(f"Saved → {out_path}")


# ----------------------------------------------------------------------
# Baseline 3: Centralized oracle — pool both datasets, single LoRA
# ----------------------------------------------------------------------

def run_centralized(seed: int, preset: str, output_dir: str, base_config: str) -> None:
    """Pool all training data, train one global LoRA."""
    import torch
    import numpy as np
    from utils.config import load_config
    from utils.logging_utils import setup_logger
    from utils.seed import set_seed
    from data.dataset_loader import FinancialDatasetLoader
    from data.preprocessing import FinancialPreprocessor
    from training.evaluator import Evaluator
    from models.finbert_lora_moe import WarmupFinBERT
    from federated.client import FederatedClient

    cfg = load_config(base_config)
    P = PRESETS[preset]
    cfg.training.seed = seed
    cfg.federated.num_rounds   = P["num_rounds"]
    cfg.federated.local_epochs = P["local_epochs"]
    cfg.federated.warmup_epochs = P["warmup_epochs"]
    cfg.federated.batch_size   = P["batch_size"]
    cfg.paths.output_dir = output_dir

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    log = setup_logger("baseline_centralized", log_dir=os.path.join(output_dir, "logs"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"CENTRALIZED baseline | seed={seed} | device={device}")

    # Load + pool
    loader = FinancialDatasetLoader(
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=seed,
    )
    pb = loader.load_phrasebank()
    tw = loader.load_twitter()

    pooled_texts  = list(pb["train"]["text"])  + list(tw["train"]["text"])
    pooled_labels = list(pb["train"]["label"]) + list(tw["train"]["label"])
    val_texts     = list(pb["val"]["text"])    + list(tw["val"]["text"])
    val_labels    = list(pb["val"]["label"])   + list(tw["val"]["label"])
    log.info(f"Pooled training: {len(pooled_texts)} examples (PB+TW)")

    preprocessor = FinancialPreprocessor(
        model_name=cfg.model.name,
        max_length=cfg.model.max_length,
    )
    train_loader = preprocessor.make_dataloader(
        pooled_texts, pooled_labels,
        batch_size=cfg.federated.batch_size, shuffle=True,
    )
    val_loader = preprocessor.make_dataloader(
        val_texts, val_labels,
        batch_size=cfg.federated.batch_size, shuffle=False,
    )

    test_loaders = {
        "phrasebank_test": preprocessor.make_dataloader(
            pb["test"]["text"], pb["test"]["label"],
            batch_size=cfg.federated.batch_size, shuffle=False,
        ),
        "twitter_test": preprocessor.make_dataloader(
            tw["test"]["text"], tw["test"]["label"],
            batch_size=cfg.federated.batch_size, shuffle=False,
        ),
    }

    # Train one centralized LoRA (uses our FederatedClient as a convenient wrapper)
    total_epochs = (
        cfg.federated.warmup_epochs
        + cfg.federated.num_rounds * cfg.federated.local_epochs
    )
    log.info(f"Training a single LoRA for {total_epochs} epochs over pooled data")

    fake_client = FederatedClient(
        client_id=0, model_name=cfg.model.name,
        train_loader=train_loader, val_loader=val_loader,
        device=device, config=cfg,
    )
    fake_client.warmup_train(total_epochs)

    # Evaluate
    final_metrics: Dict = {}
    in_dom_accs, in_dom_f1s = [], []
    for tname, tloader in test_loaders.items():
        ev = Evaluator(tloader, device)
        m = ev.evaluate_model(fake_client.warmup_model)
        # Centralized doesn't have "in-domain vs cross-domain" — it's an oracle.
        # Report the same number for in_domain/aggregate.
        agg = {"mean_accuracy": float(m["accuracy"]),
               "mean_macro_f1": float(m["macro_f1"])}
        final_metrics[tname] = {
            "per_client":   {"client_oracle": m},
            "aggregate":    agg,
            "in_domain":    agg,
            "cross_domain": {},
        }
        in_dom_accs.append(agg["mean_accuracy"])
        in_dom_f1s.append(agg["mean_macro_f1"])
        log.info(f"[{tname}] acc={agg['mean_accuracy']:.4f}  F1={agg['mean_macro_f1']:.4f}")

    final_metrics["_headline"] = {
        "in_domain_mean_accuracy": float(np.mean(in_dom_accs)),
        "in_domain_mean_macro_f1": float(np.mean(in_dom_f1s)),
    }
    log.info(f"HEADLINE: acc={final_metrics['_headline']['in_domain_mean_accuracy']:.4f}  "
             f"F1={final_metrics['_headline']['in_domain_mean_macro_f1']:.4f}")

    out_path = os.path.join(output_dir, "results", "fedlease_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"final_metrics": final_metrics,
                   "baseline": "centralized",
                   "seed": seed,
                   "total_epochs": total_epochs}, f, indent=2)
    log.info(f"Saved → {out_path}")


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------

DISPATCH = {
    "fedavg_lora":     run_fedavg_lora,
    "per_client_lora": run_per_client_lora,
    "centralized":     run_centralized,
}


def aggregate_summary(seeds: List[int], baselines: List[str], preset: str,
                      output_root: str) -> None:
    """Print and save a comparison table across baselines + FedLEASE."""
    print("\n" + "=" * 78)
    print(f"BASELINE COMPARISON — preset={preset}, seeds={seeds}")
    print("=" * 78)
    print(f"\n{'Method':<22} | "
          f"{'PB acc':>15} | {'PB F1':>15} | "
          f"{'TW acc':>15} | {'TW F1':>15}")
    print("-" * 22 + "-+-" + "-" * 15 + "-+-" + "-" * 15
          + "-+-" + "-" * 15 + "-+-" + "-" * 15)

    summary: Dict = {"preset": preset, "seeds": seeds, "methods": {}}

    # FedLEASE — read from outputs/seed_summary.json if present
    fed_path = os.path.join(output_root, "seed_summary.json")
    if os.path.exists(fed_path):
        with open(fed_path) as f:
            fed = json.load(f)
        pb_in = fed["per_test_set"].get("phrasebank_test|in_domain", {})
        tw_in = fed["per_test_set"].get("twitter_test|in_domain", {})
        row = (
            f"{'FedLEASE (ours)':<22} | "
            f"{_format_metric(pb_in['accuracy']['mean'], pb_in['accuracy']['std'], pb_in['accuracy']['n']):>15} | "
            f"{_format_metric(pb_in['macro_f1']['mean'], pb_in['macro_f1']['std'], pb_in['macro_f1']['n']):>15} | "
            f"{_format_metric(tw_in['accuracy']['mean'], tw_in['accuracy']['std'], tw_in['accuracy']['n']):>15} | "
            f"{_format_metric(tw_in['macro_f1']['mean'], tw_in['macro_f1']['std'], tw_in['macro_f1']['n']):>15}"
        )
        print(row)
        summary["methods"]["FedLEASE"] = {"phrasebank_in": pb_in, "twitter_in": tw_in}

    for b in baselines:
        pb_accs, pb_f1s, tw_accs, tw_f1s = [], [], [], []
        for s in seeds:
            p = os.path.join(output_root, f"baseline_{b}_{preset}_seed{s}",
                             "results", "fedlease_results.json")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                d = json.load(f)
            fm = d.get("final_metrics", {})
            pb = fm.get("phrasebank_test", {}).get("in_domain", {})
            tw = fm.get("twitter_test",    {}).get("in_domain", {})
            if pb: pb_accs.append(pb.get("mean_accuracy", 0)); pb_f1s.append(pb.get("mean_macro_f1", 0))
            if tw: tw_accs.append(tw.get("mean_accuracy", 0)); tw_f1s.append(tw.get("mean_macro_f1", 0))
        n = max(len(pb_accs), 1)
        def m_s(vals):
            if not vals: return (0.0, 0.0, 0)
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v-mean)**2 for v in vals) / max(len(vals)-1, 1))
            return (mean, std, len(vals))
        pba = m_s(pb_accs); pbf = m_s(pb_f1s); twa = m_s(tw_accs); twf = m_s(tw_f1s)
        row = (
            f"{b:<22} | "
            f"{_format_metric(*pba):>15} | "
            f"{_format_metric(*pbf):>15} | "
            f"{_format_metric(*twa):>15} | "
            f"{_format_metric(*twf):>15}"
        )
        print(row)
        summary["methods"][b] = {
            "phrasebank_in_domain": {"acc": pba, "f1": pbf},
            "twitter_in_domain":    {"acc": twa, "f1": twf},
        }

    summary_path = os.path.join(output_root, "baselines_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved → {summary_path}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baselines", nargs="+", default=BASELINES,
                        choices=BASELINES)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Random seed(s); use multiple for CIs")
    parser.add_argument("--preset", default="full",
                        choices=list(PRESETS.keys()))
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "configs",
                                                         "finbert_fedlease.yaml"))
    parser.add_argument("--output_root", default=os.path.join(REPO_ROOT, "outputs"))
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--summary_only", action="store_true",
                        help="Don't run anything — only aggregate existing results")
    args = parser.parse_args()

    if not args.summary_only:
        t_start = time.time()
        for baseline in args.baselines:
            for seed in args.seeds:
                out_dir = os.path.join(
                    args.output_root, f"baseline_{baseline}_{args.preset}_seed{seed}"
                )
                results = os.path.join(out_dir, "results", "fedlease_results.json")
                if args.skip_existing and os.path.exists(results):
                    print(f"[skip] {baseline} seed={seed} (exists)")
                    continue
                print(f"\n{'=' * 78}")
                print(f"Running baseline={baseline} seed={seed} preset={args.preset}")
                print(f"  → {out_dir}")
                print(f"{'=' * 78}\n", flush=True)
                DISPATCH[baseline](seed, args.preset, out_dir, args.config)
        print(f"\nAll baseline runs done in {(time.time() - t_start) / 60:.1f} min")

    aggregate_summary(args.seeds, args.baselines, args.preset, args.output_root)


if __name__ == "__main__":
    main()
