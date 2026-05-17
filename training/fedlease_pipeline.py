"""
FedLEASE end-to-end training pipeline.

Orchestrates:
  Phase 1 — Initialisation (warmup training + clustering + expert init)
  Phase 2 — Iterative federated training (T communication rounds)

Follows Algorithm 1 from the FedLEASE paper exactly.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.preprocessing import FinancialPreprocessor
from federated.client import FederatedClient
from federated.server import FederatedServer
from training.evaluator import Evaluator
from utils.logging_utils import get_logger
from utils.metrics import MetricTracker
from utils.seed import set_seed

logger = get_logger("fedlease.pipeline")


class FedLEASEPipeline:
    """
    Coordinates the complete FedLEASE federated learning experiment.

    Parameters
    ----------
    config : ConfigDict
    client_data_list : list of per-client data dicts (from FederatedDataPartitioner)
    test_loaders : optional dict of {name: DataLoader} for held-out evaluation
    device : torch.device
    """

    def __init__(
        self,
        config: Any,
        client_data_list: List[Dict],
        test_loaders: Optional[Dict[str, DataLoader]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.client_data_list = client_data_list
        self.test_loaders = test_loaders or {}

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        set_seed(config.training.seed)

        self.output_dir = config.paths.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.preprocessor = FinancialPreprocessor(
            model_name=config.model.name,
            max_length=config.model.max_length,
        )

        self.num_clients = config.federated.num_clients
        self.num_rounds = config.federated.num_rounds
        self.local_epochs = config.federated.local_epochs
        self.warmup_epochs = config.federated.warmup_epochs
        self.batch_size = config.federated.batch_size

        # Build client DataLoaders
        self.client_loaders = self._build_client_loaders()

        # Build clients
        self.clients: Dict[int, FederatedClient] = {}
        for c_data in client_data_list:
            cid = c_data["client_id"]
            self.clients[cid] = FederatedClient(
                client_id=cid,
                model_name=config.model.name,
                train_loader=self.client_loaders[cid]["train"],
                val_loader=self.client_loaders[cid]["val"],
                device=device,
                config=config,
            )

        self.server = FederatedServer(config, output_dir=self.output_dir)
        self.metric_tracker = MetricTracker()

        # Results storage
        self.round_results: List[Dict] = []
        self.init_results: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """
        Execute the complete FedLEASE training procedure.

        Returns a results dict with history and final metrics.
        """
        logger.info(f"=== FedLEASE Pipeline | device={self.device} ===")
        logger.info(f"Clients: {self.num_clients}, Rounds: {self.num_rounds}")

        # Phase 1: Initialisation
        self.init_results = self._run_initialization()

        # Phase 2: Iterative training
        for round_t in range(1, self.num_rounds + 1):
            round_result = self._run_round(round_t)
            self.round_results.append(round_result)

            # Log per-round summary
            avg_val = np.mean([r["val_accuracy"] for r in round_result["client_stats"]])
            logger.info(f"Round {round_t:3d}/{self.num_rounds} | avg_val_acc={avg_val:.4f}")
            self.metric_tracker.update({"avg_val_accuracy": avg_val}, prefix=f"round")

            # Checkpoint server every 5 rounds
            if round_t % 5 == 0:
                self.server.save_state(round_t)

        # Final evaluation
        final_metrics = self._run_final_evaluation()
        self._save_results(final_metrics)

        return {
            "init": self.init_results,
            "round_results": self.round_results,
            "final_metrics": final_metrics,
            "comm_stats": self.server.communication_summary(),
        }

    # ------------------------------------------------------------------
    # Phase 1: Initialisation
    # ------------------------------------------------------------------

    def _run_initialization(self) -> Dict:
        """
        Warmup training → B-matrix collection → clustering → expert init.
        """
        logger.info("--- Phase 1: Initialisation ---")
        client_ids = list(self.clients.keys())

        # Step 1: Each client performs E_warmup epochs of local LoRA training
        logger.info(f"Warmup training ({self.warmup_epochs} epochs per client)")
        for cid, client in self.clients.items():
            client.warmup_train(self.warmup_epochs)

        # Step 2: Collect B matrices and (A, B) matrices from all clients
        client_b_matrices: Dict = {}
        client_ab_matrices: Dict = {}
        for cid, client in self.clients.items():
            client_b_matrices[cid] = client.get_b_matrices()
            client_ab_matrices[cid] = client.get_warmup_ab_matrices()

        # Step 3: Server clustering and expert initialisation
        init_info = self.server.run_initialization(
            client_b_matrices,
            client_ab_matrices,
            client_ids,
        )

        logger.info(
            f"Optimal M = {init_info['n_experts']} experts | "
            f"cluster assignments: {init_info['client_assignments']}"
        )
        return init_info

    # ------------------------------------------------------------------
    # Phase 2: Single communication round
    # ------------------------------------------------------------------

    def _run_round(self, round_t: int) -> Dict:
        """Execute one complete communication round."""
        client_uploads: List[Tuple] = []
        client_stats: List[Dict] = []

        # Anneal Gumbel-softmax temperature across rounds.
        # Gumbel noise is only useful AFTER the router has a basic shape.
        # For the first 30% of rounds we keep plain softmax (use_gumbel=False)
        # and a high temperature; then we switch on Gumbel and anneal
        # T: 1.0 → 0.1. This avoids the unstable noisy-router regime early on.
        total_rounds = max(self.num_rounds, 1)
        warmup_router_rounds = max(1, total_rounds // 3)
        if round_t < warmup_router_rounds:
            router_temp = 1.0
            use_gumbel = False
        else:
            t_frac = (round_t - warmup_router_rounds) / max(
                total_rounds - warmup_router_rounds - 1, 1
            )
            router_temp = 1.0 * (1.0 - t_frac) + 0.1 * t_frac
            use_gumbel = True

        for cid, client in self.clients.items():
            # Server broadcasts to this client
            payload = self.server.broadcast(cid)

            # Client receives and sets up its model
            client.setup_fedlease_model(
                n_experts=payload["n_experts"],
                expert_states=payload["expert_states"],
                router_state=payload["router_state"],
                assigned_expert_idx=payload["assigned_expert_idx"],
                head_states=payload.get("head_states"),
            )

            # Configure router temperature + Gumbel mode for this round
            try:
                from models.adaptive_router import AdaptiveTopMRouter
                for m in client.fedlease_model.modules():
                    if isinstance(m, AdaptiveTopMRouter):
                        m.anneal_temperature(router_temp)
                        m.use_gumbel = use_gumbel
            except Exception:
                pass

            # Client trains locally
            stats = client.local_train(self.local_epochs)
            client_stats.append({
                "client_id": cid,
                "val_accuracy": stats["val_accuracy"],
                "final_loss": stats["epoch_stats"][-1]["loss"] if stats["epoch_stats"] else 0.0,
            })

            # Client prepares upload
            upload = client.get_upload_payload()
            client_uploads.append(upload)

        # Server aggregates
        dataset_sizes = {cid: c.dataset_size for cid, c in self.clients.items()}
        self.server.aggregate_round(client_uploads, dataset_sizes=dataset_sizes)

        return {
            "round": round_t,
            "client_stats": client_stats,
            "timestamp": time.time(),
        }

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------

    def _run_final_evaluation(self) -> Dict:
        """Evaluate each client's model on held-out test sets.

        Produces three aggregates per test set:
          - "aggregate"        — naive mean across ALL clients (legacy)
          - "in_domain"        — mean across only the clients whose source
                                  matches the test set (the canonical FL
                                  reporting protocol, and what should be
                                  used in the headline table)
          - "cross_domain"     — mean across clients whose source does NOT
                                  match (zero-shot transfer metric)
        """
        logger.info("--- Final Evaluation ---")
        results: Dict = {}

        # Build {cid: source_tag} lookup ("phrasebank" / "twitter")
        client_source = {
            c["client_id"]: c.get("source", "unknown")
            for c in self.client_data_list
        }

        # Map test-set name → matching source tag
        # ("phrasebank_test" → "phrasebank", "twitter_test" → "twitter")
        def _source_for_test(name: str) -> str:
            for tag in ("phrasebank", "twitter"):
                if tag in name.lower():
                    return tag
            return ""

        for name, test_loader in self.test_loaders.items():
            evaluator = Evaluator(test_loader, self.device)
            per_client: Dict = {}

            for cid, client in self.clients.items():
                if client.fedlease_model is None:
                    continue
                metrics = evaluator.evaluate_model(client.fedlease_model)
                metrics["source"] = client_source.get(cid, "unknown")
                per_client[f"client_{cid}"] = metrics

            agg_all = evaluator.aggregate_metrics(per_client) if per_client else {}

            # Split per_client into in-domain and cross-domain subsets
            target_src = _source_for_test(name)
            in_domain_subset = {
                k: v for k, v in per_client.items()
                if v.get("source") == target_src
            }
            cross_domain_subset = {
                k: v for k, v in per_client.items()
                if v.get("source") and v.get("source") != target_src
            }
            agg_in = (
                evaluator.aggregate_metrics(in_domain_subset)
                if in_domain_subset else {}
            )
            agg_cross = (
                evaluator.aggregate_metrics(cross_domain_subset)
                if cross_domain_subset else {}
            )

            results[name] = {
                "per_client":   per_client,
                "aggregate":    agg_all,          # legacy — all clients
                "in_domain":    agg_in,           # canonical metric
                "cross_domain": agg_cross,        # zero-shot transfer
            }

            logger.info(
                f"[{name}] in-domain mean_acc={agg_in.get('mean_accuracy', 0):.4f}  "
                f"macro_f1={agg_in.get('mean_macro_f1', 0):.4f}  "
                f"| cross-domain mean_acc={agg_cross.get('mean_accuracy', 0):.4f}  "
                f"macro_f1={agg_cross.get('mean_macro_f1', 0):.4f}  "
                f"| all mean_acc={agg_all.get('mean_accuracy', 0):.4f}"
            )

        # ---- Headline summary across BOTH test sets (in-domain only) ----
        try:
            in_dom_accs = [
                results[name]["in_domain"].get("mean_accuracy", 0.0)
                for name in results
                if results[name]["in_domain"]
            ]
            in_dom_f1s = [
                results[name]["in_domain"].get("mean_macro_f1", 0.0)
                for name in results
                if results[name]["in_domain"]
            ]
            if in_dom_accs:
                results["_headline"] = {
                    "in_domain_mean_accuracy": float(np.mean(in_dom_accs)),
                    "in_domain_mean_macro_f1": float(np.mean(in_dom_f1s)),
                }
                logger.info(
                    f"=== HEADLINE in-domain: "
                    f"acc={results['_headline']['in_domain_mean_accuracy']:.4f}  "
                    f"macro_f1={results['_headline']['in_domain_mean_macro_f1']:.4f} ==="
                )
        except Exception as exc:
            logger.warning(f"Headline summary failed: {exc}")

        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _build_client_loaders(self) -> Dict[int, Dict[str, DataLoader]]:
        """Build train/val DataLoaders for every client."""
        loaders: Dict[int, Dict[str, DataLoader]] = {}
        for c_data in self.client_data_list:
            cid = c_data["client_id"]
            loaders[cid] = self.preprocessor.prepare_client_loaders(
                c_data,
                batch_size=self.batch_size,
            )
        return loaders

    def _save_results(self, final_metrics: Dict) -> None:
        """Persist all results to JSON."""
        out_path = os.path.join(self.output_dir, "results", "fedlease_results.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Convert non-serialisable objects
        serialisable = {
            "final_metrics": final_metrics,
            "round_history": [
                {
                    "round": r["round"],
                    "client_val_accs": [s["val_accuracy"] for s in r["client_stats"]],
                    "avg_val_acc": float(np.mean([s["val_accuracy"] for s in r["client_stats"]])),
                }
                for r in self.round_results
            ],
            "comm_stats": self.server.communication_summary(),
            "n_experts": self.server.n_experts,
            "cluster_assignments": {
                str(k): v for k, v in self.server.client_assignments.items()
            },
        }

        with open(out_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info(f"Results saved to {out_path}")

    def get_training_curves(self) -> Dict[str, List[float]]:
        """Return per-round avg validation accuracy for plotting."""
        rounds = [r["round"] for r in self.round_results]
        avg_accs = [
            float(np.mean([s["val_accuracy"] for s in r["client_stats"]]))
            for r in self.round_results
        ]
        return {"rounds": rounds, "avg_val_accuracy": avg_accs}

    def get_expert_utilisation_history(self) -> List[Dict]:
        """Placeholder — actual utilisation collected from clients if needed."""
        return []
