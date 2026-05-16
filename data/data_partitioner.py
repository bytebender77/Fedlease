"""Federated data partitioning strategies for heterogeneous client simulation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import DatasetDict


class FederatedDataPartitioner:
    """
    Partition financial datasets across federated clients.

    Supports:
      - heterogeneous: some clients get PhraseBank, others get Twitter
      - iid:           random balanced split across all clients
      - non_iid:       Dirichlet-based label skew
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def partition(
        self,
        phrasebank_splits: DatasetDict,
        twitter_splits: DatasetDict,
        num_clients: int,
        strategy: str = "heterogeneous",
        num_phrasebank_clients: int = 5,
        num_twitter_clients: int = 5,
        dirichlet_alpha: float = 0.5,
    ) -> List[Dict]:
        """
        Returns a list of per-client data dicts, each containing:
          'train_texts', 'train_labels', 'val_texts', 'val_labels',
          'source' (dataset name), 'client_id'
        """
        assert num_phrasebank_clients + num_twitter_clients == num_clients

        if strategy == "heterogeneous":
            return self._heterogeneous_partition(
                phrasebank_splits,
                twitter_splits,
                num_phrasebank_clients,
                num_twitter_clients,
            )
        elif strategy == "iid":
            return self._iid_partition(
                phrasebank_splits,
                twitter_splits,
                num_clients,
            )
        elif strategy == "non_iid":
            return self._dirichlet_partition(
                phrasebank_splits,
                twitter_splits,
                num_clients,
                alpha=dirichlet_alpha,
            )
        else:
            raise ValueError(f"Unknown partition strategy: {strategy}")

    # ------------------------------------------------------------------
    # Heterogeneous: different dataset sources per client group
    # ------------------------------------------------------------------

    def _heterogeneous_partition(
        self,
        phrasebank_splits: DatasetDict,
        twitter_splits: DatasetDict,
        num_phrasebank_clients: int,
        num_twitter_clients: int,
    ) -> List[Dict]:
        clients = []

        # Phrasebank clients
        pb_train_texts = list(phrasebank_splits["train"]["text"])
        pb_train_labels = list(phrasebank_splits["train"]["label"])
        pb_val_texts = list(phrasebank_splits["val"]["text"])
        pb_val_labels = list(phrasebank_splits["val"]["label"])

        pb_indices = self._shuffle_indices(len(pb_train_texts))
        pb_chunks = np.array_split(pb_indices, num_phrasebank_clients)
        pb_val_idx = self._shuffle_indices(len(pb_val_texts))
        pb_val_chunks = np.array_split(pb_val_idx, num_phrasebank_clients)

        for cid, (chunk, val_chunk) in enumerate(zip(pb_chunks, pb_val_chunks)):
            clients.append({
                "client_id": cid,
                "source": "phrasebank",
                "train_texts": [pb_train_texts[i] for i in chunk],
                "train_labels": [pb_train_labels[i] for i in chunk],
                "val_texts": [pb_val_texts[i] for i in val_chunk],
                "val_labels": [pb_val_labels[i] for i in val_chunk],
            })

        # Twitter clients
        tw_train_texts = list(twitter_splits["train"]["text"])
        tw_train_labels = list(twitter_splits["train"]["label"])
        tw_val_texts = list(twitter_splits["val"]["text"])
        tw_val_labels = list(twitter_splits["val"]["label"])

        tw_indices = self._shuffle_indices(len(tw_train_texts))
        tw_chunks = np.array_split(tw_indices, num_twitter_clients)
        tw_val_idx = self._shuffle_indices(len(tw_val_texts))
        tw_val_chunks = np.array_split(tw_val_idx, num_twitter_clients)

        for i, (chunk, val_chunk) in enumerate(zip(tw_chunks, tw_val_chunks)):
            cid = num_phrasebank_clients + i
            clients.append({
                "client_id": cid,
                "source": "twitter",
                "train_texts": [tw_train_texts[j] for j in chunk],
                "train_labels": [tw_train_labels[j] for j in chunk],
                "val_texts": [tw_val_texts[j] for j in val_chunk],
                "val_labels": [tw_val_labels[j] for j in val_chunk],
            })

        return clients

    # ------------------------------------------------------------------
    # IID: balanced random split across all data
    # ------------------------------------------------------------------

    def _iid_partition(
        self,
        phrasebank_splits: DatasetDict,
        twitter_splits: DatasetDict,
        num_clients: int,
    ) -> List[Dict]:
        all_train_texts = list(phrasebank_splits["train"]["text"]) + \
                          list(twitter_splits["train"]["text"])
        all_train_labels = list(phrasebank_splits["train"]["label"]) + \
                           list(twitter_splits["train"]["label"])
        all_val_texts = list(phrasebank_splits["val"]["text"]) + \
                        list(twitter_splits["val"]["text"])
        all_val_labels = list(phrasebank_splits["val"]["label"]) + \
                         list(twitter_splits["val"]["label"])

        train_idx = self._shuffle_indices(len(all_train_texts))
        val_idx = self._shuffle_indices(len(all_val_texts))

        train_chunks = np.array_split(train_idx, num_clients)
        val_chunks = np.array_split(val_idx, num_clients)

        clients = []
        for cid, (t_chunk, v_chunk) in enumerate(zip(train_chunks, val_chunks)):
            clients.append({
                "client_id": cid,
                "source": "mixed",
                "train_texts": [all_train_texts[i] for i in t_chunk],
                "train_labels": [all_train_labels[i] for i in t_chunk],
                "val_texts": [all_val_texts[i] for i in v_chunk],
                "val_labels": [all_val_labels[i] for i in v_chunk],
            })
        return clients

    # ------------------------------------------------------------------
    # Non-IID: Dirichlet label distribution
    # ------------------------------------------------------------------

    def _dirichlet_partition(
        self,
        phrasebank_splits: DatasetDict,
        twitter_splits: DatasetDict,
        num_clients: int,
        alpha: float = 0.5,
        num_classes: int = 3,
    ) -> List[Dict]:
        all_train_texts = list(phrasebank_splits["train"]["text"]) + \
                          list(twitter_splits["train"]["text"])
        all_train_labels = list(phrasebank_splits["train"]["label"]) + \
                           list(twitter_splits["train"]["label"])
        all_val_texts = list(phrasebank_splits["val"]["text"]) + \
                        list(twitter_splits["val"]["text"])
        all_val_labels = list(phrasebank_splits["val"]["label"]) + \
                         list(twitter_splits["val"]["label"])

        # Group training indices by class
        class_indices = [[] for _ in range(num_classes)]
        for idx, lbl in enumerate(all_train_labels):
            class_indices[lbl].append(idx)
        for c in range(num_classes):
            self.rng.shuffle(class_indices[c])

        # Dirichlet allocation
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            proportions = self.rng.dirichlet(alpha=np.full(num_clients, alpha))
            c_idx = class_indices[c]
            n = len(c_idx)
            cumulative = (np.cumsum(proportions) * n).astype(int)
            cumulative = np.clip(cumulative, 0, n)
            splits = np.split(c_idx, cumulative[:-1])
            for cid, chunk in enumerate(splits):
                client_indices[cid].extend(chunk.tolist())

        # Val: random uniform
        val_idx = self._shuffle_indices(len(all_val_texts))
        val_chunks = np.array_split(val_idx, num_clients)

        clients = []
        for cid in range(num_clients):
            t_idx = client_indices[cid]
            v_chunk = val_chunks[cid]
            clients.append({
                "client_id": cid,
                "source": "non_iid",
                "train_texts": [all_train_texts[i] for i in t_idx],
                "train_labels": [all_train_labels[i] for i in t_idx],
                "val_texts": [all_val_texts[i] for i in v_chunk],
                "val_labels": [all_val_labels[i] for i in v_chunk],
            })
        return clients

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _shuffle_indices(self, n: int) -> np.ndarray:
        idx = np.arange(n)
        self.rng.shuffle(idx)
        return idx

    def get_partition_stats(self, client_data_list: List[Dict]) -> Dict:
        """Compute per-client label distribution statistics."""
        stats = {}
        for c in client_data_list:
            cid = c["client_id"]
            labels = np.array(c["train_labels"])
            unique, counts = np.unique(labels, return_counts=True)
            stats[f"client_{cid}"] = {
                "source": c["source"],
                "train_size": len(labels),
                "val_size": len(c["val_labels"]),
                "label_distribution": {int(u): int(cnt) for u, cnt in zip(unique, counts)},
            }
        return stats
