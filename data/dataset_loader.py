"""Dataset loader for FedLEASE.

Financial PhraseBank loading notes
------------------------------------
The ``financial_phrasebank`` HuggingFace repo uses a legacy loading script
that ``datasets>=3.0`` refuses to execute.  Neither ``trust_remote_code=True``
(now rejected) nor alternative repo mirrors solve the problem because every
known mirror also ships the ``.py`` script file.

Solution: bypass ``load_dataset()`` entirely for PhraseBank.
We use ``huggingface_hub.hf_hub_download`` to download the raw zip that the
loading script itself would have downloaded, then parse it directly.
``huggingface_hub`` is always available (it is a dependency of
``transformers``) and never executes scripts — it just fetches files.

Twitter Financial News Sentiment does NOT use a loading script, so
``load_dataset()`` works fine for that dataset.
"""

from __future__ import annotations

import io
import os
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split

LABEL_NAMES            = ["negative", "neutral", "positive"]
LABEL_MAP_PHRASEBANK   = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_MAP_TWITTER      = {"Bearish": 0, "Neutral": 1, "Bullish": 2}
PHRASEBANK_INT_TO_NAME = {0: "negative", 1: "neutral", 2: "positive"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hf_token() -> Optional[str]:
    """Return HuggingFace token from env if set."""
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )


def _parse_phrasebank_txt(content: str) -> Tuple[List[str], List[int]]:
    """Parse Financial PhraseBank raw text format.

    Each line is ``sentence@sentiment_label`` where label is one of
    ``negative``, ``neutral``, ``positive``.
    """
    lmap = {"negative": 0, "neutral": 1, "positive": 2}
    texts: List[str]  = []
    labels: List[int] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or "@" not in line:
            continue
        *parts, label_str = line.rsplit("@", 1)
        sentence  = "@".join(parts).strip()   # rejoin in case sentence had @
        label_str = label_str.strip().lower()
        if sentence and label_str in lmap:
            texts.append(sentence)
            labels.append(lmap[label_str])
    return texts, labels


def _load_phrasebank_via_hf_hub(config: str = "sentences_allagree") -> Tuple[List[str], List[int]]:
    """Download Financial PhraseBank directly via huggingface_hub.

    Approach:
    1. List all files in the ``financial_phrasebank`` dataset repo.
    2. Download whichever zip / txt file matches the requested config.
    3. Parse the ``sentence@label`` text format.

    This never calls ``load_dataset()`` so the loading-script ban is irrelevant.
    """
    from huggingface_hub import hf_hub_download, list_repo_files  # always available

    token = _hf_token()
    repo_id = "financial_phrasebank"

    print(f"  [loader] Listing files in {repo_id}...", flush=True)
    try:
        all_files = list(list_repo_files(repo_id, repo_type="dataset", token=token))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot list files in {repo_id}: {exc}\n"
            "Check your internet connection or set HF_TOKEN env var."
        ) from exc

    print(f"  [loader] Repo files: {all_files}", flush=True)

    # ── Try 1: look for a pre-extracted .txt matching the config ────────────
    config_key = config.replace("sentences_", "")   # e.g. "allagree"
    txt_candidates = [
        f for f in all_files
        if config_key in f.lower() and f.endswith(".txt")
    ]
    if txt_candidates:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=txt_candidates[0],
            repo_type="dataset",
            token=token,
        )
        with open(local_path, "r", encoding="latin-1") as fh:
            content = fh.read()
        texts, labels = _parse_phrasebank_txt(content)
        if texts:
            print(f"  [loader] Loaded {len(texts)} sentences from {txt_candidates[0]}", flush=True)
            return texts, labels

    # ── Try 2: download zip and extract matching txt ─────────────────────────
    zip_candidates = [f for f in all_files if f.endswith(".zip")]
    if not zip_candidates:
        raise RuntimeError(
            f"No .zip or matching .txt found in {repo_id}.\n"
            f"Files present: {all_files}"
        )

    zip_name   = zip_candidates[0]
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=zip_name,
        repo_type="dataset",
        token=token,
    )
    print(f"  [loader] Downloaded {zip_name}", flush=True)

    with zipfile.ZipFile(local_path) as zf:
        zip_entries = zf.namelist()
        print(f"  [loader] Zip contents: {zip_entries}", flush=True)

        # Find the txt file matching the requested config
        txt_entry = next(
            (
                n for n in zip_entries
                if config_key in n.lower() and n.endswith(".txt")
            ),
            None,
        )
        if txt_entry is None:
            raise RuntimeError(
                f"Could not find a .txt for config={config!r} inside {zip_name}.\n"
                f"Zip contents: {zip_entries}"
            )

        with zf.open(txt_entry) as fh:
            content = fh.read().decode("latin-1")

    texts, labels = _parse_phrasebank_txt(content)
    print(f"  [loader] Parsed {len(texts)} sentences from {txt_entry}", flush=True)
    return texts, labels


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class FinancialDatasetLoader:
    """Loads and stratified-splits both financial-NLP datasets."""

    def __init__(
        self,
        val_ratio:  float           = 0.10,
        test_ratio: float           = 0.20,
        seed:       int             = 42,
        cache_dir:  Optional[str]   = None,
    ) -> None:
        self.val_ratio  = val_ratio
        self.test_ratio = test_ratio
        self.seed       = seed
        self.cache_dir  = cache_dir

    # ------------------------------------------------------------------
    def load_phrasebank(self) -> DatasetDict:
        """Return {train, val, test} for Financial PhraseBank (sentences_allagree).

        Downloads the raw zip directly via ``huggingface_hub`` — bypasses
        ``load_dataset()`` and therefore the loading-script ban in datasets>=3.0.
        """
        texts, labels = _load_phrasebank_via_hf_hub("sentences_allagree")
        return self._split(texts, labels, source_tag="phrasebank")

    def load_twitter(self) -> DatasetDict:
        """Return {train, val, test} for Twitter Financial News Sentiment.

        Bypasses ``load_dataset()`` to avoid pyarrow / datasets-cache
        segfaults observed on some Windows installations. Downloads the
        raw Parquet files directly via ``huggingface_hub``.
        """
        from huggingface_hub import hf_hub_download, list_repo_files
        import pandas as pd

        token = _hf_token()
        repo_id = "zeroshot/twitter-financial-news-sentiment"
        print(f"  [loader] Listing files in {repo_id}...", flush=True)
        try:
            all_files = list(list_repo_files(repo_id, repo_type="dataset", token=token))
        except Exception as exc:
            raise RuntimeError(f"Cannot list files in {repo_id}: {exc}") from exc
        print(f"  [loader] Twitter repo files: {all_files}", flush=True)

        # Find train + validation Parquet/CSV files
        data_files = [f for f in all_files if f.endswith((".csv", ".parquet"))]
        if not data_files:
            raise RuntimeError(f"No CSV/Parquet found in {repo_id}: {all_files}")

        # Native:  0=Bearish, 1=Bullish, 2=Neutral
        # Unified: 0=negative, 1=neutral, 2=positive
        label_remap = {0: 0, 1: 2, 2: 1}
        all_texts:  List[str] = []
        all_labels: List[int] = []

        for fname in data_files:
            local_path = hf_hub_download(
                repo_id=repo_id, filename=fname,
                repo_type="dataset", token=token,
            )
            if fname.endswith(".parquet"):
                df = pd.read_parquet(local_path)
            else:
                df = pd.read_csv(local_path)
            print(f"  [loader] Loaded {fname}: {len(df)} rows", flush=True)
            # Column names may vary; normalise
            text_col  = "text"  if "text"  in df.columns else df.columns[0]
            label_col = "label" if "label" in df.columns else df.columns[-1]
            all_texts.extend(df[text_col].astype(str).tolist())
            all_labels.extend([label_remap[int(lbl)] for lbl in df[label_col]])

        print(f"  [loader] Total Twitter examples: {len(all_texts)}", flush=True)
        return self._split(all_texts, all_labels, source_tag="twitter")

    def load_all(self) -> Tuple[DatasetDict, DatasetDict]:
        return self.load_phrasebank(), self.load_twitter()

    def get_statistics(self, dataset_splits: DatasetDict) -> Dict:
        stats: Dict = {}
        for split_name, split_data in dataset_splits.items():
            labels = split_data["label"]
            unique, counts = np.unique(labels, return_counts=True)
            stats[split_name] = {
                "total": len(labels),
                "class_counts": {
                    LABEL_NAMES[int(u)]: int(c) for u, c in zip(unique, counts)
                },
                "class_ratios": {
                    LABEL_NAMES[int(u)]: float(c) / len(labels)
                    for u, c in zip(unique, counts)
                },
            }
        return stats

    # ------------------------------------------------------------------
    def _split(
        self,
        texts:      List[str],
        labels:     List[int],
        source_tag: str,
    ) -> DatasetDict:
        indices = list(range(len(texts)))

        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_ratio,
            stratify=labels,
            random_state=self.seed,
        )
        train_val_labels = [labels[i] for i in train_val_idx]
        val_frac = self.val_ratio / (1.0 - self.test_ratio)

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_frac,
            stratify=train_val_labels,
            random_state=self.seed,
        )

        def make_split(idxs: List[int]) -> Dataset:
            return Dataset.from_dict({
                "text":   [texts[i]  for i in idxs],
                "label":  [labels[i] for i in idxs],
                "source": [source_tag] * len(idxs),
            })

        return DatasetDict({
            "train": make_split(train_idx),
            "val":   make_split(val_idx),
            "test":  make_split(test_idx),
        })
