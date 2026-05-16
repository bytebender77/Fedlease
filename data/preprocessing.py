"""Tokenisation and PyTorch Dataset wrappers for financial sentiment data."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from .dataset_loader import LABEL_NAMES


class FinancialSentimentDataset(Dataset):
    """PyTorch Dataset wrapping tokenised financial text."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get(
                "token_type_ids", torch.zeros(self.max_length, dtype=torch.long)
            ).squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class FinancialPreprocessor:
    """
    Tokenises datasets and returns DataLoaders ready for training/eval.

    Uses ProsusAI/finbert tokenizer with max_length=128, truncation, and
    dynamic collation via padding.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        max_length: int = 128,
        seed: int = 42,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.seed = seed
        self.label_names = LABEL_NAMES

    def make_dataset(
        self,
        texts: List[str],
        labels: List[int],
    ) -> FinancialSentimentDataset:
        return FinancialSentimentDataset(texts, labels, self.tokenizer, self.max_length)

    def make_dataloader(
        self,
        texts: List[str],
        labels: List[int],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        from utils.seed import seed_worker
        import numpy as np

        dataset = self.make_dataset(texts, labels)
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=seed_worker if num_workers > 0 else None,
            generator=generator,
            pin_memory=True,
        )

    def prepare_client_loaders(
        self,
        client_data: Dict[str, List],
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> Dict[str, DataLoader]:
        """
        Create train/val DataLoaders from a client's data dict.

        client_data must have keys: 'train_texts', 'train_labels',
                                    'val_texts', 'val_labels'
        """
        loaders: Dict[str, DataLoader] = {}

        if "train_texts" in client_data:
            loaders["train"] = self.make_dataloader(
                client_data["train_texts"],
                client_data["train_labels"],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )

        if "val_texts" in client_data:
            loaders["val"] = self.make_dataloader(
                client_data["val_texts"],
                client_data["val_labels"],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        return loaders

    @property
    def tokenizer_obj(self) -> AutoTokenizer:
        return self.tokenizer

    def get_token_length_stats(self, texts: List[str]) -> Dict[str, float]:
        """Compute token length statistics for a list of texts."""
        import numpy as np

        lengths = []
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            lengths.append(len(ids))
        lengths = np.array(lengths)
        return {
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "max": float(np.max(lengths)),
            "min": float(np.min(lengths)),
            "p95": float(np.percentile(lengths, 95)),
            "truncated_pct": float(np.mean(lengths > self.max_length) * 100),
        }
