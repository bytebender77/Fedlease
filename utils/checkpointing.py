from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
) -> str:
    """Save training state to disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)

    meta = {k: str(v) for k, v in state.items() if not isinstance(v, (torch.Tensor, dict, list))}
    meta_path = path.replace(".pt", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return path


def load_checkpoint(
    path: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load checkpoint from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    map_location = device or torch.device("cpu")
    state = torch.load(path, map_location=map_location)
    return state


def save_model_state(
    model: torch.nn.Module,
    path: str,
    extra: Optional[Dict] = None,
) -> None:
    """Save model state dict with optional extra metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, Any] = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_model_state(
    model: torch.nn.Module,
    path: str,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> torch.nn.Module:
    """Load model weights from checkpoint."""
    state = load_checkpoint(path, device)
    model_state = state.get("model_state_dict", state)
    model.load_state_dict(model_state, strict=strict)
    return model
