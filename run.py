#!/usr/bin/env python3
"""FedLEASE — one-command entrypoint.

Usage (after `pip install -r requirements.txt`):

    python run.py                       # full preset, default config
    python run.py --preset quick        # 10 rounds, ~10 min smoke test
    python run.py --preset medium       # 15 rounds, ~25 min
    python run.py --preset full         # 25 rounds, ~40 min (paper run)
    python run.py --preset debug        # 3 rounds, ~3 min
    python run.py --no-cuda             # force CPU
    python run.py --output_dir my_run   # custom output directory

This script:
  1. Loads HF_TOKEN from .env (if present) so the dataset download works
  2. Picks the right device automatically (CUDA / Apple MPS / CPU)
  3. Applies a preset to the YAML config in-memory (no file edits)
  4. Forwards everything else to experiments/run_fedlease.py
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)


PRESETS = {
    "debug":  dict(num_rounds=3,  local_epochs=1, warmup_epochs=1, batch_size=16, max_experts=4),
    "quick":  dict(num_rounds=10, local_epochs=1, warmup_epochs=2, batch_size=32, max_experts=6),
    "medium": dict(num_rounds=15, local_epochs=2, warmup_epochs=3, batch_size=32, max_experts=8),
    "full":   dict(num_rounds=25, local_epochs=2, warmup_epochs=3, batch_size=32, max_experts=8),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FedLEASE one-command runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="full",
        help="Training scale preset (default: full)",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "configs", "finbert_fedlease.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(REPO_ROOT, "outputs", "headline_run"),
        help="Directory to write outputs (created if missing)",
    )
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument(
        "--partition",
        choices=["heterogeneous", "iid", "non_iid"],
        default=None,
        help="Override partition strategy",
    )
    args, unknown = parser.parse_known_args()

    P = PRESETS[args.preset]
    print(f"[run.py] Preset: {args.preset}")
    print(f"[run.py]   rounds={P['num_rounds']} local_epochs={P['local_epochs']} "
          f"warmup={P['warmup_epochs']} batch={P['batch_size']} max_experts={P['max_experts']}")
    print(f"[run.py] Output: {args.output_dir}")

    # Forward to the existing experiments/run_fedlease.py via argv mutation
    forwarded = [
        sys.argv[0],
        "--config", args.config,
        "--output_dir", args.output_dir,
        "--num_rounds", str(P["num_rounds"]),
        "--local_epochs", str(P["local_epochs"]),
        "--warmup_epochs", str(P["warmup_epochs"]),
        "--batch_size", str(P["batch_size"]),
        "--max_experts", str(P["max_experts"]),
    ]
    if args.no_cuda:
        forwarded.append("--no_cuda")
    if args.seed is not None:
        forwarded.extend(["--seed", str(args.seed)])
    if args.partition is not None:
        forwarded.extend(["--partition", args.partition])
    forwarded.extend(unknown)  # pass through any extra flags

    sys.argv = forwarded

    # Import and run
    from experiments.run_fedlease import main as run_main
    run_main()


if __name__ == "__main__":
    main()
