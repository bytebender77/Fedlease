#!/usr/bin/env bash
# FedLEASE one-command setup for macOS / Linux / WSL.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh

set -e

echo "=========================================="
echo "  FedLEASE setup (macOS / Linux / WSL)"
echo "=========================================="

# 1. Python version check
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi
PY_VER=$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[1/4] Using Python $PY_VER ($PY)"

# 2. Virtual env
if [ ! -d ".venv" ]; then
    echo "[2/4] Creating virtual environment .venv/"
    "$PY" -m venv .venv
else
    echo "[2/4] Reusing existing .venv/"
fi

# Activate venv for the rest of this script
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. Install dependencies
echo "[3/4] Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip wheel setuptools >/dev/null

# Detect CUDA on Linux/WSL; on macOS install the standard wheel (uses MPS or CPU)
if [ "$(uname)" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "    CUDA GPU detected — installing torch with CUDA 12.1 wheels"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "    No CUDA detected — installing default torch wheels"
    pip install torch torchvision
fi

pip install -r requirements.txt

# 4. .env
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "[4/4] Created .env from .env.example — please edit it and add your HF_TOKEN"
else
    echo "[4/4] .env already exists (or no .env.example) — skipping"
fi

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env and set HF_TOKEN=hf_xxxxxxxx"
echo "  2. Activate the venv:   source .venv/bin/activate"
echo "  3. Run the experiment:  python run.py --preset full"
echo ""
