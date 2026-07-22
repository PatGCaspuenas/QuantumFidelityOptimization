#!/usr/bin/env bash
# Python environment setup for figure generation (scripts/plots.ipynb).
# Usage: bash setup_python.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
KERNEL_NAME="qfo-venv"
KERNEL_DISPLAY_NAME="Python 3 (QuantumFidelityOptimization)"

echo "========================================"
echo "  Python figure-generation environment setup"
echo "========================================"

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3 and re-run." >&2
    exit 1
fi

echo ""
echo "[1/3] Creating virtual environment at $VENV_DIR ..."
if [ -d "$VENV_DIR" ]; then
    echo "  Already exists, reusing."
else
    python3 -m venv "$VENV_DIR"
fi

echo ""
echo "[2/3] Installing pinned dependencies from requirements.txt ..."
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -q -r "$REPO_DIR/requirements.txt"

echo ""
echo "[3/3] Registering Jupyter kernel '$KERNEL_NAME' ..."
"$VENV_DIR/bin/python" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"

echo ""
echo "Python setup complete."
echo "  Activate with: source $VENV_DIR/bin/activate"
echo "  Open scripts/plots.ipynb in Jupyter and select the '$KERNEL_DISPLAY_NAME' kernel"
echo "  Or launch directly: $VENV_DIR/bin/jupyter notebook scripts/plots.ipynb"
