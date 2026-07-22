#!/usr/bin/env bash
# Python environment setup for figure generation (scripts/plots.ipynb).
# Usage: bash setup_python.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
KERNEL_NAME="qfo-venv"
KERNEL_DISPLAY_NAME="Python 3 (QuantumFidelityOptimization)"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=12   # numpy/scipy in requirements.txt require >=3.12

echo "========================================"
echo "  Python figure-generation environment setup"
echo "========================================"

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3 and re-run." >&2
    exit 1
fi

echo ""
echo "[1/4] Checking Python version (need >= $MIN_PYTHON_MAJOR.$MIN_PYTHON_MINOR) ..."
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= ($MIN_PYTHON_MAJOR, $MIN_PYTHON_MINOR) else 1)"; then
    echo "ERROR: python3 is $(python3 --version 2>&1 | awk '{print $2}'), but requirements.txt (numpy/scipy) needs >= $MIN_PYTHON_MAJOR.$MIN_PYTHON_MINOR." >&2
    echo "  Install a newer Python (e.g. via pyenv, deadsnakes PPA, or python.org) and re-run." >&2
    exit 1
fi
echo "  Found: $(python3 --version)"

echo ""
echo "[2/4] Checking for LaTeX (needed by matplotlib's text.usetex=True in scripts/plots.ipynb) ..."
if command -v pdflatex &>/dev/null && command -v dvipng &>/dev/null; then
    echo "  Found: $(command -v pdflatex), $(command -v dvipng)"
else
    echo "  WARNING: pdflatex and/or dvipng not found on PATH."
    echo "  scripts/plots.ipynb needs a LaTeX distribution to render figure text and will fail without one. Install:"
    echo "    Debian/Ubuntu: sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super"
    echo "    macOS (Homebrew): brew install --cask basictex && sudo tlmgr update --self && sudo tlmgr install dvipng type1cm cm-super"
    echo "    Or any TeX Live / MiKTeX install that provides pdflatex + dvipng on PATH."
fi

echo ""
echo "[3/4] Creating virtual environment at $VENV_DIR ..."
if [ -d "$VENV_DIR" ]; then
    echo "  Already exists, reusing."
else
    python3 -m venv "$VENV_DIR"
fi

echo ""
echo "  Installing pinned dependencies from requirements.txt ..."
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -q -r "$REPO_DIR/requirements.txt"

echo ""
echo "[4/4] Registering Jupyter kernel '$KERNEL_NAME' ..."
"$VENV_DIR/bin/python" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"

echo ""
echo "Python setup complete."
echo "  Activate with: source $VENV_DIR/bin/activate"
echo "  Open scripts/plots.ipynb in Jupyter and select the '$KERNEL_DISPLAY_NAME' kernel"
echo "  Or launch directly: $VENV_DIR/bin/jupyter notebook scripts/plots.ipynb"
