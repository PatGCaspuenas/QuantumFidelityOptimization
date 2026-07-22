#!/usr/bin/env bash
# Automated setup for QuantumFidelityOptimization
# Usage: bash setup.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JULIA_VERSION="1.11.6"

echo "========================================"
echo "  QuantumFidelityOptimization Setup"
echo "  Julia $JULIA_VERSION"
echo "========================================"

# ── 1. Install juliaup if not present ────────────────────────────────────────
echo ""
echo "[1/4] Checking juliaup..."
if ! command -v juliaup &>/dev/null; then
    echo "  juliaup not found — installing..."
    curl -fsSL https://install.julialang.org | sh -s -- --yes
    # Add to PATH for this session (installer sets up shell profile, but not the current script)
    export PATH="$HOME/.juliaup/bin:$PATH"
    echo "  juliaup installed."
else
    echo "  juliaup already installed: $(juliaup --version 2>/dev/null || echo 'unknown version')"
    # Ensure juliaup bin is in PATH
    export PATH="$HOME/.juliaup/bin:$PATH"
fi

# ── 2. Install and default Julia 1.11.6 ──────────────────────────────────────
echo ""
echo "[2/4] Setting up Julia $JULIA_VERSION..."

if juliaup list 2>/dev/null | grep -q "^$JULIA_VERSION"; then
    echo "  Julia $JULIA_VERSION already installed."
else
    echo "  Installing Julia $JULIA_VERSION via juliaup..."
    juliaup add "$JULIA_VERSION"
fi

juliaup default "$JULIA_VERSION"

# Locate julia binary
if command -v julia &>/dev/null; then
    JULIA_BIN="julia"
elif [ -f "$HOME/.juliaup/bin/julia" ]; then
    JULIA_BIN="$HOME/.juliaup/bin/julia"
else
    echo "ERROR: julia binary not found after juliaup setup. Try opening a new terminal and re-running."
    exit 1
fi

echo "  Using: $($JULIA_BIN --version)"

# ── 3. Run Julia package setup ───────────────────────────────────────────────
echo ""
echo "[3/4] Running Julia package setup (this may take several minutes)..."
"$JULIA_BIN" --project="$REPO_DIR" "$REPO_DIR/setup_pkg.jl"

# ── 4. Python environment for figure generation (scripts/plots.ipynb) ────────
echo ""
echo "[4/4] Setting up Python environment for figure generation..."
bash "$REPO_DIR/setup_python.sh"
