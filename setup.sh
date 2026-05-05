#!/usr/bin/env bash
# Automated setup for QuantumFidelityOptimization
# Usage: bash setup.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JULIA_VERSION="1.11.6"
VENV_DIR="$REPO_DIR/.venv"

echo "========================================"
echo "  QuantumFidelityOptimization Setup"
echo "  Julia $JULIA_VERSION"
echo "========================================"

# ── 0. Patch juliacall for Julia 1.11 compatibility ─────────────────────────
# juliapkg <=0.1.23 adds OpenSSL_jll = "=2.0.0" to the Julia env, which has no
# match in Julia 1.11 (bundled at 0.0.0). Remove the entry so Julia uses its own.
echo ""
echo "[0/3] Patching juliacall juliapkg.json for Julia 1.11..."
JULIACALL_JSON="$(find "$VENV_DIR" -path "*/juliacall/juliapkg.json" 2>/dev/null | head -1)"
if [ -n "$JULIACALL_JSON" ]; then
    python3 - "$JULIACALL_JSON" <<'EOF'
import json, sys
path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
pkgs = data.get("packages", {})
if "OpenSSL_jll" in pkgs:
    del pkgs["OpenSSL_jll"]
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"  Removed OpenSSL_jll from {path}")
else:
    print(f"  OpenSSL_jll not present in {path}, nothing to do.")
EOF
else
    echo "  juliacall not found in $VENV_DIR — skipping patch."
fi

# ── 1. Install juliaup if not present ────────────────────────────────────────
echo ""
echo "[1/3] Checking juliaup..."
if ! command -v juliaup &>/dev/null; then
    echo "  juliaup not found — installing..."
    _juliaup_installer="$(mktemp /tmp/juliaup_install.XXXXXX.sh)"
    curl -fsSL https://install.julialang.org -o "$_juliaup_installer"
    sh "$_juliaup_installer" --yes
    rm -f "$_juliaup_installer"
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
echo "[2/3] Setting up Julia $JULIA_VERSION..."

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
echo "[3/3] Running Julia package setup (this may take several minutes)..."
"$JULIA_BIN" --project="$REPO_DIR" "$REPO_DIR/setup_pkg.jl"
