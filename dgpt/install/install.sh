#!/bin/sh
# One-line installer for a dgpt worker node (macOS / Linux).
#   curl -fsSL https://<where you host this>/install.sh | sh
#   DGPT_SRC=/path/to/dgpt-0.1.0-py3-none-any.whl sh install.sh     # from a wheel you were given
# Installs uv (which brings its own Python), then the worker into an isolated tool environment.
set -e
SRC="${DGPT_SRC:-dgpt @ git+https://github.com/YOUR_ORG/distribute}"   # replace with your repo or wheel URL

if ! command -v uv >/dev/null 2>&1; then
  echo "[dgpt] installing uv (Python package manager; also fetches Python if needed)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Linux without an NVIDIA GPU: use the CPU-only torch index and avoid a 2 GB CUDA download.
INDEX=""
if [ "$(uname -s)" = "Linux" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
  INDEX="--index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match"
fi

echo "[dgpt] installing worker from: $SRC"
# shellcheck disable=SC2086
uv tool install --force --python 3.12 $INDEX "$SRC"

echo
echo "[dgpt] installed. Join a pool with:"
echo "    dgpt-worker --coordinator http://HOST:8000            # add --token XYZ if the pool requires one"
echo "  (device is auto-detected: cuda > mps > cpu; add --device cpu or --threads N to limit it)"
command -v dgpt-worker >/dev/null 2>&1 || echo "[dgpt] open a new shell or run: export PATH=\"\$HOME/.local/bin:\$PATH\""
