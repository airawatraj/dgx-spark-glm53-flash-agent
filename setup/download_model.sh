#!/usr/bin/env bash
# setup/download_model.sh - Download GLM-5.3-Flash split GGUF files
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-unsloth/GLM-5.3-Flash-GGUF}"
QUANT="${QUANT:-UD-IQ3_XXS}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/models/$MODEL}"

echo "================================================================="
echo "=== Downloading GLM-5.3-Flash GGUF ==="
echo "================================================================="
echo "  Model ID:   $MODEL"
echo "  Quant:      $QUANT"
echo "  Target dir: $MODEL_DIR"
echo "  Expected:   ~120GB for UD-IQ3_XXS split GGUF"
echo

mkdir -p "$MODEL_DIR"

if find "$MODEL_DIR" -type f -name "*${QUANT}*.gguf" 2>/dev/null | grep -q .; then
  echo "Found existing $QUANT GGUF files:"
  find "$MODEL_DIR" -type f -name "*${QUANT}*.gguf" | sed 's/^/  /'
  exit 0
fi

if command -v hf >/dev/null 2>&1; then
  hf download "$MODEL" --local-dir "$MODEL_DIR" --include "*${QUANT}*"
elif command -v uvx >/dev/null 2>&1; then
  uvx hf download "$MODEL" --local-dir "$MODEL_DIR" --include "*${QUANT}*"
else
  echo "ERROR: Install huggingface_hub CLI or uvx first."
  echo "Try: python3 -m pip install -U 'huggingface_hub[cli]'"
  exit 1
fi

echo
echo "Download complete:"
find "$MODEL_DIR" -type f -name "*${QUANT}*.gguf" | sed 's/^/  /'

