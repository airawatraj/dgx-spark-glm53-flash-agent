#!/usr/bin/env bash
# setup/install.sh - DGX Spark GLM-5.3-Flash benchmark preflight
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-unsloth/GLM-5.3-Flash-GGUF}"
QUANT="${QUANT:-UD-IQ3_XXS}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/models/$MODEL}"

echo "================================================================="
echo "=== DGX Spark GLM-5.3-Flash Benchmark Preflight ==="
echo "================================================================="
echo "  Model:      $MODEL"
echo "  Quant:      $QUANT"
echo "  Model dir:  $MODEL_DIR"
echo

echo "[1/5] Checking Docker..."
if command -v docker >/dev/null 2>&1; then
  docker version --format '  Docker Server Version: {{.Server.Version}}' || true
else
  echo "  WARNING: Docker not found. Local llama.cpp launch can still work."
fi

echo
echo "[2/5] Checking NVIDIA GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
else
  echo "  WARNING: nvidia-smi not found. Verify DGX Spark GPU driver on host."
fi

echo
echo "[3/5] Checking uv / uvx..."
if ! command -v uv >/dev/null 2>&1; then
  echo "  NOTE: uv is not installed. Installing via astral.sh..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
echo "  uv version: $(uv --version 2>/dev/null || echo 'available after shell reload')"

echo
echo "[4/5] Checking Hugging Face authentication..."
if command -v uvx >/dev/null 2>&1 && uvx hf auth whoami >/dev/null 2>&1; then
  echo "  Hugging Face authenticated: OK"
else
  echo "  NOTE: Not authenticated or uvx hf unavailable. If needed, run:"
  echo "    uvx hf auth login"
fi

echo
echo "[5/5] Checking model files..."
if find "$MODEL_DIR" -type f -name "*${QUANT}*.gguf" 2>/dev/null | grep -q .; then
  echo "  Found $QUANT GGUF files under $MODEL_DIR"
else
  echo "  Model files not found yet. Download with:"
  echo "    bash setup/download_model.sh"
fi

echo
echo "================================================================="
echo "Preflight complete."
echo "Next:"
echo "  1. bash setup/download_model.sh"
echo "  2. bash setup/build_llama_cpp.sh"
echo "  3. bash setup/start_llama_server.sh"
echo "================================================================="

