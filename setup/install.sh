#!/usr/bin/env bash
# setup/install.sh - DGX Spark Cogni-Brain (GLM-5.3-Flash) benchmark preflight
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-unsloth/GLM-5.3-Flash-GGUF}"
QUANT="${QUANT:-UD-IQ3_XXS}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/models/$MODEL}"

echo "================================================================="
echo "=== DGX Spark Cogni-Brain (GLM-5.3-Flash) Preflight ==="
echo "================================================================="
echo "  Target Model:      $MODEL"
echo "  Target Quant:      $QUANT"
echo "  Model Directory:   $MODEL_DIR"
echo

echo "[1/6] Checking Docker & Container Toolkit..."
if command -v docker >/dev/null 2>&1; then
  docker version --format '  Docker Server Version: {{.Server.Version}}' || true
  if docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
    echo "  NVIDIA Container Toolkit: OK"
  else
    echo "  NOTE: GPU container pass-through verification skipped or requires permissions."
  fi
else
  echo "  ERROR: Docker not found. Install Docker and NVIDIA Container Toolkit on DGX Spark."
  exit 1
fi

echo
echo "[2/6] Checking NVIDIA GPU (Grace-Blackwell GB10)..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
else
  echo "  WARNING: nvidia-smi not found. Verify DGX Spark GPU driver on host."
fi

echo
echo "[3/6] Checking Unified Memory & Swap Status..."
if command -v free >/dev/null 2>&1; then
  TOTAL_MB=$(free -m | awk '/^Mem:/ {print $2}')
  AVAIL_MB=$(free -m | awk '/^Mem:/ {print $7}')
  echo "  Total Unified Memory:  ${TOTAL_MB} MB"
  echo "  Available Memory:      ${AVAIL_MB} MB"
fi
if command -v swapon >/dev/null 2>&1; then
  SWAP_ACTIVE=$(swapon --show | wc -l || echo 0)
  if [ "$SWAP_ACTIVE" -le 1 ]; then
    echo "  Swap: DISABLED (default on DGX Spark)"
    echo "  NOTE: With swap disabled, memory allocations exceeding physical RAM will trigger OOM."
    echo "        UD-IQ3_XXS requires ~120GB static weights. UD-IQ2_XXS (~102GB) provides safe fallback."
  else
    echo "  Swap: Active"
  fi
fi

echo
echo "[4/6] Checking uv / uvx..."
if ! command -v uv >/dev/null 2>&1; then
  echo "  NOTE: uv is not installed. Installing via astral.sh..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
echo "  uv version: $(uv --version 2>/dev/null || echo 'available after shell reload')"

echo
echo "[5/6] Checking Hugging Face authentication..."
if command -v uvx >/dev/null 2>&1 && uvx hf auth whoami >/dev/null 2>&1; then
  echo "  Hugging Face authenticated: OK"
else
  echo "  NOTE: Unauthenticated. unsloth/GLM-5.3-Flash-GGUF is public, but login is recommended:"
  echo "    uvx hf auth login"
fi

echo
echo "[6/6] Checking downloaded model files..."
SHARDS=0
EXPECTED_SHARDS=0
if [ -d "$MODEL_DIR" ]; then
  SHARDS=$(find "$MODEL_DIR" -type f -name "*${QUANT}*.gguf" 2>/dev/null | wc -l | tr -d '[:space:]')
  TOTAL_FROM_NAME=$(find "$MODEL_DIR" -type f -name "*${QUANT}*.gguf" 2>/dev/null | grep -o -E -- '-of-[0-9]+' | head -n 1 | sed 's/-of-0*//' || true)
  if [ -n "$TOTAL_FROM_NAME" ]; then
    EXPECTED_SHARDS="$TOTAL_FROM_NAME"
  fi
fi
SHARDS="${SHARDS:-0}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-0}"

if [ "$SHARDS" -gt 0 ] && [ "$EXPECTED_SHARDS" -gt 0 ] && [ "$SHARDS" -ge "$EXPECTED_SHARDS" ]; then
  echo "  Found all $SHARDS/$EXPECTED_SHARDS GGUF shards under $MODEL_DIR"
elif [ "$SHARDS" -gt 0 ]; then
  TOTAL_DISP="${EXPECTED_SHARDS}"
  [ "$TOTAL_DISP" -eq 0 ] && TOTAL_DISP="?"
  echo "  Found partial shards ($SHARDS/$TOTAL_DISP). Run download to complete:"
  echo "    bash setup/download_model.sh"
else
  echo "  Model files not found yet. Download with:"
  echo "    bash setup/download_model.sh"
fi

echo
echo "================================================================="
echo "Preflight complete."
echo "Deployment steps on DGX Spark:"
echo "  1. bash setup/download_model.sh"
echo "  2. bash docker/build.sh"
echo "  3. bash docker/start.sh"
echo "  4. bash benchmark/smoke_test.sh"
echo "================================================================="
