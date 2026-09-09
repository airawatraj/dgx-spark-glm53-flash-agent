#!/usr/bin/env bash
# setup/download_model.sh - Download GLM-5.3-Flash split GGUF files for DGX Spark
#
# Available Quantizations & Memory Footprint on DGX Spark (128GB Unified Memory):
#   - UD-IQ3_XXS: ~120.4 GB (3-bit; ~94% RAM; requires strict KV cache tuning)
#   - UD-Q2_K_XL: ~108.7 GB (2-bit high-quality; ~85% RAM; leaves ~19 GB headroom)
#   - UD-IQ2_XXS: ~101.8 GB (2-bit; ~79% RAM; recommended safe baseline with swap disabled)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-unsloth/GLM-5.3-Flash-GGUF}"
QUANT="${QUANT:-UD-IQ3_XXS}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/models/$MODEL}"

echo "================================================================="
echo "=== Downloading GLM-5.3-Flash GGUF Shards for DGX Spark ==="
echo "================================================================="
echo "  Hugging Face Repo: $MODEL"
echo "  Quantization:      $QUANT"
echo "  Target Directory:  $MODEL_DIR"
echo

mkdir -p "$MODEL_DIR"

EXISTING_SHARDS=$(find "$MODEL_DIR" -type f -name "*${QUANT}*.gguf" 2>/dev/null | wc -l | tr -d '[:space:]')
EXISTING_SHARDS="${EXISTING_SHARDS:-0}"
EXPECTED_SHARDS=$(find "$MODEL_DIR" -type f -name "*${QUANT}*.gguf" 2>/dev/null | grep -o -E -- '-of-[0-9]+' | head -n 1 | sed 's/-of-0*//' || true)
EXPECTED_SHARDS="${EXPECTED_SHARDS:-0}"

if [ "$EXISTING_SHARDS" -gt 0 ] && [ "$EXPECTED_SHARDS" -gt 0 ] && [ "$EXISTING_SHARDS" -ge "$EXPECTED_SHARDS" ]; then
  echo "Found all $EXISTING_SHARDS/$EXPECTED_SHARDS existing $QUANT GGUF shards under $MODEL_DIR:"
  find "$MODEL_DIR" -type f -name "*${QUANT}*.gguf" | sort | sed 's/^/  /'
  echo
  echo "Download already complete (idempotent exit)."
  exit 0
elif [ "$EXISTING_SHARDS" -gt 0 ]; then
  TOTAL_DISP="${EXPECTED_SHARDS}"
  [ "$TOTAL_DISP" -eq 0 ] && TOTAL_DISP="?"
  echo "Found partial download ($EXISTING_SHARDS/$TOTAL_DISP shard(s)). Resuming download..."
fi

echo "Downloading $QUANT shards from $MODEL..."
if command -v hf >/dev/null 2>&1; then
  hf download "$MODEL" --local-dir "$MODEL_DIR" --include "*${QUANT}*.gguf"
elif command -v uvx >/dev/null 2>&1; then
  uvx hf download "$MODEL" --local-dir "$MODEL_DIR" --include "*${QUANT}*.gguf"
else
  echo "ERROR: Install huggingface_hub CLI or uvx first."
  echo "Try: uv tool install huggingface_hub"
  exit 1
fi

echo
echo "Download verified:"
find "$MODEL_DIR" -type f -name "*${QUANT}*.gguf" | sort | sed 's/^/  /'
echo "================================================================="
