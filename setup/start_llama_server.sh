#!/usr/bin/env bash
# setup/start_llama_server.sh - Start a local OpenAI-compatible llama-server
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-unsloth/GLM-5.3-Flash-GGUF}"
QUANT="${QUANT:-UD-IQ3_XXS}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
CTX_SIZE="${CTX_SIZE:-131072}"
MODEL_ALIAS="${MODEL_ALIAS:-GLM-5.3-Flash}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/models/$MODEL}"
LLAMA_SERVER="${LLAMA_SERVER:-$REPO_DIR/llama.cpp/build/bin/llama-server}"

if [ ! -x "$LLAMA_SERVER" ]; then
  echo "ERROR: llama-server not found at $LLAMA_SERVER"
  echo "Run: bash setup/build_llama_cpp.sh"
  exit 1
fi

MODEL_FILE="$(find "$MODEL_DIR" -type f \( -name "*${QUANT}*00001*.gguf" -o -name "*${QUANT}*.gguf" \) 2>/dev/null | sort | head -1)"
if [ -z "$MODEL_FILE" ]; then
  echo "ERROR: Could not find $QUANT GGUF under $MODEL_DIR"
  echo "Run: bash setup/download_model.sh"
  exit 1
fi

echo "Starting GLM-5.3-Flash llama-server"
echo "  Model file: $MODEL_FILE"
echo "  Alias:      $MODEL_ALIAS"
echo "  Host/port:  $HOST:$PORT"
echo "  Context:    $CTX_SIZE"
echo

exec "$LLAMA_SERVER" \
  --model "$MODEL_FILE" \
  --alias "$MODEL_ALIAS" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx-size "$CTX_SIZE" \
  --temp 1.0 \
  --top-p 0.95
