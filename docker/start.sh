#!/usr/bin/env bash
# docker/start.sh — Start GLM-5.3-Flash llama-server on DGX Spark
# Based on Unsloth docs: https://unsloth.ai/docs/models/glm-5.3-flash
#
# OOM-prevention knobs (override via env vars):
#   CTX_SIZE        - token context window (default 16384; NOT 131072 on first boot)
#   PARALLEL        - concurrent request slots (default 1; each slot costs ~KV cache)
#   BATCH_SIZE      - max batch tokens for prompt processing (default 512)
#   UBATCH          - micro-batch for CUDA kernel dispatch (default 256)
#   MTP_DRAFT       - multi-token prediction draft steps; 0=off, 2=on (default 0)
#   REASONING_EFFORT - "low" | "medium" | "high" | "max" (default "high")
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-glm53-flash-dgx-spark:latest}"
CONTAINER="${CONTAINER:-glm53-flash}"
MODEL="${MODEL:-unsloth/GLM-5.3-Flash-GGUF}"
QUANT="${QUANT:-UD-IQ3_XXS}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/models/$MODEL}"
PORT="${PORT:-8000}"
MODEL_ALIAS="${MODEL_ALIAS:-GLM-5.3-Flash}"

# ── OOM-prevention defaults ──────────────────────────────────────────────────
# Start conservatively. Context 131072 will OOM on first boot with 120GB model.
# Verify memory headroom with: bash docker/status.sh   (check nvidia-smi line)
CTX_SIZE="${CTX_SIZE:-16384}"
PARALLEL="${PARALLEL:-1}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH="${UBATCH:-256}"
MTP_DRAFT="${MTP_DRAFT:-0}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
# ─────────────────────────────────────────────────────────────────────────────

MODEL_FILE="$(find "$MODEL_DIR" -type f \( -name "*${QUANT}*00001*.gguf" -o -name "*${QUANT}*.gguf" \) 2>/dev/null | sort | head -1)"
if [ -z "$MODEL_FILE" ]; then
  echo "ERROR: Could not find $QUANT GGUF under $MODEL_DIR"
  echo "Run: bash setup/download_model.sh"
  exit 1
fi
MODEL_REL="${MODEL_FILE#$MODEL_DIR/}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Docker image $IMAGE not found. Building..."
  bash docker/build.sh
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

EXTRA_ARGS=()
if [ "${MTP_DRAFT}" -gt 0 ]; then
  EXTRA_ARGS+=(--draft-max "${MTP_DRAFT}")
fi

docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "$PORT:8000" \
  -v "$MODEL_DIR:/models:ro" \
  "$IMAGE" \
  --model "/models/$MODEL_REL" \
  --alias "$MODEL_ALIAS" \
  --host 0.0.0.0 \
  --port 8000 \
  --ctx-size "$CTX_SIZE" \
  --parallel "$PARALLEL" \
  --batch-size "$BATCH_SIZE" \
  --ubatch-size "$UBATCH" \
  --no-mmap \
  --temp 1.0 \
  --top-p 0.95 \
  --chat-template-kwargs "{\"reasoning_effort\":\"${REASONING_EFFORT}\"}" \
  "${EXTRA_ARGS[@]}"

echo "================================================================="
echo "Started $CONTAINER on http://localhost:$PORT"
echo "  Model:            $MODEL_ALIAS ($QUANT)"
echo "  CTX:              $CTX_SIZE tokens"
echo "  Parallel:         $PARALLEL slot(s)"
echo "  Batch:            $BATCH_SIZE / ubatch $UBATCH"
echo "  MTP draft:        ${MTP_DRAFT:-off}"
echo "  Reasoning effort: $REASONING_EFFORT"
echo ""
echo "Check memory:  bash docker/status.sh"
echo "Logs:          docker logs -f $CONTAINER"
echo "================================================================="
