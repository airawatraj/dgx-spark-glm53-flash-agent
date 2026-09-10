#!/usr/bin/env bash
# docker/start.sh — Start GLM-5.3-Flash llama-server on DGX Spark
# Serves model as Cogni-Brain in container spark-brain.
#
# Memory Architecture Notes (NVIDIA DGX Spark / Grace-Blackwell GB10):
#   - Total Unified Memory: 128 GB (LPDDR5x shared CPU + GPU).
#   - SWAP IS DISABLED on DGX Spark by default. Any allocation spike beyond 128GB
#     causes immediate OOM kill of the llama-server process.
#   - Static weights for UD-IQ3_XXS are ~120.37 GB (~94% of physical RAM).
#   - Reserving >90% or pushing .95 memory fractions will fail.
#   - KV Cache Quantization (q4_0 / q8_0) is enabled by default to save 2-4 GB.
#   - Safe fallback quant: UD-IQ2_XXS (~101.8 GB) leaves 20+ GB headroom.
#
# Environment variables for tuning:
#   CONTAINER        - Container name (default: spark-brain)
#   PORT             - Host port (default: 8000)
#   MODEL_ALIAS      - Served model name for OpenAI API (default: Cogni-Brain)
#   QUANT            - Quantization profile (default: auto-detected / UD-IQ2_XXS; optional: UD-IQ3_XXS)
#   CTX_SIZE         - Token context window (default: 32768; max: 131072 with tuned KV)
#   PARALLEL         - Concurrent request slots (default: 1)
#   CACHE_TYPE_K     - KV cache K quantization: "f16" | "q8_0" | "q4_0" (default: q4_0)
#   CACHE_TYPE_V     - KV cache V quantization: "f16" | "q8_0" | "q4_0" (default: q4_0)
#   NGL              - Number of GPU layers to offload (default: 999 = all layers)
#   BATCH_SIZE       - Max batch tokens for prompt processing (default: 512)
#   UBATCH           - Micro-batch for CUDA kernel dispatch (default: 256)
#   MTP_DRAFT        - Multi-token prediction draft steps; 0=off, 2=on (default: 0)
#   REASONING_EFFORT - "low" | "high" | "max" (default: max)
#   USE_MMAP         - 1=enable mmap (recommended with swap disabled), 0=no-mmap (default: 1)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-glm53-flash-dgx-spark:latest}"
CONTAINER="${CONTAINER:-spark-brain}"
MODEL="${MODEL:-unsloth/GLM-5.3-Flash-GGUF}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/models/$MODEL}"
PORT="${PORT:-8000}"
MODEL_ALIAS="${MODEL_ALIAS:-Cogni-Brain}"

# Detect existing download or default to UD-IQ2_XXS (safe 102GB baseline)
if [ -z "${QUANT:-}" ]; then
  if find "$MODEL_DIR" -type f -name "*UD-IQ2_XXS*.gguf" 2>/dev/null | grep -q .; then
    QUANT="UD-IQ2_XXS"
  elif find "$MODEL_DIR" -type f -name "*UD-IQ3_XXS*.gguf" 2>/dev/null | grep -q .; then
    QUANT="UD-IQ3_XXS"
  else
    QUANT="UD-IQ2_XXS"
  fi
fi

# ── Memory & Safety Tuning Defaults (DGX Spark) ──────────────────────────────
CTX_SIZE="${CTX_SIZE:-32768}"
PARALLEL="${PARALLEL:-1}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q4_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q4_0}"
NGL="${NGL:-999}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH="${UBATCH:-256}"
MTP_DRAFT="${MTP_DRAFT:-0}"
REASONING_EFFORT="${REASONING_EFFORT:-max}"
USE_MMAP="${USE_MMAP:-1}"
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================="
echo "=== Starting Cogni-Brain (GLM-5.3-Flash) on DGX Spark ==="
echo "================================================================="
echo "  Container:         $CONTAINER"
echo "  Served Model:      $MODEL_ALIAS"
echo "  Quantization:      $QUANT"
echo "  Context Size:      $CTX_SIZE tokens"
echo "  Parallel Slots:    $PARALLEL"
echo "  KV Cache Quant:    K=$CACHE_TYPE_K, V=$CACHE_TYPE_V"
echo "  GPU Offload:       -ngl $NGL (GB10 Grace-Blackwell)"
echo "  Batch / Micro:     $BATCH_SIZE / $UBATCH"
echo "  MTP Draft:         ${MTP_DRAFT:-off}"
echo "  Reasoning Effort:  $REASONING_EFFORT"
echo "  MMAP:              $([ "$USE_MMAP" -eq 1 ] && echo 'enabled (safe)' || echo 'disabled (--no-mmap)')"
echo

# Memory preflight check
if command -v free >/dev/null 2>&1; then
  AVAIL_MB=$(free -m | awk '/^Mem:/ {print $7}')
  echo "  Host Available RAM: ${AVAIL_MB} MB"
  if [ "$QUANT" = "UD-IQ3_XXS" ] && [ "$AVAIL_MB" -lt 122000 ]; then
    echo "  WARNING: Available memory (${AVAIL_MB} MB) is tight for UD-IQ3_XXS (~120GB)."
    echo "  Note: Swap is disabled on DGX Spark. If OOM occurs, switch to:"
    echo "    QUANT=UD-IQ2_XXS bash docker/start.sh"
    echo
  fi
  if [ "$PARALLEL" -gt 1 ] && [ "$QUANT" = "UD-IQ3_XXS" ]; then
    echo "  WARNING: Running PARALLEL=$PARALLEL with UD-IQ3_XXS exceeds safe 128GB headroom."
    echo "  For multi-slot parallel serving, use:"
    echo "    QUANT=UD-IQ2_XXS PARALLEL=$PARALLEL bash docker/start.sh"
    echo
  fi
fi

# Locate GGUF shard 1
MODEL_FILE="$(find "$MODEL_DIR" -type f \( -name "*${QUANT}*00001*.gguf" -o -name "*${QUANT}*.gguf" \) 2>/dev/null | sort | head -1)"
if [ -z "$MODEL_FILE" ]; then
  echo "ERROR: Could not find $QUANT GGUF under $MODEL_DIR"
  echo "Run: bash setup/download_model.sh"
  exit 1
fi
MODEL_REL="${MODEL_FILE#$MODEL_DIR/}"
echo "  Found model file:  $MODEL_REL"

# Verify image
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Docker image $IMAGE not found. Building..."
  bash docker/build.sh
fi

# Idempotent cleanup of existing container(s)
echo "Cleaning up any existing container ($CONTAINER, glm53-flash)..."
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker rm -f glm53-flash >/dev/null 2>&1 || true

EXTRA_ARGS=()
if [ "${MTP_DRAFT}" -gt 0 ]; then
  EXTRA_ARGS+=(--draft-max "${MTP_DRAFT}")
fi

if [ "${USE_MMAP}" -eq 0 ]; then
  EXTRA_ARGS+=(--no-mmap)
fi

echo "Starting $CONTAINER on port $PORT..."
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
  --n-gpu-layers "$NGL" \
  --ctx-size "$CTX_SIZE" \
  --parallel "$PARALLEL" \
  --cache-type-k "$CACHE_TYPE_K" \
  --cache-type-v "$CACHE_TYPE_V" \
  --batch-size "$BATCH_SIZE" \
  --ubatch-size "$UBATCH" \
  --no-cache-prompt \
  --cache-reuse 0 \
  --no-context-shift \
  --slot-prompt-similarity 1.1 \
  --temp 1.0 \
  --top-p 0.95 \
  --chat-template-kwargs "{\"reasoning_effort\":\"${REASONING_EFFORT}\"}" \
  "${EXTRA_ARGS[@]}"

echo
echo "================================================================="
echo "✓ Container $CONTAINER started successfully."
echo "  Endpoint:     http://localhost:$PORT/v1"
echo "  Served Model: $MODEL_ALIAS"
echo
echo "Check status:   bash docker/status.sh"
echo "Follow logs:    docker logs -f $CONTAINER"
echo "Run smoke test: bash benchmark/smoke_test.sh localhost:$PORT"
echo "================================================================="
