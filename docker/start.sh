#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-glm53-flash-dgx-spark:latest}"
CONTAINER="${CONTAINER:-glm53-flash}"
MODEL="${MODEL:-unsloth/GLM-5.3-Flash-GGUF}"
QUANT="${QUANT:-UD-IQ3_XXS}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/models/$MODEL}"
PORT="${PORT:-8000}"
CTX_SIZE="${CTX_SIZE:-131072}"
MODEL_ALIAS="${MODEL_ALIAS:-GLM-5.3-Flash}"

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
  --temp 1.0 \
  --top-p 0.95

echo "Started $CONTAINER on http://localhost:$PORT"
echo "Logs: docker logs -f $CONTAINER"
