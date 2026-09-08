#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-docs/runtime_snapshot_$(date +%Y%m%d_%H%M%S).txt}"
CONTAINER="${CONTAINER:-glm53-flash}"

{
  echo "Date: $(date -Is)"
  echo
  echo "== git =="
  git status --short || true
  git -C llama.cpp rev-parse HEAD 2>/dev/null || true
  echo
  echo "== nvidia-smi =="
  nvidia-smi 2>/dev/null || true
  echo
  echo "== docker =="
  docker ps --filter "name=$CONTAINER" 2>/dev/null || true
  echo
  echo "== logs =="
  docker logs --tail 120 "$CONTAINER" 2>/dev/null || true
} | tee "$OUT"

echo "Wrote $OUT"

