#!/usr/bin/env bash
# docker/status.sh — Inspect Cogni-Brain (spark-brain) container, memory & health
set -euo pipefail

CONTAINER="${CONTAINER:-spark-brain}"
PORT="${PORT:-8000}"

echo "================================================================="
echo "=== Cogni-Brain (Container: $CONTAINER) Status ==="
echo "================================================================="
echo

# ── Container Lifecycle Check ────────────────────────────────────────────────
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  STARTED_AT=$(docker inspect "$CONTAINER" --format '{{.State.StartedAt}}')
  STATUS=$(docker inspect "$CONTAINER" --format '{{.State.Status}}')
  echo "  Container:       RUNNING ($STATUS, started $STARTED_AT)"
elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  STATUS=$(docker inspect "$CONTAINER" --format '{{.State.Status}}')
  echo "  Container:       STOPPED ($STATUS)"
else
  echo "  Container:       NOT FOUND"
fi

# ── API Health & Model Checks ────────────────────────────────────────────────
if curl -sf -m 3 "http://localhost:$PORT/health" >/dev/null 2>&1; then
  echo "  API Health:      HEALTHY (http://localhost:$PORT)"
else
  echo "  API Health:      NOT READY / NOT REACHABLE on port $PORT"
fi

MODELS_JSON="$(curl -sf -m 3 "http://localhost:$PORT/v1/models" 2>/dev/null || true)"
if [[ -n "$MODELS_JSON" ]]; then
  SERVED_MODELS="$(printf '%s\n' "$MODELS_JSON" | grep -o '"id":"[^"]*"' | cut -d'"' -f4 | tr '\n' ' ')"
  echo "  Served Models:   ${SERVED_MODELS:-Cogni-Brain}"
fi

echo
echo "=== System Unified Memory (DGX Spark) ==="
if command -v free >/dev/null 2>&1; then
  free -h
else
  echo "  'free' utility not available"
fi

if command -v swapon >/dev/null 2>&1; then
  echo
  echo "=== Swap Status ==="
  swapon --show || echo "  No swap active (swap disabled on DGX Spark)"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "=== NVIDIA GPU / Unified Memory Pool ==="
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
fi

echo
echo "=== Recent Container Logs (last 20 lines) ==="
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  docker logs "$CONTAINER" --tail 20 2>&1 | sed 's/^/  /'
else
  echo "  No container found"
fi
echo "================================================================="
