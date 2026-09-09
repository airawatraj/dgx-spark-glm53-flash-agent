#!/usr/bin/env bash
# docker/stop.sh — Stop and remove Cogni-Brain (spark-brain) container
set -euo pipefail

CONTAINER="${CONTAINER:-spark-brain}"

echo "=== Stopping $CONTAINER ==="

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "Stopping container $CONTAINER ..."
  docker stop "$CONTAINER" >/dev/null 2>&1 || true
  docker rm "$CONTAINER" >/dev/null 2>&1 || true
  echo "✓ Removed $CONTAINER."
else
  echo "Container $CONTAINER is not running."
fi

# Clean up legacy container name if still present
if docker ps -a --format '{{.Names}}' | grep -q '^glm53-flash$'; then
  docker rm -f glm53-flash >/dev/null 2>&1 || true
  echo "✓ Removed legacy glm53-flash container."
fi
