#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-glm53-flash}"

docker ps --filter "name=$CONTAINER"
echo
docker logs --tail 80 "$CONTAINER" 2>/dev/null || true
echo
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

