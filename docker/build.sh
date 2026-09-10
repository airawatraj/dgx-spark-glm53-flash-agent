#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-glm53-flash-dgx-spark:latest}"

docker build "$@" -t "$IMAGE" -f "$REPO_DIR/docker/Dockerfile" "$REPO_DIR"

