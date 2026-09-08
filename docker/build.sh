#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-glm53-flash-dgx-spark:latest}"

docker build -t "$IMAGE" -f docker/Dockerfile .

