#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-glm53-flash}"
docker rm -f "$CONTAINER"

