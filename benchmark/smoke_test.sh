#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-localhost:8000}"
MODEL="${MODEL:-GLM-5.3-Flash}"

curl -sS "http://$TARGET/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly: GLM smoke OK\"}],
    \"temperature\": 0,
    \"max_tokens\": 32
  }" | tee /tmp/glm53-smoke.json

echo

