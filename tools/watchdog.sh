#!/usr/bin/env bash
set -euo pipefail

TARGET="${TARGET:-localhost:8000}"
MODEL="${MODEL:-GLM-5.3-Flash}"

payload="$(mktemp)"
cat >"$payload" <<JSON
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": "Return exactly: healthy"}],
  "temperature": 0,
  "max_tokens": 16
}
JSON

if curl -fsS "http://$TARGET/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "@$payload" | grep -qi "healthy"; then
  echo "OK"
else
  echo "FAILED"
  exit 1
fi

