#!/usr/bin/env bash
# benchmark/smoke_test.sh — Quick verification of Cogni-Brain server health, coherence, prefill & decode
#
# Usage:
#   bash benchmark/smoke_test.sh
#   bash benchmark/smoke_test.sh localhost:8000
set -euo pipefail

EP="${1:-localhost:8000}"
BASE="http://$EP"
MODEL="${MODEL:-Cogni-Brain}"

echo "================================================================="
echo "=== DGX Spark Cogni-Brain (GLM-5.3-Flash) Smoke Test ($EP) ==="
echo "================================================================="
echo "  Target Endpoint: $BASE"
echo "  Served Model:    $MODEL"
echo

echo "[1/4] Health Check (waiting for model load if booting)..."
MAX_WAIT="${MAX_WAIT:-120}"
WAITED=0
while [ "$WAITED" -lt "$MAX_WAIT" ]; do
  if curl -sf -m 2 "$BASE/health" >/dev/null 2>&1; then
    break
  fi
  sleep 3
  WAITED=$((WAITED + 3))
  printf "  Waiting for server readiness (%ds/%ds)...\r" "$WAITED" "$MAX_WAIT"
done

if curl -sf -m 2 "$BASE/health" >/dev/null 2>&1; then
  echo "  ✓ Server is healthy (/health OK)"
else
  echo
  echo "  ✗ Server is not ready at $BASE/health after ${MAX_WAIT}s"
  echo "  Check container logs with: docker logs --tail 50 spark-brain"
  exit 1
fi

echo
echo "[2/4] Coherence Test..."
COHERENCE_RESP=$(curl -s -m 120 "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"The capital of France is\"}],
    \"temperature\": 0,
    \"max_tokens\": 32
  }")

echo "$COHERENCE_RESP" | python3 -c '
import json, sys
try:
    r = json.load(sys.stdin)
    msg = r["choices"][0]["message"]
    txt = msg.get("content") or msg.get("reasoning_content") or ""
    print("  Output:", repr(txt.strip()[:100]))
    if not txt.strip():
        print("  ✗ Coherence test returned empty content")
        sys.exit(1)
    print("  ✓ Coherence test passed")
except Exception as e:
    print("  ✗ Coherence test failed:", e)
    sys.exit(1)
'

echo
echo "[3/4] Prefill Test (TTFT on ~1,000 token prompt)..."
python3 - "$BASE" "$MODEL" <<'PY'
import json, sys, time, urllib.request

base = sys.argv[1]
model = sys.argv[2]
# ~1000 tokens
prompt = "The DGX Spark benchmark harness tests unified memory inference. " * 100
t0 = time.perf_counter()
payload = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 1,
    "temperature": 0.0,
}).encode("utf-8")
req = urllib.request.Request(
    f"{base}/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        res = json.load(resp)
        dt = time.perf_counter() - t0
        u = res.get("usage", {})
        prompt_tokens = u.get("prompt_tokens", 1000)
        rate = prompt_tokens / dt if dt > 0 else 0
        print(f"  ✓ {prompt_tokens} tokens prefilled in {dt:.2f}s => {rate:.0f} tok/s prefill speed (TTFT: {dt*1000:.0f} ms)")
except Exception as e:
    print(f"  ✗ Prefill test failed: {e}")
    sys.exit(1)
PY

echo
echo "[4/4] Single-Stream Decode Test (128 tokens)..."
python3 - "$BASE" "$MODEL" <<'PY'
import json, sys, time, urllib.request

base = sys.argv[1]
model = sys.argv[2]
t0 = time.perf_counter()
payload = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "Explain unified memory in three sentences."}],
    "max_tokens": 128,
    "temperature": 0.0,
}).encode("utf-8")
req = urllib.request.Request(
    f"{base}/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        res = json.load(resp)
        dt = time.perf_counter() - t0
        u = res.get("usage", {})
        completion_tokens = u.get("completion_tokens", 0)
        if completion_tokens == 0:
            msg = res["choices"][0]["message"]
            text = msg.get("content") or msg.get("reasoning_content") or ""
            completion_tokens = len(text.split()) * 1.33
        rate = completion_tokens / dt if dt > 0 else 0
        print(f"  ✓ ~{completion_tokens:.0f} tokens decoded in {dt:.2f}s => {rate:.1f} tok/s decode speed")
except Exception as e:
    print(f"  ✗ Decode test failed: {e}")
    sys.exit(1)
PY

echo
echo "================================================================="
echo "✓ All smoke test checks passed successfully."
echo "================================================================="
