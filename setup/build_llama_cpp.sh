#!/usr/bin/env bash
# setup/build_llama_cpp.sh - Build llama.cpp with GLM-5.3-Flash support
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$REPO_DIR/llama.cpp}"
LLAMA_BRANCH="${LLAMA_BRANCH:-glm5next/upstream}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/unslothai/llama.cpp}"
CUDA="${CUDA:-ON}"

echo "================================================================="
echo "=== Building llama.cpp for GLM-5.3-Flash ==="
echo "================================================================="
echo "  Repo:   $LLAMA_REPO"
echo "  Branch: $LLAMA_BRANCH"
echo "  Dir:    $LLAMA_DIR"
echo "  CUDA:   $CUDA"
echo

if [ ! -d "$LLAMA_DIR/.git" ]; then
  git clone --branch "$LLAMA_BRANCH" "$LLAMA_REPO" "$LLAMA_DIR"
else
  git -C "$LLAMA_DIR" fetch origin "$LLAMA_BRANCH"
  git -C "$LLAMA_DIR" checkout "$LLAMA_BRANCH"
  git -C "$LLAMA_DIR" pull --ff-only
fi

cmake "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA="$CUDA"

cmake --build "$LLAMA_DIR/build" --config Release -j --clean-first \
  --target llama-cli llama-server llama-gguf-split

echo
echo "Build complete."
"$LLAMA_DIR/build/bin/llama-server" --version || true

