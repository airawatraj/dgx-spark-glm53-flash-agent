# Benchmarking GLM-5.3-Flash on DGX Spark

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![Base Model](https://img.shields.io/badge/base%20model-GLM--5.3--Flash%20%28ox--alpha%29-limegreen)
![Runtime](https://img.shields.io/badge/runtime-llama.cpp%20%2B%20GGUF-orange)
![Hardware](https://img.shields.io/badge/hardware-NVIDIA%20DGX%20Spark-brightgreen?logo=nvidia&logoColor=white)
![Memory](https://img.shields.io/badge/unified%20memory-128GB-blue)
![Quantization](https://img.shields.io/badge/quantization-UD--IQ3__XXS-purple)
![Context](https://img.shields.io/badge/context-1M%20max-lightgrey)

This repository documents benchmarking [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) on a single **NVIDIA DGX Spark / GB10** with **128GB unified memory**.

The primary target is the documented 3-bit `UD-IQ3_XXS` GGUF profile, which Unsloth lists at about 120GB and recommends for 128GB-class systems. GLM-5.3-Flash, also known as `ox-alpha`, is a 320B total parameter / 18B active multimodal MoE model with a maximum context window of `1,048,576` tokens.

> Personal workstation benchmark harness. Not for enterprise use. Use at your own risk.

---

## Target Profile

| Dimension | Target |
|---|---|
| Served model name | `GLM-5.3-Flash` |
| Runtime | Unsloth llama.cpp branch or compatible upstream once merged |
| Quant | `UD-IQ3_XXS` |
| Expected model size | ~120GB split GGUF |
| Hardware | NVIDIA DGX Spark / GB10, 128GB unified memory |
| API surface | OpenAI-compatible `/v1/chat/completions` and `/v1/completions` |
| Sampling default | `temperature=1.0`, `top_p=0.95` |
| Reasoning effort | `max` baseline, sweep `low`, `high`, `max` |
| MTP | Benchmark both off and draft-token `2` where runtime support is available |

---

## Quick Start

### 1. Preflight

```bash
bash setup/install.sh
```

### 2. Download GLM-5.3-Flash GGUF

```bash
# Run inside tmux on remote SSH.
bash setup/download_model.sh
```

Defaults to:

```bash
MODEL=unsloth/GLM-5.3-Flash-GGUF
QUANT=UD-IQ3_XXS
```

### 3. Build llama.cpp with GLM-5.3-Flash support

```bash
bash setup/build_llama_cpp.sh
```

The script follows Unsloth's documented GLM-5.3-Flash branch:

```bash
git clone --branch glm5next/upstream https://github.com/unslothai/llama.cpp
```

### 4. Start OpenAI-compatible server

```bash
bash docker/start.sh
```

For a local non-Docker launch after `setup/build_llama_cpp.sh`:

```bash
bash setup/start_llama_server.sh
```

### 5. Smoke Test

```bash
bash benchmark/smoke_test.sh localhost:8000
```

### 6. Run Benchmarks

```bash
# TTFT, decode TPS, concurrency, context sweep:
uv run benchmark/benchmark_speed.py

# Reasoning-effort comparison:
uv run benchmark/benchmark_reasoning.py

# Tool calling / agentic sanity suite:
uv run benchmark/benchmark_smarts.py
```

---

## Benchmark Plan

1. Establish a stable `UD-IQ3_XXS` boot profile on DGX Spark with enough memory headroom to avoid host pressure.
2. Measure baseline single-stream decode, TTFT, prompt processing, and context scaling at 4K, 16K, 64K, 128K, 256K, and higher if stable.
3. Sweep reasoning modes: `low`, `high`, `max`.
4. Sweep MTP support when available: off, `n=2`, and optionally `n=3`.
5. Run tool-eval-bench short/trials/perf profiles against the same OpenAI-compatible endpoint.
6. Record every verified run in `docs/RESULTS.md` before promoting numbers into this README.

---

## OpenAI-Compatible API Usage

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "GLM-5.3-Flash",
    "messages": [
      {"role": "user", "content": "Explain the performance tradeoffs of 1-bit vs 3-bit GGUF inference."}
    ],
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 512,
    "chat_template_kwargs": {"reasoning_effort": "max"}
  }'
```

---

## Repository Layout

```text
benchmark/      OpenAI-compatible speed, reasoning, smoke, and tool-eval benchmarks
docker/         Container build/start/stop/status helpers
docs/           Results ledger and operational notes
setup/          Host preflight, model download, llama.cpp build, local server launch
tools/          Snapshot and watchdog utilities
assets/         Benchmark screenshots and charts
```

---

## Source Notes

- Unsloth GLM-5.3-Flash docs: https://unsloth.ai/docs/models/glm-5.3-flash
- GGUF model repo: https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF
- llama.cpp support branch: https://github.com/unslothai/llama.cpp/tree/glm5next/upstream

---

## License

Code and scripts: MIT License.

