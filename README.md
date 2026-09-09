# Benchmarking Cogni-Brain (GLM-5.3-Flash) on DGX Spark

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![Base Model](https://img.shields.io/badge/base%20model-GLM--5.3--Flash%20%28ox--alpha%29-limegreen)
![Served Model](https://img.shields.io/badge/served%20model-Cogni--Brain-brightgreen)
![Container](https://img.shields.io/badge/container-spark--brain-blue)
![Runtime](https://img.shields.io/badge/runtime-llama.cpp%20%2B%20GGUF-orange)
![Hardware](https://img.shields.io/badge/hardware-NVIDIA%20DGX%20Spark-brightgreen?logo=nvidia&logoColor=white)
![Memory](https://img.shields.io/badge/unified%20memory-128GB%20%28swap%20disabled%29-purple)
![Quantization](https://img.shields.io/badge/quantization-UD--IQ3__XXS%20%2F%20UD--IQ2__XXS-blueviolet)
![Context](https://img.shields.io/badge/context-8K%20default%20(up%20to%2032K)-blue)

This repository documents running and benchmarking [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) on a single **NVIDIA DGX Spark / GB10** (128 GB unified memory), served as **`Cogni-Brain`** inside container **`spark-brain`**.

GLM-5.3-Flash (code named **`ox-alpha`**) is a 320B total parameter / 18B active multimodal MoE model with linear/sparse hybrid attention. While the upstream model architecture supports up to 1M tokens in cluster environments, on a single DGX Spark (128 GB Unified Memory, swap disabled) with a 102–120 GB model footprint, context is configured to **8,192 tokens by default** (scaling up to **32,768 tokens** under `UD-IQ2_XXS` with quantized KV cache) to guarantee zero OOM risk.

> ⚠️ **Personal workstation benchmark harness. Not for enterprise use. Use at your own risk.**

---

## Target Profile

| Dimension | Specification | Notes |
|---|---|---|
| **Served Model Name** | `Cogni-Brain` | OpenAI-compatible endpoint alias |
| **Docker Container** | `spark-brain` | Lifecycle managed via `docker/*.sh` |
| **Port** | `8000` | Exposed at `http://localhost:8000/v1` |
| **Runtime** | Unsloth `llama.cpp` (`glm5next/upstream`) | Native Grace-Blackwell SM 120 CUDA build |
| **Primary Quant** | `UD-IQ3_XXS` (~120.37 GB) | 3-bit GGUF (single-slot benchmark profile) |
| **Safe Fallback Quant** | `UD-IQ2_XXS` (~101.84 GB) | 2-bit GGUF (recommended daily baseline; ~20 GB free) |
| **Context Window (DGX Spark)** | `8,192` default (up to `32,768`) | Sized to fit 128 GB unified memory budget without swap |
| **KV Cache Quantization** | `q4_0` / `q8_0` | Default: `q4_0` (preserves memory headroom) |
| **Hardware** | NVIDIA DGX Spark / GB10 | Grace-Blackwell ARM + Blackwell GPU, 128GB Unified Memory |
| **Swap Status** | **Disabled** | Host memory allocations must remain strictly within 128 GB |
| **Sampling Defaults** | `temperature=1.0`, `top_p=0.95` | Task default; DeepSWE: `temp=0.95`, `top_p=1.0` |
| **Reasoning Effort** | `max` baseline | Sweeps: `low`, `high`, `max` |
| **MTP Draft** | `0` (off) or `2` (draft tokens) | Multi-Token Prediction speculative decoding support |

---

## Quick Start (DGX Spark Single Node)

### 1. Preflight
Verify Docker, NVIDIA Container Toolkit, GPU drivers, unified memory, and swap status:

```bash
bash setup/install.sh
```

### 2. Download Model Shards
Download split GGUF shards into `models/unsloth/GLM-5.3-Flash-GGUF`:

```bash
# Primary 3-bit profile (~120GB):
bash setup/download_model.sh

# Or 2-bit profile (~102GB, recommended if host RAM headroom is tight):
QUANT=UD-IQ2_XXS bash setup/download_model.sh
```

### 3. Build Docker Image
Build the CUDA container with native Grace-Blackwell (SM 120 / Blackwell GB10) support:

```bash
bash docker/build.sh
```

### 4. Start Cogni-Brain Server
Launches `llama-server` in container `spark-brain`:

```bash
# Recommended daily driver baseline (~102 GB, ~20 GB free headroom, supports PARALLEL=2):
QUANT=UD-IQ2_XXS bash docker/start.sh

# Or launch with max-fidelity 3-bit profile (~120 GB, tight headroom, single-slot only):
bash docker/start.sh
```

Container management:
- Status & memory check: `bash docker/status.sh`
- Follow logs: `docker logs -f spark-brain`
- Stop container: `bash docker/stop.sh`

### 5. Smoke Test
Run the 4-phase verification suite (Health check, Coherence probe, Prefill tok/s, Decode tok/s):

```bash
bash benchmark/smoke_test.sh localhost:8000
```

### 6. Run Benchmarks

```bash
# 1. Single-stream decode speed, TTFT, concurrency, and context sweep:
uv run benchmark/benchmark_speed.py

# 2. Spark Arena / llama-benchy multi-depth sweep:
uv run benchmark/benchmark_speed_arena.py

# 3. Agentic tool calling benchmark (tool-eval-bench):
uv run benchmark/benchmark_smarts.py
```

---

## Operational Guidelines (DGX Spark)

- **Memory Profiles & Disabled Swap**: DGX Spark runs with swap disabled by default. Allocations exceeding physical RAM trigger an immediate kernel OOM kill. Host overhead (OS, Docker, NVIDIA UVM drivers) consumes ~5–7 GB.
  - **`UD-IQ2_XXS` (~101.8 GB static)**: Recommended production baseline. Leaves ~20 GB headroom for KV cache and supports true multi-slot serving (`PARALLEL=2`).
  - **`UD-IQ3_XXS` (~120.4 GB static)**: High-fidelity benchmark profile. Leaves ~2 GB physical RAM headroom; strictly requires single-slot (`PARALLEL=1`).
- **Parallel Concurrency vs Queueing**: `docker/start.sh` defaults to `--parallel 1` to protect unified memory. When running concurrency tests in `benchmark/benchmark_speed.py`, requests are queued unless started with multi-slot serving: `QUANT=UD-IQ2_XXS PARALLEL=2 bash docker/start.sh`.
- **Linear Attention / Recurrent Safeguards**: GLM-5.3-Flash uses hybrid linear attention with recurrent compressor states. `docker/start.sh` enforces `--no-cache-prompt`, `--cache-reuse 0`, `--no-context-shift`, and `--slot-prompt-similarity 1.1` to prevent recurrent state mismatch errors across queries.
- **KV Cache Quantization**: `docker/start.sh` passes `--cache-type-k q4_0 --cache-type-v q4_0` by default to preserve unified memory headroom.
- **Sampling Defaults**: Documented task default is `temperature=1.0`, `top_p=0.95`.
- **Reasoning Modes**: Supports `low`, `high`, and `max`. Default is `max`.

---

## Benchmark Results (Placeholders — Awaiting On-Device Run)

> ℹ️ *This model has not been tested on DGX Spark yet. The tables below are placeholders awaiting verified runs on hardware.*

### 1. Baseline Speed & Latency (`benchmark/benchmark_speed.py`)

| Profile | Quant | Context | Decode Speed | TTFT | Status |
|---|---|---|:---:|:---:|:---:|
| Baseline ($n=0$) | `UD-IQ3_XXS` | 8,192 | *Pending run* | *Pending run* | Untested |
| Baseline ($n=0$) | `UD-IQ2_XXS` | 8,192 | *Pending run* | *Pending run* | Untested |
| MTP Draft ($n=2$) | `UD-IQ3_XXS` | 8,192 | *Pending run* | *Pending run* | Untested |

### 2. Full Arena Context Sweep (`benchmark/benchmark_speed_arena.py`)

*Awaiting run. Results will be saved to `benchmark/results_arena.csv`.*

### 3. Agentic Quality (`benchmark/benchmark_smarts.py`)

*Awaiting run via tool-eval-bench.*

---

## OpenAI-Compatible API Usage

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Cogni-Brain",
    "messages": [
      {"role": "user", "content": "Explain the architectural advantages of 18B active parameter MoE routing on unified memory."}
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
assets/         Benchmark screenshots and performance charts
benchmark/      Speed, Spark Arena sweep, smarts/tool-eval, and smoke test
docker/         Dockerfile (SM 12.1), build, start, status, and stop helpers
setup/          DGX Spark preflight and model download scripts
```

---

## License

MIT License.
