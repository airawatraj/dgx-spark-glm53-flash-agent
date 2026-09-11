# Experiments & Architectural Frontiers: Cogni-Brain (GLM-5.3-Flash) on Single DGX Spark

This document captures empirical benchmark evaluations, memory profiling, and experimental configurations for running [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) (320B total / 18B active MoE) served as **`Cogni-Brain`** on a single **NVIDIA DGX Spark / GB10** (128 GB Unified Memory, swap disabled).

The current production baseline is documented in [README.md](README.md). This document serves as the research and frontier lab notes for stress-testing hardware limits.

---

## 1. Verified Production Baselines (Release v1.0.0)

The empirical test suite below was collected directly on a single DGX Spark GB10 running `UD-IQ2_XXS` (~101.8 GB) in container `spark-brain` with 32K default context and `q4_0` quantized KV cache.

### A. Speed & Context Depth Sweep (`benchmark/benchmark_speed.py`)

- **Single-Stream Decode TPS**: **18.7 tok/s** average (Peak: **18.8 tok/s**)
- **Time to First Token (TTFT)**: **707 ms** steady state
- **Prefill Throughput**: **223 tok/s** (~1,113 tokens prefilled in 5.00s)
- **Concurrency (1 & 2 Streams)**: 17.9 tok/s (1 stream) / 17.8 tok/s (2 streams, serialized under single-slot)
- **Context Depth Scaling**:
  - ~1K tokens: **18.7 tok/s** (TTFT: 4,110 ms)
  - ~4K tokens: **18.4 tok/s** (TTFT: 14,536 ms)
  - ~8K tokens: **18.1 tok/s** (TTFT: 26,103 ms)
  - ~16K tokens: **17.5 tok/s** (TTFT: 47,659 ms)

![Cogni-Brain Speed Benchmark](assets/benchmark_speed.png)

---

### B. Agentic Tool-Calling & Reasoning (`benchmark/benchmark_smarts.py`)

Evaluated against `tool-eval-bench` across 15 real-world tool scenarios:

- **Overall Score**: **100 / 100** (Rating: ★★★★★ Excellent)
- **Scenario Pass Rate**: **15 / 15 passed (30 / 30 points, 100% pass rate)**
- **Tool Selection**: **100%** (6/6)
- **Parameter Precision**: **100%** (6/6)
- **Multi-Step Chains**: **100%** (6/6)
- **Restraint & Refusal**: **100%** (6/6)
- **Error Recovery**: **100%** (6/6)
- **Deployability**: **73 / 100** (median turn: 12.3s)
- **Evaluation Tokens**: 32,685 tokens in 459.9s (0.9 pts / 1K tokens)

![Cogni-Brain Tool Eval Benchmark](assets/benchmark_smarts_eval.png)
![Cogni-Brain Tool Eval Summary](assets/benchmark_smarts_results.png)

---

### C. Verified Spark Arena Benchmark (32K Multi-Depth Sweep)

[![Spark Arena Benchmark](assets/spark_arena_glm5_3_flash.png)](https://spark-arena.com/benchmark/8b333138-2035-44d0-80da-cc1ee71faef3)

> 🔗 **Interactive Online Submission:** [spark-arena.com/benchmark/8b333138-2035-44d0-80da-cc1ee71faef3](https://spark-arena.com/benchmark/8b333138-2035-44d0-80da-cc1ee71faef3) · Recorded on single NVIDIA DGX Spark (128 GB Unified Memory) with `UD-IQ2_XXS` and 32K context window.

---

## 2. Frontier Experiment: Pushing Context Limits to 64K (`65,536` Tokens)

### A. Architectural & Memory Feasibility

On standard dense Transformer architectures, running 64K context on a 320B-class model would demand >50 GB of KV cache alone, making it impossible on a 128 GB machine.

GLM-5.3-Flash enables extended context through its **hybrid linear/sparse attention topology**:
1. **34 Kimi Delta Attention (KDA) Layers**: Linear attention layers maintain fixed recurrent state buffers ($O(1)$ memory complexity). Context length expansion adds zero KV cache overhead for these 34 layers.
2. **11 MLA / DSA Layers**: Only 11 layers allocate dynamic $O(N)$ KV cache buffers.
3. **IndexPool Pooling**: Compresses sparse indexer key vectors by 4×, drastically shrinking the attention routing table.

#### 64K Memory Budget (DGX Spark 128 GB Unified RAM, Swap Disabled)

| Component | 32K Baseline | 64K Extended Target | 131K Extreme Limit | Notes |
|---|:---:|:---:|:---:|---|
| **Static Weights (`UD-IQ2_XXS`)** | 101.8 GB | 101.8 GB | 101.8 GB | GGUF resident in unified memory |
| **OS + Drivers + Docker** | ~5.5 GB | ~5.5 GB | ~5.5 GB | Grace ARM + Blackwell UVM drivers |
| **KV Cache (`q4_0`) + Indexer** | ~2.2 GB | **~4.8 GB** | ~18–24 GB | 11 MLA layers + IndexPool cache |
| **Recurrent States (34 KDA)** | ~0.5 GB | **~0.5 GB** | ~0.5 GB | Fixed $O(1)$ recurrent buffer |
| **Workspace & FlashAttention** | ~0.8 GB | **~1.2 GB** | ~2.5 GB | CUDA execution scratchpad |
| **Total Peak Memory** | **~110.8 GB** | **~113.8 GB** | **~128.3+ GB (OOM)** | Strict physical RAM limit: 128.0 GB |
| **Safe Headroom** | **~17.2 GB** | **~14.2 GB (SAFE)** | **< 0 GB (FAIL)** | Swap disabled: zero margin for error |

> **Conclusion**: While 131K breaches the physical 128 GB ceiling due to indexer cache overhead, **64K (`65536`) has a predicted ~95% confidence of stable operation**, retaining >14 GB of safe headroom.

---

### B. Experimental Execution Runbook: 64K Context Push

Run the following procedure on the DGX Spark host to execute the 64K context experiment.

#### Step 1: Preflight Host Memory Check
Confirm swap is disabled and host baseline memory is clear:
```bash
# Verify baseline system footprint
bash docker/status.sh
free -h
```
Ensure available memory is $\ge 120\text{ GB}$ before loading the model.

#### Step 2: Launch Cogni-Brain with 64K Context Window
Restart the container with `CTX_SIZE=65536` and strictly `PARALLEL=1`:
```bash
# 1. Stop current container
bash docker/stop.sh

# 2. Start container with 64K context window
QUANT=UD-IQ2_XXS CTX_SIZE=65536 PARALLEL=1 bash docker/start.sh
```

#### Step 3: Inspect Model Loading & Cache Allocation
Monitor the initial boot logs to record exact tensor and KV cache allocations:
```bash
docker logs spark-brain | grep -E "(kv cache|model size|total size|CUDA|warming up)"
```
Verify that `total VRAM used` does not exceed **114 GB**.

#### Step 4: Run the 64K Spark Arena Sweep in `tmux`
Launch `llama-benchy` across context depths up to 65,535 tokens:
```bash
# 1. Open a persistent tmux session
tmux new -s arena-64k

# 2. Run the extended arena sweep
uv run benchmark/benchmark_speed_arena.py \
  --depth 0 2048 4096 8192 16384 32768 65535 \
  --concurrency 1 \
  --save-result benchmark/results_arena_64k.csv

# 3. Detach from tmux session:
#    Press: Ctrl+b, then d

# 4. Check progress anytime:
tmux attach -t arena-64k
```

#### Step 5: Synthetic Long-Context Coherence Probe
Send a synthetic deep-context request to probe stability under sustained load:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --max-time 900 \
  -d '{
    "model": "Cogni-Brain",
    "messages": [
      {"role": "system", "content": "You are Cogni-Brain running a 64K context verification probe on DGX Spark."},
      {"role": "user", "content": "Confirm that your KV cache and linear recurrent states are coherent across a 64K sequence length."}
    ],
    "max_tokens": 128,
    "temperature": 1.0,
    "top_p": 0.95
  }'
```

---

### C. Diagnostic Indicators & Rollback

| Symptom | Root Cause | Immediate Action |
|---|---|---|
| **Container exit `137` (SIGKILL)** | Kernel OOM killer triggered (exceeded 128 GB) | Revert to 32K: `QUANT=UD-IQ2_XXS CTX_SIZE=32768 bash docker/start.sh` |
| **`CUDA error: out of memory`** | KV or scratch allocation failed during model init | Reduce `--batch-size` to `1024` or reduce CTX to `40960` |
| **HTTP 400 `Prompt is too large`** | Total tokens (`depth + pp + tg`) exceeded `--ctx-size` | Ensure server `CTX_SIZE` $\ge$ `depth + pp + tg` |
| **Client Socket Timeout** | Long prefill at depth 64K took >120s | Set client `timeout=900.0` (15 minutes) |

---

## 3. Experiment: MTP Speculative Decoding

GLM-5.3-Flash includes a 2-token speculative draft head. The forum community reports **26–38 effective tok/s** with speculative decoding on similar quantizations (vs ~18 tok/s base). `docker/start.sh` supports this via `MTP_DRAFT`.

### Runbook: MTP Benchmark

```bash
# 1. Stop current container
bash docker/stop.sh

# 2. Start with MTP speculative decoding enabled (2 draft tokens)
MTP_DRAFT=2 bash docker/start.sh

# 3. Run smoke test to verify MTP is active
bash benchmark/smoke_test.sh localhost:8000

# 4. Run full tool-eval with spec-bench to measure acceptance rate
tmux new -s mtp
uv run benchmark/benchmark_smarts.py --spec-bench --perf
# Detach: Ctrl+b, then d

# 5. Compare results:
#    - Effective tok/s vs baseline 18.7 tok/s
#    - Acceptance rate (α) per prompt type
#    - Draft window utilization (τ / win)
```

Expected metrics to compare against baseline (`MTP=0`):

| Metric | Baseline (`MTP=0`) | Target (`MTP=2`) |
|---|:---:|:---:|
| **Decode tok/s** | 18.7 | ~28–34 (estimated) |
| **Acceptance rate (α)** | N/A | 50–80% (prompt-dependent) |
| **TTFT** | 707 ms | May increase slightly (draft head overhead) |

---

## 4. Experiment: Quantization Quality Comparison (IQ2 vs IQ3)

Aggressive quantization degrades quality. Community comparisons show EXL3 2-bit GLM-5.3-Flash scoring **34% IFEval** (vs 76% for FP8 Qwen3.8-Flash-Next). This experiment measures the quality cost of quantization for your two profiles.

### Runbook: Quality Comparison

```bash
# ── Profile 1: UD-IQ2_XXS (daily driver) ────────────────────────────
bash docker/stop.sh
QUANT=UD-IQ2_XXS bash docker/start.sh

# Run full tool-eval + academic benchmarks (in tmux, ~60–90 min):
tmux new -s quality-iq2
uv run benchmark/benchmark_smarts.py --gsm8k --mmlu --ifeval --gsm8k-limit 50 --mmlu-limit 50 --ifeval-limit 100
# Detach: Ctrl+b, then d

# ── Profile 2: UD-IQ3_XXS (high fidelity, 16K context only) ─────────
bash docker/stop.sh
QUANT=UD-IQ3_XXS CTX_SIZE=16384 PARALLEL=1 bash docker/start.sh

# Run same benchmarks for direct comparison:
tmux new -s quality-iq3
uv run benchmark/benchmark_smarts.py --gsm8k --mmlu --ifeval --gsm8k-limit 50 --mmlu-limit 50 --ifeval-limit 100
# Detach: Ctrl+b, then d
```

Key quality metrics to compare:

| Metric | Community Reference (EXL3 2-bit) | UD-IQ2_XXS | UD-IQ3_XXS |
|---|:---:|:---:|:---:|
| **Tool-Eval (69 scenarios)** | 85/100 | TBD | TBD |
| **GSM8K (math)** | 96% | TBD | TBD |
| **MMLU (knowledge)** | 70% | TBD | TBD |
| **IFEval (instruction following)** | 34% | TBD | TBD |

---

## 5. Future Research Directions

1. **True Multi-Slot Serving (`PARALLEL=2`)**:
   - Benchmark memory envelope when running dual concurrent slots under `CTX_SIZE=16384`.
2. **131K Extreme Context Investigation**:
   - Investigate whether dynamic KV offloading or `iq2_s` quantized KV cache can bring the 131K indexer footprint within the 128 GB threshold.
3. **KV Cache Precision Study**:
   - Compare `q4_0` vs `q8_0` vs `f16` KV cache impact on tool-eval and IFEval quality. Community consensus is that higher-precision KV cache is critical for GLM-5.3-Flash long-context quality.
