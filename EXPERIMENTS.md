# Experiments & Benchmark Evidence: GLM-5.3-Flash on Single DGX Spark

This document is the working lab notebook for benchmarking `unsloth/GLM-5.3-Flash-GGUF` on a single NVIDIA DGX Spark / GB10 with 128GB unified memory.

No benchmark number should be promoted to the README until it is repeatable and has a timestamped entry in `docs/RESULTS.md`.

---

## 1. Initial Hypotheses

| Hypothesis | Test |
|---|---|
| `UD-IQ3_XXS` is the practical 128GB baseline | Boot server, record memory headroom, complete smoke test |
| MTP `n=2` improves long-context decode without destabilizing output | Compare MTP off vs `n=2` at 4K, 16K, 64K |
| Reasoning effort changes quality and latency profile | Sweep `low`, `high`, `max` on fixed prompts |
| Single-stream decode remains usable at deep context | Sweep 4K to 256K+ prompts |
| Tool-calling quality is competitive for local agent loops | Run `benchmark/benchmark_smarts.py --mode trials` |

---

## 2. Run Matrix

| Profile | Quant | Reasoning | MTP | Contexts | Status |
|---|---|---:|---:|---|---|
| baseline | `UD-IQ3_XXS` | max | off | 4K, 16K, 64K, 128K | planned |
| mtp2 | `UD-IQ3_XXS` | max | 2 | 4K, 16K, 64K, 128K | planned |
| low-thinking | `UD-IQ3_XXS` | low | off | short, 16K | planned |
| high-thinking | `UD-IQ3_XXS` | high | off | short, 16K | planned |
| max-thinking | `UD-IQ3_XXS` | max | off | short, 16K | planned |

---

## 3. Measurement Priorities

- Speed: TTFT, decode TPS, aggregate concurrency TPS, prompt processing TPS.
- Context: max successful context, output coherence, failure mode at limit.
- Smarts: tool-eval score, deterministic trial variance, reasoning mode deltas.
- Reliability: boot time, memory headroom, server wedges, malformed output, aborted streams.

---

## 4. Known Constraints

- The 3-bit split GGUF is near the edge of 128GB systems; close other large processes before serving.
- Long context runs may need reduced concurrency and conservative batch sizing.
- Runtime flags for GLM-5.3-Flash and MTP are moving quickly; pin the exact llama.cpp commit in every result entry.

