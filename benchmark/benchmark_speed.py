#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""
DGX Spark / Cogni-Brain (GLM-5.3-Flash) speed, concurrency, and context benchmark.

Usage:
  uv run benchmark/benchmark_speed.py
  uv run benchmark/benchmark_speed.py --host localhost --port 8000 --model Cogni-Brain
"""

import argparse
import json
import statistics
import threading
import time
from datetime import datetime

import requests


COLORS = {
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "reset": "\033[0m",
    "dim": "\033[2m",
}


def c(text, color):
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def header(title):
    line = "-" * 64
    print(f"\n{c(line, 'cyan')}")
    print(c(f"  {title}", "bold"))
    print(c(line, "cyan"))


def result_line(label, value, unit="", color="green"):
    print(f"  {c(label.ljust(32), 'dim')} {c(str(value), color)} {unit}")


def make_prompt(target_tokens):
    base = ("DGX Spark local inference benchmark on Grace-Blackwell. " * 50).split()
    n_words = max(10, int(target_tokens / 1.33))
    words = (base * ((n_words // len(base)) + 1))[:n_words]
    return " ".join(words) + "\n\nSummarize the benchmark hardware in one sentence."


def stream_chat(host, port, model, prompt, max_tokens=256, reasoning_effort="max", timeout=300):
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"reasoning_effort": reasoning_effort},
        "cache_prompt": False,
    }

    t_start = time.perf_counter()
    t_first = None
    chunks_content = 0
    chunks_reasoning = 0

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                return None, None, 0, 0, f"HTTP {response.status_code}: {response.text[:200]}"

            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or delta.get("thinking") or ""

                if content or reasoning:
                    if t_first is None:
                        t_first = time.perf_counter()
                    if content:
                        chunks_content += 1
                    if reasoning:
                        chunks_reasoning += 1

        t_end = time.perf_counter()
        total_chunks = chunks_content + chunks_reasoning
        ttft = (t_first - t_start) * 1000 if t_first else None
        gen_time = (t_end - t_first) if t_first else max(t_end - t_start, 1e-6)
        tps = total_chunks / gen_time if gen_time > 0 else 0.0

        return ttft, tps, total_chunks, chunks_reasoning, None

    except Exception as exc:
        return None, None, 0, 0, str(exc)


def test_baseline(args):
    header("1. BASELINE SINGLE-STREAM CHAT")
    prompt = "Explain why unified memory architectures benefit large mixture-of-experts models in three concise paragraphs."
    tps_runs = []
    ttft_runs = []

    for run in range(1, args.runs + 1):
        ttft, tps, total_chunks, reasoning_chunks, err = stream_chat(
            args.host, args.port, args.model, prompt, args.max_tokens, args.reasoning_effort
        )
        if err:
            print(f"  Run {run}: {c('FAILED - ' + err, 'red')}")
            continue
        tps_runs.append(tps)
        if ttft is not None:
            ttft_runs.append(ttft)
        info = f"TTFT={ttft:.0f}ms chunks={total_chunks}"
        if reasoning_chunks > 0:
            info += f" ({reasoning_chunks} reasoning)"
        print(f"  Run {run}: {c(f'{tps:.1f} tok/s', 'green')} {c(info, 'dim')}")

    if tps_runs:
        result_line("Average decode proxy", f"{statistics.mean(tps_runs):.1f}", "tok/s")
        result_line("Peak decode proxy", f"{max(tps_runs):.1f}", "tok/s")
    if ttft_runs:
        result_line("Average TTFT", f"{statistics.mean(ttft_runs):.0f}", "ms")


def test_concurrency(args):
    header("2. CONCURRENCY SCALING")
    for streams in args.concurrency:
        results = []

        def worker():
            prompt = "Provide an architectural overview of modern transformer KV cache compression techniques."
            results.append(
                stream_chat(args.host, args.port, args.model, prompt, args.max_tokens, args.reasoning_effort)
            )

        started = time.perf_counter()
        threads = [threading.Thread(target=worker) for _ in range(streams)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.perf_counter() - started

        ok = [r for r in results if r[4] is None]
        total_tokens = sum(r[2] for r in ok)
        agg = total_tokens / elapsed if elapsed > 0 else 0.0
        color = "green" if len(ok) == streams else "yellow"
        print(f"  {streams} stream(s): {c(f'{agg:.1f} aggregate tok/s', color)} ({len(ok)}/{streams} OK in {elapsed:.1f}s)")


def test_context(args):
    header("3. CONTEXT WINDOW SCALING SWEEP")
    for target in args.contexts:
        prompt = make_prompt(target)
        print(f"  Testing ~{target:,} token prompt... ", end="", flush=True)
        started = time.perf_counter()
        ttft, tps, total_chunks, _, err = stream_chat(
            args.host, args.port, args.model, prompt, 64, args.reasoning_effort, timeout=args.timeout
        )
        elapsed = time.perf_counter() - started
        if err:
            print(c(f"FAILED ({err[:120]})", "red"))
            break
        print(c(f"OK TTFT={ttft:.0f}ms decode={tps:.1f} tok/s elapsed={elapsed:.1f}s", "green"))


def main():
    parser = argparse.ArgumentParser(description="DGX Spark Cogni-Brain (GLM-5.3-Flash) benchmark")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Cogni-Brain")
    parser.add_argument("--reasoning-effort", choices=["low", "high", "max"], default="max")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--concurrency", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--contexts", type=int, nargs="*", default=[1024, 4096, 8192, 16384])
    args = parser.parse_args()

    header("DGX SPARK / COGNI-BRAIN (GLM-5.3-FLASH) BENCHMARK")
    result_line("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    result_line("Endpoint", f"http://{args.host}:{args.port}/v1")
    result_line("Served Model", args.model)
    result_line("Reasoning effort", args.reasoning_effort)

    test_baseline(args)
    test_concurrency(args)
    test_context(args)


if __name__ == "__main__":
    main()
