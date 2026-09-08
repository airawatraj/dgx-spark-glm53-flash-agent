#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""
DGX Spark / GLM-5.3-Flash speed, concurrency, and context benchmark.

Usage:
  uv run benchmark/benchmark_speed.py
  uv run benchmark/benchmark_speed.py --host localhost --port 8000 --model GLM-5.3-Flash
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
    base = ("DGX Spark local inference benchmark. " * 100).split()
    n_words = max(10, int(target_tokens / 1.33))
    words = (base * ((n_words // len(base)) + 1))[:n_words]
    return " ".join(words) + "\n\nSummarize the benchmark setup in one sentence."


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
    }

    t_start = time.perf_counter()
    t_first = None
    chunks = 0
    text_parts = []

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                return None, None, 0, f"HTTP {response.status_code}: {response.text[:200]}"

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
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content") or ""
                if content:
                    if t_first is None:
                        t_first = time.perf_counter()
                    chunks += 1
                    text_parts.append(content)

        t_end = time.perf_counter()
        ttft = (t_first - t_start) * 1000 if t_first else None
        gen_time = (t_end - t_first) if t_first else max(t_end - t_start, 1e-6)
        tps = chunks / gen_time
        return ttft, tps, chunks, None
    except Exception as exc:
        return None, None, 0, str(exc)


def test_baseline(args):
    header("1. BASELINE SINGLE-STREAM CHAT")
    prompt = "Explain the practical tradeoffs of running a 3-bit MoE model on 128GB unified memory."
    tps_runs = []
    ttft_runs = []

    for run in range(1, args.runs + 1):
        ttft, tps, chunks, err = stream_chat(
            args.host, args.port, args.model, prompt, args.max_tokens, args.reasoning_effort
        )
        if err:
            print(f"  Run {run}: {c('FAILED - ' + err, 'red')}")
            continue
        tps_runs.append(tps)
        if ttft is not None:
            ttft_runs.append(ttft)
        print(f"  Run {run}: {c(f'{tps:.1f} chunks/s', 'green')} TTFT={ttft:.0f}ms chunks={chunks}")

    if tps_runs:
        result_line("Average decode proxy", f"{statistics.mean(tps_runs):.1f}", "chunks/s")
        result_line("Peak decode proxy", f"{max(tps_runs):.1f}", "chunks/s")
    if ttft_runs:
        result_line("Average TTFT", f"{statistics.mean(ttft_runs):.0f}", "ms")


def test_concurrency(args):
    header("2. CONCURRENCY SCALING")
    for streams in args.concurrency:
        results = []

        def worker():
            prompt = "Write a concise technical note about long-context inference on local hardware."
            results.append(stream_chat(args.host, args.port, args.model, prompt, args.max_tokens, args.reasoning_effort))

        started = time.perf_counter()
        threads = [threading.Thread(target=worker) for _ in range(streams)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.perf_counter() - started

        ok = [r for r in results if r[3] is None]
        chunks = sum(r[2] for r in ok)
        agg = chunks / elapsed if elapsed else 0.0
        color = "green" if len(ok) == streams else "yellow"
        print(f"  {streams} streams: {c(f'{agg:.1f} aggregate chunks/s', color)} ({len(ok)}/{streams} OK)")


def test_context(args):
    header("3. CONTEXT WINDOW SWEEP")
    for target in args.contexts:
        prompt = make_prompt(target)
        print(f"  Testing ~{target:,} token prompt... ", end="", flush=True)
        started = time.perf_counter()
        ttft, tps, chunks, err = stream_chat(
            args.host, args.port, args.model, prompt, 64, args.reasoning_effort, timeout=args.timeout
        )
        elapsed = time.perf_counter() - started
        if err:
            print(c(f"FAILED ({err[:120]})", "red"))
            break
        print(c(f"OK TTFT={ttft:.0f}ms decode={tps:.1f} chunks/s elapsed={elapsed:.1f}s", "green"))


def main():
    parser = argparse.ArgumentParser(description="DGX Spark GLM-5.3-Flash benchmark")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="GLM-5.3-Flash")
    parser.add_argument("--reasoning-effort", choices=["low", "high", "max"], default="max")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--concurrency", type=int, nargs="*", default=[2, 4])
    parser.add_argument("--contexts", type=int, nargs="*", default=[4096, 16384, 65536, 131072])
    args = parser.parse_args()

    header("DGX SPARK / GLM-5.3-FLASH BENCHMARK")
    result_line("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    result_line("Endpoint", f"http://{args.host}:{args.port}/v1")
    result_line("Model", args.model)
    result_line("Reasoning effort", args.reasoning_effort)

    test_baseline(args)
    test_concurrency(args)
    test_context(args)


if __name__ == "__main__":
    main()

