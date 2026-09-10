#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""
Run tool-eval-bench against the local Cogni-Brain (GLM-5.3-Flash) endpoint.

Modes:
  full  — Full 69-scenario tool-eval-bench suite (community standard, default).
  short — Quick 15-scenario subset for smoke testing.
  perf  — Full suite with embedded llama-benchy throughput profiling.
  trials — Repeated runs for statistical variance analysis.

Academic quality benchmarks (optional, combinable):
  --gsm8k    — Grade School Math (8-shot CoT)
  --mmlu     — Massive Multitask Language Understanding (5-shot)
  --ifeval   — Instruction Following Evaluation

Speculative decoding benchmark:
  --spec-bench — MTP speculative decoding acceptance rate and effective tok/s

Usage:
  # Full 69-scenario suite (default, recommended):
  uv run benchmark/benchmark_smarts.py

  # Quick 15-scenario smoke test:
  uv run benchmark/benchmark_smarts.py --mode short

  # Full suite + all academic benchmarks + speculative decoding + throughput:
  uv run benchmark/benchmark_smarts.py --perf --gsm8k --mmlu --ifeval --spec-bench

  # Statistical trials with seed:
  uv run benchmark/benchmark_smarts.py --mode trials --seed 42 --trials 3
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime


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


def build_command(args):
    command = [
        "uv",
        "tool",
        "run",
        "--from",
        "tool-eval-bench",
        "tool-eval-bench",
        "--base-url",
        args.base_url,
        "--model",
        args.model,
    ]

    # Mode selection
    if args.mode == "short":
        command.append("--short")
    elif args.mode == "perf":
        command.append("--perf")
    elif args.mode == "trials":
        command.extend(["--seed", str(args.seed), "--trials", str(args.trials)])
    # mode == "full" passes no mode flag (full suite is the default)

    # Academic quality benchmarks (optional, combinable)
    if args.gsm8k:
        command.append("--gsm8k")
        if args.gsm8k_limit:
            command.extend(["--gsm8k-limit", str(args.gsm8k_limit)])

    if args.mmlu:
        command.append("--mmlu")
        if args.mmlu_limit:
            command.extend(["--mmlu-limit", str(args.mmlu_limit)])

    if args.ifeval:
        command.append("--ifeval")
        if args.ifeval_limit:
            command.extend(["--ifeval-limit", str(args.ifeval_limit)])

    # Speculative decoding benchmark
    if args.spec_bench:
        command.append("--spec-bench")

    # Performance profiling (also available as standalone flag outside perf mode)
    if args.perf and args.mode != "perf":
        command.append("--perf")

    # Timeout override (important for slow reasoning models)
    if args.timeout:
        command.extend(["--timeout", str(args.timeout)])

    return command


def main():
    parser = argparse.ArgumentParser(
        description="Cogni-Brain (GLM-5.3-Flash) tool-eval benchmark",
        epilog=(
            "Examples:\n"
            "  Full suite:          uv run benchmark/benchmark_smarts.py\n"
            "  Quick smoke test:    uv run benchmark/benchmark_smarts.py --mode short\n"
            "  Full + academics:    uv run benchmark/benchmark_smarts.py --gsm8k --mmlu --ifeval\n"
            "  Full + everything:   uv run benchmark/benchmark_smarts.py --perf --gsm8k --mmlu --ifeval --spec-bench\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Cogni-Brain")
    parser.add_argument(
        "--mode",
        choices=["full", "short", "perf", "trials"],
        default="full",
        help="Benchmark mode: full (69 scenarios, default), short (15), perf (full + throughput), trials (repeated runs)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trials mode")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials for trials mode")

    # Academic quality benchmarks
    academic = parser.add_argument_group("academic quality benchmarks")
    academic.add_argument("--gsm8k", action="store_true", help="Run GSM8K math reasoning benchmark (8-shot CoT)")
    academic.add_argument("--gsm8k-limit", type=int, default=None, help="Limit number of GSM8K questions (default: full)")
    academic.add_argument("--mmlu", action="store_true", help="Run MMLU knowledge benchmark (5-shot)")
    academic.add_argument("--mmlu-limit", type=int, default=None, help="Limit number of MMLU questions (default: full)")
    academic.add_argument("--ifeval", action="store_true", help="Run IFEval instruction following benchmark")
    academic.add_argument("--ifeval-limit", type=int, default=None, help="Limit number of IFEval prompts (default: full)")

    # Speculative decoding
    parser.add_argument("--spec-bench", action="store_true", help="Run MTP speculative decoding benchmark")

    # Performance
    parser.add_argument("--perf", action="store_true", help="Embed llama-benchy throughput profiling in the run")
    parser.add_argument("--timeout", type=int, default=None, help="Per-turn timeout in seconds (default: tool-eval-bench default)")

    args = parser.parse_args()

    # Build feature list for display
    features = [args.mode]
    if args.gsm8k:
        features.append("gsm8k")
    if args.mmlu:
        features.append("mmlu")
    if args.ifeval:
        features.append("ifeval")
    if args.spec_bench:
        features.append("spec-bench")
    if args.perf or args.mode == "perf":
        features.append("perf")

    print(c("=" * 72, "cyan"))
    print(c(f"  Cogni-Brain (GLM-5.3-Flash) Tool Eval Benchmark", "bold"))
    print(c("=" * 72, "cyan"))
    print(f"  Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Base URL:   {args.base_url}")
    print(f"  Model:      {args.model}")
    print(f"  Mode:       {args.mode} ({'69 scenarios' if args.mode != 'short' else '15 scenarios'})")
    print(f"  Features:   {', '.join(features)}")
    if args.timeout:
        print(f"  Timeout:    {args.timeout}s per turn")
    print()

    if args.mode == "full":
        print(c("  Running the full 69-scenario suite (community standard).", "dim"))
        print(c("  This takes ~30–50 min on DGX Spark. Run in tmux for long sessions.", "dim"))
        if args.gsm8k or args.mmlu or args.ifeval:
            print(c("  Academic benchmarks add ~15–60 min depending on limits.", "dim"))
        print()

    if shutil.which("uv") is None:
        print(c("ERROR: uv is not installed or not on PATH", "red"))
        sys.exit(1)

    command = build_command(args)
    print(f"  {c('Command:', 'dim')} {' '.join(command)}")
    print()

    try:
        completed = subprocess.run(command, check=False)
    except KeyboardInterrupt:
        print(f"\n{c('Benchmark interrupted by user', 'yellow')}")
        sys.exit(130)
    except FileNotFoundError:
        print(f"\n{c('Failed to start uv', 'red')}")
        sys.exit(1)

    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
