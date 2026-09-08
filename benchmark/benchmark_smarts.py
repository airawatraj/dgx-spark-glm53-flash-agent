#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""
Run tool-eval-bench against the local GLM-5.3-Flash OpenAI-compatible endpoint.

Usage:
  uv run benchmark/benchmark_smarts.py
  uv run benchmark/benchmark_smarts.py --mode trials --seed 42 --trials 3
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime


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
    ]
    if args.mode == "short":
        command.append("--short")
    elif args.mode == "perf":
        command.append("--perf")
    elif args.mode == "trials":
        command.extend(["--seed", str(args.seed), "--trials", str(args.trials)])
    return command


def main():
    parser = argparse.ArgumentParser(description="GLM-5.3-Flash tool-eval benchmark")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--mode", choices=["short", "perf", "trials"], default="short")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    print("=" * 72)
    print("GLM-5.3-Flash Smarts / Tool Eval")
    print("=" * 72)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base URL:  {args.base_url}")
    print(f"Mode:      {args.mode}")

    if shutil.which("uv") is None:
        print("ERROR: uv is not installed or not on PATH")
        sys.exit(1)

    command = build_command(args)
    print("Command:   " + " ".join(command))
    print()

    completed = subprocess.run(command, check=False)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()

