#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""Sweep GLM-5.3-Flash reasoning_effort values against one endpoint."""

import argparse
import json
import time

import requests


def run_once(base_url, model, effort, prompt):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 512,
        "stream": False,
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    started = time.perf_counter()
    response = requests.post(f"{base_url}/chat/completions", json=payload, timeout=600)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"].get("content", "")
    usage = data.get("usage", {})
    return elapsed, usage, content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="GLM-5.3-Flash")
    parser.add_argument(
        "--prompt",
        default="Design a robust benchmark plan for a 128GB local MoE model server. Be concise.",
    )
    args = parser.parse_args()

    for effort in ["low", "high", "max"]:
        elapsed, usage, content = run_once(args.base_url, args.model, effort, args.prompt)
        print("=" * 72)
        print(f"reasoning_effort={effort} elapsed={elapsed:.2f}s usage={json.dumps(usage)}")
        print("-" * 72)
        print(content[:2000])


if __name__ == "__main__":
    main()

