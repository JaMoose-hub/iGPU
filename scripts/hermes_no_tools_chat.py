#!/usr/bin/env python3
"""Run one Hermes text turn against the local iGPU model with tools disabled."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from hermes_cli.oneshot import _run_agent


def detect_windows_host_ip() -> str:
    route = subprocess.check_output(["ip", "route"], text=True, timeout=5)
    for line in route.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "default" and parts[1] == "via":
            return parts[2]
    raise RuntimeError("Could not detect Windows host IP from WSL default route.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--model", default="gemma-4-E4B-it-Q4_K_M")
    parser.add_argument("--prompt", default="")
    args = parser.parse_args()

    prompt = args.prompt or sys.stdin.read()
    if not prompt.strip():
        raise SystemExit("Prompt is empty.")

    base_url = args.base_url.strip()
    if not base_url:
        base_url = f"http://{detect_windows_host_ip()}:{args.api_port}/v1"

    os.environ["CUSTOM_BASE_URL"] = base_url
    os.environ.setdefault("OPENAI_API_KEY", "no-key-required")

    response = _run_agent(
        prompt,
        model=args.model,
        provider="custom",
        toolsets=["__igpu_no_tools__"],
        use_config_toolsets=False,
    )
    print(response or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
