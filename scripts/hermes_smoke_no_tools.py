#!/usr/bin/env python3
"""Smoke-test Hermes against the local llama.cpp OpenAI-compatible endpoint.

This intentionally disables Hermes tools. It verifies provider/model wiring
without paying the large prompt cost of agent tool schemas on a small iGPU
model.
"""

from __future__ import annotations

import argparse
import os

from hermes_cli.oneshot import _run_agent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="gemma-4-E4B-it-Q4_K_M")
    parser.add_argument("--prompt", default="Reply with exactly: HERMES_OK")
    args = parser.parse_args()

    os.environ.setdefault("CUSTOM_BASE_URL", args.base_url)
    os.environ.setdefault("OPENAI_API_KEY", "no-key-required")

    response = _run_agent(
        args.prompt,
        model=args.model,
        provider="custom",
        toolsets=["__igpu_no_tools__"],
        use_config_toolsets=False,
    )
    print(response or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
