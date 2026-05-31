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


def apply_runtime_config_overrides(base_url: str, model: str, max_tokens: int, context_length: int) -> None:
    import hermes_cli.config as hermes_config

    original_load_config = hermes_config.load_config

    def load_config_with_igpu_overrides(*args, **kwargs):
        config = original_load_config(*args, **kwargs)
        model_config = config.get("model")
        if not isinstance(model_config, dict):
            model_config = {}
        model_config.update(
            {
                "provider": "custom",
                "base_url": base_url,
                "default": model,
                "api_key": "no-key-required",
                "api_mode": "chat_completions",
            }
        )
        if max_tokens > 0:
            model_config["max_tokens"] = max_tokens
        if context_length > 0:
            model_config["context_length"] = context_length
        config["model"] = model_config
        return config

    hermes_config.load_config = load_config_with_igpu_overrides


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="qwen3.5-9b-q4_k_m")
    parser.add_argument("--prompt", default="Reply with exactly: HERMES_OK")
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("HERMES_MAX_TOKENS", "160")))
    parser.add_argument("--context-length", type=int, default=int(os.environ.get("HERMES_CONTEXT_LENGTH", "65536")))
    parser.add_argument("--api-timeout", type=int, default=int(os.environ.get("HERMES_API_TIMEOUT", "600")))
    parser.add_argument(
        "--api-call-stale-timeout",
        type=int,
        default=int(os.environ.get("HERMES_API_CALL_STALE_TIMEOUT", "600")),
    )
    args = parser.parse_args()

    os.environ.setdefault("CUSTOM_BASE_URL", args.base_url)
    os.environ.setdefault("OPENAI_API_KEY", "no-key-required")
    os.environ["HERMES_API_TIMEOUT"] = str(args.api_timeout)
    os.environ["HERMES_API_CALL_STALE_TIMEOUT"] = str(args.api_call_stale_timeout)

    apply_runtime_config_overrides(args.base_url, args.model, args.max_tokens, args.context_length)

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
