#!/usr/bin/env python3
"""Run one Hermes agent turn with the web toolset enabled."""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import subprocess
import sys
from typing import Any

from hermes_cli.oneshot import _run_agent


def detect_windows_host_ip() -> str:
    route = subprocess.check_output(["ip", "route"], text=True, timeout=5)
    for line in route.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "default" and parts[1] == "via":
            return parts[2]
    raise RuntimeError("Could not detect Windows host IP from WSL default route.")


def parse_toolsets(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


def image_file_to_data_url(path: str) -> str:
    with open(path, "rb") as file:
        data = file.read()
    mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--model", default="qwen3.5-9b-q4_k_m")
    parser.add_argument("--toolsets", default=os.environ.get("HERMES_AGENT_TOOLSETS", "web"))
    parser.add_argument("--prompt", default="")
    parser.add_argument("--image-file", default="")
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("HERMES_AGENT_MAX_TOKENS", "360")))
    parser.add_argument("--context-length", type=int, default=int(os.environ.get("HERMES_CONTEXT_LENGTH", "65536")))
    parser.add_argument("--api-timeout", type=int, default=int(os.environ.get("HERMES_API_TIMEOUT", "900")))
    parser.add_argument(
        "--api-call-stale-timeout",
        type=int,
        default=int(os.environ.get("HERMES_API_CALL_STALE_TIMEOUT", "900")),
    )
    args = parser.parse_args()

    prompt = args.prompt or sys.stdin.read()
    if not prompt.strip():
        raise SystemExit("Prompt is empty.")

    base_url = args.base_url.strip()
    use_config_model = os.environ.get("HERMES_USE_CONFIG_MODEL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not base_url and not use_config_model:
        base_url = f"http://{detect_windows_host_ip()}:{args.api_port}/v1"

    os.environ["HERMES_API_TIMEOUT"] = str(args.api_timeout)
    os.environ["HERMES_API_CALL_STALE_TIMEOUT"] = str(args.api_call_stale_timeout)
    os.environ.setdefault("HERMES_YOLO_MODE", "1")
    os.environ.setdefault("HERMES_ACCEPT_HOOKS", "1")

    if use_config_model:
        model = None
        provider = None
    else:
        os.environ["CUSTOM_BASE_URL"] = base_url
        os.environ.setdefault("OPENAI_API_KEY", "no-key-required")
        apply_runtime_config_overrides(base_url, args.model, args.max_tokens, args.context_length)
        model = args.model
        provider = "custom"

    agent_prompt: Any = prompt
    image_file = args.image_file.strip()
    if image_file:
        agent_prompt = [
            {"type": "image_url", "image_url": {"url": image_file_to_data_url(image_file)}},
            {"type": "text", "text": prompt},
        ]

    response = _run_agent(
        agent_prompt,
        model=model,
        provider=provider,
        toolsets=parse_toolsets(args.toolsets),
        use_config_toolsets=False,
    )
    print(response or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
