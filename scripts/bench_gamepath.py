#!/usr/bin/env python3
"""Benchmark GamePath local RAG storage and optional Hermes-agent chat paths."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import mean, median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_GAME_ID = "benchmark_gamepath"
GAMEPATH_DB = Path(os.environ.get("IGPU_GAMEPATH_DIR", str(PROJECT_ROOT / "gamepath"))) / "gamepath.sqlite"


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(base_url: str, path: str, timeout: int = 10) -> dict[str, Any]:
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def stream_chat(base_url: str, payload: dict[str, Any], timeout: int) -> str:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks: list[str] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if "content" in event:
                chunks.append(str(event["content"]))
    return "".join(chunks)


def find_backend_pid(port: int) -> int | None:
    command = (
        f"$c = Get-NetTCPConnection -LocalPort {int(port)} -State Listen -ErrorAction SilentlyContinue | "
        "Select-Object -First 1; if ($c) { $c.OwningProcess }"
    )
    try:
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", command],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        return int(output) if output else None
    except Exception:
        return None


def process_metrics(pid: int | None) -> dict[str, float | int | None]:
    if not pid:
        return {"pid": None, "cpu_seconds": None, "rss_bytes": None}
    try:
        import psutil  # type: ignore

        proc = psutil.Process(pid)
        times = proc.cpu_times()
        return {
            "pid": pid,
            "cpu_seconds": float(times.user + times.system),
            "rss_bytes": int(proc.memory_info().rss),
        }
    except Exception:
        pass

    command = (
        f"$p = Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue; "
        "if ($p) { [pscustomobject]@{ CPU=$p.CPU; WorkingSet64=$p.WorkingSet64 } | ConvertTo-Json -Compress }"
    )
    try:
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", command],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        if not output:
            raise RuntimeError("No process output")
        data = json.loads(output)
        return {
            "pid": pid,
            "cpu_seconds": float(data.get("CPU") or 0.0),
            "rss_bytes": int(data.get("WorkingSet64") or 0),
        }
    except Exception:
        return {"pid": pid, "cpu_seconds": None, "rss_bytes": None}


def db_size() -> int:
    return GAMEPATH_DB.stat().st_size if GAMEPATH_DB.exists() else 0


def timed_case(name: str, func, pid: int | None) -> dict[str, Any]:
    before = process_metrics(pid)
    before_size = db_size()
    started = time.perf_counter()
    error = None
    detail: dict[str, Any] = {}
    try:
        detail = func() or {}
    except Exception as exc:
        error = repr(exc)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    after = process_metrics(pid)
    cpu_before = before.get("cpu_seconds")
    cpu_after = after.get("cpu_seconds")
    cpu_delta = (
        float(cpu_after) - float(cpu_before)
        if cpu_before is not None and cpu_after is not None
        else None
    )
    cpu_percent = (cpu_delta / (elapsed_ms / 1000.0) * 100.0) if cpu_delta is not None and elapsed_ms > 0 else None
    return {
        "case": name,
        "elapsed_ms": round(elapsed_ms, 2),
        "cpu_seconds_delta": round(cpu_delta, 4) if cpu_delta is not None else None,
        "cpu_percent_est": round(cpu_percent, 2) if cpu_percent is not None else None,
        "rss_before_mib": round((int(before.get("rss_bytes") or 0) / 1048576), 2) if before.get("rss_bytes") else None,
        "rss_after_mib": round((int(after.get("rss_bytes") or 0) / 1048576), 2) if after.get("rss_bytes") else None,
        "db_size_before_bytes": before_size,
        "db_size_after_bytes": db_size(),
        "error": error,
        "detail": detail,
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [sample for sample in samples if not sample.get("error")]
    latencies = [float(sample["elapsed_ms"]) for sample in ok]
    cpu = [float(sample["cpu_seconds_delta"]) for sample in ok if sample.get("cpu_seconds_delta") is not None]
    return {
        "runs": len(samples),
        "ok": len(ok),
        "errors": len(samples) - len(ok),
        "avg_ms": round(mean(latencies), 2) if latencies else None,
        "p50_ms": round(median(latencies), 2) if latencies else None,
        "p95_ms": round(percentile(latencies, 0.95), 2) if latencies else None,
        "min_ms": round(min(latencies), 2) if latencies else None,
        "max_ms": round(max(latencies), 2) if latencies else None,
        "avg_cpu_seconds_delta": round(mean(cpu), 4) if cpu else None,
        "max_rss_after_mib": max((sample.get("rss_after_mib") or 0 for sample in ok), default=None),
    }


def cleanup_benchmark_data(game_id: str) -> None:
    if not GAMEPATH_DB.exists():
        return
    with sqlite3.connect(GAMEPATH_DB) as conn:
        rows = conn.execute(
            "SELECT id, markdown_path FROM gamepath_entries WHERE game_id = ?",
            (game_id,),
        ).fetchall()
        for entry_id, markdown_path in rows:
            conn.execute("DELETE FROM gamepath_fts WHERE entry_id = ?", (entry_id,))
            conn.execute("DELETE FROM gamepath_entries WHERE id = ?", (entry_id,))
            if markdown_path:
                path = (PROJECT_ROOT / markdown_path).resolve() if not Path(markdown_path).is_absolute() else Path(markdown_path)
                try:
                    if path.exists() and PROJECT_ROOT.resolve() in path.resolve().parents:
                        path.unlink()
                except Exception:
                    pass
        conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark GamePath local DB and optional Hermes-agent paths.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--include-agent", action="store_true")
    parser.add_argument("--agent-timeout", type=int, default=180)
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "docs" / "gamepath_benchmark_results.json"))
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    port = int(base_url.rsplit(":", 1)[-1].split("/", 1)[0]) if ":" in base_url.rsplit("/", 1)[-1] else 8000
    pid = args.pid or find_backend_pid(port)
    result: dict[str, Any] = {
        "base_url": base_url,
        "backend_pid": pid,
        "runs_per_case": args.runs,
        "game_id": args.game_id,
        "health": {},
        "samples": {},
        "summary": {},
    }

    agent_game_id = f"{args.game_id}_agent_miss"

    try:
        result["health"] = get_json(base_url, "/health")
    except Exception as exc:
        result["health_error"] = repr(exc)

    if not args.keep_data:
        cleanup_benchmark_data(args.game_id)
        cleanup_benchmark_data(agent_game_id)

    seed_question = "GamePath benchmark item usage guide"
    seed_answer = "This benchmark hint explains that the test item is used to validate local GamePath read/write behavior without invoking Hermes."

    cases: list[tuple[str, Any]] = []
    cases.append(
        (
            "A_no_agent_write_api",
            lambda: post_json(
                base_url,
                "/gamepath/add",
                {
                    "game_id": args.game_id,
                    "question": f"{seed_question} {time.time_ns()}",
                    "title": "Benchmark item usage",
                    "answer_summary": seed_answer,
                    "tags": ["benchmark", "item", "usage"],
                    "spoiler_level": "none",
                    "source_type": "benchmark",
                    "agent_used": False,
                },
            ).get("item", {}),
        )
    )
    cases.append(
        (
            "B_no_agent_read_api",
            lambda: post_json(
                base_url,
                "/gamepath/search",
                {
                    "game_id": args.game_id,
                    "query": "benchmark item usage guide",
                    "spoiler_level": "low",
                    "limit": 5,
                },
            ),
        )
    )
    if args.include_agent:
        cases.append(
            (
                "C_agent_miss_chat_write",
                lambda: {
                    "content": stream_chat(
                        base_url,
                        {
                            "game_id": agent_game_id,
                            "message": f"請幫我去網路上查一下攻略：synthetic agent-miss item {time.time_ns()} 用途，請給無暴雷提示",
                            "use_guides": True,
                            "use_memory": True,
                        },
                        args.agent_timeout,
                    )[:500]
                },
            )
        )
        cases.append(
            (
                "D_agent_enabled_local_hit_chat",
                lambda: {
                    "content": stream_chat(
                        base_url,
                        {
                            "game_id": args.game_id,
                            "message": "benchmark item usage guide 這個物品用途是什麼？",
                            "use_guides": True,
                            "use_memory": True,
                        },
                        args.agent_timeout,
                    )[:500]
                },
            )
        )

    for name, func in cases:
        result["samples"][name] = []
        for _ in range(max(1, args.runs)):
            sample = timed_case(name, func, pid)
            result["samples"][name].append(sample)
        result["summary"][name] = summarize(result["samples"][name])

    try:
        result["final_health"] = get_json(base_url, "/health")
    except Exception as exc:
        result["final_health_error"] = repr(exc)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Benchmark results written to {output}")

    if not args.keep_data:
        cleanup_benchmark_data(args.game_id)
        cleanup_benchmark_data(agent_game_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
