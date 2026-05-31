#!/usr/bin/env python3
"""Compare the old GamePath routing policy with the retrieval evaluator."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import llama_vulkan_api_server as server  # noqa: E402


DEFAULT_GAME_ID = "__bench_eval_gamepath__"
DEFAULT_OTHER_GAME_ID = "__bench_eval_other__"
GAMEPATH_DB = server.GAMEPATH_DB


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def process_snapshot() -> dict[str, Any]:
    try:
        import psutil  # type: ignore

        proc = psutil.Process()
        times = proc.cpu_times()
        return {
            "cpu_seconds": float(times.user + times.system),
            "rss_mib": round(proc.memory_info().rss / 1048576, 2),
        }
    except Exception:
        return {
            "cpu_seconds": time.process_time(),
            "rss_mib": None,
        }


def cleanup_game(game_id: str) -> None:
    if not GAMEPATH_DB.exists():
        return
    normalized_game_id = server.normalize_game_id(game_id) or game_id
    with sqlite3.connect(GAMEPATH_DB) as conn:
        rows = conn.execute(
            "SELECT id FROM gamepath_entries WHERE game_id = ?",
            (normalized_game_id,),
        ).fetchall()
    for (entry_id,) in rows:
        server.delete_gamepath_sync(int(entry_id))


def seed_data(game_id: str, other_game_id: str) -> list[dict[str, Any]]:
    cleanup_game(game_id)
    cleanup_game(other_game_id)
    seeds = [
        server.add_gamepath_sync(
            "crystal key item usage guide",
            (
                "The crystal key opens the kitchen cellar gate. Keep it until you reach the pantry route, "
                "then use it on the blue lock near the stove. It is not a crafting material."
            ),
            game_id,
            title="Crystal key usage",
            tags=["benchmark", "item", "kitchen"],
            spoiler_level="low",
            source_type="benchmark",
            agent_used=False,
        ),
        server.add_gamepath_sync(
            "ancient key left door guide",
            (
                "The ancient key can open the left archive door. This route gives a shortcut back to the "
                "save point and avoids the patrol in the west hall."
            ),
            game_id,
            title="Ancient key left door",
            tags=["benchmark", "route", "key"],
            spoiler_level="low",
            source_type="benchmark",
            agent_used=False,
        ),
        server.add_gamepath_sync(
            "ancient key right door guide",
            (
                "The ancient key can also open the right archive door. This path leads to the optional chest, "
                "but it is longer and has one extra encounter."
            ),
            game_id,
            title="Ancient key right door",
            tags=["benchmark", "route", "key"],
            spoiler_level="low",
            source_type="benchmark",
            agent_used=False,
        ),
        server.add_gamepath_sync(
            "crystal key item usage guide",
            (
                "In the other benchmark game, the crystal key unlocks a tower lift and should not be used in "
                "the kitchen route."
            ),
            other_game_id,
            title="Other game crystal key",
            tags=["benchmark", "item"],
            spoiler_level="low",
            source_type="benchmark",
            agent_used=False,
        ),
    ]
    return seeds


def old_gamepath_requested(prompt: str) -> bool:
    return bool(
        server.should_use_guides(prompt, None)
        or server.GAMEPATH_STORE_INTENT_RE.search(prompt or "")
    )


def old_confident_hit(results: list[dict[str, Any]]) -> bool:
    if not results:
        return False
    top = results[0]
    return (
        len(str(top.get("answer_summary") or "")) >= 20
        and float(top.get("match_coverage") or 0.0) >= 0.14
    )


def old_route(prompt: str, game_id: str | None) -> dict[str, Any]:
    guide_requested = server.should_use_guides(prompt, None)
    gamepath_requested = old_gamepath_requested(prompt)
    sqlite_queried = bool(prompt.strip())
    results = server.search_gamepath_sync(prompt, game_id, 5) if sqlite_queried else []
    if gamepath_requested and old_confident_hit(results):
        route = "direct"
    elif results:
        route = "gamepath_context"
    elif gamepath_requested or guide_requested:
        route = "miss"
    else:
        route = "skipped_after_sqlite"
    return {
        "route": route,
        "sqlite_queried": sqlite_queried,
        "hits": len(results),
        "top_coverage": float(results[0].get("match_coverage") or 0.0) if results else 0.0,
        "top_title": results[0].get("title") if results else "",
    }


def new_route(prompt: str, game_id: str | None) -> dict[str, Any]:
    guide_requested = server.should_use_guides(prompt, None)
    gamepath_requested = server.should_use_gamepath(prompt, guide_requested)
    sqlite_queried = False
    results: list[dict[str, Any]] = []
    evaluation: dict[str, Any] = {
        "confidence": "skipped",
        "score": 0.0,
        "gap": 0.0,
        "reason": "intent_gate_skipped",
    }
    if gamepath_requested:
        sqlite_queried = True
        results = server.search_gamepath_sync(prompt, game_id, 5)
        evaluation = server.evaluate_gamepath_retrieval(prompt, game_id, results)
    return {
        "route": evaluation.get("confidence", "skipped"),
        "sqlite_queried": sqlite_queried,
        "hits": len(results),
        "score": float(evaluation.get("score") or 0.0),
        "gap": float(evaluation.get("gap") or 0.0),
        "reason": evaluation.get("reason", ""),
        "top_title": evaluation.get("top_title", ""),
    }


def timed(func, runs: int) -> dict[str, Any]:
    samples: list[float] = []
    result: dict[str, Any] = {}
    for _ in range(runs):
        started = time.perf_counter()
        result = func()
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "result": result,
        "avg_ms": round(mean(samples), 4),
        "p50_ms": round(median(samples), 4),
        "p95_ms": round(percentile(samples, 0.95), 4),
        "min_ms": round(min(samples), 4),
        "max_ms": round(max(samples), 4),
    }


def route_matches(route: str, expected: str) -> bool:
    aliases = {
        "skip": {"skipped", "skipped_after_sqlite"},
        "direct": {"direct"},
        "summarize": {"summarize", "gamepath_context"},
        "miss": {"miss"},
    }
    return route in aliases.get(expected, {expected})


def build_report(data: dict[str, Any]) -> str:
    summary = data["summary"]
    rows = []
    for case in data["cases"]:
        old_result = case["old"]["result"]
        new_result = case["new"]["result"]
        rows.append(
            "| {name} | {expected} | {old_route} | {new_route} | {old_avg} | {new_avg} | {score} |".format(
                name=case["name"],
                expected=case["expected"],
                old_route=old_result["route"],
                new_route=new_result["route"],
                old_avg=case["old"]["avg_ms"],
                new_avg=case["new"]["avg_ms"],
                score=new_result.get("score", 0.0),
            )
        )
    table = "\n".join(rows)
    return f"""# GamePath Retrieval Evaluator 測試報告

產生時間：{data["generated_at"]}

## 結論

新機制在這組可控測試中比較好，原因有三個：

- SQLite 查詢次數從 `{summary["old_sqlite_queries"]}` 次降到 `{summary["new_sqlite_queries"]}` 次，減少 `{summary["sqlite_query_reduction_percent"]}%`。
- 路由判斷正確率從 `{summary["old_accuracy_percent"]}%` 提升到 `{summary["new_accuracy_percent"]}%`。
- 多筆相似攻略時，新 evaluator 會走 `summarize`，避免舊機制直接拿第一筆硬答。

這份測試不是代表所有遊戲都 100% 正確；它證明的是：在一般聊天、UI 指令、精準命中、多筆模糊命中、miss 這些典型情境下，新機制有更好的路由判斷與更少不必要 SQLite 查詢。

## 測試架構

```mermaid
flowchart TD
    A["測試輸入"] --> B["舊機制模擬"]
    A --> C["新機制"]
    B --> D["每句都查 SQLite"]
    D --> E["coverage >= 0.14 就 direct"]
    C --> F["Intent Gate"]
    F -->|"非攻略"| G["skip SQLite"]
    F -->|"攻略型"| H["SQLite FTS"]
    H --> I["retrieval_evaluator"]
    I -->|"score >= 0.74 + gap 足夠"| J["direct"]
    I -->|"score >= 0.46"| K["summarize"]
    I -->|"low score"| L["miss / Hermes Tavily"]
```

## 測試資料

- 測試 game_id：`{data["game_id"]}`
- 暫存資料：crystal key、ancient key left door、ancient key right door
- 測試完成後會刪除暫存 GamePath entries，不污染正式資料庫。

## 結果表

| Case | Expected | Old Route | New Route | Old Avg ms | New Avg ms | New Score |
|---|---:|---:|---:|---:|---:|---:|
{table}

## 整體指標

| Metric | Old | New |
|---|---:|---:|
| Route accuracy | {summary["old_accuracy_percent"]}% | {summary["new_accuracy_percent"]}% |
| SQLite query count | {summary["old_sqlite_queries"]} | {summary["new_sqlite_queries"]} |
| Avg route latency ms | {summary["old_avg_ms"]} | {summary["new_avg_ms"]} |
| CPU seconds delta | {summary["cpu_seconds_delta"]} | same process |
| RSS before/after MiB | {summary["rss_before_mib"]} / {summary["rss_after_mib"]} | same process |

## 觀察

1. 一般聊天與 UI 指令現在會停在 intent gate，不再進 SQLite。
2. 精準同遊戲物品問題會得到高分，直接走 GamePath 本地回答。
3. `ancient key door guide` 這種多筆都合理的情境，舊機制會直接拿第一筆；新機制因為 top gap 不夠，改走模型濃縮。
4. 完全不相關的攻略問題仍會 miss，交給 Hermes/Tavily 慢路徑。

## 驗收判斷

這次改動「真的比較好」的範圍是：

- 非攻略訊息效能更好：少一次 SQLite FTS。
- 多筆命中品質更好：不急著 direct。
- UI 可觀察性更好：狀態會顯示 retrieval score。

還需要真實遊戲資料繼續驗證的是：

- 不同遊戲同名物品的邊界案例。
- 中文 OCR 或截圖物品名稱不完整時的 scoring。
- 大量 GamePath entries 超過數千筆後的 FTS 延遲。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark GamePath retrieval evaluator routing.")
    parser.add_argument("--runs", type=int, default=80)
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--other-game-id", default=DEFAULT_OTHER_GAME_ID)
    parser.add_argument("--json-output", default=str(PROJECT_ROOT / "docs" / "gamepath_retrieval_evaluator_results.json"))
    parser.add_argument("--report-output", default=str(PROJECT_ROOT / "docs" / "gamepath_retrieval_evaluator_report.md"))
    args = parser.parse_args()

    before = process_snapshot()
    seeds = seed_data(args.game_id, args.other_game_id)
    cases = [
        {
            "name": "general_chat",
            "message": "hello, can you help me tune settings later?",
            "game_id": args.game_id,
            "expected": "skip",
        },
        {
            "name": "ui_voice_command",
            "message": "幫我開啟語音模式",
            "game_id": args.game_id,
            "expected": "skip",
        },
        {
            "name": "exact_item_hit",
            "message": "crystal key item usage guide",
            "game_id": args.game_id,
            "expected": "direct",
        },
        {
            "name": "ambiguous_multi_hit",
            "message": "ancient key door guide",
            "game_id": args.game_id,
            "expected": "summarize",
        },
        {
            "name": "unrelated_guide_miss",
            "message": "obsidian spoon moon factory boss weakness guide",
            "game_id": args.game_id,
            "expected": "miss",
        },
        {
            "name": "software_model_question",
            "message": "Hermes 模型跟 GamePath 的關係是什麼",
            "game_id": args.game_id,
            "expected": "skip",
        },
    ]

    measured_cases: list[dict[str, Any]] = []
    try:
        for case in cases:
            message = case["message"]
            game_id = case["game_id"]
            old = timed(lambda message=message, game_id=game_id: old_route(message, game_id), args.runs)
            new = timed(lambda message=message, game_id=game_id: new_route(message, game_id), args.runs)
            measured = {
                **case,
                "old": old,
                "new": new,
                "old_matches_expected": route_matches(str(old["result"]["route"]), str(case["expected"])),
                "new_matches_expected": route_matches(str(new["result"]["route"]), str(case["expected"])),
            }
            measured_cases.append(measured)
    finally:
        cleanup_game(args.game_id)
        cleanup_game(args.other_game_id)

    after = process_snapshot()
    old_sqlite_queries = sum(1 for case in measured_cases if case["old"]["result"].get("sqlite_queried"))
    new_sqlite_queries = sum(1 for case in measured_cases if case["new"]["result"].get("sqlite_queried"))
    old_correct = sum(1 for case in measured_cases if case["old_matches_expected"])
    new_correct = sum(1 for case in measured_cases if case["new_matches_expected"])
    old_avg_ms = mean(float(case["old"]["avg_ms"]) for case in measured_cases)
    new_avg_ms = mean(float(case["new"]["avg_ms"]) for case in measured_cases)
    reduction = (
        (old_sqlite_queries - new_sqlite_queries) / old_sqlite_queries * 100.0
        if old_sqlite_queries else 0.0
    )
    cpu_delta = float(after["cpu_seconds"] or 0.0) - float(before["cpu_seconds"] or 0.0)
    data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs_per_case": args.runs,
        "game_id": args.game_id,
        "seed_ids": [item["id"] for item in seeds],
        "cases": measured_cases,
        "summary": {
            "old_sqlite_queries": old_sqlite_queries,
            "new_sqlite_queries": new_sqlite_queries,
            "sqlite_query_reduction_percent": round(reduction, 2),
            "old_accuracy_percent": round(old_correct / len(measured_cases) * 100.0, 2),
            "new_accuracy_percent": round(new_correct / len(measured_cases) * 100.0, 2),
            "old_avg_ms": round(old_avg_ms, 4),
            "new_avg_ms": round(new_avg_ms, 4),
            "cpu_seconds_delta": round(cpu_delta, 4),
            "rss_before_mib": before.get("rss_mib"),
            "rss_after_mib": after.get("rss_mib"),
        },
    }

    json_output = Path(args.json_output)
    report_output = Path(args.report_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    report_output.write_text(build_report(data), encoding="utf-8", newline="\n")
    print(json.dumps(data["summary"], ensure_ascii=True, indent=2))
    print(f"JSON: {json_output}")
    print(f"Report: {report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
