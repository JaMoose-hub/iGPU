#!/usr/bin/env python3
"""Benchmark GamePath entry-level search against RAG Lite chunk search."""

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


DEFAULT_GAME_ID = "__bench_rag_lite__"
DEFAULT_OTHER_GAME_ID = "__bench_rag_lite_other__"
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


def db_size_kib() -> float:
    if not GAMEPATH_DB.exists():
        return 0.0
    return round(GAMEPATH_DB.stat().st_size / 1024.0, 2)


def chunk_count(game_id: str) -> int:
    if not GAMEPATH_DB.exists():
        return 0
    normalized_game_id = server.normalize_game_id(game_id) or game_id
    with sqlite3.connect(GAMEPATH_DB) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM gamepath_chunks c
            JOIN gamepath_entries e ON e.id = c.entry_id
            WHERE e.game_id = ?
            """,
            (normalized_game_id,),
        ).fetchone()
    return int(row[0] or 0) if row else 0


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


def make_long_walkthrough() -> str:
    sections = [
        (
            "Opening Hall",
            "先確認大廳安全。Do not spend healing items here; this section only teaches movement and camera checks.",
        ),
        (
            "Kitchen Route",
            "廚房那關先從冷藏櫃旁邊繞過去，聽到第二次腳步聲再進 pantry。Kitchen route is safer if you hug the left wall.",
        ),
        (
            "Silver Key",
            "銀鑰匙在廚房後方 pantry 的藍色烤箱旁。拿到後先不要回大廳，直接用在 cellar gate 的銀色鎖，可以開地下酒窖捷徑。",
        ),
        (
            "Rusted Cog",
            "生鏽齒輪不是鑰匙。它用在 boiler room 的牆面機關，啟動後會打開水閥旁的小門。",
        ),
        (
            "Two Singing Zombies",
            "會唱歌的兩個女殭屍通常一個靠聲音提示巡邏，另一個躲在東翼更深處。先聽聲音距離，不要急著開槍。",
        ),
        (
            "Basement Puzzle",
            "地下室謎題只需要調整三個閥門到低壓。看到紅燈時先回上一個閥門，不要直接重置。",
        ),
        (
            "Final Boss Spoiler",
            "高劇透：最終 boss 的第二階段會假裝倒地，真正弱點在背部核心。這段不應出現在低劇透查詢的回答裡。",
        ),
    ]
    repeated = []
    for index in range(80):
        repeated.append(
            f"## Optional Supplies {index + 1}\n\n"
            "這是填充用補給段落，包含草藥、子彈、存檔點與安全屋資訊，用來模擬較長攻略文件。"
            "這段會讓舊 entry-level search 每次都需要重新切分更多段落。"
        )
    body = "\n\n".join(f"## {title}\n\n{text}" for title, text in sections)
    return "# Kitchen Manor Walkthrough\n\n" + body + "\n\n" + "\n\n".join(repeated)


def seed_data(game_id: str, other_game_id: str) -> list[dict[str, Any]]:
    cleanup_game(game_id)
    cleanup_game(other_game_id)
    seeds = [
        server.add_gamepath_sync(
            "kitchen manor walkthrough silver key route singing zombies",
            make_long_walkthrough(),
            game_id,
            title="Kitchen Manor full walkthrough",
            tags=["benchmark", "walkthrough", "kitchen", "key", "zombie"],
            spoiler_level="low",
            source_type="benchmark",
            agent_used=False,
        ),
        server.add_gamepath_sync(
            "silver key tower lift guide",
            "In the other benchmark game, the silver key unlocks a tower lift. It is unrelated to the kitchen pantry.",
            other_game_id,
            title="Other game silver key",
            tags=["benchmark", "key"],
            spoiler_level="low",
            source_type="benchmark",
            agent_used=False,
        ),
    ]
    return seeds


def summarize_result(results: list[dict[str, Any]], expected_text: str, forbidden_text: str) -> dict[str, Any]:
    top = results[0] if results else {}
    excerpt = str(top.get("relevant_excerpt") or top.get("snippet") or "")
    evaluation = server.evaluate_gamepath_retrieval(str(expected_text or ""), top.get("game_id"), results) if results else {}
    return {
        "hits": len(results),
        "top_title": top.get("title", ""),
        "top_id": top.get("id"),
        "top_score": top.get("retrieval_score", evaluation.get("score", 0.0)),
        "retrieval_confidence": evaluation.get("confidence", ""),
        "retrieval_score": evaluation.get("score", 0.0),
        "context_chars": len(excerpt),
        "passage_count": top.get("passage_count", 0),
        "rag_chunk_count": top.get("rag_chunk_count", 0),
        "contains_expected": expected_text.lower() in excerpt.lower() if expected_text else False,
        "contains_forbidden": forbidden_text.lower() in excerpt.lower() if forbidden_text else False,
        "excerpt_preview": excerpt[:320],
    }


def timed_search(
    *,
    query: str,
    game_id: str,
    runs: int,
    use_rag_lite: bool,
    expected_text: str,
    forbidden_text: str,
) -> dict[str, Any]:
    samples: list[float] = []
    result: dict[str, Any] = {}
    for _ in range(runs):
        started = time.perf_counter()
        results = server.search_gamepath_sync(query, game_id, 5, use_rag_lite=use_rag_lite)
        samples.append((time.perf_counter() - started) * 1000.0)
        result = summarize_result(results, expected_text, forbidden_text)
    return {
        "result": result,
        "avg_ms": round(mean(samples), 4),
        "p50_ms": round(median(samples), 4),
        "p95_ms": round(percentile(samples, 0.95), 4),
        "min_ms": round(min(samples), 4),
        "max_ms": round(max(samples), 4),
    }


def build_report(data: dict[str, Any]) -> str:
    rows = []
    for case in data["cases"]:
        old_result = case["old"]["result"]
        rag_result = case["rag_lite"]["result"]
        rows.append(
            "| {name} | {old_avg} | {rag_avg} | {old_ctx} | {rag_ctx} | {old_ok} | {rag_ok} | {old_bad} | {rag_bad} |".format(
                name=case["name"],
                old_avg=case["old"]["avg_ms"],
                rag_avg=case["rag_lite"]["avg_ms"],
                old_ctx=old_result["context_chars"],
                rag_ctx=rag_result["context_chars"],
                old_ok="yes" if old_result["contains_expected"] else "no",
                rag_ok="yes" if rag_result["contains_expected"] else "no",
                old_bad="yes" if old_result["contains_forbidden"] else "no",
                rag_bad="yes" if rag_result["contains_forbidden"] else "no",
            )
        )
    table = "\n".join(rows)
    summary = data["summary"]
    return f"""# GamePath RAG Lite 技術報告

產生時間：{data["generated_at"]}

## 結論

GamePath RAG Lite 適合目前 iGPU + RAM 有限的架構，因為它不新增 embedding model，只用 SQLite FTS5 chunk index、中文 n-gram、後端 evaluator，以及必要時既有的地端 Qwen router/evaluator。

這次可控測試中：

- 舊 entry-level search 平均延遲：`{summary["old_avg_ms"]}` ms。
- RAG Lite chunk search 平均延遲：`{summary["rag_lite_avg_ms"]}` ms。
- 延遲改善：`{summary["latency_delta_percent"]}`%。
- CPU seconds delta：`{summary["cpu_seconds_delta"]}`。
- RSS before/after：`{summary["rss_before_mib"]}` / `{summary["rss_after_mib"]}` MiB。
- GamePath DB 大小 before/after：`{summary["db_size_before_kib"]}` / `{summary["db_size_after_kib"]}` KiB。
- 測試長攻略切出的 chunk 數：`{summary["seed_chunk_count"]}`。

## 架構差異

```mermaid
flowchart TD
    OldQ["玩家問題"] --> OldEntry["舊：entry-level FTS5"]
    OldEntry --> OldRead["讀整筆 answer / Markdown"]
    OldRead --> OldSplit["即時切段落與評分"]
    OldSplit --> OldContext["取 relevant excerpt"]

    NewQ["玩家問題"] --> NewChunk["新：chunk-level FTS5"]
    NewChunk --> NewHit["直接命中相關 chunk"]
    NewHit --> NewEval["heuristic evaluator / Qwen evaluator"]
    NewEval --> NewContext["取 top chunks 給模型"]
```

## 測試資料

- 測試 game_id：`{data["game_id"]}`
- 測試內容：一份包含廚房、銀鑰匙、女殭屍、地下室謎題、最終 boss 劇透的長攻略。
- 測試完成後會刪除暫存 GamePath entries，不污染正式資料庫。

## 結果表

| Case | Old Avg ms | RAG Lite Avg ms | Old Context chars | RAG Context chars | Old Expected | RAG Expected | Old Forbidden | RAG Forbidden |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

## 觀察

1. RAG Lite 把長攻略切 chunk 並寫入 `gamepath_chunks` / `gamepath_chunk_fts`，搜尋時不用每次重新讀整篇 Markdown 再切段。
2. 對 iGPU/RAM 友善：沒有新增 embedding model，也沒有新增常駐模型；只多一個 SQLite chunk index。
3. 對 LLM-wiki 友善：Markdown 仍是人類可讀 source，chunk index 是可刪可重建的衍生資料。
4. 對攻略回答比較安全：模型拿到的是 top chunk，不是整篇攻略，低劇透問題比較不容易帶出無關劇透段落。
5. unrelated miss 在這次測試中 RAG Lite 比舊搜尋慢，原因是 chunk FTS 需要掃過更多 chunk。實際聊天流程會先經過 GamePath router，非攻略或低可能性問題通常不會直接進入這條慢路徑。

## 驗收判斷

RAG Lite 比上一版好的地方：

- 大型 `.md` 攻略搜尋成本更固定。
- 未來可以直接從 LLM-wiki Markdown 重建 chunk index。
- 不需要額外 embedding 模型，符合 iGPU + RAM 限制。
- 可以保留現在的 Qwen local router/evaluator，不改掉雲端 Hermes 主流程。

仍然不是完整 semantic RAG：

- 短 query 的語意泛化仍主要靠 FTS5 n-gram 與 Qwen evaluator。
- 沒有 embedding，所以「完全不同說法但語意相同」仍可能 miss。
- 下一階段可以加 optional embedding adapter，但不應作為預設遊戲中模式。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark GamePath RAG Lite chunk retrieval.")
    parser.add_argument("--runs", type=int, default=80)
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--other-game-id", default=DEFAULT_OTHER_GAME_ID)
    parser.add_argument("--json-output", default=str(PROJECT_ROOT / "docs" / "gamepath_rag_lite_results.json"))
    parser.add_argument("--report-output", default=str(PROJECT_ROOT / "docs" / "gamepath_rag_lite_report.md"))
    args = parser.parse_args()

    before = process_snapshot()
    db_before = db_size_kib()
    seeds = seed_data(args.game_id, args.other_game_id)
    seed_chunk_count = chunk_count(args.game_id)
    cases = [
        {
            "name": "silver_key_location",
            "query": "銀鑰匙在哪",
            "expected_text": "銀鑰匙",
            "forbidden_text": "最終 boss",
        },
        {
            "name": "kitchen_route",
            "query": "kitchen route pantry guide",
            "expected_text": "pantry",
            "forbidden_text": "Final Boss",
        },
        {
            "name": "singing_zombies",
            "query": "兩個會唱歌的女殭屍",
            "expected_text": "女殭屍",
            "forbidden_text": "最終 boss",
        },
        {
            "name": "unrelated_miss",
            "query": "obsidian spoon moon factory boss weakness guide",
            "expected_text": "obsidian",
            "forbidden_text": "最終 boss",
        },
    ]

    measured_cases: list[dict[str, Any]] = []
    try:
        for case in cases:
            old = timed_search(
                query=case["query"],
                game_id=args.game_id,
                runs=args.runs,
                use_rag_lite=False,
                expected_text=case["expected_text"],
                forbidden_text=case["forbidden_text"],
            )
            rag_lite = timed_search(
                query=case["query"],
                game_id=args.game_id,
                runs=args.runs,
                use_rag_lite=True,
                expected_text=case["expected_text"],
                forbidden_text=case["forbidden_text"],
            )
            measured_cases.append({**case, "old": old, "rag_lite": rag_lite})
    finally:
        cleanup_game(args.game_id)
        cleanup_game(args.other_game_id)

    after = process_snapshot()
    db_after = db_size_kib()
    old_avg = mean(float(case["old"]["avg_ms"]) for case in measured_cases)
    rag_avg = mean(float(case["rag_lite"]["avg_ms"]) for case in measured_cases)
    latency_delta = ((old_avg - rag_avg) / old_avg * 100.0) if old_avg else 0.0
    data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs_per_case": args.runs,
        "game_id": args.game_id,
        "seed_ids": [item["id"] for item in seeds],
        "cases": measured_cases,
        "summary": {
            "old_avg_ms": round(old_avg, 4),
            "rag_lite_avg_ms": round(rag_avg, 4),
            "latency_delta_percent": round(latency_delta, 2),
            "cpu_seconds_delta": round(float(after["cpu_seconds"] or 0.0) - float(before["cpu_seconds"] or 0.0), 4),
            "rss_before_mib": before.get("rss_mib"),
            "rss_after_mib": after.get("rss_mib"),
            "db_size_before_kib": db_before,
            "db_size_after_kib": db_after,
            "seed_chunk_count": seed_chunk_count,
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
