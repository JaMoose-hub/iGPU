#!/usr/bin/env python3
"""Compare previous GamePath RAG Lite search with metadata-scoped search.

The benchmark uses an isolated temporary GamePath/Memory database. It seeds
50 GamePath entries plus memory rows, then compares:

- Old path: chunk FTS search without metadata scope/ranking.
- New path: current search_gamepath_sync with scope filters, source_quality,
  trust_state, and metadata ranking.
- Old memory context: search all memory kinds.
- New memory context: memory_kinds_for_chat policy.
"""

from __future__ import annotations

import argparse
import gc
import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import llama_vulkan_api_server as server  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "gamepath_scope_policy_report.md"
RESULTS_PATH = PROJECT_ROOT / "docs" / "gamepath_scope_policy_results.json"


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
        cpu = proc.cpu_times()
        return {
            "cpu_seconds": round(float(cpu.user + cpu.system), 4),
            "rss_mib": round(proc.memory_info().rss / 1048576, 2),
        }
    except Exception:
        return {
            "cpu_seconds": round(time.process_time(), 4),
            "rss_mib": None,
        }


def configure_isolated_store(root: Path) -> None:
    gamepath_dir = root / "gamepath"
    memory_dir = root / "memory_cache"
    server.GAMEPATH_DIR = gamepath_dir
    server.GAMEPATH_DB = gamepath_dir / "gamepath.sqlite"
    server.GAMEPATH_NOTES_DIR = gamepath_dir / "notes"
    server.MEMORY_CACHE_DIR = memory_dir
    server.MEMORY_DB = memory_dir / "memory.sqlite"


def build_answer(title: str, expected: str, kind: str, noise: str = "") -> str:
    return (
        f"{title}\n"
        f"類型：{kind}\n"
        f"無劇透提示：{expected}\n"
        f"下一步：先確認區域名稱、物品描述或敵人動作，再執行這個提示。\n"
        f"注意：不要把其他遊戲或其他版本的同名內容混用。\n"
        f"{noise}".strip()
    )


def make_seed_entries() -> list[dict[str, Any]]:
    target_game = "RESIDENT_EVIL_requiem"
    other_games = ["ELDEN_RING_nightreign", "SILENT_HILL_f", "MONSTER_HUNTER_wilds"]
    seeds: list[dict[str, Any]] = [
        {
            "key": "kitchen_butcher_boss",
            "game_id": target_game,
            "title": "廚房屠夫怪打法",
            "question": "廚房的屠夫怪怎麼打",
            "answer": build_answer("廚房屠夫怪打法", "拉開距離，等它揮刀硬直後繞到側面打弱點。", "boss"),
            "tags": ["boss", "kitchen", "butcher"],
            "entity_type": "boss",
            "area": "廚房",
            "source_quality": 0.86,
        },
        {
            "key": "kitchen_route_noise",
            "game_id": target_game,
            "title": "廚房怪門路線",
            "question": "廚房的怪門怎麼打開路線",
            "answer": build_answer(
                "廚房怪門路線",
                "這是門與路線提示，不是戰鬥打法；先找牆上的閥門再繞到儲藏室。",
                "route",
                "關鍵字包含：廚房 怪 怎麼 打 開 門 路線。",
            ),
            "tags": ["route", "kitchen", "door"],
            "entity_type": "route",
            "area": "廚房",
            "source_quality": 0.58,
        },
        {
            "key": "west_hall_fat_boss",
            "game_id": target_game,
            "title": "西翼大廳胖胖怪打法",
            "question": "西翼大廳胖胖怪物要怎麼打",
            "answer": build_answer("西翼大廳胖胖怪打法", "利用柱子卡位，等衝撞撞牆後再打背部。", "boss"),
            "tags": ["boss", "west hall"],
            "entity_type": "boss",
            "area": "西翼大廳",
            "source_quality": 0.84,
        },
        {
            "key": "silver_key_item",
            "game_id": target_game,
            "title": "銀鑰匙用途",
            "question": "銀鑰匙能用在哪",
            "answer": build_answer("銀鑰匙用途", "銀鑰匙用在地下酒窖入口的銀色鎖，不是大廳的銅門。", "item"),
            "tags": ["item", "key", "cellar"],
            "entity_type": "item",
            "entity_name": "銀鑰匙",
            "area": "地下酒窖",
            "source_quality": 0.88,
        },
        {
            "key": "silver_key_wrong_game",
            "game_id": other_games[0],
            "title": "銀鑰匙升降梯",
            "question": "銀鑰匙能用在哪",
            "answer": build_answer("銀鑰匙升降梯", "另一款遊戲的銀鑰匙用在塔樓升降梯。", "item"),
            "tags": ["item", "key"],
            "entity_type": "item",
            "entity_name": "銀鑰匙",
            "area": "塔樓",
            "source_quality": 0.82,
        },
        {
            "key": "red_light_mechanic",
            "game_id": target_game,
            "title": "紅光區域互動",
            "question": "紅光區域怎麼互動",
            "answer": build_answer("紅光區域互動", "先關掉旁邊電箱，再用紫外線燈照地面符號。", "mechanic"),
            "tags": ["mechanic", "red light"],
            "entity_type": "mechanic",
            "area": "紅光區域",
            "source_quality": 0.8,
        },
        {
            "key": "hospital_route",
            "game_id": target_game,
            "title": "醫院那關路線",
            "question": "醫院那關要怎麼破",
            "answer": build_answer("醫院那關路線", "先去護理站拿保險絲，再回中央走廊開電梯。", "route"),
            "tags": ["route", "hospital"],
            "entity_type": "route",
            "area": "醫院",
            "source_quality": 0.78,
        },
        {
            "key": "music_zombie_enemy",
            "game_id": target_game,
            "title": "會唱歌的女殭屍",
            "question": "兩個會唱歌的女殭屍名字是什麼",
            "answer": build_answer("會唱歌的女殭屍", "一個在西翼附近巡邏，另一個在東翼深處；目前不要臆造正式名字。", "enemy"),
            "tags": ["enemy", "zombie"],
            "entity_type": "enemy",
            "entity_name": "會唱歌的女殭屍",
            "source_quality": 0.76,
        },
        {
            "key": "rusted_cog_item",
            "game_id": target_game,
            "title": "生鏽齒輪用途",
            "question": "生鏽齒輪能做什麼",
            "answer": build_answer("生鏽齒輪用途", "生鏽齒輪用在鍋爐房牆面機關，能打開水閥旁的小門。", "item"),
            "tags": ["item", "cog", "boiler"],
            "entity_type": "item",
            "entity_name": "生鏽齒輪",
            "area": "鍋爐房",
            "source_quality": 0.83,
        },
        {
            "key": "boiler_puzzle",
            "game_id": target_game,
            "title": "鍋爐房閥門謎題",
            "question": "鍋爐房三個閥門怎麼解",
            "answer": build_answer("鍋爐房閥門謎題", "把左中右調成低壓、高壓、低壓；看到紅燈就回上一個閥門。", "puzzle"),
            "tags": ["puzzle", "boiler"],
            "entity_type": "puzzle",
            "area": "鍋爐房",
            "source_quality": 0.81,
        },
    ]

    entity_cycle = ["item", "boss", "route", "mechanic", "puzzle", "npc", "map", "material"]
    area_cycle = ["廚房", "醫院", "西翼大廳", "地下酒窖", "鍋爐房", "庭院", "資料室", "東翼走廊"]
    for index in range(40):
        entity_type = entity_cycle[index % len(entity_cycle)]
        area = area_cycle[index % len(area_cycle)]
        game_id = target_game if index < 24 else other_games[index % len(other_games)]
        title = f"{area} 測試攻略 {index + 1:02d}"
        question = f"{area} {entity_type} 測試關鍵字 怎麼處理 {index + 1:02d}"
        answer = build_answer(
            title,
            f"這是 {area} 的 {entity_type} 測試資料，編號 {index + 1:02d}，用來製造相似但不應置頂的候選。",
            entity_type,
            "重複詞：廚房 怪 銀鑰匙 紅光 醫院 怎麼 打 用途 路線。",
        )
        seeds.append(
            {
                "key": f"noise_{index + 1:02d}",
                "game_id": game_id,
                "title": title,
                "question": question,
                "answer": answer,
                "tags": [entity_type, area, "benchmark"],
                "entity_type": entity_type,
                "area": area,
                "source_quality": 0.42 + (index % 5) * 0.04,
            }
        )
    return seeds[:50]


def seed_gamepath() -> dict[str, int]:
    keys: dict[str, int] = {}
    for seed in make_seed_entries():
        item = server.add_gamepath_sync(
            seed["question"],
            seed["answer"],
            seed["game_id"],
            title=seed["title"],
            tags=seed["tags"],
            spoiler_level=seed.get("spoiler_level", "low"),
            source_type="benchmark",
            agent_used=False,
            version=seed.get("version"),
            area=seed.get("area"),
            entity_type=seed.get("entity_type"),
            entity_name=seed.get("entity_name"),
            source_quality=seed.get("source_quality"),
        )
        keys[seed["key"]] = int(item["id"])
    return keys


def seed_memory() -> None:
    server.ensure_memory_db()
    memory_rows = [
        ("玩家目前進度是西翼大廳。", "state", "auto", 4),
        ("玩家偏好是無劇透提示。", "preference", "auto", 4),
        ("任務目標：廚房屠夫怪；下一步：找弱點並避免正面硬打。", "task", "boss,kitchen", 4),
        ("任務目標：銀鑰匙；下一步：確認地下酒窖入口。", "task", "item,key", 4),
        ("玩家筆記：之前提到紅光區域可能需要紫外線燈。", "note", "red light", 3),
    ]
    for content, kind, tags, importance in memory_rows:
        server.add_memory_sync(content, "RESIDENT_EVIL_requiem", kind, tags, importance)


def row_to_item(row: sqlite3.Row, query: str) -> dict[str, Any]:
    return {
        "id": int(row["entry_id"]),
        "game_id": row["game_id"],
        "title": row["title"],
        "question": row["question"],
        "answer_summary": row["answer_summary"],
        "snippet": server.make_snippet(row["chunk_content"] or row["answer_summary"], query),
        "markdown_path": row["markdown_path"],
        "tags": row["tags"],
        "version": row["version"],
        "area": row["area"],
        "entity_type": row["entity_type"],
        "entity_name": row["entity_name"],
        "spoiler_level": row["spoiler_level"],
        "source_type": row["source_type"],
        "source_quality": float(row["source_quality"] or 0.5),
        "agent_used": bool(row["agent_used"]),
        "trust_state": row["trust_state"],
        "dispute_count": int(row["dispute_count"] or 0),
        "last_feedback": row["last_feedback"],
        "last_feedback_at": row["last_feedback_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "score": float(row["chunk_score"]),
        "match_coverage": 0.0,
        "rag_lite": True,
    }


def old_search_gamepath(query: str, game_id: str | None, limit: int = 5, spoiler_level: str = "low") -> list[dict[str, Any]]:
    query = query.strip()
    if not query:
        return []
    match = server.fts_query(server.gamepath_query_text(query))
    if not match:
        return []
    normalized_game_id = server.normalize_game_id(game_id)
    sql = (
        "SELECT e.id AS entry_id, e.game_id, e.title, e.question, e.answer_summary, "
        "e.markdown_path, e.tags, e.version, e.area, e.entity_type, e.entity_name, "
        "e.spoiler_level, e.source_type, e.source_quality, e.agent_used, "
        "e.trust_state, e.dispute_count, e.last_feedback, e.last_feedback_at, "
        "e.created_at, e.updated_at, c.id AS chunk_id, c.chunk_index, c.heading, "
        "c.content AS chunk_content, c.char_count, bm25(gamepath_chunk_fts) AS chunk_score "
        "FROM gamepath_chunk_fts "
        "JOIN gamepath_chunks c ON c.id = gamepath_chunk_fts.chunk_id "
        "JOIN gamepath_entries e ON e.id = c.entry_id "
        "WHERE gamepath_chunk_fts MATCH ? AND e.spoiler_rank <= ?"
    )
    params: list[Any] = [match, server.spoiler_rank(spoiler_level)]
    if normalized_game_id:
        sql += " AND e.game_id IN (?, 'global')"
        params.append(normalized_game_id)
    sql += " ORDER BY chunk_score LIMIT ?"
    params.append(80)
    with sqlite3.connect(server.GAMEPATH_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()

    grouped: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for row in rows:
        entry_id = int(row["entry_id"])
        item = grouped.get(entry_id)
        chunk_content = str(row["chunk_content"] or "")
        coverage = server.gamepath_term_coverage(query, "\n".join([row["title"], row["question"], chunk_content, row["tags"]]))
        if item is None:
            item = row_to_item(row, query)
            item["match_coverage"] = round(coverage, 3)
            item["rag_chunk_hits"] = []
            grouped[entry_id] = item
            order.append(entry_id)
        item["match_coverage"] = round(max(float(item.get("match_coverage") or 0.0), coverage), 3)
        item["rag_chunk_hits"].append(
            {
                "chunk_id": int(row["chunk_id"]),
                "heading": row["heading"],
                "content": chunk_content,
                "score": float(row["chunk_score"]),
                "coverage": round(coverage, 3),
            }
        )

    results: list[dict[str, Any]] = []
    for entry_id in order:
        item = grouped[entry_id]
        chunks = list(item.pop("rag_chunk_hits", []))
        chunks.sort(key=lambda chunk: (float(chunk.get("score") or 0.0)))
        excerpts = []
        for chunk in chunks[:3]:
            excerpts.append(server.make_snippet(str(chunk.get("content") or ""), query, max_len=server.GAMEPATH_PASSAGE_MAX_CHARS))
        relevant_excerpt = "\n\n---\n\n".join(excerpt for excerpt in excerpts if excerpt).strip()
        item["relevant_excerpt"] = relevant_excerpt
        item["context_char_count"] = len(relevant_excerpt)
        item["rag_chunk_count"] = len(chunks)
        item["metadata_match_score"] = 0.0
        results.append(item)
        if len(results) >= limit:
            break
    return results


TEST_CASES = [
    ("kitchen_boss_short", "廚房那個怪", "kitchen_butcher_boss"),
    ("kitchen_boss_action", "廚房怪怎麼處理", "kitchen_butcher_boss"),
    ("kitchen_door_route", "廚房怪門怎麼開", "kitchen_route_noise"),
    ("west_hall_boss", "西翼大廳胖胖怪物怎麼打", "west_hall_fat_boss"),
    ("silver_key_short", "銀鑰匙", "silver_key_item"),
    ("silver_key_usage", "銀色鑰匙用途", "silver_key_item"),
    ("red_short", "紅光要怎麼弄", "red_light_mechanic"),
    ("red_interact", "紅光區域怎麼互動", "red_light_mechanic"),
    ("hospital_short", "醫院下一步", "hospital_route"),
    ("singing_zombies", "兩個會唱歌的女殭屍名字是什麼", "music_zombie_enemy"),
    ("rusted_cog", "生鏽齒輪能做什麼", "rusted_cog_item"),
    ("boiler_short", "三個閥門", "boiler_puzzle"),
]


def summarize_ranking(results: list[dict[str, Any]], expected_id: int) -> dict[str, Any]:
    ids = [int(item.get("id") or 0) for item in results]
    rank = ids.index(expected_id) + 1 if expected_id in ids else 0
    top = results[0] if results else {}
    return {
        "top_id": int(top.get("id") or 0),
        "top_title": top.get("title", ""),
        "rank": rank,
        "top1": bool(rank == 1),
        "hit3": bool(rank and rank <= 3),
        "mrr": round(1.0 / rank, 4) if rank else 0.0,
        "context_chars": int(top.get("context_char_count") or len(str(top.get("relevant_excerpt") or ""))),
        "metadata_match_score": float(top.get("metadata_match_score") or 0.0),
        "match_coverage": float(top.get("match_coverage") or 0.0),
    }


def time_search(fn, query: str, game_id: str, expected_id: int, runs: int) -> dict[str, Any]:
    samples: list[float] = []
    summary: dict[str, Any] = {}
    for _ in range(runs):
        started = time.perf_counter()
        results = fn(query, game_id, 5)
        samples.append((time.perf_counter() - started) * 1000.0)
        summary = summarize_ranking(results, expected_id)
    return {
        **summary,
        "avg_ms": round(mean(samples), 4),
        "p50_ms": round(median(samples), 4),
        "p95_ms": round(percentile(samples, 0.95), 4),
        "min_ms": round(min(samples), 4),
        "max_ms": round(max(samples), 4),
    }


def memory_policy_compare() -> list[dict[str, Any]]:
    cases = [
        ("general_guide", "廚房的怪怎麼打"),
        ("task_intent", "我目前任務下一步是什麼"),
        ("note_intent", "你記得我之前說過紅光區域什麼嗎"),
    ]
    rows = []
    for name, query in cases:
        old_results = server.search_memory_sync(query, "RESIDENT_EVIL_requiem", None, 8)
        new_kinds = server.memory_kinds_for_chat(query)
        new_results = server.search_memory_sync(query, "RESIDENT_EVIL_requiem", new_kinds, 8)
        rows.append(
            {
                "case": name,
                "query": query,
                "old_kinds": "all",
                "new_kinds": new_kinds,
                "old_hits": len(old_results),
                "new_hits": len(new_results),
                "old_task_hits": sum(1 for item in old_results if item.get("kind") == "task"),
                "new_task_hits": sum(1 for item in new_results if item.get("kind") == "task"),
            }
        )
    return rows


def run_benchmark(runs: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gamepath-scope-bench-") as tmp:
        configure_isolated_store(Path(tmp))
        before = process_snapshot()
        expected_ids = seed_gamepath()
        seed_memory()
        server.ensure_gamepath_db()
        server.ensure_memory_db()

        cases: list[dict[str, Any]] = []
        for name, query, expected_key in TEST_CASES:
            expected_id = expected_ids[expected_key]
            old = time_search(old_search_gamepath, query, "RESIDENT_EVIL_requiem", expected_id, runs)
            new = time_search(
                lambda q, g, limit: server.search_gamepath_sync(q, g, limit),
                query,
                "RESIDENT_EVIL_requiem",
                expected_id,
                runs,
            )
            cases.append({"case": name, "query": query, "expected_id": expected_id, "old": old, "new": new})

        after = process_snapshot()
        conn = sqlite3.connect(server.GAMEPATH_DB)
        try:
            entry_count = int(conn.execute("SELECT COUNT(*) FROM gamepath_entries").fetchone()[0])
            chunk_count = int(conn.execute("SELECT COUNT(*) FROM gamepath_chunks").fetchone()[0])
            db_size_kib = round(server.GAMEPATH_DB.stat().st_size / 1024.0, 2)
        finally:
            conn.close()

        old_top1 = mean([1.0 if case["old"]["top1"] else 0.0 for case in cases])
        new_top1 = mean([1.0 if case["new"]["top1"] else 0.0 for case in cases])
        old_hit3 = mean([1.0 if case["old"]["hit3"] else 0.0 for case in cases])
        new_hit3 = mean([1.0 if case["new"]["hit3"] else 0.0 for case in cases])
        old_mrr = mean([case["old"]["mrr"] for case in cases])
        new_mrr = mean([case["new"]["mrr"] for case in cases])
        old_p50 = mean([case["old"]["p50_ms"] for case in cases])
        new_p50 = mean([case["new"]["p50_ms"] for case in cases])
        old_p95 = mean([case["old"]["p95_ms"] for case in cases])
        new_p95 = mean([case["new"]["p95_ms"] for case in cases])
        before_cpu = float(before.get("cpu_seconds") or 0.0)
        after_cpu = float(after.get("cpu_seconds") or 0.0)
        before_rss = before.get("rss_mib")
        after_rss = after.get("rss_mib")
        data = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "runs_per_case": runs,
            "seed_entries": entry_count,
            "chunk_count": chunk_count,
            "db_size_kib": db_size_kib,
            "process_before": before,
            "process_after": after,
            "summary": {
                "old_top1": round(old_top1, 4),
                "new_top1": round(new_top1, 4),
                "old_hit3": round(old_hit3, 4),
                "new_hit3": round(new_hit3, 4),
                "old_mrr": round(old_mrr, 4),
                "new_mrr": round(new_mrr, 4),
                "old_avg_ms": round(mean([case["old"]["avg_ms"] for case in cases]), 4),
                "new_avg_ms": round(mean([case["new"]["avg_ms"] for case in cases]), 4),
                "old_p50_ms": round(old_p50, 4),
                "new_p50_ms": round(new_p50, 4),
                "old_p95_ms": round(old_p95, 4),
                "new_p95_ms": round(new_p95, 4),
                "cpu_seconds_delta": round(after_cpu - before_cpu, 4),
                "rss_mib_delta": round(float(after_rss) - float(before_rss), 2)
                if before_rss is not None and after_rss is not None
                else None,
            },
            "cases": cases,
            "memory_policy": memory_policy_compare(),
        }
        gc.collect()
        return data


def markdown_report(data: dict[str, Any]) -> str:
    summary = data["summary"]
    lines = [
        "# GamePath Scope Policy Benchmark",
        "",
        f"產生時間：`{data['generated_at']}`",
        "",
        "## 結論",
        "",
        "這份測試用隔離的暫存 GamePath/Memory DB，建立 50 筆攻略資料，比較前一版 RAG Lite 搜尋路徑與新版 metadata scope + ranking 路徑。",
        "",
        "比較方式：前一版路徑以 `chunk FTS only` 模擬，也就是不使用 metadata scope filter、source_quality、trust_state、metadata ranking；新版路徑直接呼叫目前的 `search_gamepath_sync()`。兩邊使用同一份暫存 SQLite，因此差異主要來自搜尋演算法，而不是資料庫內容。",
        "",
        "| 指標 | 前一版 | 新版 |",
        "| --- | ---: | ---: |",
        f"| Top-1 準確率 | {summary['old_top1']:.2%} | {summary['new_top1']:.2%} |",
        f"| Hit@3 | {summary['old_hit3']:.2%} | {summary['new_hit3']:.2%} |",
        f"| MRR@5 | {summary['old_mrr']:.4f} | {summary['new_mrr']:.4f} |",
        f"| 平均延遲 ms | {summary['old_avg_ms']} | {summary['new_avg_ms']} |",
        f"| 平均 p50 延遲 ms | {summary['old_p50_ms']} | {summary['new_p50_ms']} |",
        f"| 平均 p95 延遲 ms | {summary['old_p95_ms']} | {summary['new_p95_ms']} |",
        "",
        "## 測試資料",
        "",
        f"- GamePath 測試資料筆數：`{data['seed_entries']}`",
        f"- RAG Lite chunks：`{data['chunk_count']}`",
        f"- SQLite 大小：`{data['db_size_kib']}` KiB",
        "- 資料組成：10 筆目標攻略 + 40 筆同遊戲/跨遊戲干擾攻略。",
        f"- 每個查詢、每條路徑重跑次數：`{data['runs_per_case']}`",
        f"- 測試查詢數：`{len(data['cases'])}`",
        "- 隔離性：使用暫存 DB，不修改正式 GamePath/Memory。",
        f"- Benchmark process CPU delta：`{summary['cpu_seconds_delta']}` seconds",
        f"- Benchmark process RSS RAM delta：`{summary['rss_mib_delta']}` MiB",
        "",
        "## 搜尋案例",
        "",
        "| 案例 | 查詢 | 正解 ID | 前一版 Top | 新版 Top | 前一版排名 | 新版排名 | 前一版 ms | 新版 ms |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for case in data["cases"]:
        lines.append(
            "| {case} | {query} | {expected} | {old_top} | {new_top} | {old_rank} | {new_rank} | {old_ms} | {new_ms} |".format(
                case=case["case"],
                query=case["query"],
                expected=case["expected_id"],
                old_top=case["old"]["top_title"],
                new_top=case["new"]["top_title"],
                old_rank=case["old"]["rank"],
                new_rank=case["new"]["rank"],
                old_ms=case["old"]["avg_ms"],
                new_ms=case["new"]["avg_ms"],
            )
        )
    lines.extend(
        [
            "",
            "## 玩家記憶 Policy",
            "",
            "| 案例 | 前一版 kinds | 新版 kinds | 前一版命中 | 新版命中 | 前一版 task 命中 | 新版 task 命中 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in data["memory_policy"]:
        lines.append(
            "| {case} | {old_kinds} | {new_kinds} | {old_hits} | {new_hits} | {old_task_hits} | {new_task_hits} |".format(
                case=row["case"],
                old_kinds=row["old_kinds"],
                new_kinds=", ".join(row["new_kinds"]),
                old_hits=row["old_hits"],
                new_hits=row["new_hits"],
                old_task_hits=row["old_task_hits"],
                new_task_hits=row["new_task_hits"],
            )
        )
    lines.extend(
        [
            "",
            "## 解讀",
            "",
            "- 新版會先推測 `entity_type`、`area`、`version` 等 scope，再套 SQLite filter 與 metadata ranking。",
            "- 前一版主要依賴 chunk FTS 排序；字面相近但語意不同的資料，容易排在正確攻略前面。",
            "- 新版玩家記憶 policy 會避免一般聊天被舊的 `task` 記憶污染；只有玩家問任務、下一步、目前目標時才拉 task。",
            "- 這不是取代 embedding RAG，而是低 RAM/iGPU 友善的精準度前置層；之後可以再接 embedding/reranker。",
            "",
            "## 剩餘風險",
            "",
            "- Metadata 推斷是輕量規則，未來大量匯入資料時，最好由 importer 或 Agent 明確提供 `game_id`、`version`、`area`、`entity_type`、`entity_name`。",
            "- 如果 metadata 標錯，strict filter 可能藏掉正確資料；聊天流程目前採非 strict fallback 降低風險。",
            "- 若未來進到數萬筆長文攻略，仍建議加可選的 embedding/reranker adapter。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    data = run_benchmark(max(1, args.runs))
    RESULTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(markdown_report(data), encoding="utf-8", newline="\n")
    print(json.dumps(data["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
