#!/usr/bin/env python3
"""Stress benchmark for the GamePath SQLite/RAG Lite retrieval algorithm.

The benchmark is isolated from the production GamePath database by redirecting
llama_vulkan_api_server.GAMEPATH_* paths to a temporary directory.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REPORT_PATH = PROJECT_ROOT / "docs" / "gamepath_stress_report.md"
RESULTS_PATH = PROJECT_ROOT / "docs" / "gamepath_stress_results.json"

TARGET_GAME = "GAMEPATH_STRESS_ARPG"
OTHER_GAMES = ["MISTFALL_ARCHIVE", "IRON_RIFT", "STARLIGHT_EXILE", "NOCTURNE_FIELD"]
ENTITY_TYPES = ["boss", "item", "route", "puzzle", "mechanic", "npc", "map", "material", "enemy", "quest", "location", "character"]
AREAS = [
    "霧港廚房",
    "西翼大廳",
    "地下酒窖",
    "紅光中庭",
    "舊醫院",
    "鍋爐房",
    "鏡湖碼頭",
    "鐘塔頂層",
    "黑森林入口",
    "資料室",
    "東翼走廊",
    "沉沒禮拜堂",
    "療養院二樓",
    "主廚餐廳",
    "生化武器儲存室",
]


GOLDEN: list[dict[str, Any]] = [
    {
        "key": "singing_zombie",
        "title": "唱歌女殭屍位置",
        "question": "唱歌女殭屍在哪出沒",
        "alt": "二樓聽到歌聲是哪隻怪",
        "answer": "唱歌女殭屍主要在療養院二樓 Bar & Lounge 附近。若在一樓聽到樓上歌聲，就往二樓找；東翼路線往主席辦公室時還會再遇到一隻。",
        "area": "療養院二樓",
        "entity_type": "enemy",
        "entity_name": "唱歌女殭屍",
        "tags": ["enemy", "location", "guide"],
    },
    {
        "key": "kitchen_butcher",
        "title": "廚房屠夫怪打法",
        "question": "廚房拿刀屠夫怪怎麼打",
        "alt": "廚房那個拿刀的一直追我怎麼安全處理",
        "answer": "保持中距離，等屠夫連砍第三下後繞到右側，打背後發亮弱點。不要貪刀，聽到喘息聲再補一輪。",
        "area": "霧港廚房",
        "entity_type": "boss",
        "entity_name": "屠夫怪",
        "tags": ["boss", "enemy", "guide"],
    },
    {
        "key": "silver_key",
        "title": "銀鑰匙用途",
        "question": "銀鑰匙能用在哪",
        "alt": "這把銀色鑰匙是不是該留著",
        "answer": "銀鑰匙用在地下酒窖入口的銀色鎖，不是西翼大廳的銅門。先從廚房後門繞到酒窖。",
        "area": "地下酒窖",
        "entity_type": "item",
        "entity_name": "銀鑰匙",
        "tags": ["item", "route", "guide"],
    },
    {
        "key": "red_light",
        "title": "紅光中庭互動",
        "question": "紅光中庭紅光怎麼互動",
        "alt": "紅色光一直擋路我要先動哪個東西",
        "answer": "先關牆邊電箱，再用紫外線燈照地面符號。紅光變短閃時才能安全穿過。",
        "area": "紅光中庭",
        "entity_type": "mechanic",
        "entity_name": "紅光機制",
        "tags": ["mechanic", "puzzle", "guide"],
    },
    {
        "key": "hospital_next",
        "title": "舊醫院下一步",
        "question": "舊醫院下一步要做什麼",
        "alt": "我在醫院繞很久接下來應該去哪",
        "answer": "先去護理站拿保險絲，再回中央走廊開電梯。不要先進地下室，會少一個回程捷徑。",
        "area": "舊醫院",
        "entity_type": "route",
        "entity_name": "醫院路線",
        "tags": ["route", "quest", "guide"],
    },
    {
        "key": "boiler_valves",
        "title": "鍋爐房三閥門解法",
        "question": "鍋爐房三個閥門順序",
        "alt": "三個轉盤我轉到快瘋了順序是什麼",
        "answer": "左閥低壓、中閥高壓、右閥低壓。看到紅燈就退回上一個閥門，等聲音變低再轉下一個。",
        "area": "鍋爐房",
        "entity_type": "puzzle",
        "entity_name": "三閥門",
        "tags": ["puzzle", "mechanic", "guide"],
    },
    {
        "key": "mirror_fragment",
        "title": "鏡湖碼頭地圖碎片",
        "question": "鏡湖碼頭地圖碎片在哪",
        "alt": "碼頭附近那張地圖碎片藏在哪個角落",
        "answer": "地圖碎片在第二個木棧橋下方的箱子裡。先推開藍色漁網，從側邊小坡下去。",
        "area": "鏡湖碼頭",
        "entity_type": "map",
        "entity_name": "地圖碎片",
        "tags": ["map", "location", "guide"],
    },
    {
        "key": "clock_guard",
        "title": "鐘塔守門人打法",
        "question": "鐘塔守門人弱點在哪",
        "alt": "鐘塔那個守門的到底要打哪裡才有效",
        "answer": "守門人的弱點在背後鐘擺核心。等它敲鐘後會短暫跪地，從左側樓梯繞背輸出。",
        "area": "鐘塔頂層",
        "entity_type": "boss",
        "entity_name": "守門人",
        "tags": ["boss", "weakness", "guide"],
    },
    {
        "key": "moonsilver_branch",
        "title": "月銀枝用途",
        "question": "月銀枝這個素材能做什麼",
        "alt": "月銀枝先別賣嗎它能做什麼",
        "answer": "月銀枝用來升級靜音護符，也能交給資料室 NPC 換一個無暴雷提示。前期建議先留一根。",
        "area": "黑森林入口",
        "entity_type": "material",
        "entity_name": "月銀枝",
        "tags": ["material", "item", "guide"],
    },
    {
        "key": "white_npc",
        "title": "資料室白衣 NPC",
        "question": "資料室白衣 NPC 對話選哪個",
        "alt": "資料室那個白衣NPC我該選哪一句",
        "answer": "第一次選「我只是路過」，第二次選「你在找誰」。這樣會開無暴雷提示線，不會直接跳劇情。",
        "area": "資料室",
        "entity_type": "npc",
        "entity_name": "白衣 NPC",
        "tags": ["npc", "dialogue", "guide"],
    },
    {
        "key": "chapel_quest",
        "title": "沉沒禮拜堂支線目標",
        "question": "沉沒禮拜堂支線下一步",
        "alt": "禮拜堂支線現在只剩鐘聲下一步是什麼",
        "answer": "先找三個沒有點火的燭台，只點亮靠近水痕的那一個。之後回入口聽鐘聲方向。",
        "area": "沉沒禮拜堂",
        "entity_type": "quest",
        "entity_name": "禮拜堂支線",
        "tags": ["quest", "route", "guide"],
    },
    {
        "key": "hidden_door",
        "title": "西翼大廳暗門入口",
        "question": "西翼大廳暗門入口在哪",
        "alt": "西翼大廳好像有暗門我要看哪裡",
        "answer": "暗門在兩幅破損肖像中間。先把左邊肖像扶正，再按右下角裂縫。",
        "area": "西翼大廳",
        "entity_type": "location",
        "entity_name": "暗門入口",
        "tags": ["location", "puzzle", "guide"],
    },
]

NEGATIVE_QUERIES = [
    "星霜羅盤最新版本用途",
    "第七章黑曜飛龍弱點",
    "雲端排行榜 meta 配裝",
    "這不是攻略問題只是心情不好",
    "幫我打開 Game Search 視窗",
    "目前 patch 2.4 有沒有改保險箱密碼",
]


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": round(mean(values), 3),
        "p50": round(median(values), 3),
        "p95": round(percentile(values, 0.95), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def rss_mib() -> float:
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / 1048576.0
    except Exception:
        return 0.0


def set_gamepath_dir(server: Any, path: Path) -> None:
    server.GAMEPATH_DIR = path
    server.GAMEPATH_DB = path / "gamepath.sqlite"
    server.GAMEPATH_NOTES_DIR = path / "notes"


def db_counts(server: Any) -> dict[str, Any]:
    db = server.GAMEPATH_DB
    if not db.exists():
        return {"entries": 0, "chunks": 0, "fts": 0, "db_kib": 0.0}
    with sqlite3.connect(db) as conn:
        entries = conn.execute("SELECT COUNT(*) FROM gamepath_entries").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM gamepath_chunks").fetchone()[0]
        fts = conn.execute("SELECT COUNT(*) FROM gamepath_fts").fetchone()[0]
    return {
        "entries": int(entries),
        "chunks": int(chunks),
        "fts": int(fts),
        "db_kib": round(db.stat().st_size / 1024, 2),
    }


def filler_entry(index: int) -> dict[str, Any]:
    entity_type = ENTITY_TYPES[index % len(ENTITY_TYPES)]
    area = AREAS[index % len(AREAS)]
    game_id = TARGET_GAME if index % 5 else OTHER_GAMES[index % len(OTHER_GAMES)]
    entity = f"{area}{entity_type}{index:04d}"
    tags = [entity_type, "guide"]
    if entity_type in {"boss", "enemy"}:
        tags.append("enemy")
    if entity_type in {"item", "material"}:
        tags.append("item")
    title = f"壓力資料 {area} {entity_type} {index:04d}"
    question = f"{area} {entity} {entity_type} 攻略 線索 {index:04d}"
    answer = (
        f"這是壓力測試干擾資料 {index:04d}。區域是 {area}，主題是 {entity_type}，"
        f"關鍵物件是 {entity}。Hint 1：先確認路線。Hint 2：再檢查互動點。Hint 3：必要時回到上一個安全房。"
    )
    return {
        "game_id": game_id,
        "title": title,
        "question": question,
        "answer": answer,
        "area": area,
        "entity_type": entity_type,
        "entity_name": entity,
        "tags": tags,
        "source_type": "manual_stress",
        "source_quality": 0.58 + ((index % 9) / 100.0),
    }


def seed_dataset(server: Any, size: int) -> tuple[dict[str, int], list[float]]:
    expected_ids: dict[str, int] = {}
    write_ms: list[float] = []
    for item in GOLDEN:
        started = time.perf_counter()
        saved = server.add_gamepath_sync(
            item["question"],
            item["answer"],
            TARGET_GAME,
            title=f"[GOLD] {item['title']}",
            tags=item["tags"],
            spoiler_level="low",
            source_type="manual_stress",
            agent_used=False,
            area=item["area"],
            entity_type=item["entity_type"],
            entity_name=item["entity_name"],
            source_quality=0.72,
        )
        write_ms.append((time.perf_counter() - started) * 1000.0)
        expected_ids[item["key"]] = int(saved["id"])

    filler_count = max(0, size - len(GOLDEN))
    for index in range(filler_count):
        item = filler_entry(index)
        started = time.perf_counter()
        server.add_gamepath_sync(
            item["question"],
            item["answer"],
            item["game_id"],
            title=item["title"],
            tags=item["tags"],
            spoiler_level="low",
            source_type=item["source_type"],
            agent_used=False,
            area=item["area"],
            entity_type=item["entity_type"],
            entity_name=item["entity_name"],
            source_quality=item["source_quality"],
        )
        write_ms.append((time.perf_counter() - started) * 1000.0)
    return expected_ids, write_ms


def query_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for item in GOLDEN:
        cases.append(
            {
                "key": item["key"],
                "kind": "exact",
                "query": item["question"],
                "tags": [item["entity_type"]],
            }
        )
        cases.append(
            {
                "key": item["key"],
                "kind": "paraphrase",
                "query": item["alt"],
                "tags": [item["entity_type"]],
            }
        )
        noisy_tag = "enemy" if item["entity_type"] != "enemy" else "location"
        cases.append(
            {
                "key": item["key"],
                "kind": "noisy_tag",
                "query": item["alt"],
                "tags": [noisy_tag],
            }
        )
    return cases


def rank_of(results: list[dict[str, Any]], expected_id: int) -> int | None:
    for index, item in enumerate(results, 1):
        if int(item.get("id") or 0) == expected_id:
            return index
    return None


def run_search_suite(server: Any, expected_ids: dict[str, int], repeats: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    by_kind: dict[str, list[bool]] = defaultdict(list)
    for case in query_cases():
        expected_id = expected_ids[case["key"]]
        for run in range(repeats):
            started = time.perf_counter()
            results = server.search_gamepath_multi_query_sync(
                case["query"],
                TARGET_GAME,
                5,
                tags=case["tags"],
                spoiler_level="low",
            )
            evaluation = server.evaluate_gamepath_retrieval(case["query"], TARGET_GAME, results)
            elapsed = (time.perf_counter() - started) * 1000.0
            latencies.append(elapsed)
            evaluated = list(evaluation.get("results") or results)
            rank = rank_of(evaluated, expected_id)
            hit = rank == 1
            by_kind[case["kind"]].append(hit)
            top = evaluated[0] if evaluated else {}
            rows.append(
                {
                    "query": case["query"],
                    "kind": case["kind"],
                    "expected_id": expected_id,
                    "top_id": int(evaluation.get("top_id") or 0),
                    "top_title": evaluation.get("top_title"),
                    "rank": rank,
                    "hit_top1": hit,
                    "hit_top3": bool(rank and rank <= 3),
                    "hit_top5": bool(rank and rank <= 5),
                    "confidence": evaluation.get("confidence"),
                    "score": evaluation.get("score"),
                    "gap": evaluation.get("gap"),
                    "coverage": top.get("match_coverage"),
                    "core_overlap": top.get("core_overlap"),
                    "elapsed_ms": round(elapsed, 3),
                    "run": run,
                }
            )
    top1 = [row["hit_top1"] for row in rows]
    top3 = [row["hit_top3"] for row in rows]
    top5 = [row["hit_top5"] for row in rows]
    mrr_values = [(1.0 / row["rank"]) if row["rank"] else 0.0 for row in rows]
    return {
        "summary": {
            "cases": len(rows),
            "top1": round(sum(top1) / len(top1), 4) if rows else 0.0,
            "hit3": round(sum(top3) / len(top3), 4) if rows else 0.0,
            "hit5": round(sum(top5) / len(top5), 4) if rows else 0.0,
            "mrr5": round(mean(mrr_values), 4) if mrr_values else 0.0,
            "latency_ms": summarize(latencies),
            "by_kind": {
                kind: round(sum(values) / len(values), 4)
                for kind, values in sorted(by_kind.items())
            },
        },
        "rows": rows,
    }


def run_negative_suite(server: Any, repeats: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for query in NEGATIVE_QUERIES:
        for run in range(repeats):
            started = time.perf_counter()
            results = server.search_gamepath_multi_query_sync(query, TARGET_GAME, 5, spoiler_level="low")
            evaluation = server.evaluate_gamepath_retrieval(query, TARGET_GAME, results)
            elapsed = (time.perf_counter() - started) * 1000.0
            latencies.append(elapsed)
            false_positive = bool(results) and str(evaluation.get("confidence")) in {"direct", "summarize"}
            rows.append(
                {
                    "query": query,
                    "run": run,
                    "result_count": len(results),
                    "confidence": evaluation.get("confidence"),
                    "top_id": evaluation.get("top_id"),
                    "top_title": evaluation.get("top_title"),
                    "false_positive": false_positive,
                    "elapsed_ms": round(elapsed, 3),
                }
            )
    fp = [row["false_positive"] for row in rows]
    return {
        "summary": {
            "cases": len(rows),
            "false_positive_rate": round(sum(fp) / len(fp), 4) if fp else 0.0,
            "latency_ms": summarize(latencies),
        },
        "rows": rows,
    }


def run_duplicate_suite(server: Any, expected_ids: dict[str, int]) -> dict[str, Any]:
    before = db_counts(server)["entries"]
    rows: list[dict[str, Any]] = []
    for item in GOLDEN[: min(6, len(GOLDEN))]:
        query = item["alt"]
        answer = item["answer"] + " 這是模擬 Hermes 第二次濃縮後的相似提示。"
        started = time.perf_counter()
        saved = server.add_gamepath_sync(
            query,
            answer,
            TARGET_GAME,
            title=f"duplicate probe {item['key']}",
            tags=["auto", "hermes", "guide", item["entity_type"]],
            spoiler_level="low",
            source_type="hermes_agent_web",
            agent_used=True,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        rows.append(
            {
                "key": item["key"],
                "expected_id": expected_ids[item["key"]],
                "returned_id": int(saved.get("id") or 0),
                "status": saved.get("status"),
                "elapsed_ms": round(elapsed, 3),
            }
        )
    after = db_counts(server)["entries"]
    ok = [row["status"] == "duplicate_existing" and row["returned_id"] == row["expected_id"] for row in rows]
    return {
        "summary": {
            "checks": len(rows),
            "ok_rate": round(sum(ok) / len(ok), 4) if ok else 0.0,
            "entry_count_before": before,
            "entry_count_after": after,
            "count_unchanged": before == after,
            "latency_ms": summarize([row["elapsed_ms"] for row in rows]),
        },
        "rows": rows,
    }


def run_size(server: Any, root: Path, size: int, repeats: int) -> dict[str, Any]:
    test_dir = root / f"size_{size}"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    set_gamepath_dir(server, test_dir)
    server.ensure_gamepath_db()
    gc.collect()
    cpu_before = time.process_time()
    rss_before = rss_mib()
    started = time.perf_counter()
    expected_ids, write_ms = seed_dataset(server, size)
    seed_elapsed_ms = (time.perf_counter() - started) * 1000.0
    counts_after_seed = db_counts(server)
    search = run_search_suite(server, expected_ids, repeats)
    negative = run_negative_suite(server, max(1, min(repeats, 3)))
    duplicate = run_duplicate_suite(server, expected_ids)
    cpu_after = time.process_time()
    rss_after = rss_mib()
    return {
        "size": size,
        "counts": db_counts(server),
        "counts_after_seed": counts_after_seed,
        "seed_elapsed_ms": round(seed_elapsed_ms, 3),
        "write_latency_ms": summarize(write_ms),
        "search": search["summary"],
        "search_rows": search["rows"],
        "search_rows_sample": search["rows"][:18],
        "search_failures": [row for row in search["rows"] if not row.get("hit_top1")],
        "negative": negative["summary"],
        "negative_rows_sample": negative["rows"][:12],
        "duplicate": duplicate["summary"],
        "duplicate_rows": duplicate["rows"],
        "cpu_seconds_delta": round(cpu_after - cpu_before, 4),
        "rss_before_mib": round(rss_before, 2),
        "rss_after_mib": round(rss_after, 2),
        "rss_delta_mib": round(rss_after - rss_before, 2),
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_report(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GamePath 壓力測試技術報告")
    lines.append("")
    lines.append(f"產生時間：`{results['generated_at']}`")
    lines.append("")
    lines.append("## 結論")
    best = results["sizes"][-1]
    lines.append(
        f"這次用隔離 SQLite 壓到 `{best['size']}` 筆本地攻略資料。核心 GamePath 檢索在最大資料量下 "
        f"Top-1 準確度為 `{pct(best['search']['top1'])}`，Hit@5 為 `{pct(best['search']['hit5'])}`，"
        f"搜尋 p95 為 `{best['search']['latency_ms']['p95']}` ms。"
    )
    lines.append("")
    lines.append(
        "判斷：目前演算法對「數千筆本地攻略包 + 小筆記」是夠用的；真正拖慢聊天體驗的通常不是 SQLite/RAG Lite，"
        "而是地端 Qwen 意圖判斷與提示整理。如果資料量上萬或大量同義詞不共享字面線索，就需要下一階段快取或 optional semantic reranker。"
    )
    lines.append("")
    lines.append("## 測試設計")
    lines.append("")
    lines.append("- 使用暫存 `IGPU_GAMEPATH_DIR`，不修改正式 `gamepath/gamepath.sqlite`。")
    lines.append("- 每個資料量都重新建立 SQLite、FTS5、chunk index 與 Markdown notes。")
    lines.append("- Golden 正解資料：12 筆，涵蓋 enemy、boss、item、route、puzzle、mechanic、npc、map、material、quest、location。")
    lines.append("- 干擾資料：同遊戲與不同遊戲混合，包含泛用攻略包、同區域、同 tag、不同 entity。")
    lines.append("- 查詢型態：exact、paraphrase、noisy_tag；另測 hard negative 與重複寫入。")
    lines.append("")
    lines.append("## 壓力測試總覽")
    lines.append("")
    lines.append("| 資料量 | DB KiB | 寫入 avg/p95 ms | 搜尋 Top-1 | Hit@5 | MRR@5 | 搜尋 avg/p95 ms | 負例誤命中 | 去重成功 | RSS delta MiB | CPU sec |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in results["sizes"]:
        lines.append(
            "| {size} | {db_kib:.2f} | {wavg}/{wp95} | {top1} | {hit5} | {mrr} | {savg}/{sp95} | {fp} | {dup} | {rss} | {cpu} |".format(
                size=row["size"],
                db_kib=row["counts"]["db_kib"],
                wavg=row["write_latency_ms"]["avg"],
                wp95=row["write_latency_ms"]["p95"],
                top1=pct(row["search"]["top1"]),
                hit5=pct(row["search"]["hit5"]),
                mrr=f"{row['search']['mrr5']:.4f}",
                savg=row["search"]["latency_ms"]["avg"],
                sp95=row["search"]["latency_ms"]["p95"],
                fp=pct(row["negative"]["false_positive_rate"]),
                dup=pct(row["duplicate"]["ok_rate"]),
                rss=row["rss_delta_mib"],
                cpu=row["cpu_seconds_delta"],
            )
        )
    lines.append("")
    lines.append("## 分類準確度")
    lines.append("")
    lines.append("| 資料量 | exact | paraphrase | noisy_tag |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for row in results["sizes"]:
        by_kind = row["search"]["by_kind"]
        lines.append(
            f"| {row['size']} | {pct(by_kind.get('exact', 0.0))} | {pct(by_kind.get('paraphrase', 0.0))} | {pct(by_kind.get('noisy_tag', 0.0))} |"
        )
    lines.append("")
    lines.append("## 代表查詢樣本")
    lines.append("")
    sample = results["sizes"][-1]["search_rows_sample"]
    lines.append("| Query | Kind | Expected | Top | Rank | Score | Coverage | ms |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sample[:12]:
        lines.append(
            f"| {row['query']} | {row['kind']} | {row['expected_id']} | {row['top_id']} | {row['rank']} | {row['score']} | {row['coverage']} | {row['elapsed_ms']} |"
        )
    lines.append("")
    failures = results["sizes"][-1].get("search_failures") or []
    lines.append("## Top-1 失敗案例")
    lines.append("")
    if not failures:
        lines.append("最大資料量下沒有 Top-1 失敗案例。")
    else:
        lines.append("| Query | Kind | Expected | Top | Rank | Score | Coverage |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in failures[:12]:
            lines.append(
                f"| {row['query']} | {row['kind']} | {row['expected_id']} | {row['top_id']} | {row['rank']} | {row['score']} | {row['coverage']} |"
            )
    lines.append("")
    lines.append("## Hard Negative 觀察")
    lines.append("")
    lines.append(
        "Hard negative 是刻意把非攻略、UI 指令、最新版/網路意圖丟進純 GamePath 檢索。"
        "這不是正式 `/chat` 的完整路徑，因為正式流程前面還有 Qwen intent router。"
    )
    lines.append("")
    lines.append("| 資料量 | 負例誤命中 | 負例 avg/p95 ms |")
    lines.append("| ---: | ---: | ---: |")
    for row in results["sizes"]:
        lines.append(
            f"| {row['size']} | {pct(row['negative']['false_positive_rate'])} | {row['negative']['latency_ms']['avg']}/{row['negative']['latency_ms']['p95']} |"
        )
    lines.append("")
    lines.append(
        "壓力測試顯示：如果繞過 router、無腦把每句話都丟 SQLite，部分泛用負例會被誤判為可整理的本地內容，"
        "而且在 5000 筆時無命中掃描 p95 會明顯變慢。因此生產流程必須保留 Qwen intent gate："
        "UI/一般聊天/最新版網路意圖先分流，不要每句都查 GamePath。"
    )
    lines.append("")
    lines.append("## 重複寫入測試")
    lines.append("")
    last_dup = results["sizes"][-1]["duplicate"]
    lines.append(
        f"最大資料量下，對 6 筆既有攻略用 Hermes 風格相似問法再次寫入：成功辨識重複 `{pct(last_dup['ok_rate'])}`，"
        f"entry count `{last_dup['entry_count_before']} -> {last_dup['entry_count_after']}`。"
    )
    lines.append("")
    lines.append("## 解讀")
    lines.append("")
    lines.append("1. SQLite FTS5 + chunk RAG Lite 對數千筆攻略的核心檢索成本仍在遊戲互動可接受範圍，通常低於一次地端 Qwen 呼叫。")
    lines.append("2. 這版修正後，metadata/tag 不再壓過文字覆蓋率；noisy tag 壓力測試可用來防止大攻略包壓掉精準小筆記。")
    lines.append("3. 重複寫入已能擋住語意相近的 Hermes 回答，避免 GamePath 被同一題不同問法洗版。")
    lines.append("4. 目前弱點是純字面/中文 n-gram 還不是完整語意向量；如果玩家完全換一套說法、沒有共通詞，仍可能需要 Qwen retrieval evaluator 或未來 optional embedding。")
    lines.append("5. 生產聊天流程的總延遲會比本報告高，因為 `/chat` 還包含 Qwen intent router、Qwen hint summary 或 Hermes；本報告主要測 GamePath 演算法本體。")
    lines.append("")
    lines.append("## 建議門檻")
    lines.append("")
    lines.append("- 目前可支撐：數千筆本地攻略、小型攻略包、玩家常見中文相似問法。")
    lines.append("- 建議警戒：超過 10k 筆或大量長篇 wiki 匯入後，應加查詢快取、熱門 entry cache、或 optional reranker。")
    lines.append("- 下一步：把生產 `/chat` 的 Qwen intent/hint 結果做短 TTL cache，這會比繼續壓 SQLite 更能改善玩家體感。")
    lines.append("")
    lines.append("## 產物")
    lines.append("")
    lines.append(f"- JSON raw results：`{RESULTS_PATH}`")
    lines.append(f"- Markdown report：`{REPORT_PATH}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="100,500,2000,5000")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--keep-db", action="store_true")
    args = parser.parse_args()

    sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]
    os.environ.setdefault("IGPU_LOCAL_ROUTER_ENABLED", "0")
    import llama_vulkan_api_server as server  # noqa: WPS433

    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    temp_parent = Path(tempfile.mkdtemp(prefix="gamepath_stress_", dir=str(PROJECT_ROOT / "logs")))
    results: dict[str, Any] = {
        "generated_at": generated_at,
        "sizes_requested": sizes,
        "repeats": args.repeats,
        "temp_parent": str(temp_parent),
        "sizes": [],
    }
    try:
        for size in sizes:
            print(f"[stress] running size={size} repeats={args.repeats}", flush=True)
            results["sizes"].append(run_size(server, temp_parent, size, args.repeats))
            gc.collect()
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        REPORT_PATH.write_text(render_report(results), encoding="utf-8")
        print(f"[stress] wrote {RESULTS_PATH}")
        print(f"[stress] wrote {REPORT_PATH}")
    finally:
        if not args.keep_db:
            shutil.rmtree(temp_parent, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
