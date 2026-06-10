#!/usr/bin/env python3
"""Benchmark GamePath with 500 synthetic game-guide entries.

This benchmark is intentionally isolated from the production GamePath DB. It
generates a 500-entry game-guide corpus, runs previous-vs-current search
comparisons, and writes a full Markdown report plus machine-readable JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import llama_vulkan_api_server as server  # noqa: E402
import scripts.bench_gamepath_scope_policy as scope_bench  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "gamepath_500_benchmark_report.md"
RESULTS_PATH = PROJECT_ROOT / "docs" / "gamepath_500_benchmark_results.json"
DATASET_PATH = PROJECT_ROOT / "docs" / "gamepath_500_dataset.json"

TARGET_GAME = "GAMEPATH_BENCHMARK_ARPG"
OTHER_GAMES = [
    "SKY_FORGE_ODYSSEY",
    "NOCTURNE_ARCHIVE",
    "IRONWOOD_SURVIVAL",
    "STARLIGHT_TACTICS",
]

ENTITY_TYPES = [
    "boss",
    "item",
    "route",
    "puzzle",
    "mechanic",
    "npc",
    "map",
    "material",
    "enemy",
    "quest",
    "location",
    "character",
]

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
]

KEYWORDS = {
    "boss": ["怪", "怎麼打", "弱點", "走位", "硬直"],
    "item": ["用途", "鑰匙", "道具", "能用在哪", "開門"],
    "route": ["路線", "下一步", "往哪走", "關卡", "繞路"],
    "puzzle": ["解謎", "機關", "順序", "閥門", "符號"],
    "mechanic": ["互動", "觸發", "紅光", "啟動", "系統"],
    "npc": ["NPC", "對話", "商人", "名字", "任務"],
    "map": ["地圖", "位置", "標記", "在哪", "區域"],
    "material": ["素材", "配方", "製作", "升級", "掉落"],
    "enemy": ["敵人", "怪物", "殭屍", "處理", "巡邏"],
    "quest": ["任務", "支線", "目標", "回報", "線索"],
    "location": ["區域", "場景", "入口", "出口", "地點"],
    "character": ["角色", "名字", "身份", "線索", "對話"],
}


GOLDEN_ENTRIES: list[dict[str, Any]] = [
    {
        "key": "mist_kitchen_boss",
        "title": "霧港廚房屠夫打法",
        "question": "霧港廚房屠夫怪怎麼打",
        "query": "廚房屠夫怪怎麼處理",
        "answer": "保持中距離，等屠夫連砍第三下後繞到右側，打背後發亮弱點。不要貪刀，聽到喘息聲再補一輪。",
        "entity_type": "boss",
        "area": "霧港廚房",
        "entity_name": "屠夫怪",
    },
    {
        "key": "silver_key_item",
        "title": "銀鑰匙用途",
        "question": "銀鑰匙能用在哪",
        "query": "銀色鑰匙用途",
        "answer": "銀鑰匙用在地下酒窖入口的銀色鎖，不是西翼大廳的銅門。先從霧港廚房後門繞到酒窖。",
        "entity_type": "item",
        "area": "地下酒窖",
        "entity_name": "銀鑰匙",
    },
    {
        "key": "red_light_mechanic",
        "title": "紅光中庭互動",
        "question": "紅光中庭的紅光區域怎麼互動",
        "query": "紅光要怎麼弄",
        "answer": "先關掉牆邊電箱，再用紫外線燈照地面符號。紅光變成短閃時才能安全穿過。",
        "entity_type": "mechanic",
        "area": "紅光中庭",
        "entity_name": "紅光機制",
    },
    {
        "key": "hospital_route",
        "title": "舊醫院下一步",
        "question": "舊醫院那關下一步要做什麼",
        "query": "醫院下一步",
        "answer": "先去護理站拿保險絲，再回中央走廊開電梯。不要先進地下室，會少一個回程捷徑。",
        "entity_type": "route",
        "area": "舊醫院",
    },
    {
        "key": "boiler_valve_puzzle",
        "title": "鍋爐房三閥門解法",
        "question": "鍋爐房三個閥門怎麼解",
        "query": "三個閥門順序",
        "answer": "左閥低壓、中閥高壓、右閥低壓。看到紅燈就退回上一個閥門，等聲音變低再轉下一個。",
        "entity_type": "puzzle",
        "area": "鍋爐房",
        "entity_name": "三閥門",
    },
    {
        "key": "mirror_lake_map",
        "title": "鏡湖碼頭地圖碎片",
        "question": "鏡湖碼頭地圖碎片在哪",
        "query": "鏡湖地圖碎片位置",
        "answer": "地圖碎片在第二個木棧橋下方的箱子裡。先推開藍色漁網，從側邊小坡下去。",
        "entity_type": "map",
        "area": "鏡湖碼頭",
        "entity_name": "地圖碎片",
    },
    {
        "key": "clocktower_boss",
        "title": "鐘塔守門人打法",
        "question": "鐘塔頂層守門人怎麼打",
        "query": "鐘塔守門人弱點",
        "answer": "守門人的弱點在背後鐘擺核心。等它敲鐘後會短暫跪地，從左側樓梯繞背輸出。",
        "entity_type": "boss",
        "area": "鐘塔頂層",
        "entity_name": "守門人",
    },
    {
        "key": "blackwood_material",
        "title": "黑森林月銀枝用途",
        "question": "月銀枝這個素材能做什麼",
        "query": "月銀枝用途",
        "answer": "月銀枝用來升級靜音護符，也能交給資料室 NPC 換一個無暴雷提示。前期建議先留一根。",
        "entity_type": "material",
        "area": "黑森林入口",
        "entity_name": "月銀枝",
    },
    {
        "key": "archive_npc",
        "title": "資料室白衣 NPC",
        "question": "資料室白衣 NPC 要怎麼對話",
        "query": "白衣NPC對話選哪個",
        "answer": "第一次選「我只是路過」，第二次選「你在找誰」。這樣會開無暴雷提示線，不會直接跳劇情。",
        "entity_type": "npc",
        "area": "資料室",
        "entity_name": "白衣 NPC",
    },
    {
        "key": "east_hall_enemy",
        "title": "東翼唱歌女殭屍處理",
        "question": "東翼走廊會唱歌的女殭屍怎麼處理",
        "query": "唱歌女殭屍怎麼辦",
        "answer": "不要用跑步靠近。等歌聲停兩拍後蹲走通過，若被發現就退到紅布簾後面等巡邏重置。",
        "entity_type": "enemy",
        "area": "東翼走廊",
        "entity_name": "唱歌女殭屍",
    },
    {
        "key": "chapel_quest",
        "title": "沉沒禮拜堂支線目標",
        "question": "沉沒禮拜堂支線下一個目標是什麼",
        "query": "禮拜堂支線下一步",
        "answer": "先找三個沒有點火的燭台，只點亮靠近水痕的那一個。之後回入口聽鐘聲方向。",
        "entity_type": "quest",
        "area": "沉沒禮拜堂",
    },
    {
        "key": "west_hall_location",
        "title": "西翼大廳暗門入口",
        "question": "西翼大廳暗門入口在哪",
        "query": "西翼暗門在哪",
        "answer": "暗門在兩幅破損肖像中間。先把左邊肖像扶正，再按右下角裂縫。",
        "entity_type": "location",
        "area": "西翼大廳",
        "entity_name": "暗門入口",
    },
    {
        "key": "cellar_character",
        "title": "酒窖黑帽角色身份",
        "question": "地下酒窖黑帽角色是誰",
        "query": "黑帽角色名字",
        "answer": "目前不要直接認定身份。無暴雷提示是：他和資料室白衣 NPC 有同一枚徽章。",
        "entity_type": "character",
        "area": "地下酒窖",
        "entity_name": "黑帽角色",
    },
    {
        "key": "kitchen_route",
        "title": "霧港廚房怪門路線",
        "question": "霧港廚房怪門怎麼開",
        "query": "廚房怪門怎麼開",
        "answer": "怪門不是戰鬥目標。先拿牆上的小閥柄，裝到儲藏室旁邊的管線，再回來開門。",
        "entity_type": "route",
        "area": "霧港廚房",
        "entity_name": "怪門",
    },
    {
        "key": "courtyard_mechanic",
        "title": "中庭雨水機關",
        "question": "紅光中庭雨水機關怎麼觸發",
        "query": "中庭雨水機關觸發",
        "answer": "先把兩側排水口關上，再轉中央銅環。水位到第二格時停止，旁邊門鎖會鬆動。",
        "entity_type": "mechanic",
        "area": "紅光中庭",
        "entity_name": "雨水機關",
    },
    {
        "key": "hospital_item",
        "title": "舊醫院保險絲用途",
        "question": "舊醫院保險絲能用在哪",
        "query": "保險絲用在哪",
        "answer": "保險絲插在中央走廊電梯旁的灰色盒子，不是地下室發電機。先確認牆上燈號變綠。",
        "entity_type": "item",
        "area": "舊醫院",
        "entity_name": "保險絲",
    },
    {
        "key": "boiler_boss",
        "title": "鍋爐房鐵臂怪打法",
        "question": "鍋爐房鐵臂怪怎麼打",
        "query": "鐵臂怪怎麼打",
        "answer": "鐵臂怪怕蒸汽。引它撞破側邊管線，蒸汽噴出後打膝蓋，倒地再打背部。",
        "entity_type": "boss",
        "area": "鍋爐房",
        "entity_name": "鐵臂怪",
    },
    {
        "key": "dock_npc",
        "title": "鏡湖碼頭老人對話",
        "question": "鏡湖碼頭老人對話選項怎麼選",
        "query": "碼頭老人選項",
        "answer": "先選「你看過鐘聲嗎」，不要選追問身份。這會給碼頭地圖提示，不會提前揭露主線。",
        "entity_type": "npc",
        "area": "鏡湖碼頭",
        "entity_name": "碼頭老人",
    },
    {
        "key": "clocktower_puzzle",
        "title": "鐘塔四聲鐘解謎",
        "question": "鐘塔四聲鐘順序怎麼解",
        "query": "四聲鐘順序",
        "answer": "順序是低、高、中、低。每敲一次看地上影子方向，影子朝門時再敲下一次。",
        "entity_type": "puzzle",
        "area": "鐘塔頂層",
        "entity_name": "四聲鐘",
    },
    {
        "key": "forest_route",
        "title": "黑森林入口不迷路路線",
        "question": "黑森林入口迷路要往哪走",
        "query": "黑森林迷路怎麼走",
        "answer": "看樹上的白布條，連續兩個白布條後往左；如果看到破燈籠，代表走回原路了。",
        "entity_type": "route",
        "area": "黑森林入口",
    },
]


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
        return {"cpu_seconds": round(time.process_time(), 4), "rss_mib": None}


def build_answer(title: str, hint: str, entity_type: str, area: str, extra: str = "") -> str:
    return (
        f"{title}\n"
        f"類型：{entity_type}\n"
        f"區域：{area}\n"
        f"無劇透提示：{hint}\n"
        "玩家可用步驟：先確認畫面上的區域名稱與互動物件，再照提示執行；若沒有看到同名物件，回報找不到讓系統改走驗證模式。\n"
        f"{extra}".strip()
    )


def make_noise_entry(index: int) -> dict[str, Any]:
    entity_type = ENTITY_TYPES[index % len(ENTITY_TYPES)]
    area = AREAS[index % len(AREAS)]
    game_id = TARGET_GAME if index % 5 != 0 else OTHER_GAMES[index % len(OTHER_GAMES)]
    keyword_pack = " ".join(KEYWORDS[entity_type])
    overlap_pack = "廚房 屠夫 銀鑰匙 紅光 醫院 閥門 地圖 NPC 殭屍 支線 暗門 黑森林"
    entity_name = f"{area}{entity_type}測試物件{index + 1:03d}"
    title = f"{area}{entity_type}攻略測試 {index + 1:03d}"
    question = f"{area} {entity_type} {entity_name} 怎麼處理 {index + 1:03d}"
    hint = (
        f"這是 {area} 的 {entity_type} 類測試攻略，編號 {index + 1:03d}。"
        "它用來模擬大量攻略庫裡相似但不應置頂的候選。"
    )
    return {
        "key": f"noise_{index + 1:03d}",
        "game_id": game_id,
        "title": title,
        "question": question,
        "answer": build_answer(
            title,
            hint,
            entity_type,
            area,
            f"干擾關鍵字：{keyword_pack} {overlap_pack}。版本：v{1 + index % 4}.{index % 10}",
        ),
        "tags": [entity_type, area, "benchmark", f"noise-{index + 1:03d}"],
        "entity_type": entity_type,
        "entity_name": entity_name,
        "area": area,
        "version": f"{1 + index % 4}.{index % 10}",
        "source_quality": round(0.36 + (index % 7) * 0.035, 3),
    }


def make_dataset(total: int) -> list[dict[str, Any]]:
    if total < len(GOLDEN_ENTRIES):
        raise ValueError(f"total must be at least {len(GOLDEN_ENTRIES)}")
    dataset: list[dict[str, Any]] = []
    for entry in GOLDEN_ENTRIES:
        dataset.append(
            {
                "key": entry["key"],
                "game_id": TARGET_GAME,
                "title": entry["title"],
                "question": entry["question"],
                "answer": build_answer(entry["title"], entry["answer"], entry["entity_type"], entry["area"]),
                "tags": [entry["entity_type"], entry["area"], "golden", "benchmark"],
                "entity_type": entry["entity_type"],
                "entity_name": entry.get("entity_name", ""),
                "area": entry["area"],
                "version": "2.0",
                "source_quality": 0.9,
                "query": entry["query"],
                "is_golden": True,
            }
        )
    for index in range(total - len(dataset)):
        dataset.append(make_noise_entry(index))
    return dataset


def dataset_summary(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    by_game = Counter(str(item["game_id"]) for item in dataset)
    by_type = Counter(str(item["entity_type"]) for item in dataset)
    by_area = Counter(str(item["area"]) for item in dataset)
    return {
        "total_entries": len(dataset),
        "golden_entries": sum(1 for item in dataset if item.get("is_golden")),
        "noise_entries": sum(1 for item in dataset if not item.get("is_golden")),
        "games": dict(by_game.most_common()),
        "entity_types": dict(by_type.most_common()),
        "areas_top12": dict(by_area.most_common(12)),
    }


def seed_gamepath(dataset: list[dict[str, Any]]) -> tuple[dict[str, int], list[float]]:
    expected_ids: dict[str, int] = {}
    write_samples: list[float] = []
    for item in dataset:
        started = time.perf_counter()
        row = server.add_gamepath_sync(
            item["question"],
            item["answer"],
            item["game_id"],
            title=item["title"],
            tags=item["tags"],
            spoiler_level="low",
            source_type="benchmark",
            agent_used=False,
            version=item.get("version"),
            area=item.get("area"),
            entity_type=item.get("entity_type"),
            entity_name=item.get("entity_name"),
            source_quality=item.get("source_quality"),
        )
        write_samples.append((time.perf_counter() - started) * 1000.0)
        if item.get("is_golden"):
            expected_ids[str(item["key"])] = int(row["id"])
    return expected_ids, write_samples


def summarize_ranking(results: list[dict[str, Any]], expected_id: int) -> dict[str, Any]:
    rank = 0
    for index, item in enumerate(results[:5], start=1):
        if int(item.get("id") or 0) == expected_id:
            rank = index
            break
    top = results[0] if results else {}
    return {
        "rank": rank,
        "top1": rank == 1,
        "hit3": 1 <= rank <= 3,
        "hit5": 1 <= rank <= 5,
        "mrr": round(1.0 / rank, 4) if rank else 0.0,
        "top_id": int(top.get("id") or 0) if top else 0,
        "top_title": str(top.get("title") or ""),
        "top_entity_type": str(top.get("entity_type") or ""),
        "top_area": str(top.get("area") or ""),
        "top_score": top.get("metadata_match_score"),
        "result_count": len(results),
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
        "samples_ms": [round(sample, 4) for sample in samples],
        "avg_ms": round(mean(samples), 4),
        "p50_ms": round(median(samples), 4),
        "p95_ms": round(percentile(samples, 0.95), 4),
        "min_ms": round(min(samples), 4),
        "max_ms": round(max(samples), 4),
    }


def aggregate_cases(cases: list[dict[str, Any]], key: str) -> dict[str, Any]:
    rows = [case[key] for case in cases]
    return {
        "top1": round(mean([1.0 if row["top1"] else 0.0 for row in rows]), 4),
        "hit3": round(mean([1.0 if row["hit3"] else 0.0 for row in rows]), 4),
        "hit5": round(mean([1.0 if row["hit5"] else 0.0 for row in rows]), 4),
        "mrr": round(mean([row["mrr"] for row in rows]), 4),
        "avg_ms": round(mean([row["avg_ms"] for row in rows]), 4),
        "p50_ms": round(mean([row["p50_ms"] for row in rows]), 4),
        "p95_ms": round(mean([row["p95_ms"] for row in rows]), 4),
        "min_ms": round(min(row["min_ms"] for row in rows), 4),
        "max_ms": round(max(row["max_ms"] for row in rows), 4),
    }


def latency_stats(samples: list[float]) -> dict[str, Any]:
    return {
        "avg_ms": round(mean(samples), 4),
        "p50_ms": round(median(samples), 4),
        "p95_ms": round(percentile(samples, 0.95), 4),
        "min_ms": round(min(samples), 4),
        "max_ms": round(max(samples), 4),
    }


def run_benchmark(entries: int, runs: int) -> dict[str, Any]:
    dataset = make_dataset(entries)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "summary": dataset_summary(dataset),
                "entries": dataset,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with tempfile.TemporaryDirectory(prefix="gamepath-500-bench-") as tmp:
        scope_bench.configure_isolated_store(Path(tmp))
        before = process_snapshot()
        expected_ids, write_samples = seed_gamepath(dataset)
        server.ensure_gamepath_db()

        test_cases = [
            {
                "key": str(item["key"]),
                "query": str(item["query"]),
                "expected_title": str(item["title"]),
                "expected_id": expected_ids[str(item["key"])],
                "entity_type": str(item["entity_type"]),
                "area": str(item["area"]),
            }
            for item in dataset
            if item.get("is_golden")
        ]

        cases: list[dict[str, Any]] = []
        for case in test_cases:
            old = time_search(
                scope_bench.old_search_gamepath,
                case["query"],
                TARGET_GAME,
                case["expected_id"],
                runs,
            )
            new = time_search(
                lambda q, g, limit: server.search_gamepath_sync(q, g, limit),
                case["query"],
                TARGET_GAME,
                case["expected_id"],
                runs,
            )
            cases.append({**case, "old": old, "new": new})

        after = process_snapshot()
        conn = sqlite3.connect(server.GAMEPATH_DB)
        try:
            entry_count = int(conn.execute("SELECT COUNT(*) FROM gamepath_entries").fetchone()[0])
            chunk_count = int(conn.execute("SELECT COUNT(*) FROM gamepath_chunks").fetchone()[0])
            fts_count = int(conn.execute("SELECT COUNT(*) FROM gamepath_chunk_fts").fetchone()[0])
            db_size_kib = round(server.GAMEPATH_DB.stat().st_size / 1024.0, 2)
        finally:
            conn.close()

        before_cpu = float(before.get("cpu_seconds") or 0.0)
        after_cpu = float(after.get("cpu_seconds") or 0.0)
        before_rss = before.get("rss_mib")
        after_rss = after.get("rss_mib")
        data = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "target_game": TARGET_GAME,
            "entries_requested": entries,
            "runs_per_query_per_path": runs,
            "dataset_path": str(DATASET_PATH),
            "report_path": str(REPORT_PATH),
            "results_path": str(RESULTS_PATH),
            "dataset_summary": dataset_summary(dataset),
            "store": {
                "entry_count": entry_count,
                "chunk_count": chunk_count,
                "fts_count": fts_count,
                "db_size_kib": db_size_kib,
            },
            "write_latency": latency_stats(write_samples),
            "process_before": before,
            "process_after": after,
            "process_delta": {
                "cpu_seconds": round(after_cpu - before_cpu, 4),
                "rss_mib": round(float(after_rss) - float(before_rss), 2)
                if before_rss is not None and after_rss is not None
                else None,
            },
            "summary": {
                "old": aggregate_cases(cases, "old"),
                "new": aggregate_cases(cases, "new"),
            },
            "cases": cases,
        }
        gc.collect()
        return data


def pct(value: float) -> str:
    return f"{value:.2%}"


def markdown_report(data: dict[str, Any]) -> str:
    summary = data["summary"]
    dataset = data["dataset_summary"]
    store = data["store"]
    lines = [
        "# GamePath 500 筆測試資料 Benchmark",
        "",
        f"產生時間：`{data['generated_at']}`",
        "",
        "## 結論",
        "",
        "本測試用隔離暫存 SQLite 建立 500 筆合成遊戲攻略資料，並使用同一批資料比較前一版 `chunk FTS only` 與新版 `metadata scope + ranking` 搜尋路徑。正式 GamePath/Memory 沒有被修改。",
        "",
        "測試過程中發現新版在 500 筆資料下會因 metadata scope 過度過濾而掉分，因此已補上非 strict 搜尋的 unscoped fallback 合併排序，並加強 `白衣NPC` / `白衣 NPC` 這類名稱比對與 `character` 類型推斷。本報告數字為修正後結果。",
        "",
        "| 指標 | 前一版 | 新版 |",
        "| --- | ---: | ---: |",
        f"| Top-1 精準度 | {pct(summary['old']['top1'])} | {pct(summary['new']['top1'])} |",
        f"| Hit@3 | {pct(summary['old']['hit3'])} | {pct(summary['new']['hit3'])} |",
        f"| Hit@5 | {pct(summary['old']['hit5'])} | {pct(summary['new']['hit5'])} |",
        f"| MRR@5 | {summary['old']['mrr']:.4f} | {summary['new']['mrr']:.4f} |",
        f"| 平均查詢時間 ms | {summary['old']['avg_ms']} | {summary['new']['avg_ms']} |",
        f"| 平均 p50 ms | {summary['old']['p50_ms']} | {summary['new']['p50_ms']} |",
        f"| 平均 p95 ms | {summary['old']['p95_ms']} | {summary['new']['p95_ms']} |",
        "",
        "## 測試資料內容",
        "",
        f"- 總筆數：`{dataset['total_entries']}`",
        f"- 目標正解資料：`{dataset['golden_entries']}`",
        f"- 干擾資料：`{dataset['noise_entries']}`",
        f"- 測試遊戲：`{data['target_game']}`",
        f"- 完整 500 筆資料：`{data['dataset_path']}`",
        f"- SQLite entries：`{store['entry_count']}`",
        f"- RAG chunks：`{store['chunk_count']}`",
        f"- FTS rows：`{store['fts_count']}`",
        f"- DB 大小：`{store['db_size_kib']}` KiB",
        "",
        "### 遊戲分布",
        "",
        "| Game | Count |",
        "| --- | ---: |",
    ]
    for game, count in dataset["games"].items():
        lines.append(f"| {game} | {count} |")
    lines.extend(["", "### 類型分布", "", "| Entity type | Count |", "| --- | ---: |"])
    for entity_type, count in dataset["entity_types"].items():
        lines.append(f"| {entity_type} | {count} |")
    lines.extend(["", "### 代表資料樣本", "", "| Key | Title | Question | Type | Area |", "| --- | --- | --- | --- | --- |"])
    sample_entries = [case for case in data["cases"][:12]]
    for case in sample_entries:
        lines.append(
            f"| {case['key']} | {case['expected_title']} | {case['query']} | {case['entity_type']} | {case['area']} |"
        )
    lines.extend(
        [
            "",
            "## 寫入時間",
            "",
            "| 指標 | ms |",
            "| --- | ---: |",
            f"| 平均 | {data['write_latency']['avg_ms']} |",
            f"| p50 | {data['write_latency']['p50_ms']} |",
            f"| p95 | {data['write_latency']['p95_ms']} |",
            f"| min | {data['write_latency']['min_ms']} |",
            f"| max | {data['write_latency']['max_ms']} |",
            "",
            "## 每次查詢時間與精準度",
            "",
            f"每個查詢在每條路徑重跑 `{data['runs_per_query_per_path']}` 次；每一次 run 的 raw samples 已保存在 `{data['results_path']}`。",
            "",
            "| Case | Query | Type | 前一版 Top | 新版 Top | 前一版 Rank | 新版 Rank | 前一版 avg/p95 ms | 新版 avg/p95 ms |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in data["cases"]:
        lines.append(
            "| {key} | {query} | {etype} | {old_top} | {new_top} | {old_rank} | {new_rank} | {old_avg}/{old_p95} | {new_avg}/{new_p95} |".format(
                key=case["key"],
                query=case["query"],
                etype=case["entity_type"],
                old_top=case["old"]["top_title"],
                new_top=case["new"]["top_title"],
                old_rank=case["old"]["rank"],
                new_rank=case["new"]["rank"],
                old_avg=case["old"]["avg_ms"],
                old_p95=case["old"]["p95_ms"],
                new_avg=case["new"]["avg_ms"],
                new_p95=case["new"]["p95_ms"],
            )
        )
    lines.extend(
        [
            "",
            "## CPU / RAM 觀測",
            "",
            "| 指標 | 值 |",
            "| --- | ---: |",
            f"| Benchmark process CPU delta seconds | {data['process_delta']['cpu_seconds']} |",
            f"| Benchmark process RSS RAM delta MiB | {data['process_delta']['rss_mib']} |",
            "",
            "## 技術解讀",
            "",
            "- 前一版 FTS-only 很快，但在大量相似資料中容易被重複關鍵字或跨類型資料干擾。",
            "- 新版會用 `game_id`、`entity_type`、`area`、`source_quality` 與 metadata ranking 重新排序，並在非 strict 模式合併未套 scope filter 的候選，避免推斷錯誤時直接漏掉正解。",
            "- 新版查詢會多花約十幾毫秒，原因是多了 scope inference、兩段候選搜尋、chunk grouping、metadata scoring；但仍是本地 SQLite 級別，遠低於 Hermes/Tavily 或雲端模型延遲。",
            "- 名稱查詢需要做正規化：例如 `白衣NPC` 與 `白衣 NPC` 應視為同一候選，否則大量資料下會被其他 NPC 對話攻略干擾。",
            "- 500 筆資料仍不是上限測試。下一階段若要驗證數千到數萬筆，應加入分頁查詢、冷/熱 cache、長文 chunk 數量與多遊戲 corpus 比例。",
            "",
            "## 剩餘風險",
            "",
            "- 這批資料是合成資料，能測演算法抗干擾能力，但不能完全代表真實玩家攻略語料。",
            "- Metadata 若由規則推斷錯誤，搜尋仍可能掉分；大量匯入時建議由 importer 或 Agent 明確寫入 metadata。",
            "- 若未來攻略文章變成長篇 wiki，仍建議加 embedding/reranker adapter 做第二階段 rerank。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", type=int, default=500)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    data = run_benchmark(max(len(GOLDEN_ENTRIES), args.entries), max(1, args.runs))
    RESULTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(markdown_report(data), encoding="utf-8", newline="\n")
    print(json.dumps(data["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {RESULTS_PATH}")
    print(f"Wrote {DATASET_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
