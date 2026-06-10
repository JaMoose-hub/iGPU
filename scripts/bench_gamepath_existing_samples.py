#!/usr/bin/env python3
"""Validate representative queries against the current production GamePath DB.

This script is read-only. It builds a small, human-readable test set from
entries that already exist in gamepath/gamepath.sqlite, then records whether
similar player questions can retrieve those saved entries.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import llama_vulkan_api_server as server  # noqa: E402


OUTPUT_PATH = PROJECT_ROOT / "docs" / "gamepath_existing_sample_tests.json"


SAMPLE_CASES: list[dict[str, Any]] = [
    {
        "label": "旅館路線",
        "expected_id": 75,
        "query": "鷦木旅館辦公室地圖要在哪拿？",
        "tags": ["route", "location"],
        "note": "攻略包路線 / 地圖取得",
    },
    {
        "label": "廚房餐廳路線",
        "expected_id": 79,
        "query": "療養院一樓西側廚房餐廳要怎麼走？",
        "tags": ["route", "enemy"],
        "note": "攻略包路線 / 敵人避開",
    },
    {
        "label": "等候室道具",
        "expected_id": 83,
        "query": "等候室後面是不是要去遊樂室兌換道具？",
        "tags": ["route", "puzzle"],
        "spoiler_level": "medium",
        "note": "道具兌換 / 路線提示",
    },
    {
        "label": "決戰怪物",
        "expected_id": 94,
        "query": "淨水場那個怪物女孩決戰要先啟動哪幾個系統？",
        "tags": ["enemy", "route"],
        "spoiler_level": "medium",
        "note": "Boss/敵人路線",
    },
    {
        "label": "彩蛋收集",
        "expected_id": 101,
        "query": "浣熊市警局豆腐先生彩蛋在哪段路線？",
        "tags": ["item", "location"],
        "note": "彩蛋 / 收集品",
    },
    {
        "label": "研究主任室謎題",
        "expected_id": 124,
        "query": "前往研究主任室在機關盒處輸入太陽太陽星星的那個長順序是哪筆？",
        "tags": ["puzzle"],
        "spoiler_level": "high",
        "note": "高劇透密碼 / 謎題答案",
    },
    {
        "label": "主廚喪屍",
        "expected_id": 128,
        "query": "遊樂室外主廚喪屍和酒吧保險箱是哪一段？",
        "tags": ["enemy", "puzzle"],
        "spoiler_level": "medium",
        "note": "敵人 + 保險箱混合段落",
    },
    {
        "label": "酒吧保險箱",
        "expected_id": 131,
        "query": "酒吧休閒室保險箱左10右80左30在哪筆攻略？",
        "tags": ["puzzle"],
        "spoiler_level": "medium",
        "note": "保險箱密碼",
    },
    {
        "label": "孢源喪屍弱點",
        "expected_id": 133,
        "query": "教堂孢源喪屍弱點是不是發光部位？",
        "tags": ["enemy"],
        "spoiler_level": "medium",
        "note": "敵人弱點",
    },
    {
        "label": "泰坦巨蛛",
        "expected_id": 134,
        "query": "NF大廈泰坦巨蛛弱點在哪？",
        "tags": ["enemy"],
        "spoiler_level": "medium",
        "note": "Boss 弱點",
    },
    {
        "label": "Hermes 自動保存",
        "expected_id": 139,
        "query": "廚房屠夫弱點是什麼？要硬打嗎？",
        "tags": ["boss", "enemy"],
        "note": "Hermes web 濃縮後保存",
    },
    {
        "label": "Vision 保存",
        "expected_id": 140,
        "query": "黑白格地板昏暗走廊這是哪裡？",
        "tags": ["vision", "mechanic"],
        "note": "截圖/vision 產生的 GamePath 筆記",
    },
    {
        "label": "唱歌女殭屍",
        "expected_id": 143,
        "query": "唱歌女殭屍在哪出沒？",
        "tags": ["route", "location"],
        "note": "相似問法命中已保存條目",
    },
]


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    results = server.search_gamepath_multi_query_sync(
        case["query"],
        "RESIDENT_EVIL_requiem",
        5,
        tags=case.get("tags"),
        spoiler_level=case.get("spoiler_level", "low"),
    )
    evaluation = server.evaluate_gamepath_retrieval(
        case["query"],
        "RESIDENT_EVIL_requiem",
        results,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    evaluated_results = list(evaluation.get("results") or results)
    expected_id = int(case["expected_id"])
    rank = None
    for index, item in enumerate(evaluated_results, 1):
        if int(item.get("id") or 0) == expected_id:
            rank = index
            break
    top = evaluated_results[0] if evaluated_results else {}
    return {
        **case,
        "top_id": int(evaluation.get("top_id") or 0),
        "top_title": evaluation.get("top_title") or "",
        "rank": rank,
        "hit_top1": rank == 1,
        "hit_top5": bool(rank and rank <= 5),
        "confidence": evaluation.get("confidence"),
        "score": evaluation.get("score"),
        "gap": evaluation.get("gap"),
        "coverage": top.get("match_coverage"),
        "core_overlap": top.get("core_overlap"),
        "elapsed_ms": elapsed_ms,
    }


def main() -> int:
    rows = [run_case(case) for case in SAMPLE_CASES]
    top1 = [row["hit_top1"] for row in rows]
    hit5 = [row["hit_top5"] for row in rows]
    elapsed = [float(row["elapsed_ms"]) for row in rows]
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "db_path": str(server.GAMEPATH_DB),
        "case_count": len(rows),
        "summary": {
            "top1": round(sum(top1) / len(top1), 4),
            "hit5": round(sum(hit5) / len(hit5), 4),
            "avg_ms": round(sum(elapsed) / len(elapsed), 3),
            "min_ms": round(min(elapsed), 3),
            "max_ms": round(max(elapsed), 3),
        },
        "rows": rows,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
