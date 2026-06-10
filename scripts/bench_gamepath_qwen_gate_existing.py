#!/usr/bin/env python3
"""Benchmark Qwen intent routing in front of the production GamePath DB.

This benchmark is read-only. It compares:

- Pure GamePath retrieval, especially hard negatives that should not query
  the local guide DB.
- The actual local Qwen intent router followed by GamePath retrieval only
  when the router chooses gamepath_query.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import llama_vulkan_api_server as server  # noqa: E402
from scripts.bench_gamepath_existing_samples import SAMPLE_CASES  # noqa: E402


OUTPUT_PATH = PROJECT_ROOT / "docs" / "gamepath_qwen_gate_existing_tests.json"
GAME_ID = "RESIDENT_EVIL_requiem"


EXTRA_POSITIVE_CASES: list[dict[str, Any]] = [
    {
        "label": "警衛室門禁卡",
        "expected_id": 78,
        "query": "警衛室和投藥室哪裡拿西側門禁卡？",
        "tags": ["item", "route"],
        "spoiler_level": "low",
        "note": "道具 / 門禁卡路線",
    },
    {
        "label": "血液研究室車庫",
        "expected_id": 85,
        "query": "血液研究室分析血液後六角扳手和車庫器官箱在哪？",
        "tags": ["puzzle", "route"],
        "spoiler_level": "medium",
        "note": "謎題 / 道具路線",
    },
    {
        "label": "隔離病房腕帶",
        "expected_id": 88,
        "query": "隔離病房和警備主管室要拿幾級身份識別腕帶？",
        "tags": ["puzzle", "route"],
        "spoiler_level": "medium",
        "note": "腕帶 / 區域推進",
    },
    {
        "label": "三樓肥碩喪屍",
        "expected_id": 89,
        "query": "三樓閣樓的肥碩喪屍那段要往哪走？",
        "tags": ["enemy", "route"],
        "spoiler_level": "medium",
        "note": "敵人 / 三樓路線",
    },
    {
        "label": "貴賓室地下研究所",
        "expected_id": 92,
        "query": "中庭到貴賓室謎題和地下研究所是哪段攻略？",
        "tags": ["puzzle", "route"],
        "spoiler_level": "medium",
        "note": "中庭 / 地下研究所",
    },
    {
        "label": "鷹之星物流倉庫",
        "expected_id": 96,
        "query": "鷹之星物流倉庫三樓拿雪松溪公寓鑰匙和屋頂分電器在哪？",
        "tags": ["route", "item"],
        "spoiler_level": "low",
        "note": "Part2 / 鑰匙與分電器",
    },
    {
        "label": "公寓地下停車場電池",
        "expected_id": 97,
        "query": "公寓地下停車場和汙水處理設施拿電池開捲簾門怎麼走？",
        "tags": ["route", "item"],
        "spoiler_level": "low",
        "note": "電池 / 捲簾門",
    },
    {
        "label": "雪松溪公寓集裝箱",
        "expected_id": 98,
        "query": "雪松溪公寓門口BSAA集裝箱鑰匙和信號接收器在哪？",
        "tags": ["route", "item"],
        "spoiler_level": "low",
        "note": "集裝箱 / 信號接收器",
    },
    {
        "label": "加油站燃油桶",
        "expected_id": 99,
        "query": "加油站燃油桶和下水道到4號集裝箱是哪一段？",
        "tags": ["route", "item"],
        "spoiler_level": "low",
        "note": "燃油桶 / 下水道",
    },
    {
        "label": "格裡姆斯通繼電器",
        "expected_id": 100,
        "query": "格裡姆斯通大樓屋頂拿繼電器下樓引爆炸彈怎麼走？",
        "tags": ["route", "item"],
        "spoiler_level": "low",
        "note": "繼電器 / 炸彈路線",
    },
    {
        "label": "孤兒院路線",
        "expected_id": 103,
        "query": "孤兒院臥室和育兒室是哪一段路線？",
        "tags": ["route", "location"],
        "spoiler_level": "low",
        "note": "孤兒院 / 區域路線",
    },
    {
        "label": "警局暴君追逐",
        "expected_id": 104,
        "query": "浣熊市警局東側暴君追逐戰到私人收藏室怎麼走？",
        "tags": ["enemy", "route"],
        "spoiler_level": "medium",
        "note": "暴君追逐 / 警局東側",
    },
    {
        "label": "方舟伺服器室",
        "expected_id": 107,
        "query": "方舟伺服器室庫房精英衛隊指揮官是哪段？",
        "tags": ["enemy", "route"],
        "spoiler_level": "low",
        "note": "方舟 / 伺服器室",
    },
    {
        "label": "生化武器儲存室舔食者",
        "expected_id": 110,
        "query": "生化武器儲存室11和12舔食者大戰是哪段攻略？",
        "tags": ["enemy", "route"],
        "spoiler_level": "low",
        "note": "方舟 / 舔食者大戰",
    },
    {
        "label": "顯微鏡分析斷手",
        "expected_id": 122,
        "query": "血液研究室顯微鏡分析斷手是哪一步？",
        "tags": ["puzzle", "item"],
        "spoiler_level": "medium",
        "note": "斷手 / 顯微鏡",
    },
    {
        "label": "理事長室紅寶石",
        "expected_id": 127,
        "query": "理事長室拿紅寶石和月之晶石謎題怎麼解？",
        "tags": ["puzzle", "item"],
        "spoiler_level": "medium",
        "note": "紅寶石 / 月之晶石",
    },
    {
        "label": "檢查室保險箱",
        "expected_id": 129,
        "query": "檢查室保險箱右30左10右50在哪筆攻略？",
        "tags": ["puzzle"],
        "spoiler_level": "medium",
        "note": "檢查室保險箱密碼",
    },
]

POSITIVE_CASES: list[dict[str, Any]] = [*SAMPLE_CASES, *EXTRA_POSITIVE_CASES]


NEGATIVE_CASES: list[dict[str, Any]] = [
    {
        "label": "一般聊天",
        "query": "我今天心情不好，先陪我聊一下。",
        "expected": "not_gamepath",
        "note": "情緒聊天，不應查攻略庫。",
    },
    {
        "label": "UI 指令",
        "query": "打開 GamePath 視窗讓我看資料庫。",
        "expected": "not_gamepath",
        "note": "軟體控制指令，不應查 SQLite。",
    },
    {
        "label": "透明度設定",
        "query": "把視窗透明度調到 70%。",
        "expected": "not_gamepath",
        "note": "Overlay 設定，不是攻略問題。",
    },
    {
        "label": "最新版網路問題",
        "query": "最新 patch 廚房屠夫弱點有改嗎？幫我查網路。",
        "expected": "not_gamepath",
        "note": "應偏 Hermes/Tavily，不應只靠本地舊資料。",
    },
    {
        "label": "玩家記憶",
        "query": "幫我記錄我剛拿到紅寶石和地下室鑰匙。",
        "expected": "not_gamepath",
        "note": "任務/物品記憶寫入，不是搜尋攻略。",
    },
    {
        "label": "HUD 標記",
        "query": "幫我看畫面並圈出門在哪。",
        "expected": "not_gamepath",
        "note": "需要看畫面/HUD，不應用文字直接查攻略。",
    },
    {
        "label": "明確否定搜尋",
        "query": "先不要查攻略，我只是想確認你有沒有聽到。",
        "expected": "not_gamepath",
        "note": "玩家明確說不要查攻略。",
    },
    {
        "label": "架構說明",
        "query": "現在 GamePath 架構跟 Hermes 的關係再說明一次。",
        "expected": "not_gamepath",
        "note": "技術說明，不是遊戲攻略查詢。",
    },
]


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    return {
        "avg_ms": round(mean(values), 3),
        "p50_ms": round(median(values), 3),
        "p95_ms": round(percentile(values, 0.95), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def configure_router() -> dict[str, Any]:
    original = {
        "LOCAL_ROUTER_ENABLED": server.LOCAL_ROUTER_ENABLED,
        "LOCAL_ROUTER_GAMEPATH_GATE": server.LOCAL_ROUTER_GAMEPATH_GATE,
        "LOCAL_ROUTER_RETRIEVAL_EVAL": server.LOCAL_ROUTER_RETRIEVAL_EVAL,
        "LOCAL_ROUTER_ALWAYS_ROUTE": server.LOCAL_ROUTER_ALWAYS_ROUTE,
        "LOCAL_ROUTER_TIMEOUT_SECONDS": server.LOCAL_ROUTER_TIMEOUT_SECONDS,
        "LOCAL_ROUTER_DECISION_CACHE_TTL_SECONDS": server.LOCAL_ROUTER_DECISION_CACHE_TTL_SECONDS,
    }
    server.LOCAL_ROUTER_ENABLED = True
    server.LOCAL_ROUTER_GAMEPATH_GATE = True
    server.LOCAL_ROUTER_RETRIEVAL_EVAL = False
    server.LOCAL_ROUTER_ALWAYS_ROUTE = True
    server.LOCAL_ROUTER_TIMEOUT_SECONDS = 90
    server.LOCAL_ROUTER_DECISION_CACHE_TTL_SECONDS = 0
    server.local_router_decision_cache.clear()
    return original


def restore_router(original: dict[str, Any]) -> None:
    for key, value in original.items():
        setattr(server, key, value)
    server.local_router_decision_cache.clear()


def pure_gamepath_negative(case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    results = server.search_gamepath_multi_query_sync(
        case["query"],
        GAME_ID,
        5,
        spoiler_level="low",
    )
    evaluation = server.evaluate_gamepath_retrieval(case["query"], GAME_ID, results)
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    confidence = str(evaluation.get("confidence") or "miss")
    return {
        "queried_sqlite": True,
        "false_positive": bool(results) and confidence in {"direct", "summarize"},
        "confidence": confidence,
        "score": evaluation.get("score"),
        "top_id": int(evaluation.get("top_id") or 0),
        "top_title": evaluation.get("top_title") or "",
        "elapsed_ms": elapsed_ms,
        "hit_count": len(results),
    }


def search_after_router(
    query: str,
    decision: dict[str, Any],
    *,
    case: dict[str, Any] | None = None,
    use_router_query: bool = True,
) -> dict[str, Any]:
    if not decision.get("search_gamepath"):
        return {
            "queried_sqlite": False,
            "confidence": "skipped",
            "score": 0.0,
            "top_id": 0,
            "top_title": "",
            "rank": None,
            "hit_count": 0,
            "elapsed_ms": 0.0,
        }

    if use_router_query:
        search_query = str(decision.get("query") or query or "").strip()
        query_variants = list(decision.get("query_variants") or [])
        tags = server.normalize_router_tags(decision.get("tags"))
        spoiler_level = server.normalize_router_spoiler(decision.get("spoiler_level"))
    else:
        search_query = str(query or "").strip()
        query_variants = []
        tags = server.normalize_router_tags((case or {}).get("tags"))
        spoiler_level = server.normalize_spoiler_level((case or {}).get("spoiler_level", "low"))

    started = time.perf_counter()
    results = server.search_gamepath_multi_query_sync(
        search_query,
        GAME_ID,
        5,
        tags=tags or None,
        spoiler_level=spoiler_level,
        query_variants=query_variants,
    )
    if not results and tags:
        results = server.search_gamepath_multi_query_sync(
            search_query,
            GAME_ID,
            5,
            spoiler_level=spoiler_level,
            query_variants=query_variants,
        )
    if not results and spoiler_level not in {"high"}:
        results = server.search_gamepath_multi_query_sync(
            search_query,
            GAME_ID,
            5,
            tags=tags or None,
            spoiler_level="high",
            query_variants=query_variants,
        )
    evaluation = server.evaluate_gamepath_retrieval(search_query, GAME_ID, results)
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    return {
        "queried_sqlite": True,
        "query": search_query,
        "query_variants": query_variants,
        "tags": tags,
        "spoiler_level": spoiler_level,
        "confidence": evaluation.get("confidence"),
        "score": evaluation.get("score"),
        "gap": evaluation.get("gap"),
        "top_id": int(evaluation.get("top_id") or 0),
        "top_title": evaluation.get("top_title") or "",
        "results": evaluation.get("results") or results,
        "hit_count": len(evaluation.get("results") or results),
        "elapsed_ms": elapsed_ms,
    }


def run_router_decision(query: str) -> dict[str, Any]:
    guide_requested = server.should_use_guides(query, None)
    started = time.perf_counter()
    decision = server.local_router_gamepath_decision(query, GAME_ID, guide_requested)
    wall_ms = round((time.perf_counter() - started) * 1000.0, 3)
    decision_ms = round(float(decision.get("latency_ms") or wall_ms), 3)
    return {
        "decision": decision,
        "latency_ms": decision_ms,
        "wall_ms": wall_ms,
        "guide_requested": guide_requested,
    }


def run_positive_case(case: dict[str, Any]) -> dict[str, Any]:
    routed = run_router_decision(case["query"])
    search = search_after_router(
        case["query"],
        routed["decision"],
        case=case,
        use_router_query=False,
    )
    router_query_search = search_after_router(
        case["query"],
        routed["decision"],
        case=case,
        use_router_query=True,
    )
    expected_id = int(case["expected_id"])
    rank = None
    for index, item in enumerate(search.get("results") or [], 1):
        if int(item.get("id") or 0) == expected_id:
            rank = index
            break
    router_query_rank = None
    for index, item in enumerate(router_query_search.get("results") or [], 1):
        if int(item.get("id") or 0) == expected_id:
            router_query_rank = index
            break
    return {
        **case,
        "kind": "positive",
        "expected_route": "gamepath_query",
        "qwen_route": routed["decision"].get("intent_route"),
        "qwen_search_gamepath": bool(routed["decision"].get("search_gamepath")),
        "qwen_confidence": routed["decision"].get("confidence"),
        "qwen_reason": routed["decision"].get("raw_reason") or routed["decision"].get("reason"),
        "qwen_latency_ms": routed["latency_ms"],
        "sqlite_latency_ms": search["elapsed_ms"],
        "total_latency_ms": round(routed["wall_ms"] + search["elapsed_ms"], 3),
        "top_id": search["top_id"],
        "top_title": search["top_title"],
        "rank": rank,
        "hit_top1": rank == 1,
        "hit_top5": bool(rank and rank <= 5),
        "router_query_top_id": router_query_search["top_id"],
        "router_query_top_title": router_query_search["top_title"],
        "router_query_rank": router_query_rank,
        "router_query_hit_top1": router_query_rank == 1,
        "router_query_hit_top5": bool(router_query_rank and router_query_rank <= 5),
        "router_query_sqlite_latency_ms": router_query_search["elapsed_ms"],
        "route_ok": bool(routed["decision"].get("search_gamepath")),
        "retrieval_confidence": search["confidence"],
        "retrieval_score": search["score"],
        "queried_sqlite": search["queried_sqlite"],
        "search_strategy": "qwen_gate_original_query",
    }


def run_negative_case(case: dict[str, Any]) -> dict[str, Any]:
    pure = pure_gamepath_negative(case)
    routed = run_router_decision(case["query"])
    search = search_after_router(case["query"], routed["decision"])
    confidence = str(search.get("confidence") or "skipped")
    qwen_false_positive = bool(search.get("queried_sqlite")) and confidence in {"direct", "summarize"}
    return {
        **case,
        "kind": "negative",
        "pure_gamepath": pure,
        "qwen_route": routed["decision"].get("intent_route"),
        "qwen_search_gamepath": bool(routed["decision"].get("search_gamepath")),
        "qwen_confidence": routed["decision"].get("confidence"),
        "qwen_reason": routed["decision"].get("raw_reason") or routed["decision"].get("reason"),
        "qwen_latency_ms": routed["latency_ms"],
        "sqlite_latency_ms": search["elapsed_ms"],
        "total_latency_ms": round(routed["wall_ms"] + search["elapsed_ms"], 3),
        "blocked_gamepath": not bool(routed["decision"].get("search_gamepath")),
        "qwen_false_positive": qwen_false_positive,
        "retrieval_confidence": search["confidence"],
        "retrieval_score": search["score"],
        "top_id": search["top_id"],
        "top_title": search["top_title"],
        "queried_sqlite": search["queried_sqlite"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row["kind"] == "positive"]
    negatives = [row for row in rows if row["kind"] == "negative"]
    route_ok = [row["route_ok"] for row in positives]
    top1 = [row["hit_top1"] for row in positives]
    hit5 = [row["hit_top5"] for row in positives]
    router_query_top1 = [row["router_query_hit_top1"] for row in positives]
    router_query_hit5 = [row["router_query_hit_top5"] for row in positives]
    blocked = [row["blocked_gamepath"] for row in negatives]
    qwen_fp = [row["qwen_false_positive"] for row in negatives]
    pure_fp = [row["pure_gamepath"]["false_positive"] for row in negatives]
    qwen_lat = [float(row["qwen_latency_ms"]) for row in rows]
    sqlite_lat = [float(row["sqlite_latency_ms"]) for row in rows if row.get("queried_sqlite")]
    total_lat = [float(row["total_latency_ms"]) for row in rows]
    return {
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "positive_qwen_route_accuracy": round(sum(route_ok) / len(route_ok), 4) if route_ok else 0.0,
        "positive_top1": round(sum(top1) / len(top1), 4) if top1 else 0.0,
        "positive_hit5": round(sum(hit5) / len(hit5), 4) if hit5 else 0.0,
        "positive_router_query_top1": round(sum(router_query_top1) / len(router_query_top1), 4)
        if router_query_top1
        else 0.0,
        "positive_router_query_hit5": round(sum(router_query_hit5) / len(router_query_hit5), 4)
        if router_query_hit5
        else 0.0,
        "negative_qwen_block_rate": round(sum(blocked) / len(blocked), 4) if blocked else 0.0,
        "negative_qwen_false_positive_rate": round(sum(qwen_fp) / len(qwen_fp), 4) if qwen_fp else 0.0,
        "negative_pure_gamepath_false_positive_rate": round(sum(pure_fp) / len(pure_fp), 4) if pure_fp else 0.0,
        "qwen_latency": stats(qwen_lat),
        "sqlite_latency_when_queried": stats(sqlite_lat),
        "end_to_end_latency": stats(total_lat),
    }


def main() -> int:
    original = configure_router()
    try:
        router_ready = server.local_router_ready()
        rows: list[dict[str, Any]] = []
        if not router_ready:
            print("WARNING: local Qwen router is not ready; results will not represent Qwen intent routing.")
        for case in POSITIVE_CASES:
            rows.append(run_positive_case(case))
        for case in NEGATIVE_CASES:
            rows.append(run_negative_case(case))
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "db_path": str(server.GAMEPATH_DB),
            "game_id": GAME_ID,
            "router_ready": router_ready,
            "local_router_url": server.LOCAL_ROUTER_URL,
            "local_router_model": server.LOCAL_ROUTER_MODEL,
            "case_count": len(rows),
            "summary": summarize(rows),
            "rows": rows,
        }
        OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        print(OUTPUT_PATH)
        return 0
    finally:
        restore_router(original)


if __name__ == "__main__":
    raise SystemExit(main())
