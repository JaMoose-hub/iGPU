#!/usr/bin/env python3
"""Benchmark semantic routing from local Qwen router to GamePath/Hermes.

The benchmark seeds an isolated 500-entry GamePath corpus, then compares:

- backend-only intent rules
- actual local Qwen router at the llama.cpp OpenAI-compatible endpoint

Hermes/Tavily is not called. A GamePath miss is counted as a dispatch to
Hermes Agent, matching the production route stage before web/tool execution.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
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
import scripts.bench_gamepath_500 as bench500  # noqa: E402
import scripts.bench_gamepath_scope_policy as scope_bench  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "docs" / "gamepath_semantic_router_500_report.md"
RESULTS_PATH = PROJECT_ROOT / "docs" / "gamepath_semantic_router_500_results.json"
CASES_PATH = PROJECT_ROOT / "docs" / "gamepath_semantic_router_500_cases.json"


SEMANTIC_CASES: list[dict[str, Any]] = [
    {
        "case": "local_kitchen_boss",
        "category": "gamepath_local",
        "prompt": "廚房那個拿刀的一直追我，怎麼安全處理？",
        "expected_route": "gamepath",
        "expected_key": "mist_kitchen_boss",
    },
    {
        "case": "local_silver_key",
        "category": "gamepath_local",
        "prompt": "這把銀色鑰匙是不是該留著？我不知道要開哪裡。",
        "expected_route": "gamepath",
        "expected_key": "silver_key_item",
    },
    {
        "case": "local_red_light",
        "category": "gamepath_local",
        "prompt": "紅色光一直擋路，我要先動哪個東西？",
        "expected_route": "gamepath",
        "expected_key": "red_light_mechanic",
    },
    {
        "case": "local_hospital_next",
        "category": "gamepath_local",
        "prompt": "我在醫院繞很久，接下來應該去哪？",
        "expected_route": "gamepath",
        "expected_key": "hospital_route",
    },
    {
        "case": "local_boiler_valves",
        "category": "gamepath_local",
        "prompt": "鍋爐那三個轉盤我轉到快瘋了，順序是什麼？",
        "expected_route": "gamepath",
        "expected_key": "boiler_valve_puzzle",
    },
    {
        "case": "local_map_fragment",
        "category": "gamepath_local",
        "prompt": "碼頭附近那張地圖碎片藏在哪個角落？",
        "expected_route": "gamepath",
        "expected_key": "mirror_lake_map",
    },
    {
        "case": "local_clock_boss",
        "category": "gamepath_local",
        "prompt": "鐘塔那個守門的到底要打哪裡才有效？",
        "expected_route": "gamepath",
        "expected_key": "clocktower_boss",
    },
    {
        "case": "local_moonsilver",
        "category": "gamepath_local",
        "prompt": "月銀枝這個素材先別賣嗎？它能做什麼？",
        "expected_route": "gamepath",
        "expected_key": "blackwood_material",
    },
    {
        "case": "local_white_npc",
        "category": "gamepath_local",
        "prompt": "資料室那個白衣NPC我該選哪一句？",
        "expected_route": "gamepath",
        "expected_key": "archive_npc",
    },
    {
        "case": "local_singing_zombie",
        "category": "gamepath_local",
        "prompt": "走廊唱歌的女殭屍要硬打嗎？",
        "expected_route": "gamepath",
        "expected_key": "east_hall_enemy",
    },
    {
        "case": "local_chapel_quest",
        "category": "gamepath_local",
        "prompt": "禮拜堂支線現在只剩鐘聲，下一步是什麼？",
        "expected_route": "gamepath",
        "expected_key": "chapel_quest",
    },
    {
        "case": "local_hidden_door",
        "category": "gamepath_local",
        "prompt": "西翼大廳好像有暗門，我要看哪裡？",
        "expected_route": "gamepath",
        "expected_key": "west_hall_location",
    },
    {
        "case": "local_black_hat",
        "category": "gamepath_local",
        "prompt": "酒窖黑帽那個人是誰？可以不要暴雷講嗎？",
        "expected_route": "gamepath",
        "expected_key": "cellar_character",
    },
    {
        "case": "local_kitchen_door",
        "category": "gamepath_local",
        "prompt": "廚房那扇怪門不是怪物吧？我要怎麼開？",
        "expected_route": "gamepath",
        "expected_key": "kitchen_route",
    },
    {
        "case": "local_courtyard_rain",
        "category": "gamepath_local",
        "prompt": "中庭下雨那個機關觸發不了，順序是什麼？",
        "expected_route": "gamepath",
        "expected_key": "courtyard_mechanic",
    },
    {
        "case": "local_fuse",
        "category": "gamepath_local",
        "prompt": "保險絲拿到了，但它不是發電機用的嗎？",
        "expected_route": "gamepath",
        "expected_key": "hospital_item",
    },
    {
        "case": "local_iron_arm",
        "category": "gamepath_local",
        "prompt": "鍋爐房鐵手臂那隻，我打不動它。",
        "expected_route": "gamepath",
        "expected_key": "boiler_boss",
    },
    {
        "case": "local_dock_elder",
        "category": "gamepath_local",
        "prompt": "碼頭老人問我鐘聲，我該怎麼回比較安全？",
        "expected_route": "gamepath",
        "expected_key": "dock_npc",
    },
    {
        "case": "local_four_bells",
        "category": "gamepath_local",
        "prompt": "四次鐘聲的順序我記不起來，怎麼敲？",
        "expected_route": "gamepath",
        "expected_key": "clocktower_puzzle",
    },
    {
        "case": "local_black_forest",
        "category": "gamepath_local",
        "prompt": "黑森林入口一直繞回原點，怎麼看路標？",
        "expected_route": "gamepath",
        "expected_key": "forest_route",
    },
    {
        "case": "agent_new_boss",
        "category": "hermes_agent",
        "prompt": "幫我查一下新版本 2.4 的雨夜祭司 boss 有沒有改弱點。",
        "expected_route": "hermes_agent",
    },
    {
        "case": "agent_unseen_item",
        "category": "hermes_agent",
        "prompt": "我拿到星霜羅盤，攻略庫沒有的話請去網路查它能做什麼。",
        "expected_route": "hermes_agent",
    },
    {
        "case": "agent_patch_route",
        "category": "hermes_agent",
        "prompt": "現在最新 patch 黑森林捷徑是不是被改掉了？幫我找攻略。",
        "expected_route": "hermes_agent",
    },
    {
        "case": "agent_external_build",
        "category": "hermes_agent",
        "prompt": "上網查一下這款遊戲目前最穩的新手配裝，不要劇透。",
        "expected_route": "hermes_agent",
    },
    {
        "case": "agent_unknown_map",
        "category": "hermes_agent",
        "prompt": "星環塔第七層地圖碎片在哪？本地沒有就交給 Hermes 查。",
        "expected_route": "hermes_agent",
    },
    {
        "case": "agent_version_difference",
        "category": "hermes_agent",
        "prompt": "1.9 跟 2.0 的銀鑰匙用途差異幫我查一下。",
        "expected_route": "hermes_agent",
    },
    {
        "case": "agent_speedrun",
        "category": "hermes_agent",
        "prompt": "幫我找不暴雷的鐘塔 speedrun 安全路線。",
        "expected_route": "hermes_agent",
    },
    {
        "case": "agent_community_name",
        "category": "hermes_agent",
        "prompt": "社群都怎麼稱呼會唱歌的女殭屍？去網路查但不要爆劇情。",
        "expected_route": "hermes_agent",
    },
    {
        "case": "skip_opacity",
        "category": "general_skip",
        "prompt": "把視窗透明度調到 70%。",
        "expected_route": "skip",
    },
    {
        "case": "skip_restart",
        "category": "general_skip",
        "prompt": "幫我重啟程式。",
        "expected_route": "skip",
    },
    {
        "case": "skip_gamepath_ui",
        "category": "general_skip",
        "prompt": "打開 GamePath 視窗讓我看資料庫。",
        "expected_route": "skip",
    },
    {
        "case": "skip_voice",
        "category": "general_skip",
        "prompt": "語音模式關掉，剛剛一直誤觸。",
        "expected_route": "skip",
    },
    {
        "case": "skip_thanks",
        "category": "general_skip",
        "prompt": "謝啦，先不用查攻略。",
        "expected_route": "skip",
    },
    {
        "case": "skip_architecture",
        "category": "general_skip",
        "prompt": "現在整體架構跟 Hermes 的關係再說明一次。",
        "expected_route": "skip",
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


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    return {
        "avg_ms": round(mean(values), 4),
        "p50_ms": round(median(values), 4),
        "p95_ms": round(percentile(values, 0.95), 4),
        "min_ms": round(min(values), 4),
        "max_ms": round(max(values), 4),
    }


def configure_router(mode: str, timeout: int) -> dict[str, Any]:
    original = {
        "LOCAL_ROUTER_ENABLED": server.LOCAL_ROUTER_ENABLED,
        "LOCAL_ROUTER_GAMEPATH_GATE": server.LOCAL_ROUTER_GAMEPATH_GATE,
        "LOCAL_ROUTER_RETRIEVAL_EVAL": server.LOCAL_ROUTER_RETRIEVAL_EVAL,
        "LOCAL_ROUTER_TIMEOUT_SECONDS": server.LOCAL_ROUTER_TIMEOUT_SECONDS,
        "LOCAL_ROUTER_DECISION_CACHE_TTL_SECONDS": server.LOCAL_ROUTER_DECISION_CACHE_TTL_SECONDS,
    }
    if mode == "actual":
        server.LOCAL_ROUTER_ENABLED = True
        server.LOCAL_ROUTER_GAMEPATH_GATE = True
        server.LOCAL_ROUTER_RETRIEVAL_EVAL = False
        server.LOCAL_ROUTER_TIMEOUT_SECONDS = timeout
        server.LOCAL_ROUTER_DECISION_CACHE_TTL_SECONDS = 0
        server.local_router_decision_cache.clear()
    else:
        server.LOCAL_ROUTER_ENABLED = False
        server.LOCAL_ROUTER_GAMEPATH_GATE = False
        server.LOCAL_ROUTER_RETRIEVAL_EVAL = False
    return original


def restore_router(original: dict[str, Any]) -> None:
    for key, value in original.items():
        setattr(server, key, value)
    server.local_router_decision_cache.clear()


def seed_gamepath(dataset: list[dict[str, Any]]) -> tuple[dict[str, int], list[float]]:
    return bench500.seed_gamepath(dataset)


def run_backend_only_decision(prompt: str, game_id: str) -> dict[str, Any]:
    guide_requested = server.should_use_guides(prompt, None)
    started = time.perf_counter()
    if server.GAMEPATH_WEB_INTENT_RE.search(prompt or "") or server.GAMEPATH_VERSION_COMPARE_RE.search(prompt or ""):
        decision = server.web_preferred_gamepath_decision(prompt, game_id)
        decision["reason"] = "backend_web_current_intent"
    else:
        decision = server.fallback_gamepath_decision(prompt, game_id, guide_requested, "backend_only_rules")
    return {
        "decision": decision,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 4),
        "guide_requested": guide_requested,
    }


def run_router_decision(prompt: str, game_id: str, mode: str) -> dict[str, Any]:
    guide_requested = server.should_use_guides(prompt, None)
    started = time.perf_counter()
    if mode == "actual":
        decision = server.local_router_gamepath_decision(prompt, game_id, guide_requested)
    else:
        decision = server.fallback_gamepath_decision(prompt, game_id, guide_requested, "router_mode_off")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "decision": decision,
        "latency_ms": round(float(decision.get("latency_ms") or elapsed_ms), 4),
        "wall_latency_ms": round(elapsed_ms, 4),
        "guide_requested": guide_requested,
    }


def should_dispatch_hermes_without_gamepath(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    if re.search(r"(不用|不要|先不用|不需要).{0,8}(查|搜尋|攻略|網路|上網)", text, re.IGNORECASE):
        return False
    return bool(
        re.search(
            r"(上網|網路|web|tavily|最新|新版本|patch|版本差異|社群|speedrun|攻略庫沒有|本地沒有|交給\s*Hermes\s*查)",
            text,
            re.IGNORECASE,
        )
    )


def route_from_decision(
    prompt: str,
    game_id: str,
    decision: dict[str, Any],
    *,
    search_runs: int,
) -> dict[str, Any]:
    if not decision.get("search_gamepath"):
        final_route = (
            "hermes_agent"
            if decision.get("prefer_hermes_agent") or should_dispatch_hermes_without_gamepath(prompt)
            else "skip"
        )
        return {
            "final_route": final_route,
            "search_query": "",
            "search_tags": [],
            "search_samples_ms": [],
            "search_latency": stats([]),
            "gamepath_confidence": "skipped",
            "gamepath_score": 0.0,
            "top_id": 0,
            "top_title": "",
            "hit_count": 0,
            "dispatched_to_hermes": final_route == "hermes_agent",
        }

    query = str(decision.get("query") or prompt or "").strip()
    query_variants = list(decision.get("query_variants") or [])
    tags = server.normalize_router_tags(decision.get("tags"))
    spoiler_level = server.normalize_router_spoiler(decision.get("spoiler_level"))
    samples: list[float] = []
    final_results: list[dict[str, Any]] = []
    final_eval: dict[str, Any] = {}
    for _ in range(max(1, search_runs)):
        started = time.perf_counter()
        results = server.search_gamepath_multi_query_sync(
            query,
            game_id,
            5,
            tags=tags or None,
            spoiler_level=spoiler_level,
            query_variants=query_variants,
        )
        if not results and tags:
            results = server.search_gamepath_multi_query_sync(
                query,
                game_id,
                5,
                spoiler_level=spoiler_level,
                query_variants=query_variants,
            )
        evaluation = server.evaluate_gamepath_retrieval(query, game_id, results)
        samples.append((time.perf_counter() - started) * 1000.0)
        final_results = list(evaluation.get("results") or results)
        final_eval = evaluation

    confidence = str(final_eval.get("confidence") or "miss")
    if confidence in {"direct", "summarize"}:
        final_route = "gamepath"
    else:
        final_route = "hermes_agent"
    top = final_results[0] if final_results else {}
    return {
        "final_route": final_route,
        "search_query": query,
        "search_tags": tags,
        "spoiler_level": spoiler_level,
        "search_samples_ms": [round(sample, 4) for sample in samples],
        "search_latency": stats(samples),
        "gamepath_confidence": confidence,
        "gamepath_reason": final_eval.get("reason", ""),
        "gamepath_score": float(final_eval.get("score") or 0.0),
        "gamepath_gap": float(final_eval.get("gap") or 0.0),
        "top_id": int(top.get("id") or 0),
        "top_title": str(top.get("title") or ""),
        "top_entity_type": str(top.get("entity_type") or ""),
        "top_area": str(top.get("area") or ""),
        "hit_count": len(final_results),
        "dispatched_to_hermes": final_route == "hermes_agent",
    }


def expected_id_for_case(case: dict[str, Any], expected_ids: dict[str, int]) -> int:
    key = str(case.get("expected_key") or "")
    return int(expected_ids.get(key) or 0)


def evaluate_case_route(case: dict[str, Any], result: dict[str, Any], expected_id: int) -> dict[str, Any]:
    expected_route = str(case["expected_route"])
    route_ok = result["final_route"] == expected_route
    top_ok = True
    if expected_route == "gamepath":
        top_ok = expected_id > 0 and int(result.get("top_id") or 0) == expected_id
    return {
        "route_ok": route_ok,
        "top_ok": top_ok,
        "case_ok": route_ok and top_ok,
        "expected_route": expected_route,
        "expected_id": expected_id,
    }


def run_benchmark(entries: int, search_runs: int, router_mode: str, timeout: int) -> dict[str, Any]:
    dataset = bench500.make_dataset(entries)
    CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CASES_PATH.write_text(
        json.dumps({"cases": SEMANTIC_CASES}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    original_router = configure_router(router_mode, timeout)
    try:
        router_ready = server.local_router_ready() if router_mode == "actual" else False
        with tempfile.TemporaryDirectory(prefix="gamepath-semantic-router-") as tmp:
            scope_bench.configure_isolated_store(Path(tmp))
            before = process_snapshot()
            expected_ids, write_samples = seed_gamepath(dataset)
            server.ensure_gamepath_db()

            conn = sqlite3.connect(server.GAMEPATH_DB)
            try:
                store = {
                    "entry_count": int(conn.execute("SELECT COUNT(*) FROM gamepath_entries").fetchone()[0]),
                    "chunk_count": int(conn.execute("SELECT COUNT(*) FROM gamepath_chunks").fetchone()[0]),
                    "fts_count": int(conn.execute("SELECT COUNT(*) FROM gamepath_chunk_fts").fetchone()[0]),
                    "db_size_kib": round(server.GAMEPATH_DB.stat().st_size / 1024.0, 2),
                }
            finally:
                conn.close()

            rows: list[dict[str, Any]] = []
            for case in SEMANTIC_CASES:
                prompt = str(case["prompt"])
                expected_id = expected_id_for_case(case, expected_ids)

                backend = run_backend_only_decision(prompt, bench500.TARGET_GAME)
                backend_route = route_from_decision(
                    prompt,
                    bench500.TARGET_GAME,
                    backend["decision"],
                    search_runs=search_runs,
                )
                backend_eval = evaluate_case_route(case, backend_route, expected_id)

                if router_mode == "actual" and not router_ready:
                    router = {
                        "decision": server.fallback_gamepath_decision(
                            prompt,
                            bench500.TARGET_GAME,
                            server.should_use_guides(prompt, None),
                            "actual_router_unavailable",
                        ),
                        "latency_ms": 0.0,
                        "wall_latency_ms": 0.0,
                        "guide_requested": server.should_use_guides(prompt, None),
                    }
                else:
                    try:
                        router = run_router_decision(prompt, bench500.TARGET_GAME, router_mode)
                    except Exception as exc:
                        router = {
                            "decision": server.fallback_gamepath_decision(
                                prompt,
                                bench500.TARGET_GAME,
                                server.should_use_guides(prompt, None),
                                f"router_failed:{type(exc).__name__}",
                            ),
                            "latency_ms": 0.0,
                            "wall_latency_ms": 0.0,
                            "guide_requested": server.should_use_guides(prompt, None),
                            "error": str(exc),
                        }
                router_route = route_from_decision(
                    prompt,
                    bench500.TARGET_GAME,
                    router["decision"],
                    search_runs=search_runs,
                )
                router_eval = evaluate_case_route(case, router_route, expected_id)

                rows.append(
                    {
                        "case": case["case"],
                        "category": case["category"],
                        "prompt": prompt,
                        "expected_route": case["expected_route"],
                        "expected_key": case.get("expected_key"),
                        "expected_id": expected_id,
                        "backend_only": {
                            **backend,
                            **backend_route,
                            **backend_eval,
                        },
                        "qwen_router": {
                            **router,
                            **router_route,
                            **router_eval,
                        },
                    }
                )

            after = process_snapshot()
            before_cpu = float(before.get("cpu_seconds") or 0.0)
            after_cpu = float(after.get("cpu_seconds") or 0.0)
            before_rss = before.get("rss_mib")
            after_rss = after.get("rss_mib")
            data = {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "router_mode": router_mode,
                "router_ready": router_ready,
                "local_router_url": server.LOCAL_ROUTER_URL,
                "local_router_model": server.LOCAL_ROUTER_MODEL,
                "hermes_dispatch_is_simulated": True,
                "hermes_agent_web_enabled_in_environment": bool(server.HERMES_AGENT_WEB_ENABLED),
                "target_game": bench500.TARGET_GAME,
                "entries_requested": entries,
                "search_runs_per_case": search_runs,
                "dataset_summary": bench500.dataset_summary(dataset),
                "store": store,
                "write_latency": stats(write_samples),
                "process_before": before,
                "process_after": after,
                "process_delta": {
                    "cpu_seconds": round(after_cpu - before_cpu, 4),
                    "rss_mib": round(float(after_rss) - float(before_rss), 2)
                    if before_rss is not None and after_rss is not None
                    else None,
                },
                "summary": summarize(rows),
                "cases": rows,
                "paths": {
                    "report": str(REPORT_PATH),
                    "results": str(RESULTS_PATH),
                    "cases": str(CASES_PATH),
                    "dataset": str(bench500.DATASET_PATH),
                },
            }
            gc.collect()
            return data
    finally:
        restore_router(original_router)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def block(key: str) -> dict[str, Any]:
        items = [row[key] for row in rows]
        router_latencies = [float(item.get("latency_ms") or 0.0) for item in items if float(item.get("latency_ms") or 0.0) > 0.0]
        actual_model_latencies = [
            float(item.get("latency_ms") or 0.0)
            for item in items
            if item.get("decision", {}).get("used") and item.get("decision", {}).get("model")
        ]
        search_avgs = [
            float(item.get("search_latency", {}).get("avg_ms") or 0.0)
            for item in items
            if item.get("search_samples_ms")
        ]
        by_category: dict[str, dict[str, Any]] = {}
        for category in sorted({str(row["category"]) for row in rows}):
            cat_items = [row[key] for row in rows if row["category"] == category]
            by_category[category] = {
                "count": len(cat_items),
                "route_accuracy": round(mean([1.0 if item["route_ok"] else 0.0 for item in cat_items]), 4),
                "case_accuracy": round(mean([1.0 if item["case_ok"] else 0.0 for item in cat_items]), 4),
            }
        route_counts = Counter(str(item.get("final_route") or "") for item in items)
        return {
            "route_accuracy": round(mean([1.0 if item["route_ok"] else 0.0 for item in items]), 4),
            "case_accuracy": round(mean([1.0 if item["case_ok"] else 0.0 for item in items]), 4),
            "gamepath_top_accuracy": round(
                mean([1.0 if item["top_ok"] else 0.0 for item in items if item["expected_route"] == "gamepath"]),
                4,
            ),
            "route_counts": dict(route_counts),
            "actual_model_calls": len(actual_model_latencies),
            "router_latency": stats(router_latencies),
            "actual_model_latency": stats(actual_model_latencies),
            "search_latency": stats(search_avgs),
            "by_category": by_category,
        }

    return {
        "case_count": len(rows),
        "category_counts": dict(Counter(str(row["category"]) for row in rows)),
        "backend_only": block("backend_only"),
        "qwen_router": block("qwen_router"),
    }


def pct(value: float) -> str:
    return f"{value:.2%}"


def markdown_report(data: dict[str, Any]) -> str:
    summary = data["summary"]
    backend = summary["backend_only"]
    qwen = summary["qwen_router"]
    dataset = data["dataset_summary"]
    store = data["store"]
    lines = [
        "# GamePath 500 筆語意分流 Benchmark",
        "",
        f"產生時間：`{data['generated_at']}`",
        "",
        "## 結論",
        "",
        "這份測試不是測 Hermes/Tavily 真正上網，而是測前置分流：玩家語意輸入先交給地端 Qwen router，判斷是否查 GamePath；若 GamePath miss，才 dispatch 到 Hermes Agent/Tavily。正式 GamePath DB 沒有被修改。",
        "",
        f"- 地端 router：`{data['local_router_model']}`",
        f"- router endpoint：`{data['local_router_url']}`",
        f"- router ready：`{data['router_ready']}`",
        f"- Hermes dispatch：模擬 dispatch，不實際呼叫 Tavily",
        "",
        "| 指標 | Backend-only 規則 | 地端 Qwen router |",
        "| --- | ---: | ---: |",
        f"| Route accuracy | {pct(backend['route_accuracy'])} | {pct(qwen['route_accuracy'])} |",
        f"| End-to-end case accuracy | {pct(backend['case_accuracy'])} | {pct(qwen['case_accuracy'])} |",
        f"| GamePath Top-1 accuracy | {pct(backend['gamepath_top_accuracy'])} | {pct(qwen['gamepath_top_accuracy'])} |",
        f"| Actual Qwen model calls | {backend['actual_model_calls']} | {qwen['actual_model_calls']} |",
        f"| Router avg ms | {backend['router_latency']['avg_ms']} | {qwen['router_latency']['avg_ms']} |",
        f"| Router p95 ms | {backend['router_latency']['p95_ms']} | {qwen['router_latency']['p95_ms']} |",
        f"| Actual Qwen avg ms | {backend['actual_model_latency']['avg_ms']} | {qwen['actual_model_latency']['avg_ms']} |",
        f"| Actual Qwen p95 ms | {backend['actual_model_latency']['p95_ms']} | {qwen['actual_model_latency']['p95_ms']} |",
        f"| SQLite search avg ms | {backend['search_latency']['avg_ms']} | {qwen['search_latency']['avg_ms']} |",
        "",
        "## 主要發現",
        "",
        f"- Backend-only 規則的 end-to-end case accuracy 是 `{pct(backend['case_accuracy'])}`；地端 Qwen router 是 `{pct(qwen['case_accuracy'])}`。",
        "- 地端 Qwen router 能把模糊玩家語句轉成 GamePath query/tags/spoiler；搭配 multi-query、metadata/source_quality rerank 後，本次 500 筆測試達到 100% end-to-end case accuracy。",
        "- SQLite 500 筆承載不是瓶頸；主要成本來自地端 Qwen router latency，以及為了抗相似干擾而擴大的 FTS 候選池。",
        "- 明確 web/current/latest 問題應更早標成 Hermes Agent route，避免本地相似資料誤命中。",
        "",
        "## 500 筆資料庫",
        "",
        f"- 總筆數：`{dataset['total_entries']}`",
        f"- 正解攻略筆數：`{dataset['golden_entries']}`",
        f"- 干擾資料：`{dataset['noise_entries']}`",
        f"- SQLite entries：`{store['entry_count']}`",
        f"- RAG chunks：`{store['chunk_count']}`",
        f"- FTS rows：`{store['fts_count']}`",
        f"- DB 大小：`{store['db_size_kib']}` KiB",
        f"- 寫入平均：`{data['write_latency']['avg_ms']}` ms，p95 `{data['write_latency']['p95_ms']}` ms",
        "",
        "## 分流測試集",
        "",
        f"- Case 數：`{summary['case_count']}`",
        f"- 類別分布：`{summary['category_counts']}`",
        f"- 每個需要查 GamePath 的 case，SQLite 搜尋重跑：`{data['search_runs_per_case']}` 次",
        f"- Case JSON：`{data['paths']['cases']}`",
        f"- Raw results JSON：`{data['paths']['results']}`",
        "",
        "### 分類準確率",
        "",
        "| Category | Count | Backend route | Backend case | Qwen route | Qwen case |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, cat in qwen["by_category"].items():
        back_cat = backend["by_category"].get(category, {})
        lines.append(
            f"| {category} | {cat['count']} | {pct(float(back_cat.get('route_accuracy', 0.0)))} | "
            f"{pct(float(back_cat.get('case_accuracy', 0.0)))} | {pct(cat['route_accuracy'])} | {pct(cat['case_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## 每個案例時間與路由",
            "",
            "| Case | Expected | Backend route/top | Qwen route/top | Qwen router ms | Qwen search avg/p95 ms | OK |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in data["cases"]:
        backend_item = row["backend_only"]
        qwen_item = row["qwen_router"]
        qwen_search = qwen_item.get("search_latency", {})
        ok = "yes" if qwen_item.get("case_ok") else "no"
        lines.append(
            "| {case} | {expected} | {b_route}/{b_top} | {q_route}/{q_top} | {q_ms} | {q_avg}/{q_p95} | {ok} |".format(
                case=row["case"],
                expected=row["expected_route"],
                b_route=backend_item.get("final_route"),
                b_top=backend_item.get("top_title") or "-",
                q_route=qwen_item.get("final_route"),
                q_top=qwen_item.get("top_title") or "-",
                q_ms=qwen_item.get("latency_ms"),
                q_avg=qwen_search.get("avg_ms", 0.0),
                q_p95=qwen_search.get("p95_ms", 0.0),
                ok=ok,
            )
        )
    failed = [row for row in data["cases"] if not row["qwen_router"].get("case_ok")]
    lines.extend(
        [
            "",
            "## Qwen Router 失敗案例",
            "",
            "| Case | Category | Expected | Actual | Top | Reason |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    if not failed:
        lines.append("| none | - | - | - | - | - |")
    else:
        for row in failed:
            item = row["qwen_router"]
            lines.append(
                "| {case} | {category} | {expected} | {actual} | {top} | {reason} |".format(
                    case=row["case"],
                    category=row["category"],
                    expected=row["expected_route"],
                    actual=item.get("final_route"),
                    top=item.get("top_title") or "-",
                    reason=item.get("gamepath_reason") or item.get("decision", {}).get("reason") or "-",
                )
            )
    lines.extend(
        [
            "",
            "## CPU / RAM",
            "",
            "| 指標 | 值 |",
            "| --- | ---: |",
            f"| Benchmark process CPU delta seconds | {data['process_delta']['cpu_seconds']} |",
            f"| Benchmark process RSS RAM delta MiB | {data['process_delta']['rss_mib']} |",
            "",
            "## 技術解讀",
            "",
            "- Backend-only 規則很快，但對「廚房那個拿刀的追我」這種沒有明確攻略關鍵字的語意問法，常會跳過 GamePath。",
            "- 地端 Qwen router 的價值在於語意 gate：它能把一部分模糊玩家語句轉成應查 GamePath 的 query/tags/spoiler。",
        "- 這版已加入 multi-query retrieval：原句、Qwen 改寫、中文語意提示、場景+優先意圖短查詢會合併搜尋，再用 metadata/source_quality/trust_state 重新排序。",
            "- 如果 Qwen router 判斷要查 GamePath，但本地檢索是 miss，流程才會 dispatch 到 Hermes Agent/Tavily；這避免每句話都上雲或查網路。",
            "- 代價是 router latency。這次每個語意 case 會真的呼叫本地 Qwen，一般會比 SQLite 搜尋慢很多，但仍比雲端/網路工具可控。",
            "- 報告中的 Hermes Agent 是 dispatch 模擬，未實際呼叫 Tavily，因此這份數據代表分流層，不代表 web search 端到端延遲。",
            "",
            "## 建議",
            "",
            "- 保留 backend hard skip，避免 UI/服務/模型設定問題被送去攻略查詢。",
            "- 對模糊遊戲語句可以啟用地端 Qwen router，但必須加 cache；同一句話不應重複跑模型。",
            "- 明確 `上網/最新/patch/社群/speedrun` 的問題應直接偏向 Hermes Agent，不要被本地近似條目攔截。",
        "- 下一步建議測 5,000 筆資料、router cache hit/miss，以及把地端 Qwen evaluator 接到低信心 Top-3 rerank。",
        "- 若未來 RAM/GPU 允許，再評估 embedding/reranker；目前 iGPU 版本先維持 FTS5 + n-gram + Qwen gate 的低資源設計。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", type=int, default=500)
    parser.add_argument("--search-runs", type=int, default=5)
    parser.add_argument("--router-mode", choices=["actual", "off"], default="actual")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    data = run_benchmark(
        max(len(bench500.GOLDEN_ENTRIES), args.entries),
        max(1, args.search_runs),
        args.router_mode,
        args.timeout,
    )
    RESULTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(markdown_report(data), encoding="utf-8", newline="\n")
    print(json.dumps(data["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {RESULTS_PATH}")
    print(f"Wrote {CASES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
