#!/usr/bin/env python3
"""Import a summarized Resident Evil Requiem guide pack into GamePath.

The importer intentionally does not store full web article text. It fetches the
two source pages, splits them by guide section, extracts route / item / puzzle
signals, and writes concise player-facing summaries into GamePath.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import llama_vulkan_api_server as server  # noqa: E402


GAME_ID = "RESIDENT_EVIL_requiem"
SOURCE_TYPE = "manual_web_import"
SOURCE_QUALITY = 0.74
REPORT_PATH = PROJECT_ROOT / "docs" / "gamepath_guide_pack_validation_report.md"

SOURCES = [
    {
        "part": "part1",
        "title": "惡靈古堡 9 安魂曲 全收集圖文流程攻略 part1",
        "url": "https://www.entertainment14.net/blog/post/111006575-%E6%83%A1%E9%9D%88%E5%8F%A4%E5%A0%A1-9-%E5%AE%89%E9%AD%82%E6%9B%B2-%E5%85%A8%E6%94%B6%E9%9B%86%E5%9C%96%E6%96%87%E6%B5%81%E7%A8%8B%E6%94%BB%E7%95%A5",
    },
    {
        "part": "part2",
        "title": "惡靈古堡 9 安魂曲 全收集圖文流程攻略 part2",
        "url": "https://www.entertainment14.net/blog/post/111006808-%E6%83%A1%E9%9D%88%E5%8F%A4%E5%A0%A1-9-%E5%AE%89%E9%AD%82%E6%9B%B2-%E5%85%A8%E6%94%B6%E9%9B%86%E5%9C%96%E6%96%87%E6%B5%81%E7%A8%8B%E6%94%BB%E7%95%A5-part2",
    },
]

STOP_HEADINGS = {
    "發佈留言 取消回覆",
    "文章搜尋",
    "近期熱門遊戲攻略",
    "廣告",
}

META_HEADINGS = {
    "本頁收集進度：",
    "本頁路線：",
}

VALIDATION_QUERIES = [
    "鷦木旅館老舊鑰匙在哪",
    "療養院護士站螺絲刀怎麼拿",
    "一樓西側廚房餐廳怎麼過",
    "理事長室機關盒答案",
    "主廚喪屍怎麼打",
    "東側門禁卡在哪",
    "血液研究室謎題怎麼解",
    "酒吧休閒室保險箱密碼",
    "地下保險箱怎麼開",
    "泰坦巨蛛弱點在哪",
    "鷹之星物流倉庫下一步",
    "浣熊市警局尋寶遊戲",
    "暴君追逐戰怎麼辦",
    "方舟兩個保險箱",
    "最終謎題怎麼解",
]


@dataclass
class SourceSection:
    part: str
    source_url: str
    heading: str
    route: str = ""
    collectibles: list[str] = field(default_factory=list)
    body: list[str] = field(default_factory=list)


class ArticleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[str, str]] = []
        self._tag: str | None = None
        self._buf: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav", "footer"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in {"h1", "h2", "h3", "p", "li"}:
            self._tag = tag
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if self._tag == tag:
            text = html.unescape("".join(self._buf))
            text = re.sub(r"\s+", " ", text).strip()
            if text and text not in {"廣告", "* * *"} and not text.startswith("Image"):
                self.rows.append((tag, text))
            self._tag = None
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._tag and not self._skip_depth:
            self._buf.append(data)


def fetch_rows(url: str) -> list[tuple[str, str]]:
    response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 GamePathImporter/1.0"})
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding or "utf-8"
    parser = ArticleTextParser()
    parser.feed(response.text)
    return parser.rows


def parse_sections(source: dict[str, str]) -> list[SourceSection]:
    rows = fetch_rows(source["url"])
    sections: list[SourceSection] = []
    current: SourceSection | None = None
    mode = "body"
    started = False

    for tag, text in rows:
        if tag == "h1":
            started = True
            continue
        if not started:
            continue
        if tag == "h3":
            if text in STOP_HEADINGS:
                break
            if text in META_HEADINGS:
                mode = "collectibles" if "收集" in text else "route"
                continue
            current = SourceSection(part=source["part"], source_url=source["url"], heading=text)
            sections.append(current)
            mode = "body"
            continue
        if not current:
            continue
        if mode == "collectibles":
            current.collectibles.append(text)
            continue
        if mode == "route" and not current.route:
            current.route = text
            mode = "body"
            continue
        current.body.append(text)

    return [section for section in sections if is_gameplay_section(section)]


def is_gameplay_section(section: SourceSection) -> bool:
    heading = section.heading.strip()
    if not heading or heading in META_HEADINGS or heading in STOP_HEADINGS:
        return False
    if heading.startswith("全文主要路線"):
        return True
    text = "\n".join([heading, section.route, *section.collectibles, *section.body[:8]])
    return bool(re.search(r"(療養院|浣熊市|旅館|警局|方舟|結局|謎題|喪屍|怪物|保險箱|收集|攻略|大廈|倉庫|孤兒院|槍械店|植物)", text))


def unique_keep_order(items: list[str], limit: int = 20) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for raw in items:
        text = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def route_steps(route: str) -> list[str]:
    if not route:
        return []
    parts = re.split(r"[→>＞]+", route)
    return unique_keep_order([part.strip(" ，,。") for part in parts if part.strip()], 10)


def bracket_terms(text: str) -> list[str]:
    terms = re.findall(r"〖([^〗]{1,60})〗", text)
    cleaned = []
    for term in terms:
        term = re.sub(r"\s+", " ", term).strip()
        term = re.sub(r"^(檔|文件|武器|古錢幣|黑白浣熊先生)(\d+/\d+)?", r"\1\2", term)
        cleaned.append(term)
    return unique_keep_order(cleaned, 28)


def short_fact_patterns(text: str) -> list[str]:
    facts: list[str] = []
    compact_lines = [line.strip() for line in re.split(r"[\n\r]+", text) if line.strip()]
    search_text = "\n".join(compact_lines)
    patterns = [
        r"([^。\n]{0,24}保險箱密碼[:：][^。\n]{1,48})",
        r"([^。\n]{0,24}密碼[:：][^。\n]{1,48})",
        r"([^。\n]{0,24}答案[:：][^。\n]{1,48})",
        r"([^。\n]{0,24}弱點(?:在|是)[^。\n]{1,36})",
        r"([^。\n]{0,24}需要古錢幣[:：]?\s*\d+個[^。\n]{0,20})",
    ]
    for pattern in patterns:
        facts.extend(re.findall(pattern, search_text))
    return unique_keep_order([re.sub(r"\s+", " ", fact).strip(" ，,。") for fact in facts], 8)


def entity_type_for(section: SourceSection, text: str) -> str:
    if re.search(r"(結局|謎題|保險箱|密碼|機關盒|尋寶|代碼)", text):
        return "puzzle"
    if re.search(r"(boss|暴君|泰坦巨蛛|主廚|舔食者|精英衛隊|怪物|喪屍|植物|指揮官)", text, re.I):
        return "enemy"
    if re.search(r"(鑰匙|門禁卡|操控杆|晶石|道具|武器|古錢幣|浣熊先生)", text):
        return "item"
    return "location"


def spoiler_level_for(section: SourceSection, text: str) -> str:
    heading = section.heading
    if re.search(r"(結局|最終|完整解謎|壞結局|好結局)", heading + text):
        return "high"
    if re.search(r"(答案|密碼|代碼|保險箱|決戰|boss|暴君|弱點|謎題)", heading + text, re.I):
        return "medium"
    return "low"


def tags_for(section: SourceSection, entity_type: str) -> list[str]:
    tags = ["guide_pack", "entertainment14_summary", "route", entity_type, section.part]
    text = "\n".join([section.heading, section.route, *section.collectibles, *section.body])
    if "古錢幣" in text or "黑白浣熊先生" in text or "文件" in text or "檔" in text:
        tags.append("collectibles")
    if re.search(r"(保險箱|密碼|答案|謎題|機關盒)", text):
        tags.append("puzzle")
    if re.search(r"(喪屍|怪物|暴君|泰坦巨蛛|舔食者|boss)", text, re.I):
        tags.append("enemy")
    return unique_keep_order(tags, 12)


def area_for(section: SourceSection) -> str:
    heading = section.heading
    if "-" in heading:
        return heading.split("-", 1)[1].strip()[:80]
    return heading[:80]


def build_entry(section: SourceSection) -> dict[str, Any]:
    full_text = "\n".join([section.heading, section.route, *section.collectibles, *section.body])
    steps = route_steps(section.route)
    terms = bracket_terms(full_text)
    facts = short_fact_patterns(full_text)
    entity_type = entity_type_for(section, full_text)
    spoiler_level = spoiler_level_for(section, full_text)
    tags = tags_for(section, entity_type)

    lines = [
        "攻略包匯入摘要（已改寫，非網頁原文全文）。",
        f"- 區域/主題：{section.heading}",
    ]
    if steps:
        lines.append("- 路線骨架：" + " → ".join(steps[:8]))
    if terms:
        lines.append("- 關鍵物品/收集/文件：" + "、".join(terms[:18]))
    if facts:
        lines.append("- 明確答案/弱點提示：" + "；".join(facts[:5]))
    lines.extend(
        [
            "- 使用方式：玩家問路線、卡關、道具用途、收集品位置或敵人打法時，先用這筆本地資料定位段落。",
            "- 回答規則：預設先給低劇透提示；玩家明確要求密碼、答案、結局或完整步驟時，再揭露高劇透資訊。",
            f"- 來源：Entertainment14 / 遊民星空整理，{section.part}；此條目只保存濃縮提示與索引線索。",
        ]
    )

    question_terms = unique_keep_order([section.heading, *steps[:4], *terms[:8], *facts[:3]], 18)
    question = " ".join(question_terms) + " 攻略 路線 收集品 卡關"

    return {
        "title": f"攻略包：{section.heading}",
        "question": question,
        "answer_summary": "\n".join(lines),
        "game_id": GAME_ID,
        "tags": tags,
        "spoiler_level": spoiler_level,
        "source_type": SOURCE_TYPE,
        "agent_used": False,
        "version": "",
        "area": area_for(section),
        "entity_type": entity_type,
        "entity_name": "",
        "source_quality": SOURCE_QUALITY,
        "source_url": section.source_url,
        "part": section.part,
    }


def collect_entries(limit: int | None = None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for source in SOURCES:
        for section in parse_sections(source):
            entries.append(build_entry(section))
    if limit:
        return entries[:limit]
    return entries


def import_entries(entries: list[dict[str, Any]], dry_run: bool = False) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for entry in entries:
        if dry_run:
            results.append({"status": "dry_run", "id": None, **entry})
            continue
        stable_question = existing_pack_question(entry["title"]) or entry["question"]
        item = server.add_gamepath_sync(
            stable_question,
            entry["answer_summary"],
            entry["game_id"],
            title=entry["title"],
            tags=entry["tags"],
            spoiler_level=entry["spoiler_level"],
            source_type=entry["source_type"],
            agent_used=entry["agent_used"],
            version=entry["version"],
            area=entry["area"],
            entity_type=entry["entity_type"],
            entity_name=entry["entity_name"],
            source_quality=entry["source_quality"],
        )
        results.append({"source_url": entry["source_url"], "part": entry["part"], **item})
    return results


def existing_pack_question(title: str) -> str:
    if not server.GAMEPATH_DB.exists():
        return ""
    with sqlite3.connect(server.GAMEPATH_DB) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT question
            FROM gamepath_entries
            WHERE game_id = ? AND source_type = ? AND title = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (server.normalize_game_id(GAME_ID), SOURCE_TYPE, title),
        ).fetchone()
    return str(row["question"] or "") if row else ""


def cleanup_duplicate_pack_entries() -> list[dict[str, Any]]:
    if not server.GAMEPATH_DB.exists():
        return []
    deleted: list[dict[str, Any]] = []
    with sqlite3.connect(server.GAMEPATH_DB) as conn:
        conn.row_factory = sqlite3.Row
        groups = conn.execute(
            """
            SELECT title, GROUP_CONCAT(id) AS ids, COUNT(*) AS count
            FROM gamepath_entries
            WHERE game_id = ? AND source_type = ?
            GROUP BY title
            HAVING COUNT(*) > 1
            """,
            (server.normalize_game_id(GAME_ID), SOURCE_TYPE),
        ).fetchall()
    for group in groups:
        ids = [int(item) for item in str(group["ids"] or "").split(",") if item.strip()]
        if len(ids) <= 1:
            continue
        keep_id = max(ids)
        for entry_id in ids:
            if entry_id == keep_id:
                continue
            deleted_item = server.delete_gamepath_sync(entry_id)
            if deleted_item:
                deleted.append(deleted_item)
    return deleted


def count_pack_entries() -> tuple[int, int]:
    server.ensure_gamepath_db()
    with sqlite3.connect(server.GAMEPATH_DB) as conn:
        entry_count = conn.execute(
            "SELECT COUNT(*) FROM gamepath_entries WHERE game_id = ? AND source_type = ?",
            (server.normalize_game_id(GAME_ID), SOURCE_TYPE),
        ).fetchone()[0]
        chunk_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM gamepath_chunks c
            JOIN gamepath_entries e ON e.id = c.entry_id
            WHERE e.game_id = ? AND e.source_type = ?
            """,
            (server.normalize_game_id(GAME_ID), SOURCE_TYPE),
        ).fetchone()[0]
    return int(entry_count or 0), int(chunk_count or 0)


def validate_queries(queries: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        started = time.perf_counter()
        results = server.search_gamepath_multi_query_sync(query, GAME_ID, 5, spoiler_level="high")
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        evaluation = server.evaluate_gamepath_retrieval(query, GAME_ID, results)
        top = (evaluation.get("results") or results or [{}])[0] if (evaluation.get("results") or results) else {}
        rows.append(
            {
                "query": query,
                "elapsed_ms": elapsed_ms,
                "confidence": evaluation.get("confidence"),
                "score": evaluation.get("score"),
                "top_id": top.get("id"),
                "top_title": top.get("title"),
                "top_area": top.get("area"),
                "source_type": top.get("source_type"),
                "rag_lite": bool(top.get("rag_lite")),
            }
        )
    return rows


def write_report(
    import_results: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    dry_run: bool,
    duplicate_deletes: list[dict[str, Any]] | None = None,
) -> None:
    duplicate_deletes = duplicate_deletes or []
    entry_count, chunk_count = count_pack_entries() if not dry_run else (0, 0)
    created = sum(1 for item in import_results if item.get("status") == "created")
    updated = sum(1 for item in import_results if item.get("status") == "updated")
    status_label = "dry-run" if dry_run else "imported"
    lines = [
        "# GamePath 攻略包匯入驗證報告",
        "",
        f"產生時間：{time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "",
        "## 目的",
        "",
        "驗證一般玩家下載整理好的攻略包後，GamePath 是否能把長篇流程攻略轉成可搜尋、可局部命中的本地知識庫。",
        "本次匯入只保存改寫後摘要、路線骨架、關鍵物品/收集品/謎題線索，不保存網站原文全文。",
        "",
        "## 匯入來源",
        "",
        "- Entertainment14：惡靈古堡 9 安魂曲 全收集圖文流程攻略 part1",
        "- Entertainment14：惡靈古堡 9 安魂曲 全收集圖文流程攻略 part2",
        "",
        "## 匯入結果",
        "",
        f"- 狀態：`{status_label}`",
        f"- 本次產生 entries：`{len(import_results)}`",
        f"- created：`{created}`",
        f"- updated：`{updated}`",
        f"- duplicate cleanup deleted：`{len(duplicate_deletes)}`",
        f"- DB 內攻略包 entries：`{entry_count}`",
        f"- DB 內攻略包 chunks：`{chunk_count}`",
        f"- game_id：`{GAME_ID}`",
        f"- source_type：`{SOURCE_TYPE}`",
        "",
        "## 驗證查詢",
        "",
        "| Query | ms | confidence | score | top entry | area | rag_lite |",
        "|---|---:|---|---:|---|---|---|",
    ]
    for row in validation:
        title = str(row.get("top_title") or "").replace("|", "\\|")
        area = str(row.get("top_area") or "").replace("|", "\\|")
        score = row.get("score")
        score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else ""
        lines.append(
            f"| {row['query']} | {row['elapsed_ms']} | {row.get('confidence') or ''} | {score_text} | #{row.get('top_id') or ''} {title} | {area} | {row.get('rag_lite')} |"
        )
    lines.extend(
        [
            "",
            "## 判斷",
            "",
            "- 若查詢命中 `source_type=manual_web_import` 且 `rag_lite=True`，代表玩家下載攻略包後可以走本地 GamePath/RAG Lite。",
            "- 若 confidence 是 `summarize`，代表本地資料有相關段落，但應交給模型濃縮，不應直接吐整篇攻略。",
            "- 若 confidence 是 `miss`，代表本地資料不足，才應交給 Hermes/Tavily 或請玩家提供更多場景資訊。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Parse and report without writing GamePath.")
    parser.add_argument("--limit", type=int, default=0, help="Limit imported entries for smoke testing.")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation queries.")
    parser.add_argument("--no-cleanup", action="store_true", help="Do not remove duplicate pack entries by title.")
    args = parser.parse_args()

    entries = collect_entries(args.limit or None)
    import_results = import_entries(entries, dry_run=args.dry_run)
    duplicate_deletes = [] if args.dry_run or args.no_cleanup else cleanup_duplicate_pack_entries()
    validation = [] if args.no_validate else validate_queries(VALIDATION_QUERIES)
    write_report(import_results, validation, args.dry_run, duplicate_deletes)

    created = sum(1 for item in import_results if item.get("status") == "created")
    updated = sum(1 for item in import_results if item.get("status") == "updated")
    entry_count, chunk_count = count_pack_entries() if not args.dry_run else (0, 0)
    print(
        json.dumps(
            {
                "status": "dry_run" if args.dry_run else "ok",
                "entries": len(import_results),
                "created": created,
                "updated": updated,
                "duplicate_deleted": len(duplicate_deletes),
                "pack_entry_count": entry_count,
                "pack_chunk_count": chunk_count,
                "report": str(REPORT_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
