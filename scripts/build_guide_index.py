#!/usr/bin/env python
"""Build the CPU-only local guide index for the in-game RAG path."""

from __future__ import annotations

import argparse
import html
import re
import sqlite3
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUIDES_ROOT = PROJECT_ROOT / "game_guides"
GUIDE_CACHE = PROJECT_ROOT / "guide_cache"
GUIDE_DB = GUIDE_CACHE / "guide.sqlite"
SUPPORTED_SUFFIXES = {".md", ".txt", ".html", ".htm"}
CJK_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+")


def normalize_game_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise ValueError("game_id cannot be empty")
    return cleaned[:80]


def cjk_ngrams(text: str) -> list[str]:
    tokens: list[str] = []
    for match in CJK_RE.findall(text):
        compact = re.sub(r"\s+", "", match)
        for size in (2, 3):
            if len(compact) >= size:
                tokens.extend(compact[index : index + size] for index in range(len(compact) - size + 1))
        if len(compact) == 1:
            tokens.append(compact)
    return tokens


def expand_search_text(*parts: str) -> str:
    text = "\n".join(part for part in parts if part)
    return text + "\n" + " ".join(cjk_ngrams(text))


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp950", "big5", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return html.unescape(text)


def split_chunks(text: str, max_chars: int = 1400) -> list[tuple[str, str]]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    chunks: list[tuple[str, str]] = []
    section = "General"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        body = "\n".join(buffer).strip()
        if body:
            while len(body) > max_chars:
                chunks.append((section, body[:max_chars].strip()))
                body = body[max_chars:].strip()
            if body:
                chunks.append((section, body))
        buffer = []

    for line in lines:
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if heading:
            flush()
            section = heading.group(1).strip()[:160] or "General"
            continue
        buffer.append(line)
        if sum(len(part) for part in buffer) >= max_chars:
            flush()
    flush()
    return chunks


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS guide_fts USING fts5(
            game_id UNINDEXED,
            title,
            source_path UNINDEXED,
            section,
            content,
            tags,
            updated_at UNINDEXED,
            search_text,
            tokenize='unicode61'
        )
        """
    )
    conn.commit()


def index_game(conn: sqlite3.Connection, game_id: str, game_dir: Path) -> tuple[int, int]:
    conn.execute("DELETE FROM guide_fts WHERE game_id = ?", (game_id,))
    files = sorted(path for path in game_dir.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES)
    chunk_count = 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for path in files:
        text = read_text(path)
        if path.suffix.lower() in {".html", ".htm"}:
            text = strip_html(text)
        title = path.stem[:160]
        rel_path = path.relative_to(PROJECT_ROOT).as_posix()
        for section, content in split_chunks(text):
            search_text = expand_search_text(game_id, title, section, content)
            conn.execute(
                """
                INSERT INTO guide_fts(game_id, title, source_path, section, content, tags, updated_at, search_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (game_id, title, rel_path, section, content, "", now, search_text),
            )
            chunk_count += 1
    conn.commit()
    return len(files), chunk_count


def discover_games(game_id: str | None) -> list[tuple[str, Path]]:
    if game_id:
        normalized = normalize_game_id(game_id)
        return [(normalized, GUIDES_ROOT / normalized)]
    if not GUIDES_ROOT.exists():
        return []
    return [
        (normalize_game_id(path.name), path)
        for path in sorted(GUIDES_ROOT.iterdir())
        if path.is_dir()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CPU-only SQLite FTS guide index.")
    parser.add_argument("--game-id", help="Only index one game folder under game_guides/.")
    args = parser.parse_args()

    GUIDE_CACHE.mkdir(exist_ok=True)
    games = discover_games(args.game_id)
    if not games:
        print(f"No guide folders found under {GUIDES_ROOT}")
        return 1

    total_files = 0
    total_chunks = 0
    with sqlite3.connect(GUIDE_DB) as conn:
        init_db(conn)
        for game_id, game_dir in games:
            if not game_dir.exists():
                print(f"Skipping missing guide folder: {game_dir}")
                continue
            file_count, chunk_count = index_game(conn, game_id, game_dir)
            total_files += file_count
            total_chunks += chunk_count
            print(f"{game_id}: indexed {file_count} files, {chunk_count} chunks")

    print(f"Guide index ready: {GUIDE_DB}")
    print(f"Total: {total_files} files, {total_chunks} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
