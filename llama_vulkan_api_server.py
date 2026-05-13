import asyncio
import base64
import html
import io
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import mss
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel


LLAMA_HOST = os.environ.get("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "18080"))
API_HOST = os.environ.get("IGPU_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("IGPU_API_PORT", "8000"))
MODEL_ALIAS = os.environ.get("LLAMA_MODEL_ALIAS", "gemma-4-E4B-it-Q4_K_M")
VULKAN_DEVICE = os.environ.get("GGML_VK_VISIBLE_DEVICES", "0")
LLAMA_CTX_SIZE = os.environ.get("LLAMA_CTX_SIZE", "32768")
LLAMA_PARALLEL = os.environ.get("LLAMA_PARALLEL", "").strip()
LLAMA_CACHE_RAM = os.environ.get("LLAMA_CACHE_RAM", "").strip()
CHAT_BACKEND = os.environ.get("IGPU_CHAT_BACKEND", "hermes").strip().lower()
HERMES_WSL_DISTRO = os.environ.get("HERMES_WSL_DISTRO", "Ubuntu-24.04")
HERMES_TIMEOUT_SECONDS = int(os.environ.get("HERMES_TIMEOUT_SECONDS", "180"))
ENABLE_LOCAL_TOOLS = os.environ.get(
    "IGPU_ENABLE_LOCAL_TOOLS",
    "0" if CHAT_BACKEND == "hermes" else "1",
).strip().lower() in {"1", "true", "yes", "on"}

ASSET_ROOT = Path(
    os.environ.get(
        "LLAMA_GEMMA4_HOME",
        str(Path(os.environ["LOCALAPPDATA"]) / "llama-gemma4-e4b"),
    )
)
LLAMA_DIR = ASSET_ROOT / "tools" / "llama.cpp-vulkan"
LLAMA_SERVER = LLAMA_DIR / "llama-server.exe"
MODEL_PATH = ASSET_ROOT / "models" / "gemma-4-E4B-it-Q4_K_M.gguf"
MMPROJ_PATH = ASSET_ROOT / "models" / "mmproj-BF16.gguf"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LLAMA_LOG = LOG_DIR / "llama-server.log"
GENERATED_DIR = Path(__file__).resolve().parent / "generated_files"
PROJECT_ROOT = Path(__file__).resolve().parent
HERMES_CHAT_SCRIPT = PROJECT_ROOT / "scripts" / "hermes_no_tools_chat.py"
GAME_GUIDES_DIR = PROJECT_ROOT / "game_guides"
GUIDE_CACHE_DIR = PROJECT_ROOT / "guide_cache"
GUIDE_DB = GUIDE_CACHE_DIR / "guide.sqlite"
MEMORY_CACHE_DIR = PROJECT_ROOT / "memory_cache"
MEMORY_DB = MEMORY_CACHE_DIR / "memory.sqlite"

llama_process: Optional[subprocess.Popen] = None
history: list[dict[str, Any]] = []
generate_lock = asyncio.Lock()


app = FastAPI(title="iGPU Overlay llama.cpp Vulkan Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    image_base64: Optional[str] = None
    game_id: Optional[str] = None
    use_guides: Optional[bool] = None
    use_memory: bool = True


class GuideSearchRequest(BaseModel):
    game_id: Optional[str] = None
    query: str
    limit: int = 5


class MemoryAddRequest(BaseModel):
    content: str
    game_id: Optional[str] = None
    kind: str = "note"
    tags: Optional[str] = None
    importance: int = 3


class MemorySearchRequest(BaseModel):
    query: str
    game_id: Optional[str] = None
    kinds: Optional[list[str]] = None
    limit: int = 5


def get_system_prompt() -> str:
    return (
        "你是遊戲陪伴 AI，也是電腦系統效能分析師。"
        "你用繁體中文回答，語氣親切、簡潔、像會陪玩家一起看局勢的隊友。"
        "看到遊戲截圖時，請直接描述實際畫面，再給一到兩個可執行建議；不要泛泛要求使用者再貼圖。"
    )


def llama_base_url() -> str:
    return f"http://{LLAMA_HOST}:{LLAMA_PORT}"


def post_json(url: str, payload: dict[str, Any], timeout: int = 30):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def post_raw_json(url: str, raw_payload: bytes, timeout: int = 30):
    req = urllib.request.Request(
        url,
        data=raw_payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(url: str, timeout: int = 2) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def windows_path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    posix = resolved.as_posix()
    if len(posix) >= 3 and posix[1:3] == ":/":
        drive = posix[0].lower()
        return f"/mnt/{drive}/{posix[3:]}"
    return posix


def get_hermes_system_prompt() -> str:
    return (
        "你是遊戲陪伴 AI，也是電腦系統效能分析師。"
        "你用繁體中文回答，語氣親切、簡潔，像會陪玩家一起看局勢的隊友。"
        "目前你是透過 Hermes Agent 呼叫本機 llama.cpp Vulkan 上的 Gemma4 模型。"
        "先不要使用任何工具或系統操作，只處理使用者的文字對話。"
    )


def build_hermes_prompt(prompt: str) -> str:
    recent = history[-8:]
    lines = [get_hermes_system_prompt()]
    if recent:
        lines.append("\n最近對話：")
        for item in recent:
            role = "使用者" if item.get("role") == "user" else "助理"
            content = str(item.get("content") or "").strip()
            if content:
                lines.append(f"{role}: {content}")
    lines.append("\n目前使用者訊息：")
    lines.append(prompt)
    return "\n".join(lines)


def call_hermes_no_tools(prompt: str) -> str:
    if not HERMES_CHAT_SCRIPT.exists():
        raise RuntimeError(f"Missing Hermes chat script: {HERMES_CHAT_SCRIPT}")

    project_wsl = windows_path_to_wsl(PROJECT_ROOT)
    script_wsl = windows_path_to_wsl(HERMES_CHAT_SCRIPT)
    command = (
        f"cd {shlex.quote(project_wsl)} && "
        "OPENAI_API_KEY=no-key-required "
        f"~/.hermes/hermes-agent/venv/bin/python {shlex.quote(script_wsl)} "
        f"--api-port {API_PORT} --model {shlex.quote(MODEL_ALIAS)}"
    )
    args = ["wsl.exe", "-d", HERMES_WSL_DISTRO, "--", "bash", "-lc", command]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    result = subprocess.run(
        args,
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=HERMES_TIMEOUT_SECONDS,
        creationflags=creationflags,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or f"Hermes exited with code {result.returncode}")
    output = result.stdout.strip()
    if not output:
        raise RuntimeError("Hermes returned an empty response.")
    return output


def chunk_text(text: str, size: int = 80):
    for index in range(0, len(text), size):
        yield text[index : index + size]


def llama_ready() -> bool:
    try:
        get_json(f"{llama_base_url()}/v1/models", timeout=2)
        return True
    except Exception:
        return False


def validate_assets() -> None:
    missing = [
        str(path)
        for path in (LLAMA_SERVER, MODEL_PATH, MMPROJ_PATH)
        if not path.exists()
    ]
    if missing:
        raise RuntimeError("Missing llama.cpp Vulkan assets:\n" + "\n".join(missing))


def start_llama_server() -> None:
    global llama_process
    if llama_ready():
        print(f"llama-server already running at {llama_base_url()}")
        return

    validate_assets()
    LOG_DIR.mkdir(exist_ok=True)
    log_file = open(LLAMA_LOG, "a", encoding="utf-8", errors="replace")

    env = os.environ.copy()
    env["GGML_VK_VISIBLE_DEVICES"] = VULKAN_DEVICE
    env["LLAMA_ARG_FLASH_ATTN"] = env.get("LLAMA_ARG_FLASH_ATTN", "1")
    env["LLAMA_CHAT_TEMPLATE_KWARGS"] = '{"enable_thinking":false}'

    args = [
        str(LLAMA_SERVER),
        "--model",
        str(MODEL_PATH),
        "--mmproj",
        str(MMPROJ_PATH),
        "--host",
        LLAMA_HOST,
        "--port",
        str(LLAMA_PORT),
        "--ctx-size",
        LLAMA_CTX_SIZE,
        "--n-gpu-layers",
        os.environ.get("LLAMA_GPU_LAYERS", "999"),
        "--temp",
        "1.0",
        "--top-p",
        "0.95",
        "--top-k",
        "64",
        "--alias",
        MODEL_ALIAS,
        "--jinja",
        "--reasoning",
        "off",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
        "--flash-attn",
        "on",
    ]
    if LLAMA_PARALLEL:
        args.extend(["--parallel", LLAMA_PARALLEL])
    if LLAMA_CACHE_RAM:
        args.extend(["--cache-ram", LLAMA_CACHE_RAM])

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW

    print(f"Starting llama-server on {llama_base_url()} with Vulkan device {VULKAN_DEVICE}")
    llama_process = subprocess.Popen(
        args,
        cwd=str(LLAMA_DIR),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )

    deadline = time.time() + int(os.environ.get("LLAMA_STARTUP_TIMEOUT", "360"))
    while time.time() < deadline:
        if llama_process.poll() is not None:
            raise RuntimeError(
                f"llama-server exited early with code {llama_process.returncode}. "
                f"See {LLAMA_LOG}"
            )
        if llama_ready():
            print("llama-server is ready.")
            return
        time.sleep(2)

    raise RuntimeError(f"llama-server did not become ready in time. See {LLAMA_LOG}")


def image_to_data_url(image_base64: str) -> str:
    raw = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
    image_bytes = base64.b64decode(raw)
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG", optimize=True)
    normalized = base64.b64encode(buffered.getvalue()).decode("ascii")
    return f"data:image/png;base64,{normalized}"


def build_messages(prompt: str, image_base64: Optional[str]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": get_system_prompt()}]
    if not image_base64:
        messages.extend(history[-10:])

    if image_base64:
        prompt = (
            f"{prompt}\n\n"
            "請根據圖片內容回答，最多三句；如果能看到文字、UI、角色、地圖或錯誤訊息，請直接指出。"
        )
        content = [
            {"type": "image_url", "image_url": {"url": image_to_data_url(image_base64)}},
            {"type": "text", "text": prompt},
        ]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})

    return messages


def append_history(role: str, content: str) -> None:
    history.append({"role": role, "content": content})
    if len(history) > 20:
        del history[:-20]


def is_text_file_task(prompt: str) -> bool:
    text = prompt.lower()
    wants_text_file = any(
        marker in text
        for marker in ("txt", ".txt", "文字檔", "文本檔", "純文字檔", "記事本")
    )
    wants_create = any(
        marker in prompt
        for marker in ("建立", "生成", "產生", "新增", "寫", "存成", "保存", "輸出", "幫我")
    )
    is_only_capability_question = (
        "嗎" in prompt
        and not any(marker in prompt for marker in ("幫我", "請", "替我", "幫忙"))
    )
    return wants_text_file and wants_create and not is_only_capability_question


def sanitize_txt_filename(filename: str) -> str:
    name = Path(filename.strip()).name
    allowed = []
    for char in name:
        if char.isalnum() or char in (" ", "-", "_", "."):
            allowed.append(char)
        else:
            allowed.append("_")
    clean = "".join(allowed).strip(" ._")
    if not clean:
        clean = time.strftime("generated-%Y%m%d-%H%M%S")
    if not clean.lower().endswith(".txt"):
        clean += ".txt"
    return clean[:120]


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]
    return json.loads(stripped)


def call_llama_once(messages: list[dict[str, Any]], max_tokens: int = 1536) -> str:
    payload = {
        "model": MODEL_ALIAS,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.4,
        "top_p": 0.9,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with post_json(f"{llama_base_url()}/v1/chat/completions", payload, 600) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def create_text_file_from_prompt(prompt: str) -> Path:
    messages = [
        {
            "role": "system",
            "content": (
                "你是本機文字檔產生器。請只輸出有效 JSON，不要 Markdown。"
                "JSON 格式必須是 {\"filename\":\"檔名.txt\",\"content\":\"文字檔內容\"}。"
                "filename 不可包含路徑；content 必須是使用者需要寫入 txt 的完整內容。"
            ),
        },
        {"role": "user", "content": prompt},
    ]
    output = call_llama_once(messages)

    try:
        spec = extract_json_object(output)
        filename = str(spec.get("filename") or "")
        content = str(spec.get("content") or "")
    except Exception:
        filename = ""
        content = output.strip()

    if not content:
        content = "這是一個由本機 AI 產生的文字檔。"

    GENERATED_DIR.mkdir(exist_ok=True)
    path = (GENERATED_DIR / sanitize_txt_filename(filename)).resolve()
    generated_root = GENERATED_DIR.resolve()
    if path.parent != generated_root:
        raise RuntimeError("Refusing to write outside generated_files.")

    path.write_text(content, encoding="utf-8", newline="\n")
    return path


CJK_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+")
WORD_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
GUIDE_INTENT_RE = re.compile(
    r"(攻略|怎麼打|怎麼走|在哪|哪裡|弱點|任務|素材|材料|路線|指路|地圖|boss|npc|quest|guide|route|map|weakness)",
    re.IGNORECASE,
)
OVERLAY_INTENT_RE = re.compile(
    r"(圈|圈出|圈選|框出|標記|標出|指引|導引|導航|指路|往哪|往哪走|哪邊|路線|路標|目標|目的地|箭頭|提醒|危險|門在哪|在哪裡|where|mark|circle|arrow|route|path|guide|navigate|target|objective|destination)",
    re.IGNORECASE,
)
MEMORY_ADD_RE = re.compile(
    r"(記住|幫我記|幫我記住|記一下|remember|note this|save this)",
    re.IGNORECASE,
)


def normalize_game_id(game_id: Optional[str]) -> Optional[str]:
    if not game_id:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", game_id.strip()).strip("._-")
    return cleaned[:80] or None


def clamp_limit(value: int, default: int = 5, upper: int = 20) -> int:
    try:
        number = int(value)
    except Exception:
        return default
    return max(1, min(number, upper))


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


def search_terms(text: str, max_terms: int = 32) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for raw in WORD_RE.findall(text.lower()):
        if len(raw) >= 2 and raw not in seen:
            seen.add(raw)
            terms.append(raw)
    for raw in cjk_ngrams(text):
        if raw not in seen:
            seen.add(raw)
            terms.append(raw)
    return terms[:max_terms]


def fts_query(text: str) -> str:
    terms = search_terms(text)
    escaped = [f'"{term.replace(chr(34), chr(34) + chr(34))}"' for term in terms]
    return " OR ".join(escaped)


def expand_search_text(*parts: str) -> str:
    text = "\n".join(part for part in parts if part)
    return text + "\n" + " ".join(cjk_ngrams(text))


def make_snippet(content: str, query: str, max_len: int = 220) -> str:
    compact = re.sub(r"\s+", " ", content).strip()
    if len(compact) <= max_len:
        return compact
    lowered = compact.lower()
    start = 0
    for term in search_terms(query):
        index = lowered.find(term.lower())
        if index >= 0:
            start = max(0, index - 60)
            break
    end = min(len(compact), start + max_len)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(compact) else ""
    return f"{prefix}{compact[start:end]}{suffix}"


def guide_connection() -> sqlite3.Connection:
    if not GUIDE_DB.exists():
        raise FileNotFoundError(f"Guide index not found: {GUIDE_DB}")
    conn = sqlite3.connect(GUIDE_DB)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_memory_db() -> None:
    MEMORY_CACHE_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(MEMORY_DB) as conn:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                game_id UNINDEXED,
                kind,
                content,
                tags,
                importance UNINDEXED,
                created_at UNINDEXED,
                updated_at UNINDEXED,
                search_text,
                tokenize='unicode61'
            )
            """
        )
        conn.commit()


def list_guide_games_sync() -> list[str]:
    games: set[str] = set()
    if GAME_GUIDES_DIR.exists():
        games.update(path.name for path in GAME_GUIDES_DIR.iterdir() if path.is_dir())
    if GUIDE_DB.exists():
        with guide_connection() as conn:
            rows = conn.execute("SELECT DISTINCT game_id FROM guide_fts WHERE game_id != ''").fetchall()
            games.update(str(row["game_id"]) for row in rows if row["game_id"])
    return sorted(games)


def search_guides_sync(query: str, game_id: Optional[str], limit: int = 5) -> list[dict[str, Any]]:
    query = (query or "").strip()
    if not query or not GUIDE_DB.exists():
        return []
    match = fts_query(query)
    if not match:
        return []
    normalized_game_id = normalize_game_id(game_id)
    sql = (
        "SELECT rowid, game_id, title, source_path, section, content, tags, updated_at, "
        "bm25(guide_fts) AS score FROM guide_fts WHERE guide_fts MATCH ?"
    )
    params: list[Any] = [match]
    if normalized_game_id:
        sql += " AND game_id = ?"
        params.append(normalized_game_id)
    sql += " ORDER BY score LIMIT ?"
    params.append(clamp_limit(limit, upper=10))
    try:
        with guide_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {
            "id": int(row["rowid"]),
            "game_id": row["game_id"],
            "title": row["title"],
            "section": row["section"],
            "snippet": make_snippet(row["content"], query),
            "source_path": row["source_path"],
            "tags": row["tags"],
            "score": float(row["score"]),
        }
        for row in rows
    ]


def add_memory_sync(
    content: str,
    game_id: Optional[str],
    kind: str = "note",
    tags: Optional[str] = None,
    importance: int = 3,
) -> dict[str, Any]:
    ensure_memory_db()
    normalized_game_id = normalize_game_id(game_id) or "global"
    safe_kind = re.sub(r"[^A-Za-z0-9_.-]+", "_", (kind or "note").strip().lower()).strip("._-") or "note"
    safe_importance = max(1, min(int(importance or 3), 5))
    clean_content = re.sub(r"\s+", " ", (content or "").strip())
    if not clean_content:
        raise ValueError("Memory content is empty.")
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    search_text = expand_search_text(normalized_game_id, safe_kind, clean_content, tags or "")
    with sqlite3.connect(MEMORY_DB) as conn:
        cursor = conn.execute(
            """
            INSERT INTO memory_fts(game_id, kind, content, tags, importance, created_at, updated_at, search_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (normalized_game_id, safe_kind, clean_content, tags or "", safe_importance, now, now, search_text),
        )
        conn.commit()
        row_id = cursor.lastrowid
    return {
        "id": row_id,
        "game_id": normalized_game_id,
        "kind": safe_kind,
        "content": clean_content,
        "tags": tags or "",
        "importance": safe_importance,
        "created_at": now,
    }


def search_memory_sync(
    query: str,
    game_id: Optional[str],
    kinds: Optional[list[str]] = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    query = (query or "").strip()
    if not query or not MEMORY_DB.exists():
        return []
    match = fts_query(query)
    if not match:
        return []
    normalized_game_id = normalize_game_id(game_id)
    sql = (
        "SELECT rowid, game_id, kind, content, tags, importance, created_at, updated_at, "
        "bm25(memory_fts) AS score FROM memory_fts WHERE memory_fts MATCH ?"
    )
    params: list[Any] = [match]
    if normalized_game_id:
        sql += " AND game_id IN (?, 'global')"
        params.append(normalized_game_id)
    if kinds:
        safe_kinds = [re.sub(r"[^A-Za-z0-9_.-]+", "_", kind.strip().lower()).strip("._-") for kind in kinds]
        safe_kinds = [kind for kind in safe_kinds if kind]
        if safe_kinds:
            sql += f" AND kind IN ({','.join('?' for _ in safe_kinds)})"
            params.extend(safe_kinds)
    sql += " ORDER BY importance DESC, score LIMIT ?"
    params.append(clamp_limit(limit, upper=10))
    try:
        with sqlite3.connect(MEMORY_DB) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {
            "id": int(row["rowid"]),
            "game_id": row["game_id"],
            "kind": row["kind"],
            "content": row["content"],
            "tags": row["tags"],
            "importance": int(row["importance"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "score": float(row["score"]),
        }
        for row in rows
    ]


def recent_memory_sync(game_id: Optional[str], limit: int = 10) -> list[dict[str, Any]]:
    if not MEMORY_DB.exists():
        return []
    normalized_game_id = normalize_game_id(game_id)
    sql = "SELECT rowid, game_id, kind, content, tags, importance, created_at, updated_at FROM memory_fts"
    params: list[Any] = []
    if normalized_game_id:
        sql += " WHERE game_id IN (?, 'global')"
        params.append(normalized_game_id)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(clamp_limit(limit, upper=30))
    with sqlite3.connect(MEMORY_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    return [
        {
            "id": int(row["rowid"]),
            "game_id": row["game_id"],
            "kind": row["kind"],
            "content": row["content"],
            "tags": row["tags"],
            "importance": int(row["importance"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


def should_use_guides(prompt: str, explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return explicit
    return bool(GUIDE_INTENT_RE.search(prompt or ""))


def should_use_overlay(prompt: str, image_base64: Optional[str]) -> bool:
    return bool(image_base64 and OVERLAY_INTENT_RE.search(prompt or ""))


def detect_memory_add(prompt: str) -> Optional[dict[str, str]]:
    text = re.sub(r"\s+", " ", (prompt or "").strip())
    if not text or not MEMORY_ADD_RE.search(text):
        return None
    content = MEMORY_ADD_RE.sub("", text, count=1).strip(" ：:，,。.")
    if not content:
        content = text
    lowered = text.lower()
    if any(marker in text for marker in ("不想", "喜歡", "偏好", "劇透", "設定")) or "prefer" in lowered:
        kind = "preference"
    elif any(marker in text for marker in ("現在", "目前", "卡", "做到", "目標", "進度")):
        kind = "state"
    else:
        kind = "note"
    return {"content": content[:1000], "kind": kind}


def format_rag_context(
    guide_results: list[dict[str, Any]],
    memory_results: list[dict[str, Any]],
    guide_was_requested: bool,
) -> str:
    lines: list[str] = []
    if memory_results:
        lines.append("Local player memory (CPU SQLite, use only as user-specific context):")
        for item in memory_results[:8]:
            lines.append(f"- [{item.get('kind')}] {item.get('content')}")
    if guide_results:
        lines.append("Local guide snippets (CPU SQLite, prefer these over general knowledge):")
        for index, item in enumerate(guide_results[:5], 1):
            title = item.get("title") or "Guide"
            section = item.get("section") or ""
            snippet = item.get("snippet") or ""
            source = item.get("source_path") or ""
            lines.append(f"{index}. {title} {section} ({source}): {snippet}")
    elif guide_was_requested:
        lines.append("Local guide snippets: no matching local guide was found. Do not invent guide facts.")
    return "\n".join(lines)


def build_augmented_prompt(original_prompt: str, rag_context: str) -> str:
    if not rag_context:
        return original_prompt
    return (
        f"{original_prompt}\n\n"
        "Use the local context below when it is relevant. Keep the answer concise and in Traditional Chinese.\n"
        "If local guide snippets are empty, say the local guide library has no matching entry.\n\n"
        f"{rag_context}"
    )


def build_overlay_messages(prompt: str, image_base64: str, rag_context: str) -> list[dict[str, Any]]:
    system = (
        "You are a game companion HUD planner. Return only valid JSON, no Markdown. "
        "Use Traditional Chinese for answer and labels. "
        "If you are not visually confident about coordinates, return an empty overlay items array. "
        "All coordinates must be normalized numbers from 0 to 1. "
        "JSON shape: {\"answer\":\"...\",\"overlay\":{\"duration_ms\":6000,\"items\":[...]}}. "
        "Allowed item types: circle, arrow, path, pin, label. "
        "For target/objective requests, use pin or circle with a short label. "
        "For route/navigation requests, use path with 2-6 points and optionally one arrow. "
        "For direction/guidance requests, use arrow from the player/current area to the target. "
        "For circle/mark requests, use circle around the visible object."
    )
    user_text = prompt
    if rag_context:
        user_text += "\n\nLocal context:\n" + rag_context
    user_text += (
        "\n\nIf the user asks for visual guidance, identify visible targets and provide at most 3 HUD items. "
        "For circle use x, y, radius. For arrow use from {x,y} and to {x,y}. "
        "For path use points [{x,y}]. For label/pin use x, y, label. "
        "Use circle for objects to circle, arrow for where to go next, path for route lines, and pin for targets/objectives."
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_base64)}},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def clamp_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return max(0.0, min(1.0, number))


def sanitize_overlay_item(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    item_type = str(item.get("type") or "").lower()
    item_type = {
        "target": "pin",
        "objective": "pin",
        "destination": "pin",
        "marker": "pin",
        "route": "path",
        "line": "path",
        "guidance": "arrow",
        "guide": "arrow",
        "direction": "arrow",
    }.get(item_type, item_type)
    allowed = {"circle", "arrow", "path", "pin", "label"}
    if item_type not in allowed:
        return None
    color = str(item.get("color") or "#00E5FF")
    label = str(item.get("label") or "")[:80]
    cleaned: dict[str, Any] = {"type": item_type, "color": color}
    if label:
        cleaned["label"] = label
    if item_type == "circle":
        cleaned.update(
            {
                "x": clamp_float(item.get("x")),
                "y": clamp_float(item.get("y")),
                "radius": max(0.01, min(float(item.get("radius") or 0.06), 0.35)),
            }
        )
    elif item_type == "arrow":
        src = item.get("from") or {}
        dst = item.get("to") or {}
        cleaned["from"] = {"x": clamp_float(src.get("x"), 0.45), "y": clamp_float(src.get("y"), 0.7)}
        cleaned["to"] = {"x": clamp_float(dst.get("x"), 0.55), "y": clamp_float(dst.get("y"), 0.45)}
    elif item_type == "path":
        points = item.get("points") or []
        cleaned["points"] = [
            {"x": clamp_float(point.get("x")), "y": clamp_float(point.get("y"))}
            for point in points
            if isinstance(point, dict)
        ][:8]
        if len(cleaned["points"]) < 2:
            return None
    else:
        cleaned.update({"x": clamp_float(item.get("x")), "y": clamp_float(item.get("y"))})
    return cleaned


def sanitize_overlay(raw_overlay: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw_overlay, dict):
        return None
    items = raw_overlay.get("items") or []
    cleaned_items = [
        cleaned
        for item in items
        if isinstance(item, dict)
        for cleaned in [sanitize_overlay_item(item)]
        if cleaned
    ][:5]
    if not cleaned_items:
        return None
    duration = int(raw_overlay.get("duration_ms") or 6000)
    return {"duration_ms": max(3000, min(duration, 8000)), "items": cleaned_items}


def create_overlay_response(prompt: str, image_base64: str, rag_context: str) -> dict[str, Any]:
    output = call_llama_once(build_overlay_messages(prompt, image_base64, rag_context), max_tokens=900)
    try:
        parsed = extract_json_object(output)
    except Exception:
        return {"answer": output.strip(), "overlay": None}
    answer = str(parsed.get("answer") or "").strip()
    overlay = sanitize_overlay(parsed.get("overlay"))
    if not answer:
        answer = "我已根據畫面整理提示。"
    return {"answer": answer, "overlay": overlay}


@app.on_event("startup")
async def startup_event():
    await asyncio.to_thread(start_llama_server)


@app.on_event("shutdown")
async def shutdown_event():
    if llama_process and llama_process.poll() is None:
        llama_process.terminate()


@app.get("/health")
async def health():
    return {
        "status": "ok" if llama_ready() else "loading",
        "model": MODEL_ALIAS,
        "llama_url": llama_base_url(),
        "vulkan_device": VULKAN_DEVICE,
        "resource_policy": "game",
        "llama_ctx_size": LLAMA_CTX_SIZE,
        "llama_parallel": LLAMA_PARALLEL or "auto",
        "llama_cache_ram_mib": LLAMA_CACHE_RAM or "default",
        "chat_backend": CHAT_BACKEND,
        "hermes_wsl_distro": HERMES_WSL_DISTRO if CHAT_BACKEND == "hermes" else None,
        "local_tools": ENABLE_LOCAL_TOOLS,
        "rag_backend": "cpu_sqlite_fts5",
    }


@app.get("/guides/games")
async def guide_games():
    games = await asyncio.to_thread(list_guide_games_sync)
    return {"games": games}


@app.post("/guides/search")
async def guide_search(request: GuideSearchRequest):
    results = await asyncio.to_thread(
        search_guides_sync,
        request.query,
        request.game_id,
        request.limit,
    )
    return {
        "game_id": normalize_game_id(request.game_id),
        "query": request.query,
        "results": results,
    }


@app.post("/memory/add")
async def memory_add(request: MemoryAddRequest):
    try:
        item = await asyncio.to_thread(
            add_memory_sync,
            request.content,
            request.game_id,
            request.kind,
            request.tags,
            request.importance,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok", "item": item}


@app.post("/memory/search")
async def memory_search(request: MemorySearchRequest):
    results = await asyncio.to_thread(
        search_memory_sync,
        request.query,
        request.game_id,
        request.kinds,
        request.limit,
    )
    return {
        "game_id": normalize_game_id(request.game_id),
        "query": request.query,
        "results": results,
    }


@app.get("/memory/recent")
async def memory_recent(game_id: Optional[str] = None, limit: int = 10):
    results = await asyncio.to_thread(recent_memory_sync, game_id, limit)
    return {"game_id": normalize_game_id(game_id), "results": results}


@app.get("/v1/models")
async def list_models():
    try:
        return get_json(f"{llama_base_url()}/v1/models", timeout=5)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    raw_body = await request.body()
    try:
        body = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")

    body.setdefault("model", MODEL_ALIAS)
    body.setdefault("chat_template_kwargs", {"enable_thinking": False})
    body.setdefault("max_tokens", int(os.environ.get("LLAMA_MAX_TOKENS", "512")))
    proxied_body = json.dumps(body, ensure_ascii=False).encode("utf-8")
    target_url = f"{llama_base_url()}/v1/chat/completions"

    if body.get("stream"):
        async def proxy_stream():
            response = await asyncio.to_thread(post_raw_json, target_url, proxied_body, 600)
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    line = await asyncio.to_thread(response.readline)
                    if not line:
                        break
                    yield line
            finally:
                response.close()

        return StreamingResponse(proxy_stream(), media_type="text/event-stream")

    try:
        response = await asyncio.to_thread(post_raw_json, target_url, proxied_body, 600)
        try:
            data = json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
        return JSONResponse(data)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=exc.code, detail=detail)
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/clear")
async def clear_history():
    history.clear()
    return {"status": "success", "message": "History cleared"}


@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    raise HTTPException(status_code=503, detail="Voice transcription is not wired in this llama.cpp Vulkan bridge yet.")


@app.get("/screenshot")
async def screenshot_endpoint():
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return {"image_base64": base64.b64encode(buffered.getvalue()).decode("ascii")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat")
async def chat_endpoint(fastapi_request: Request, chat_request: ChatRequest):
    prompt = chat_request.message or "請分析這張截圖。"
    game_id = normalize_game_id(chat_request.game_id)

    memory_add = detect_memory_add(prompt)
    if memory_add and not chat_request.image_base64:
        async def memory_add_event_generator():
            try:
                item = await asyncio.to_thread(
                    add_memory_sync,
                    memory_add["content"],
                    game_id,
                    memory_add["kind"],
                    "",
                    4,
                )
                message = f"已記住：{item['content']}"
                append_history("user", prompt)
                append_history("assistant", message)
                yield f"data: {json.dumps({'content': message}, ensure_ascii=False)}\n\n"
            except Exception as exc:
                error = f"記憶寫入失敗：{exc}"
                yield f"data: {json.dumps({'content': error}, ensure_ascii=False)}\n\n"

        return StreamingResponse(memory_add_event_generator(), media_type="text/event-stream")

    guide_was_requested = should_use_guides(prompt, chat_request.use_guides)
    memory_results: list[dict[str, Any]] = []
    guide_results: list[dict[str, Any]] = []
    if chat_request.use_memory:
        memory_results = await asyncio.to_thread(search_memory_sync, prompt, game_id, None, 8)
    if guide_was_requested:
        guide_results = await asyncio.to_thread(search_guides_sync, prompt, game_id, 5)

    if guide_was_requested and not guide_results and not chat_request.image_base64:
        async def no_guide_event_generator():
            game_label = game_id or "目前遊戲"
            message = f"本機攻略庫沒有找到「{game_label}」相關條目。你可以把攻略 .md/.txt/.html 放進 game_guides/{game_label}/ 後重建索引。"
            if memory_results:
                message += "\n我有找到一些玩家記憶，但沒有本機攻略片段，所以不會硬編攻略。"
            yield f"data: {json.dumps({'content': message}, ensure_ascii=False)}\n\n"

        return StreamingResponse(no_guide_event_generator(), media_type="text/event-stream")

    rag_context = format_rag_context(guide_results, memory_results, guide_was_requested)
    augmented_prompt = build_augmented_prompt(prompt, rag_context)

    if ENABLE_LOCAL_TOOLS and not chat_request.image_base64 and is_text_file_task(prompt):
        async def tool_event_generator():
            try:
                path = await asyncio.to_thread(create_text_file_from_prompt, prompt)
                message = f"已建立文字檔：{path}"
                append_history("user", prompt)
                append_history("assistant", message)
                yield f"data: {json.dumps({'content': message}, ensure_ascii=False)}\n\n"
            except Exception as exc:
                error = f"建立文字檔失敗：{exc}"
                yield f"data: {json.dumps({'content': error}, ensure_ascii=False)}\n\n"

        return StreamingResponse(tool_event_generator(), media_type="text/event-stream")

    if CHAT_BACKEND == "hermes" and not chat_request.image_base64:
        hermes_prompt = build_hermes_prompt(augmented_prompt)

        async def hermes_event_generator():
            collected = ""
            async with generate_lock:
                try:
                    collected = await asyncio.to_thread(call_hermes_no_tools, hermes_prompt)
                except subprocess.TimeoutExpired:
                    error = "Hermes 回應逾時。請稍後再試，或把 IGPU_CHAT_BACKEND 改成 llama 先走直接模型。"
                    yield f"data: {json.dumps({'content': error}, ensure_ascii=False)}\n\n"
                    return
                except Exception as exc:
                    error = f"Hermes 連線失敗：{exc}"
                    yield f"data: {json.dumps({'content': error}, ensure_ascii=False)}\n\n"
                    return

            if collected:
                append_history("user", prompt)
                append_history("assistant", collected)
                for chunk in chunk_text(collected):
                    if await fastapi_request.is_disconnected():
                        break
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0)

        return StreamingResponse(hermes_event_generator(), media_type="text/event-stream")

    if should_use_overlay(prompt, chat_request.image_base64):
        async def overlay_event_generator():
            yield f"data: {json.dumps({'content': '我正在讀取截圖，等一下我會嘗試標記位置。\\n'}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)
            async with generate_lock:
                try:
                    result = await asyncio.to_thread(
                        create_overlay_response,
                        augmented_prompt,
                        chat_request.image_base64 or "",
                        rag_context,
                    )
                except Exception as exc:
                    error = f"HUD 指引產生失敗：{exc}"
                    yield f"data: {json.dumps({'content': error}, ensure_ascii=False)}\n\n"
                    return

            answer = str(result.get("answer") or "").strip()
            overlay = result.get("overlay")
            if answer:
                append_history("user", prompt)
                append_history("assistant", answer)
                for chunk in chunk_text(answer):
                    if await fastapi_request.is_disconnected():
                        return
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0)
            if overlay:
                yield f"data: {json.dumps({'overlay': overlay}, ensure_ascii=False)}\n\n"

        return StreamingResponse(overlay_event_generator(), media_type="text/event-stream")

    messages = build_messages(augmented_prompt, chat_request.image_base64)

    payload = {
        "model": MODEL_ALIAS,
        "messages": messages,
        "stream": True,
        "max_tokens": int(
            os.environ.get(
                "LLAMA_IMAGE_MAX_TOKENS" if chat_request.image_base64 else "LLAMA_MAX_TOKENS",
                "256" if chat_request.image_base64 else "512",
            )
        ),
        "temperature": 1.0,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    async def event_generator():
        collected = ""
        if chat_request.image_base64:
            yield f"data: {json.dumps({'content': '我正在讀取截圖，先辨識畫面內容。\\n'}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)
        async with generate_lock:
            try:
                response = await asyncio.to_thread(
                    post_json,
                    f"{llama_base_url()}/v1/chat/completions",
                    payload,
                    600,
                )
            except urllib.error.URLError as exc:
                yield f"data: {json.dumps({'content': f'模型服務尚未就緒: {exc}'}, ensure_ascii=False)}\n\n"
                return

            try:
                while True:
                    if await fastapi_request.is_disconnected():
                        break
                    line = await asyncio.to_thread(response.readline)
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text.startswith("data: "):
                        continue
                    data = text[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content") or ""
                    if content:
                        collected += content
                        yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
            finally:
                response.close()

            if collected:
                append_history("user", prompt)
                append_history("assistant", collected)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
