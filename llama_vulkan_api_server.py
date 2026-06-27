import asyncio
import base64
import ctypes
import hashlib
import http.client
import html
import io
import json
import os
import re
import shlex
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import mss
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageFont, ImageGrab, ImageStat
from pydantic import BaseModel


LLAMA_HOST = os.environ.get("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "18080"))
API_HOST = os.environ.get("IGPU_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("IGPU_API_PORT", "8000"))
MODEL_ALIAS = os.environ.get("LLAMA_MODEL_ALIAS", "qwen2.5-vl-3b-instruct-q8_0")
VULKAN_DEVICE = os.environ.get("GGML_VK_VISIBLE_DEVICES", "1")
LLAMA_CTX_SIZE = os.environ.get("LLAMA_CTX_SIZE", "32768")
LLAMA_GPU_LAYERS = os.environ.get("LLAMA_GPU_LAYERS", "1")
LLAMA_FLASH_ATTN = os.environ.get("LLAMA_FLASH_ATTN", "off").strip().lower()
LLAMA_PARALLEL = os.environ.get("LLAMA_PARALLEL", "").strip()
LLAMA_CACHE_RAM = os.environ.get("LLAMA_CACHE_RAM", "").strip()
CHAT_BACKEND = os.environ.get("IGPU_CHAT_BACKEND", "llama").strip().lower()
LLAMA_AUTO_START = os.environ.get(
    "LLAMA_AUTO_START",
    "0" if CHAT_BACKEND == "hermes" else "1",
).strip().lower() in {"1", "true", "yes", "on"}
HISTORY_CONTEXT_MESSAGES = int(os.environ.get("IGPU_HISTORY_CONTEXT_MESSAGES", "30"))
IMAGE_HISTORY_CONTEXT_MESSAGES = int(os.environ.get("IGPU_IMAGE_HISTORY_CONTEXT_MESSAGES", "12"))
HISTORY_STORE_MESSAGES = int(os.environ.get("IGPU_HISTORY_STORE_MESSAGES", "80"))
HERMES_WSL_DISTRO = os.environ.get("HERMES_WSL_DISTRO", "Ubuntu-24.04")
HERMES_TIMEOUT_SECONDS = int(os.environ.get("HERMES_TIMEOUT_SECONDS", "600"))
HERMES_BASE_URL = os.environ.get("HERMES_BASE_URL", f"http://127.0.0.1:{API_PORT}/v1").strip()
HERMES_MAX_TOKENS = int(os.environ.get("HERMES_MAX_TOKENS", "160"))
HERMES_CONTEXT_LENGTH = int(os.environ.get("HERMES_CONTEXT_LENGTH", "32768"))
HERMES_USE_CONFIG_MODEL = os.environ.get("HERMES_USE_CONFIG_MODEL", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HERMES_AGENT_WEB_ENABLED = os.environ.get(
    "HERMES_AGENT_WEB_ENABLED",
    "1" if HERMES_USE_CONFIG_MODEL else "0",
).strip().lower() in {"1", "true", "yes", "on"}
HERMES_AGENT_TOOLSETS = os.environ.get("HERMES_AGENT_TOOLSETS", "web").strip() or "web"
HERMES_AGENT_MAX_TOKENS = int(os.environ.get("HERMES_AGENT_MAX_TOKENS", "360"))
OPENAI_MAX_TOKENS_CAP = int(os.environ.get("LLAMA_OPENAI_MAX_TOKENS_CAP", "0"))
ENABLE_LOCAL_TOOLS = os.environ.get(
    "IGPU_ENABLE_LOCAL_TOOLS",
    "0" if CHAT_BACKEND == "hermes" else "1",
).strip().lower() in {"1", "true", "yes", "on"}
LOCAL_ROUTER_ENABLED = os.environ.get("IGPU_LOCAL_ROUTER_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LOCAL_ROUTER_URL = os.environ.get("IGPU_LOCAL_ROUTER_URL", "http://127.0.0.1:18081").strip()
LOCAL_ROUTER_MODEL = os.environ.get("IGPU_LOCAL_ROUTER_MODEL", "qwen3.5-2b-q4_k_m").strip()
LOCAL_ROUTER_ROLE = os.environ.get("IGPU_LOCAL_ROUTER_ROLE", "user_intent_router").strip()
LOCAL_ROUTER_TIMEOUT_SECONDS = int(os.environ.get("IGPU_LOCAL_ROUTER_TIMEOUT", "20"))
LOCAL_ROUTER_GAMEPATH_GATE = os.environ.get("IGPU_LOCAL_ROUTER_GAMEPATH_GATE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LOCAL_ROUTER_GAMEPATH_MAX_CHARS = int(os.environ.get("IGPU_LOCAL_ROUTER_GAMEPATH_MAX_CHARS", "280"))
LOCAL_ROUTER_RETRIEVAL_EVAL = os.environ.get("IGPU_LOCAL_ROUTER_RETRIEVAL_EVAL", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LOCAL_ROUTER_ALWAYS_ROUTE = os.environ.get("IGPU_LOCAL_ROUTER_ALWAYS_ROUTE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LOCAL_ROUTER_DECISION_CACHE_TTL_SECONDS = int(os.environ.get("IGPU_LOCAL_ROUTER_CACHE_TTL", "600"))
LOCAL_ROUTER_INTENT_CACHE_VERSION = "intent-route-v3-screen-guard"

ASSET_ROOT = Path(
    os.environ.get(
        "LLAMA_GEMMA4_HOME",
        str(Path(os.environ["LOCALAPPDATA"]) / "llama-gemma4-e4b"),
    )
)
LLAMA_DIR = ASSET_ROOT / "tools" / "llama.cpp-vulkan"
LLAMA_SERVER = LLAMA_DIR / "llama-server.exe"
MODEL_PATH = Path(
    os.environ.get(
        "LLAMA_MODEL_PATH",
        str(ASSET_ROOT / "models" / "gemma-4-E4B-it-Q4_K_M.gguf"),
    )
)
MMPROJ_PATH = Path(
    os.environ.get(
        "LLAMA_MMPROJ_PATH",
        str(ASSET_ROOT / "models" / "mmproj-BF16.gguf"),
    )
)
LLAMA_HF_REPO = os.environ.get(
    "LLAMA_HF_REPO",
    "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q8_0",
).strip()
LLAMA_FORCE_LOCAL_MODEL = os.environ.get("LLAMA_FORCE_LOCAL_MODEL", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
if LLAMA_FORCE_LOCAL_MODEL:
    LLAMA_HF_REPO = ""
LLAMA_HF_FILE = os.environ.get("LLAMA_HF_FILE", "").strip()
LLAMA_CHAT_TEMPLATE_KWARGS = os.environ.get("LLAMA_CHAT_TEMPLATE_KWARGS", "").strip()
LLAMA_IMAGE_MIN_TOKENS = os.environ.get("LLAMA_ARG_IMAGE_MIN_TOKENS", "256").strip()
LLAMA_IMAGE_MAX_TOKENS_SERVER = os.environ.get("LLAMA_ARG_IMAGE_MAX_TOKENS", "512").strip()
LLAMA_SKIP_CHAT_PARSING = os.environ.get("LLAMA_SKIP_CHAT_PARSING", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LOG_DIR = Path(os.environ.get("IGPU_LOG_DIR", str(Path(__file__).resolve().parent / "logs")))
LLAMA_LOG = LOG_DIR / "llama-server.log"
LATEST_SCREENSHOT = LOG_DIR / "latest-screenshot.jpg"
LATEST_VISION_INPUT = LOG_DIR / "latest-vision-input.jpg"
LATEST_VISION_RETRY_INPUT = LOG_DIR / "latest-vision-retry-input.jpg"
LATEST_OCR_INPUT = LOG_DIR / "latest-ocr-input.jpg"
LATEST_OVERLAY_GRID_INPUT = LOG_DIR / "latest-overlay-grid-input.jpg"
OVERLAY_GRID_COLUMNS = 6
OVERLAY_GRID_ROWS = 4
ENABLE_OCR_CONTEXT = os.environ.get("IGPU_ENABLE_OCR_CONTEXT", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
IGNORED_CAPTURE_TITLES = (
    "Game Companion",
    "overlay-chat",
    "game-guidance-hud",
    "game-task-log",
    "game-search",
)
IGNORED_CAPTURE_PROCESSES = (
    "overlay-chat.exe",
    "textinputhost.exe",
)
GENERATED_DIR = Path(__file__).resolve().parent / "generated_files"
PROJECT_ROOT = Path(__file__).resolve().parent
HERMES_CHAT_SCRIPT = PROJECT_ROOT / "scripts" / "hermes_no_tools_chat.py"
HERMES_AGENT_WEB_SCRIPT = PROJECT_ROOT / "scripts" / "hermes_agent_web_chat.py"
GAME_GUIDES_DIR = PROJECT_ROOT / "game_guides"
GUIDE_CACHE_DIR = PROJECT_ROOT / "guide_cache"
GUIDE_DB = GUIDE_CACHE_DIR / "guide.sqlite"
GAMEPATH_DIR = Path(os.environ.get("IGPU_GAMEPATH_DIR", str(PROJECT_ROOT / "gamepath")))
GAMEPATH_DB = GAMEPATH_DIR / "gamepath.sqlite"
GAMEPATH_NOTES_DIR = GAMEPATH_DIR / "notes"
GAME_PROFILES_FILE = PROJECT_ROOT / "game_profiles.json"
MEMORY_CACHE_DIR = PROJECT_ROOT / "memory_cache"
MEMORY_DB = MEMORY_CACHE_DIR / "memory.sqlite"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
LIVE_STATE_DIR = Path(os.environ.get("IGPU_LIVE_STATE_DIR", str(RUNTIME_DIR / "state")))
LIVE_STATE_FILE = LIVE_STATE_DIR / "current_game_state.json"

llama_process: Optional[subprocess.Popen] = None
history: list[dict[str, Any]] = []
last_gamepath_reference: dict[str, Any] = {}
local_router_decision_cache: dict[str, tuple[float, dict[str, Any]]] = {}
last_active_game_window: Optional[dict[str, Any]] = None
last_active_game_detection: Optional[dict[str, Any]] = None
live_state_enabled = os.environ.get("IGPU_LIVE_STATE_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
live_state_task: Optional[asyncio.Task] = None
live_state_memory: dict[str, Any] = {}
live_state_last_signature: Optional[list[float]] = None
live_state_last_analyze_at = 0.0
live_state_last_capture_at = 0.0
live_state_busy = False
live_state_monitor: Optional[int] = None
live_state_mode = "foreground"
live_state_error_count = 0
generate_lock = asyncio.Lock()
ocr_engine: Any = None
stt_model: Any = None


def enable_windows_dpi_awareness() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


enable_windows_dpi_awareness()


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
    use_live_state: bool = True


class LiveStateStartRequest(BaseModel):
    monitor: Optional[int] = None
    mode: str = "foreground"


class LiveStateAnalyzeRequest(BaseModel):
    force: bool = True
    monitor: Optional[int] = None
    mode: str = "foreground"


class IntentRouteRequest(BaseModel):
    message: str
    game_id: Optional[str] = None
    use_guides: Optional[bool] = None


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


class GamePathAddRequest(BaseModel):
    title: Optional[str] = None
    question: str
    answer_summary: str
    game_id: Optional[str] = None
    tags: Any = None
    spoiler_level: str = "none"
    source_type: str = "manual"
    agent_used: bool = False
    version: Optional[str] = None
    area: Optional[str] = None
    entity_type: Optional[str] = None
    entity_name: Optional[str] = None
    source_quality: Optional[float] = None


class GamePathSearchRequest(BaseModel):
    query: str
    game_id: Optional[str] = None
    tags: Any = None
    spoiler_level: str = "low"
    limit: int = 5
    version: Optional[str] = None
    area: Optional[str] = None
    entity_type: Optional[str] = None
    entity_name: Optional[str] = None
    strict_metadata: bool = False


class GamePathFeedbackRequest(BaseModel):
    entry_id: Optional[int] = None
    message: str = ""
    game_id: Optional[str] = None
    state: str = "disputed"


class TaskAnalyzeRequest(BaseModel):
    message: str = ""
    image_base64: Optional[str] = None
    game_id: Optional[str] = None
    source_title: Optional[str] = None


class GameProfileLearnRequest(BaseModel):
    game_id: str
    name: Optional[str] = None
    process_name: Optional[str] = None
    process_path: Optional[str] = None
    window_title: Optional[str] = None


def get_system_prompt() -> str:
    return (
        "你是即時遊戲陪玩助理。用繁體中文回答，語氣自然、簡短、直接。"
        "一般聊天最多 2 句；遊戲建議用 1 到 3 個可執行重點。"
        "不要重複同一個字詞，不要自稱系統分析師，不要輸出亂碼。"
        "如果使用者貼截圖，先說你看到的重點，再給下一步建議。"
    )
    return (
        "你是遊戲陪伴 AI，也是電腦系統效能分析師。"
        "你用繁體中文回答，語氣親切、簡潔、像會陪玩家一起看局勢的隊友。"
        "看到遊戲截圖時，請直接描述實際畫面，再給一到兩個可執行建議；不要泛泛要求使用者再貼圖。"
    )


def llama_base_url() -> str:
    return f"http://{LLAMA_HOST}:{LLAMA_PORT}"


def local_router_v1_url(endpoint: str) -> str:
    base = LOCAL_ROUTER_URL.rstrip("/")
    clean_endpoint = endpoint.lstrip("/")
    if base.endswith("/v1"):
        return f"{base}/{clean_endpoint}"
    return f"{base}/v1/{clean_endpoint}"


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
        "You are the game companion AI. Reply in Traditional Chinese with a concise, friendly tone. "
        "You are currently called through Hermes Agent using the model configured in Hermes. "
        "Do not perform system actions or use tools in this text-only bridge unless an explicit "
        "Hermes tool route is enabled."
    )
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


def get_hermes_agent_web_system_prompt() -> str:
    return (
        "You are the Game Companion Hermes Agent. Reply in Traditional Chinese, concise and practical. "
        "You may use only the Hermes web toolset, backed by Tavily, and you must decide by yourself "
        "whether web search is needed. Do not search for every message. Search only when it clearly "
        "helps: walkthroughs, guides, item usage, version/update differences, current events, unclear "
        "game mechanics, or when the player explicitly asks to check the web. If local context is enough, "
        "answer directly without web search. Default to no-spoiler guidance: avoid story twists, later "
        "area names, character fate, endings, and surprise encounters unless the player explicitly asks "
        "for the full solution. Prefer one direct teaching hint first, then add details only when useful. "
        "Do not label answers with tiered hint markers. If the player asks for the answer directly, "
        "give a clear solution but still avoid unnecessary story spoilers. When you do "
        "search, use retrieved pages only as private background material. Condense them into useful player "
        "guidance. Do not include a sources/references/links section, source titles, or URLs unless the "
        "player explicitly asks for sources or links. Mention uncertainty or version mismatch only when it "
        "affects the advice. Never copy long passages from sources."
    )


def build_hermes_agent_web_prompt(
    prompt: str,
    *,
    game_id: str,
    rag_context: str = "",
    active_game: Optional[dict[str, Any]] = None,
) -> str:
    lines = [get_hermes_agent_web_system_prompt()]
    if history[-8:]:
        lines.append("\nRecent chat:")
        for item in history[-8:]:
            role = "Player" if item.get("role") == "user" else "Assistant"
            content = str(item.get("content") or "").strip()
            if content:
                lines.append(f"{role}: {content}")

    lines.append("\nCurrent game context:")
    if game_id:
        lines.append(f"- selected_game_id: {game_id}")
    if active_game:
        lines.append(f"- detected_game: {active_game.get('name') or active_game.get('game_id') or ''}")
        lines.append(f"- detection_confidence: {active_game.get('confidence')}")
        lines.append(f"- process_name: {active_game.get('process_name') or ''}")
        lines.append(f"- window_title: {active_game.get('window_title') or ''}")
    if rag_context:
        lines.append("\nLocal context:")
        lines.append(rag_context)

    lines.append("\nPlayer message:")
    lines.append(prompt)
    lines.append(
        "\nUse your own judgment: answer directly if enough context exists; otherwise use Tavily web "
        "search through the web toolset. For guide searches, build queries from game + platform/version "
        "+ scene/item/objective + guide/walkthrough/tips/no spoilers. If GamePath context is present, "
        "first extract only the relevant passages and turn them into a compact player hint; do not paste "
        "the whole local document. If you search, summarize the result for the player and omit "
        "references/URLs unless explicitly requested."
    )
    return "\n".join(lines)


def wants_source_details(prompt: str) -> bool:
    return bool(
        re.search(
            r"(來源|參考|連結|網址|source|sources|reference|references|link|links|url)",
            prompt or "",
            re.IGNORECASE,
        )
    )


def condense_agent_answer(answer: str, prompt: str) -> str:
    text = str(answer or "").strip()
    if not text or wants_source_details(prompt):
        return text

    source_header_re = re.compile(
        r"^\s*(?:來源|資料來源|參考|參考資料|參考來源|連結|相關連結|Sources?|References?|Links?)\s*[:：]?\s*$",
        re.IGNORECASE,
    )
    inline_source_re = re.compile(
        r"^\s*(?:來源|資料來源|參考|參考資料|參考來源|Sources?|References?|Links?)\s*[:：]",
        re.IGNORECASE,
    )
    source_line_re = re.compile(
        r"^\s*(?:[-*]|\d+[.)、])?\s*(?:https?://|\[[^\]]+\]\(https?://)",
        re.IGNORECASE,
    )
    kept: list[str] = []
    skipping_sources = False
    for line in text.splitlines():
        stripped = line.strip()
        if source_header_re.match(stripped) or inline_source_re.match(stripped):
            skipping_sources = True
            continue
        if skipping_sources:
            continue
        if source_line_re.match(stripped):
            continue
        kept.append(line)

    cleaned = "\n".join(kept).strip()
    cleaned = re.sub(
        r"\n{0,2}\s*(?:來源|資料來源|參考資料|參考來源|Sources?|References?|Links?)\s*[:：].*$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    return cleaned or text


def call_hermes_no_tools(
    prompt: str,
    *,
    image_file: Optional[Path] = None,
    max_tokens: Optional[int] = None,
) -> str:
    if not HERMES_CHAT_SCRIPT.exists():
        raise RuntimeError(f"Missing Hermes chat script: {HERMES_CHAT_SCRIPT}")

    project_wsl = windows_path_to_wsl(PROJECT_ROOT)
    script_wsl = windows_path_to_wsl(HERMES_CHAT_SCRIPT)
    token_limit = int(max_tokens or HERMES_MAX_TOKENS)
    hermes_env = {
        "HERMES_API_TIMEOUT": str(max(HERMES_TIMEOUT_SECONDS, 600)),
        "HERMES_API_CALL_STALE_TIMEOUT": str(max(HERMES_TIMEOUT_SECONDS, 600)),
        "HERMES_MAX_TOKENS": str(token_limit),
        "HERMES_CONTEXT_LENGTH": str(HERMES_CONTEXT_LENGTH),
    }
    if HERMES_USE_CONFIG_MODEL:
        hermes_env["HERMES_USE_CONFIG_MODEL"] = "1"
    else:
        hermes_env["OPENAI_API_KEY"] = "no-key-required"
        hermes_env["CUSTOM_BASE_URL"] = HERMES_BASE_URL
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in hermes_env.items())
    script_args = [
        f"--api-port {API_PORT}",
        f"--max-tokens {token_limit}",
        f"--context-length {HERMES_CONTEXT_LENGTH}",
        f"--api-timeout {max(HERMES_TIMEOUT_SECONDS, 600)}",
        f"--api-call-stale-timeout {max(HERMES_TIMEOUT_SECONDS, 600)}",
    ]
    if image_file:
        script_args.append(f"--image-file {shlex.quote(windows_path_to_wsl(image_file))}")
    if not HERMES_USE_CONFIG_MODEL:
        script_args.insert(0, f"--model {shlex.quote(MODEL_ALIAS)}")
        script_args.insert(0, f"--base-url {shlex.quote(HERMES_BASE_URL)}")
    command = (
        f"cd {shlex.quote(project_wsl)} && "
        f"{env_prefix} "
        f"~/.hermes/hermes-agent/venv/bin/python {shlex.quote(script_wsl)} "
        + " ".join(script_args)
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


def call_hermes_web_agent(
    prompt: str,
    *,
    image_file: Optional[Path] = None,
    max_tokens: Optional[int] = None,
) -> str:
    if not HERMES_AGENT_WEB_SCRIPT.exists():
        raise RuntimeError(f"Missing Hermes web agent script: {HERMES_AGENT_WEB_SCRIPT}")

    project_wsl = windows_path_to_wsl(PROJECT_ROOT)
    script_wsl = windows_path_to_wsl(HERMES_AGENT_WEB_SCRIPT)
    token_limit = int(max_tokens or HERMES_AGENT_MAX_TOKENS)
    hermes_env = {
        "HERMES_API_TIMEOUT": str(max(HERMES_TIMEOUT_SECONDS, 900)),
        "HERMES_API_CALL_STALE_TIMEOUT": str(max(HERMES_TIMEOUT_SECONDS, 900)),
        "HERMES_AGENT_MAX_TOKENS": str(token_limit),
        "HERMES_CONTEXT_LENGTH": str(HERMES_CONTEXT_LENGTH),
        "HERMES_AGENT_TOOLSETS": HERMES_AGENT_TOOLSETS,
    }
    if HERMES_USE_CONFIG_MODEL:
        hermes_env["HERMES_USE_CONFIG_MODEL"] = "1"
    else:
        hermes_env["OPENAI_API_KEY"] = "no-key-required"
        hermes_env["CUSTOM_BASE_URL"] = HERMES_BASE_URL
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in hermes_env.items())
    script_args = [
        f"--api-port {API_PORT}",
        f"--toolsets {shlex.quote(HERMES_AGENT_TOOLSETS)}",
        f"--max-tokens {token_limit}",
        f"--context-length {HERMES_CONTEXT_LENGTH}",
        f"--api-timeout {max(HERMES_TIMEOUT_SECONDS, 900)}",
        f"--api-call-stale-timeout {max(HERMES_TIMEOUT_SECONDS, 900)}",
    ]
    if image_file:
        script_args.append(f"--image-file {shlex.quote(windows_path_to_wsl(image_file))}")
    if not HERMES_USE_CONFIG_MODEL:
        script_args.insert(0, f"--model {shlex.quote(MODEL_ALIAS)}")
        script_args.insert(0, f"--base-url {shlex.quote(HERMES_BASE_URL)}")
    command = (
        f"cd {shlex.quote(project_wsl)} && "
        f"{env_prefix} "
        f"~/.hermes/hermes-agent/venv/bin/python {shlex.quote(script_wsl)} "
        + " ".join(script_args)
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
        timeout=max(HERMES_TIMEOUT_SECONDS, 900),
        creationflags=creationflags,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or f"Hermes web agent exited with code {result.returncode}")
    output = result.stdout.strip()
    if not output:
        raise RuntimeError("Hermes web agent returned an empty response.")
    return output


def hermes_prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip()
        content = message.get("content")
        prefix = "System" if role == "system" else "User"
        if isinstance(content, str):
            if content.strip():
                lines.append(f"{prefix}:\n{content.strip()}")
            continue
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str) and item.strip():
                    text_parts.append(item.strip())
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str) and text.strip():
                        text_parts.append(text.strip())
            if text_parts:
                lines.append(f"{prefix}:\n" + "\n\n".join(text_parts))
    return "\n\n".join(lines).strip()


def call_hermes_messages(
    messages: list[dict[str, Any]],
    *,
    image_file: Optional[Path] = None,
    max_tokens: Optional[int] = None,
) -> str:
    prompt = hermes_prompt_from_messages(messages)
    if not prompt:
        prompt = "請分析這張圖片並用繁體中文簡短回答。"
    return call_hermes_no_tools(prompt, image_file=image_file, max_tokens=max_tokens)


def chunk_text(text: str, size: int = 80):
    for index in range(0, len(text), size):
        yield text[index : index + size]


def sse_data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def clean_response_inline_text(text: str) -> str:
    cleaned = re.sub(r"```[\s\S]*?```", " ", str(text or ""))
    cleaned = re.sub(r"[`*_>#]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def truncate_response_short(text: str, limit: int = 118) -> str:
    cleaned = clean_response_inline_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 3)].rstrip()}..."


def parse_response_format_object(answer: str) -> Optional[dict[str, str]]:
    raw = str(answer or "").strip()
    if not raw:
        return None
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    if not raw.startswith("{"):
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    short = str(parsed.get("short") or parsed.get("Short") or "").strip()
    long = str(parsed.get("long") or parsed.get("Long") or parsed.get("content") or "").strip()
    if not short and not long:
        return None
    return {"short": short, "long": long}


def parse_response_format_labels(answer: str) -> Optional[dict[str, str]]:
    raw = str(answer or "").strip()
    if not raw:
        return None
    short_re = r"(?:短回覆|短答|短教學|摘要|short)"
    long_re = r"(?:詳細回覆|長回覆|完整回覆|long)"
    short_match = re.search(
        rf"(?:^|\n)\s*{short_re}\s*[:：]\s*([\s\S]*?)(?=\n\s*{long_re}\s*[:：]|\Z)",
        raw,
        flags=re.IGNORECASE,
    )
    long_match = re.search(
        rf"(?:^|\n)\s*{long_re}\s*[:：]\s*([\s\S]*?)(?=\n\s*{short_re}\s*[:：]|\Z)",
        raw,
        flags=re.IGNORECASE,
    )
    short = (short_match.group(1).strip() if short_match else "")
    long = (long_match.group(1).strip() if long_match else "")
    if not short and not long:
        return None
    return {"short": short, "long": long}


def extract_hint_line(answer: str, number: int = 3) -> str:
    lines = str(answer or "").splitlines()
    capture: list[str] = []
    in_target = False
    hint_re = re.compile(r"^\s*(?:[-*]\s*)?Hint\s*([123])\s*[:：]\s*(.*)$", re.IGNORECASE)
    for line in lines:
        match = hint_re.match(line)
        if match:
            if in_target:
                break
            in_target = int(match.group(1)) == number
            if in_target and match.group(2).strip():
                capture.append(match.group(2).strip())
            continue
        if in_target:
            stripped = line.strip()
            if not stripped:
                break
            capture.append(stripped)
    return truncate_response_short(" ".join(capture), 118) if capture else ""


def derive_response_short(answer: str) -> str:
    # Compatibility only: older cached answers may still contain tiered hint markers.
    hint3 = extract_hint_line(answer, 3)
    if hint3:
        return hint3
    for line in str(answer or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(GamePath\s*已[有存]|來源|資料來源|參考|References?|Sources?)", stripped, re.IGNORECASE):
            continue
        stripped = re.sub(r"^\s*(?:[-*]|\d+[.)、])\s*", "", stripped)
        stripped = re.sub(r"^\s*(?:短回覆|短答|短教學|摘要|Hint\s*[123])\s*[:：]\s*", "", stripped, flags=re.IGNORECASE)
        if stripped:
            return truncate_response_short(stripped, 118)
    compact = clean_response_inline_text(answer)
    parts = re.split(r"(?<=[。！？!?])\s+", compact, maxsplit=1)
    return truncate_response_short(parts[0] if parts else compact, 118)


def build_response_format(answer: str) -> dict[str, str]:
    parsed = parse_response_format_object(answer) or parse_response_format_labels(answer)
    long_text = (parsed or {}).get("long") or str(answer or "").strip()
    short_text = (parsed or {}).get("short") or derive_response_short(long_text)
    return {
        "short": short_text.strip(),
        "long": long_text.strip(),
    }


def response_format_event(answer: str) -> str:
    return sse_data({"response_format": build_response_format(answer)})


def lookup_status_event(stage: str, message: str, **extra: Any) -> str:
    payload = {"stage": stage, "message": message}
    payload.update(extra)
    return sse_data({"lookup_status": payload})


def gamepath_store_lookup_status_event(stored_item: dict[str, Any], game_id: Optional[str]) -> str:
    status = str(stored_item.get("status") or "").strip()
    if status == "duplicate_existing":
        return lookup_status_event(
            "gamepath_not_stored",
            "GamePath 已有相似紀錄，這次沒有新增重複條目。",
            source="gamepath",
            web_search=False,
            fast_path=True,
            entry_id=stored_item.get("id"),
            game_id=game_id,
            reason="duplicate_existing",
        )
    message = (
        "已更新既有 GamePath 相似紀錄，避免新增重複條目。"
        if status == "updated_duplicate"
        else "已濃縮並存入 GamePath，下次同類問題會走本地快取。"
    )
    return lookup_status_event(
        "gamepath_stored",
        message,
        source="gamepath",
        web_search=False,
        fast_path=False,
        entry_id=stored_item.get("id"),
        game_id=game_id,
        reason=status or "stored",
    )


def llama_ready() -> bool:
    try:
        get_json(f"{llama_base_url()}/v1/models", timeout=2)
        return True
    except Exception:
        return False


def local_router_ready() -> bool:
    if not LOCAL_ROUTER_ENABLED:
        return False
    try:
        get_json(local_router_v1_url("models"), timeout=2)
        return True
    except Exception:
        return False


def validate_assets() -> None:
    required = [LLAMA_SERVER] if LLAMA_HF_REPO else [LLAMA_SERVER, MODEL_PATH, MMPROJ_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Missing llama.cpp Vulkan assets:\n" + "\n".join(missing))


def stop_llama_server() -> None:
    global llama_process
    if llama_process and llama_process.poll() is None:
        try:
            llama_process.terminate()
            llama_process.wait(timeout=10)
        except Exception:
            try:
                llama_process.kill()
                llama_process.wait(timeout=5)
            except Exception:
                pass
    llama_process = None


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
    if LLAMA_IMAGE_MIN_TOKENS:
        env["LLAMA_ARG_IMAGE_MIN_TOKENS"] = LLAMA_IMAGE_MIN_TOKENS
    if LLAMA_IMAGE_MAX_TOKENS_SERVER:
        env["LLAMA_ARG_IMAGE_MAX_TOKENS"] = LLAMA_IMAGE_MAX_TOKENS_SERVER
    if LLAMA_CHAT_TEMPLATE_KWARGS:
        env["LLAMA_CHAT_TEMPLATE_KWARGS"] = LLAMA_CHAT_TEMPLATE_KWARGS
    else:
        env.pop("LLAMA_CHAT_TEMPLATE_KWARGS", None)

    args = [str(LLAMA_SERVER)]
    if LLAMA_HF_REPO:
        args.extend(["--hf-repo", LLAMA_HF_REPO])
        if LLAMA_HF_FILE:
            args.extend(["--hf-file", LLAMA_HF_FILE])
    else:
        args.extend(
            [
                "--model",
                str(MODEL_PATH),
                "--mmproj",
                str(MMPROJ_PATH),
            ]
        )

    args.extend(
        [
        "--host",
        LLAMA_HOST,
        "--port",
        str(LLAMA_PORT),
        "--ctx-size",
        LLAMA_CTX_SIZE,
        "--n-gpu-layers",
        LLAMA_GPU_LAYERS,
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
        "--flash-attn",
        LLAMA_FLASH_ATTN,
        ]
    )
    if LLAMA_IMAGE_MIN_TOKENS:
        args.extend(["--image-min-tokens", LLAMA_IMAGE_MIN_TOKENS])
    if LLAMA_IMAGE_MAX_TOKENS_SERVER:
        args.extend(["--image-max-tokens", LLAMA_IMAGE_MAX_TOKENS_SERVER])
    if LLAMA_CHAT_TEMPLATE_KWARGS:
        args.extend(["--chat-template-kwargs", LLAMA_CHAT_TEMPLATE_KWARGS])
    if LLAMA_SKIP_CHAT_PARSING:
        args.append("--skip-chat-parsing")
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


def restart_llama_server(reason: str) -> None:
    print(f"Restarting llama-server after transient failure: {reason}")
    stop_llama_server()
    time.sleep(1)
    start_llama_server()


def is_retryable_llama_error(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
            TimeoutError,
            socket.timeout,
            http.client.RemoteDisconnected,
        ),
    ):
        return True
    if isinstance(exc, urllib.error.HTTPError) and exc.code in {500, 502, 503, 504}:
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, urllib.error.URLError):
        return True
    text = str(exc).lower()
    return "10054" in text or "connection reset" in text or "remote end closed" in text


def bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def normalize_vision_image(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    long_edge = bounded_int_env("LLAMA_VISION_LONG_EDGE", 960, 512, 2560)
    if max(img.size) > long_edge:
        img = img.copy()
        img.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
    return img


def resize_to_long_edge(img: Image.Image, long_edge: int) -> Image.Image:
    img = img.convert("RGB")
    if max(img.size) <= long_edge:
        return img
    resized = img.copy()
    resized.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
    return resized


def encode_image_base64(
    img: Image.Image,
    *,
    image_format: str = "JPEG",
    quality: int = 72,
    save_path: Optional[Path] = None,
) -> tuple[str, str]:
    image_format = (image_format or "JPEG").upper()
    if image_format == "JPG":
        image_format = "JPEG"

    buffered = io.BytesIO()
    if image_format == "JPEG":
        img.convert("RGB").save(
            buffered,
            format="JPEG",
            quality=max(35, min(int(quality), 95)),
            optimize=True,
            subsampling=1,
        )
        mime_type = "image/jpeg"
    elif image_format == "WEBP":
        img.convert("RGB").save(
            buffered,
            format="WEBP",
            quality=max(35, min(int(quality), 95)),
            method=4,
        )
        mime_type = "image/webp"
    else:
        img.save(buffered, format="PNG", optimize=True)
        mime_type = "image/png"

    data = buffered.getvalue()
    if save_path:
        save_path.parent.mkdir(exist_ok=True)
        save_path.write_bytes(data)
    return base64.b64encode(data).decode("ascii"), mime_type


def encode_png_base64(img: Image.Image, *, save_path: Optional[Path] = None) -> str:
    encoded, _ = encode_image_base64(img, image_format="PNG", save_path=save_path)
    return encoded


def decode_image_size(image_base64: str) -> tuple[int, int]:
    raw = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
    image_bytes = base64.b64decode(raw)
    with Image.open(io.BytesIO(image_bytes)) as img:
        return int(img.width), int(img.height)


def overlay_grid_font(width: int, height: int) -> ImageFont.ImageFont:
    cell_edge = min(width / OVERLAY_GRID_COLUMNS, height / OVERLAY_GRID_ROWS)
    size = max(18, min(42, int(cell_edge * 0.18)))
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    for candidate in (
        windir / "Fonts" / "arialbd.ttf",
        windir / "Fonts" / "segoeuib.ttf",
        windir / "Fonts" / "arial.ttf",
    ):
        try:
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_overlay_grid_image(img: Image.Image) -> Image.Image:
    long_edge = bounded_int_env("LLAMA_OVERLAY_GRID_LONG_EDGE", 960, 384, 1280)
    base = resize_to_long_edge(img, long_edge).convert("RGBA")
    width, height = base.size
    draw = ImageDraw.Draw(base, "RGBA")
    line_width = max(2, min(6, int(min(width, height) / 420)))
    line_fill = (0, 210, 255, 210)
    label_font = overlay_grid_font(width, height)
    label_pad = max(5, line_width * 2)

    for index in range(OVERLAY_GRID_COLUMNS + 1):
        x = round(index * width / OVERLAY_GRID_COLUMNS)
        draw.line((x, 0, x, height), fill=line_fill, width=line_width)
    for index in range(OVERLAY_GRID_ROWS + 1):
        y = round(index * height / OVERLAY_GRID_ROWS)
        draw.line((0, y, width, y), fill=line_fill, width=line_width)

    for row in range(OVERLAY_GRID_ROWS):
        for col in range(OVERLAY_GRID_COLUMNS):
            label = f"{chr(ord('A') + col)}{row + 1}"
            left = round(col * width / OVERLAY_GRID_COLUMNS)
            top = round(row * height / OVERLAY_GRID_ROWS)
            bbox = draw.textbbox((0, 0), label, font=label_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_left = left + label_pad
            text_top = top + label_pad
            draw.rectangle(
                (
                    text_left - label_pad // 2,
                    text_top - label_pad // 2,
                    text_left + text_width + label_pad // 2,
                    text_top + text_height + label_pad // 2,
                ),
                fill=(0, 0, 0, 170),
            )
            draw.text((text_left, text_top), label, fill=(255, 255, 255, 255), font=label_font)

    return base.convert("RGB")


def image_to_data_url(image_base64: str) -> str:
    raw = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
    image_bytes = base64.b64decode(raw)
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = normalize_vision_image(img)
        normalized, mime_type = encode_image_base64(
            img,
            image_format=os.environ.get("LLAMA_VISION_IMAGE_FORMAT", "JPEG"),
            quality=bounded_int_env("LLAMA_VISION_IMAGE_QUALITY", 72, 35, 95),
            save_path=LATEST_VISION_INPUT,
        )
    return f"data:{mime_type};base64,{normalized}"


def image_to_overlay_grid_data_url(image_base64: str) -> str:
    raw = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
    image_bytes = base64.b64decode(raw)
    with Image.open(io.BytesIO(image_bytes)) as img:
        gridded = make_overlay_grid_image(img)
        normalized, mime_type = encode_image_base64(
            gridded,
            image_format=os.environ.get("LLAMA_VISION_IMAGE_FORMAT", "JPEG"),
            quality=bounded_int_env("LLAMA_OVERLAY_GRID_IMAGE_QUALITY", 68, 35, 95),
            save_path=LATEST_OVERLAY_GRID_INPUT,
        )
    return f"data:{mime_type};base64,{normalized}"


def message_has_image(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                return True
    return False


def compact_image_data_url(data_url: str, *, long_edge: int, quality: int) -> str:
    raw = data_url.split(",", 1)[1] if "," in data_url else data_url
    image_bytes = base64.b64decode(raw)
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = resize_to_long_edge(img, long_edge)
        normalized, mime_type = encode_image_base64(
            img,
            image_format="JPEG",
            quality=quality,
            save_path=LATEST_VISION_RETRY_INPUT,
        )
    return f"data:{mime_type};base64,{normalized}"


def compact_vision_messages_for_retry(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted = json.loads(json.dumps(messages))
    long_edge = bounded_int_env("LLAMA_RETRY_VISION_LONG_EDGE", 640, 384, 1280)
    quality = bounded_int_env("LLAMA_RETRY_VISION_QUALITY", 50, 35, 85)
    for message in compacted:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            image_url = item.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url")
            if isinstance(url, str) and url.startswith("data:image/"):
                image_url["url"] = compact_image_data_url(url, long_edge=long_edge, quality=quality)
    return compacted


def get_ocr_engine() -> Optional[Any]:
    global ocr_engine
    if os.environ.get("IGPU_ENABLE_OCR", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    if ocr_engine is not None:
        return ocr_engine
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception as exc:
        print(f"OCR unavailable: {exc}")
        return None
    ocr_engine = RapidOCR()
    return ocr_engine


def extract_ocr_text(image_base64: str) -> str:
    engine = get_ocr_engine()
    if engine is None:
        return ""

    raw = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
    image_bytes = base64.b64decode(raw)
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = normalize_vision_image(img)
        encode_png_base64(img, save_path=LATEST_OCR_INPUT)

    try:
        result, _ = engine(str(LATEST_OCR_INPUT))
    except Exception as exc:
        print(f"OCR failed: {exc}")
        return ""

    lines: list[str] = []
    seen: set[str] = set()
    for item in result or []:
        if len(item) < 3:
            continue
        text = str(item[1]).strip()
        try:
            score = float(item[2])
        except (TypeError, ValueError):
            score = 0.0
        if score < 0.55 or len(text) < 2:
            continue
        compact = re.sub(r"\s+", " ", text)
        if compact in seen:
            continue
        seen.add(compact)
        lines.append(compact)
        max_lines = bounded_int_env("IGPU_OCR_CONTEXT_LINES", 10, 1, 30)
        if len(lines) >= max_lines:
            break

    max_chars = bounded_int_env("IGPU_OCR_CONTEXT_CHARS", 500, 0, 1600)
    return "\n".join(lines)[:max_chars]


def format_recent_history(limit: int) -> str:
    recent = history[-limit:]
    lines: list[str] = []
    for item in recent:
        role = "玩家" if item.get("role") == "user" else "助理"
        content = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content[:800]}")
    return "\n".join(lines)


def add_context_to_prompt(prompt: str, limit: int) -> str:
    recent_context = format_recent_history(limit)
    if not recent_context:
        return prompt
    return (
        "以下是最近對話上下文，回答目前問題時必須參考；"
        "如果玩家問「剛剛、前面、上一句、我的代號」之類問題，就從這裡找答案。\n"
        f"{recent_context}\n\n"
        f"目前玩家問題: {prompt}"
    )


def build_messages(
    prompt: str,
    image_base64: Optional[str],
    ocr_text: str = "",
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": get_system_prompt()}]
    if not image_base64:
        messages.append({"role": "user", "content": add_context_to_prompt(prompt, HISTORY_CONTEXT_MESSAGES)})
        return messages

    prompt = add_context_to_prompt(prompt, IMAGE_HISTORY_CONTEXT_MESSAGES)
    vision_prompt = (
        f"{prompt}\n\n"
        "Analyze the screenshot pixels first, then answer in Traditional Chinese. "
        "For normal gameplay requests, reply with one short sentence under 45 Chinese characters. "
        "Only give a longer answer when the player explicitly asks for detailed analysis. "
        "Describe only visible objects, UI, scene layout, and immediate risks. "
        "Do not invent movement, vehicles, enemies, objectives, or actions that are not clearly visible. "
        "For first-person shooter screenshots with ammo, crosshair, minimap, or alive counters, treat the large foreground object as the player's held weapon unless wheels or a full vehicle body are clearly visible. "
        "Do not call an object a motorcycle or vehicle unless wheels, seat, and vehicle body are visible. "
        "If the foreground object is ambiguous, describe its visible shape instead of guessing. "
        "Give at most one immediate next-step suggestion."
    )
    if ocr_text:
        vision_prompt += (
            "\n\n畫面中可能可讀到的字串如下，只能當作理解圖片的內部參考。"
            "不要提到這段參考的存在；除非玩家明確詢問畫面文字，否則不要逐字列出。"
            "回答仍然必須以圖片內容和玩家問題為主：\n"
            f"{ocr_text}"
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_base64)}},
                {"type": "text", "text": vision_prompt},
            ],
        }
    )
    return messages

def append_history(role: str, content: str) -> None:
    history.append({"role": role, "content": content})
    if len(history) > HISTORY_STORE_MESSAGES:
        del history[:-HISTORY_STORE_MESSAGES]


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

    decoder = json.JSONDecoder()
    last_error: Optional[Exception] = None
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except Exception as exc:
        last_error = exc

    for match in re.finditer(r"\{", stripped):
        try:
            parsed, _ = decoder.raw_decode(stripped[match.start() :])
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            last_error = exc

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidate = stripped[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    return json.loads(stripped)


def call_llama_chat_payload(payload: dict[str, Any]) -> str:
    with post_json(f"{llama_base_url()}/v1/chat/completions", payload, 600) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"].get("content") or ""


def call_local_router_once(
    messages: list[dict[str, Any]],
    max_tokens: int = 96,
    timeout_seconds: Optional[int] = None,
    temperature: float = 0.1,
) -> str:
    if not LOCAL_ROUTER_ENABLED:
        raise RuntimeError("Local router is disabled.")
    payload = {
        "model": LOCAL_ROUTER_MODEL,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 20,
        "cache_prompt": True,
    }
    with post_json(
        local_router_v1_url("chat/completions"),
        payload,
        timeout_seconds or LOCAL_ROUTER_TIMEOUT_SECONDS,
    ) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"].get("content") or ""


def call_llama_once_no_recovery(messages: list[dict[str, Any]], max_tokens: int = 1536) -> str:
    payload = {
        "model": MODEL_ALIAS,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 20,
        "repeat_penalty": 1.0,
    }
    content = call_llama_chat_payload(payload)
    if content.strip():
        return content

    retry_payload = dict(payload)
    retry_payload.update(
        {
            "cache_prompt": True,
            "temperature": max(float(payload.get("temperature") or 0.2), 0.8),
            "min_p": 0.05,
        }
    )
    return call_llama_chat_payload(retry_payload)


def call_llama_once(messages: list[dict[str, Any]], max_tokens: int = 1536) -> str:
    attempts: list[tuple[str, list[dict[str, Any]], int]] = [("primary", messages, max_tokens)]
    if message_has_image(messages):
        retry_tokens = min(max_tokens, bounded_int_env("LLAMA_RETRY_IMAGE_RESPONSE_TOKENS", 64, 16, 512))
        attempts.append(("recovered-compact-image", compact_vision_messages_for_retry(messages), retry_tokens))
    else:
        attempts.append(("recovered", messages, max_tokens))

    last_exc: Optional[BaseException] = None
    for index, (label, attempt_messages, attempt_max_tokens) in enumerate(attempts):
        try:
            if index > 0:
                print(f"Retrying llama request with profile: {label}")
            return call_llama_once_no_recovery(attempt_messages, attempt_max_tokens)
        except Exception as exc:
            last_exc = exc
            if index >= len(attempts) - 1 or not is_retryable_llama_error(exc):
                raise
            try:
                restart_llama_server(str(exc))
            except Exception as restart_exc:
                raise RuntimeError(
                    f"llama-server connection failed and restart did not recover it: {restart_exc}"
                ) from exc

    if last_exc:
        raise last_exc
    return ""


def clean_short_answer(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    for _ in range(2):
        text = re.sub(r"([\u4e00-\u9fff]{2,6})\1+", r"\1", text)
        text = re.sub(r"\b([A-Za-z]{2,})\1+\b", r"\1", text)
    text = re.sub(r"([\u4e00-\u9fff])\1+", r"\1", text)
    text = re.sub(r"([。！？!?，,、])\1+", r"\1", text)
    text = text.replace("，、", "，")
    text = re.sub(r"([！？!?])。", r"\1", text)
    text = re.sub(r"(`)\1+", r"\1", text)
    return text


def clean_vision_answer(text: str) -> str:
    text = clean_short_answer(text)
    text = re.sub(r"\bOCR\b", "畫面文字", text, flags=re.IGNORECASE)
    for source, target in {
        "文字辨識": "畫面文字",
        "可讀文字摘錄": "畫面文字",
        "隱藏輔助上下文": "畫面內容",
        "輔助上下文": "畫面內容",
        "內部參考": "畫面內容",
    }.items():
        text = text.replace(source, target)
    return text


def sanitize_task_analysis(raw: dict[str, Any], message: str, game_id: Optional[str]) -> dict[str, Any]:
    def clean(value: Any, default: str = "") -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text[:400] if text else default

    def clean_list(value: Any, limit: int = 5) -> list[str]:
        if isinstance(value, str):
            parts = re.split(r"[\n;；]+", value)
        elif isinstance(value, list):
            parts = value
        else:
            parts = []
        items: list[str] = []
        for item in parts:
            text = clean(item)
            if text and text not in items:
                items.append(text[:160])
            if len(items) >= limit:
                break
        return items

    fallback_title = clean(message, "調查目前取得的物品")
    title = clean(raw.get("title"), fallback_title)[:80]
    item_name = clean(raw.get("item_name") or raw.get("item"), "")
    objective = clean(raw.get("objective"), title)
    why = clean(raw.get("why") or raw.get("reason"), "")
    next_steps = clean_list(raw.get("next_steps"), 4)
    if not next_steps and objective:
        next_steps = [objective]
    tags = clean_list(raw.get("tags"), 6)
    if item_name and item_name not in tags:
        tags.insert(0, item_name[:40])
    category = clean(raw.get("category"), "unknown").lower()
    if category not in {"item", "quest", "indicator", "resource", "location", "unknown"}:
        category = "unknown"
    try:
        confidence = float(raw.get("confidence", 0.45))
    except Exception:
        confidence = 0.45
    confidence = max(0.0, min(confidence, 1.0))
    summary = clean(raw.get("summary"), "")
    if not summary:
        summary = clean("；".join(part for part in [item_name, objective, why] if part), title)

    return {
        "title": title,
        "category": category,
        "item_name": item_name,
        "objective": objective,
        "why": why,
        "next_steps": next_steps,
        "tags": tags[:6],
        "confidence": confidence,
        "summary": summary,
        "game_id": normalize_game_id(game_id) or "global",
    }


def fallback_task_analysis(message: str, game_id: Optional[str], error: Optional[str] = None) -> dict[str, Any]:
    cleaned = re.sub(r"\s+", " ", (message or "").strip())
    result = sanitize_task_analysis(
        {
            "title": cleaned[:80] or "調查目前取得的物品",
            "category": "unknown",
            "objective": "先確認這個物品或指標的用途，再決定下一步",
            "why": "模型沒有穩定產生結構化任務，所以先保留為待查目標。",
            "next_steps": ["查看物品描述", "詢問這個物品能用在哪", "找到相關 NPC、配方或任務提示"],
            "tags": ["待查"],
            "confidence": 0.25,
            "summary": cleaned or "從目前畫面建立待查任務。",
        },
        message,
        game_id,
    )
    if error:
        result["warning"] = error[:300]
    return result


def build_task_analysis_messages(
    message: str,
    image_base64: Optional[str],
    ocr_text: str,
    rag_context: str,
    source_title: Optional[str],
) -> list[dict[str, Any]]:
    system = (
        "You are a game task logger. Convert the player's screenshot and note into one actionable task record. "
        "Return one compact JSON object only, no Markdown. Use Traditional Chinese. "
        "Do not invent exact game facts if they are not visible or in local context; mark uncertainty in why and lower confidence. "
        "JSON keys: title, category, item_name, objective, why, next_steps, tags, confidence, summary. "
        "category must be one of item, quest, indicator, resource, location, unknown. "
        "next_steps and tags must be arrays of short strings. confidence is 0.0 to 1.0. "
        "Focus on obtained items, visible item descriptions, quest/indicator text, what it may unlock, and what the player should check next."
    )
    user_text = (
        f"Player note: {message or '根據目前截圖建立任務目標。'}\n"
        f"Screenshot source title: {source_title or 'unknown'}\n\n"
        "Create a task goal for the player. If the screenshot shows an item description but the use is unclear, "
        "make the objective about discovering its use and list concrete checks such as recipe, NPC, quest, upgrade, or map marker."
    )
    if ocr_text:
        user_text += "\n\nReadable screenshot text:\n" + ocr_text[:2000]
    if rag_context:
        user_text += "\n\nLocal guide/player memory context:\n" + rag_context[:2500]

    if not image_base64:
        return [{"role": "system", "content": system}, {"role": "user", "content": user_text}]

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


def analyze_task_sync(request: TaskAnalyzeRequest) -> dict[str, Any]:
    message = (request.message or "").strip()
    image_base64 = request.image_base64
    if image_base64 and "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    ocr_text = ""
    if image_base64 and ENABLE_OCR_CONTEXT:
        try:
            ocr_text = extract_ocr_text(image_base64)
        except Exception as exc:
            print(f"Task OCR failed: {exc}")

    query = " ".join(part for part in [message, ocr_text] if part).strip()
    guide_results = search_guides_sync(query, request.game_id, 4) if query else []
    memory_results = search_memory_sync(query, request.game_id, ["task", "state", "note"], 4) if query else []
    rag_context = format_rag_context(guide_results, memory_results, bool(guide_results))
    messages = build_task_analysis_messages(
        message,
        image_base64,
        ocr_text,
        rag_context,
        request.source_title,
    )

    try:
        task_tokens = int(os.environ.get("LLAMA_TASK_RESPONSE_TOKENS", "360"))
        if CHAT_BACKEND == "hermes" and HERMES_USE_CONFIG_MODEL:
            output = call_hermes_messages(
                messages,
                image_file=LATEST_VISION_INPUT if image_base64 else None,
                max_tokens=task_tokens,
            )
        else:
            output = call_llama_once(messages, task_tokens)
        raw = extract_json_object(output)
        result = sanitize_task_analysis(raw, message, request.game_id)
        result["raw_response"] = output[:1200]
    except Exception as exc:
        result = fallback_task_analysis(message, request.game_id, str(exc))

    memory_text = (
        f"任務目標：{result['title']}；"
        f"物品/指標：{result.get('item_name') or '未確認'}；"
        f"目的：{result.get('objective') or result['summary']}；"
        f"下一步：{'、'.join(result.get('next_steps') or [])}"
    )
    try:
        result["memory_item"] = add_memory_sync(
            memory_text,
            request.game_id,
            "task",
            ",".join(result.get("tags") or []),
            4,
        )
    except Exception as exc:
        result["memory_warning"] = str(exc)
    return result


def get_stt_model() -> Any:
    global stt_model
    if stt_model is not None:
        return stt_model

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise RuntimeError(
            "Voice transcription dependency is missing. Install it with: "
            ".venv\\Scripts\\python.exe -m pip install faster-whisper"
        ) from exc

    model_name = os.environ.get("IGPU_STT_MODEL", "base").strip() or "base"
    device = os.environ.get("IGPU_STT_DEVICE", "cpu").strip() or "cpu"
    compute_type = os.environ.get("IGPU_STT_COMPUTE_TYPE", "int8").strip() or "int8"
    cache_dir = ASSET_ROOT / "models" / "faster-whisper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stt_model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=str(cache_dir),
    )
    return stt_model


def transcribe_audio_path(audio_path: Path) -> dict[str, Any]:
    model = get_stt_model()
    language = os.environ.get("IGPU_STT_LANGUAGE", "zh").strip()
    transcribe_kwargs: dict[str, Any] = {
        "beam_size": 1,
        "temperature": 0.0,
        "condition_on_previous_text": False,
    }
    if language and language.lower() not in {"auto", "none"}:
        transcribe_kwargs["language"] = language

    segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    return {
        "text": clean_short_answer(text),
        "language": getattr(info, "language", language or None),
        "duration": getattr(info, "duration", None),
        "model": os.environ.get("IGPU_STT_MODEL", "base").strip() or "base",
    }


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
GAMEPATH_STORE_INTENT_RE = re.compile(
    r"(guide|walkthrough|tips|item|usage|quest|boss|npc|route|map|weakness|material|location|攻略|教學|提示|用途|用法|物品|道具|任務|素材|材料|路線|地圖|弱點|打法|怎麼過|怎麼用|能做什麼|做什麼|用來幹嘛|能幹嘛|過關|那關|關卡)",
    re.IGNORECASE,
)
GAMEPATH_UNCERTAIN_RE = re.compile(
    r"(timeout|timed out|failed|error|不知道|不確定|不清楚|沒有找到|沒找到|無法確認)",
    re.IGNORECASE,
)
GAMEPATH_DISPUTE_RE = re.compile(
    r"(沒有看到|沒看到|沒有發現|沒發現|找不到|沒有你說|不是你說|不在這|路不對|位置不對|"
    r"沒有這個|沒這個|沒有那個|你講的.*沒有|你說的.*沒有|不是這樣|不對|錯了|錯誤|"
    r"版本不一樣|版本不同|not there|can't find|cannot find|not found|wrong|incorrect)",
    re.IGNORECASE,
)
GAMEPATH_UI_SKIP_RE = re.compile(
    r"(gamepath|game path|game search|webview|browser|瀏覽器|搜尋視窗|攻略視窗|path 視窗|path視窗|"
    r"任務視窗|task window|語音|麥克風|mic|voice|透明|opacity|虛擬游標|游標|cursor|"
    r"內容保護|保護內容|截圖保護|hotkey|快捷鍵|重啟|啟動程式|關閉服務|退出|打包|安裝檔|"
    r"llama|hermes|模型|backend|gpu|igpu|dgpu)",
    re.IGNORECASE,
)
GAMEPATH_SAVE_ONLY_RE = re.compile(
    r"(存下來|保存|存起來|記錄下來|刪掉|刪除|打開|開啟|關閉|刷新|有幾筆|多少資料|資料庫|狀態|機制)",
    re.IGNORECASE,
)
GAMEPATH_NEGATED_GUIDE_RE = re.compile(
    r"(不用|不要|先不用|不需要|別).{0,10}(查|搜尋|搜索|找|攻略|網路|上網|web|guide)",
    re.IGNORECASE,
)
GAMEPATH_WEB_INTENT_RE = re.compile(
    r"(上網|網路|web|tavily|最新|新版本|patch|hotfix|版本差異|更新|社群|speedrun|meta|"
    r"攻略庫沒有|本地沒有|交給\s*Hermes\s*查)",
    re.IGNORECASE,
)
GAMEPATH_VERSION_COMPARE_RE = re.compile(
    r"((?:\b\d+(?:\.\d+){1,3}\b).{0,24}(?:\u8ddf|\u548c|vs|/|,).{0,24}"
    r"(?:\b\d+(?:\.\d+){1,3}\b).{0,28}(?:\u5dee\u7570|\u4e0d\u540c|\u6539|\u6bd4\u8f03|\u67e5))|"
    r"(\u7248\u672c.{0,16}(?:\u5dee\u7570|\u4e0d\u540c|\u6bd4\u8f03))",
    re.IGNORECASE,
)
LOCAL_ROUTER_GAMEPATH_CANDIDATE_RE = re.compile(
    r"(game|guide|walkthrough|tips|quest|item|boss|map|route|where|how|stuck|puzzle|"
    r"name|character|enemy|zombie|"
    r"\u653b\u7565|\u6559\u5b78|\u63d0\u793a|\u4efb\u52d9|\u7269\u54c1|\u9053\u5177|"
    r"\u95dc\u5361|\u5361\u95dc|\u8def\u7dda|\u5730\u5716|\u540d\u5b57|\u540d\u7a31|\u89d2\u8272|\u654c\u4eba|\u602a\u7269|\u6bad\u5c4d|\u50f5\u5c4d|"
    r"\u9806\u5e8f|\u8f49\u76e4|\u95a5\u9580|\u6a5f\u95dc|\u89f8\u767c|\u8ffd|\u64cb\u8def|\u786c\u6253|\u6253\u4e0d\u52d5|\u8ff7\u8def|\u6697\u9580|"
    r"\u9418\u8072|\u9ed1\u5e3d|\u8001\u4eba|\u4fdd\u96aa\u7d72|\u9470\u5319|\u7d20\u6750|\u652f\u7dda|"
    r"\u600e|\u54ea|\u627e|\u7528)",
    re.IGNORECASE,
)
GAMEPATH_TRUST_STATES = {"unverified", "verified", "disputed", "needs_review", "deprecated"}
SPOILER_RANKS = {"none": 0, "low": 1, "medium": 2, "high": 3, "full": 4}
GAMEPATH_DIRECT_MAX_CHARS = 900
GAMEPATH_CONTEXT_MAX_CHARS = 1800
GAMEPATH_PASSAGE_MAX_CHARS = 900
GAMEPATH_ENTITY_TYPES = {
    "item",
    "quest",
    "boss",
    "npc",
    "map",
    "route",
    "puzzle",
    "mechanic",
    "material",
    "location",
    "character",
    "enemy",
}
GAMEPATH_ENTITY_TYPE_ALIASES = {
    "items": "item",
    "usage": "item",
    "weapon": "item",
    "equipment": "item",
    "resource": "material",
    "resources": "material",
    "materials": "material",
    "walkthrough": "route",
    "path": "route",
    "area": "location",
    "zone": "location",
    "monster": "enemy",
    "zombie": "enemy",
}
GAMEPATH_SOURCE_TAGS = {
    "auto",
    "hermes",
    "guide",
    "vision",
    "web",
    "manual",
    "agent",
    "tavily",
    "local",
    "llama",
    "qwen",
}
GAMEPATH_GENERIC_TERMS = {
    "guide",
    "walkthrough",
    "tips",
    "tip",
    "item",
    "usage",
    "quest",
    "boss",
    "npc",
    "route",
    "map",
    "weakness",
    "material",
    "location",
    "攻略",
    "教學",
    "提示",
    "用途",
    "用法",
    "物品",
    "道具",
    "任務",
    "素材",
    "材料",
    "路線",
    "地圖",
    "弱點",
    "打法",
    "怎麼",
    "怎麼過",
    "怎麼用",
    "哪裡",
    "在哪",
    "那關",
    "關卡",
    "過關",
}
GAMEPATH_GENERIC_TERMS.update(
    {
        "\u653b\u7565",
        "\u6559\u5b78",
        "\u63d0\u793a",
        "\u7528\u9014",
        "\u7528\u6cd5",
        "\u7269\u54c1",
        "\u9053\u5177",
        "\u4efb\u52d9",
        "\u7d20\u6750",
        "\u6750\u6599",
        "\u8def\u7dda",
        "\u5730\u5716",
        "\u5f31\u9ede",
        "\u6253\u6cd5",
        "\u600e\u9ebc",
        "\u600e\u9ebc\u904e",
        "\u600e\u9ebc\u7528",
        "\u54ea\u88e1",
        "\u5728\u54ea",
        "\u90a3\u95dc",
        "\u95dc\u5361",
        "\u904e\u95dc",
        "\u4e0b\u4e00\u6b65",
        "\u63a5\u4e0b\u4f86",
        "\u61c9\u8a72",
        "\u8981\u505a\u4ec0\u9ebc",
        "\u8981\u53bb\u54ea",
        "\u54ea\u500b",
        "\u54ea\u96bb",
        "\u54ea\u908a",
        "\u4ec0\u9ebc",
    }
)

MEMORY_TASK_CONTEXT_RE = re.compile(
    r"(task|objective|goal|todo|quest log|next step|current goal|"
    r"\u4efb\u52d9|\u76ee\u6a19|\u5f85\u8fa6|\u4e0b\u4e00\u6b65|\u76ee\u524d\u8981\u505a|\u9032\u5ea6|"
    r"\u6211\u62ff\u5230|\u6211\u53d6\u5f97|\u6211\u7372\u5f97|\u8ffd\u8e64)",
    re.IGNORECASE,
)
MEMORY_NOTE_CONTEXT_RE = re.compile(
    r"(memory|remember|note|notes|what did I say|"
    r"\u8a18\u61b6|\u7b46\u8a18|\u4f60\u8a18\u5f97|\u6211\u4e4b\u524d|\u524d\u9762|\u525b\u525b)",
    re.IGNORECASE,
)
OVERLAY_INTENT_RE = re.compile(
    r"(圈|圈出|圈選|框出|標記|標出|指引|導引|導航|指路|往哪|往哪走|哪邊|路線|路標|目標|目的地|箭頭|提醒|危險|門在哪|在哪裡|where|mark|circle|arrow|route|path|guide|navigate|target|objective|destination)",
    re.IGNORECASE,
)
VISUAL_SCENE_INTENT_RE = re.compile(
    r"(物件|東西|看到什麼|看到了什麼|畫面.*有什麼|截圖.*有什麼|有什麼|"
    r"角色|人物|敵人|怪物|道具|裝備|武器|車|門|箱子|地圖|場景|環境|畫面內容|"
    r"object|objects|thing|things|what.*see|what.*image|what.*screen|describe.*image)",
    re.IGNORECASE,
)
MEMORY_ADD_RE = re.compile(
    r"(記住|幫我記|幫我記住|記一下|記在|remember|note this|save this)",
    re.IGNORECASE,
)
FACT_KEY_PATTERN = r"(測試代號|代號|名字|暱稱|id|ID|帳號|角色|伺服器|職業|偏好|目標|進度|任務)"
IMPLICIT_MEMORY_FACT_RE = re.compile(
    rf"(?:我的|我目前的|我現在的|目前的|現在的)?\s*"
    rf"(?P<key>{FACT_KEY_PATTERN})\s*(?:是|叫|=|:|：)\s*"
    r"(?P<value>[A-Za-z0-9_.#\-_\u4e00-\u9fff ]{1,80})"
)
FACT_LOOKUP_RE = re.compile(
    rf"(?P<key>{FACT_KEY_PATTERN}).{{0,12}}(?:是什麼|多少|哪個|叫什麼|\?)",
    re.IGNORECASE,
)
USER_FACT_CONTEXT_RE = re.compile(
    r"(我的|我目前的|我現在的|玩家|使用者|我叫|我說的|剛剛.*我|前面.*我|上一句.*我|你記得.*我|記得.*我)",
    re.IGNORECASE,
)
GAME_ENTITY_FACT_CONTEXT_RE = re.compile(
    r"(這個|這兩個|那個|那些|女殭屍|殭屍|僵屍|敵人|怪物|boss|npc|角色|人物|道具|物品|地點|關卡|遊戲)",
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


def has_cjk_text(text: Any) -> bool:
    return bool(CJK_RE.search(str(text or "")))


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
    games.update(gamepath_games_sync())
    games.update(load_game_profiles_sync().get("profiles", {}).keys())
    return sorted(games)


def game_display_name(game_id: str) -> str:
    clean = str(game_id or "").strip()
    if not clean:
        return "Unknown Game"
    return re.sub(r"[_-]+", " ", clean).strip().title()


def normalize_list(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = []
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = re.sub(r"\s+", " ", str(item or "").strip())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text[:260])
    return result


def normalize_game_profiles(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    raw_profiles = raw.get("profiles") if isinstance(raw.get("profiles"), dict) else {}
    profiles: dict[str, dict[str, Any]] = {}
    for raw_game_id, value in raw_profiles.items():
        game_id = normalize_game_id(str(raw_game_id))
        if not game_id or not isinstance(value, dict):
            continue
        profiles[game_id] = {
            "name": str(value.get("name") or game_display_name(game_id)).strip()[:120],
            "aliases": normalize_list(value.get("aliases")),
            "processes": [item.lower() for item in normalize_list(value.get("processes"))],
            "window_titles": normalize_list(value.get("window_titles")),
            "process_paths": normalize_list(value.get("process_paths")),
            "learned": bool(value.get("learned")),
            "updated_at": str(value.get("updated_at") or ""),
        }
    mappings_raw = raw.get("process_mappings") if isinstance(raw.get("process_mappings"), dict) else {}
    process_mappings: dict[str, str] = {}
    for process_name, mapped_game_id in mappings_raw.items():
        game_id = normalize_game_id(str(mapped_game_id))
        process_key = Path(str(process_name or "").strip()).name.lower()
        if game_id and process_key:
            process_mappings[process_key] = game_id
    return {"profiles": profiles, "process_mappings": process_mappings}


def load_game_profiles_sync() -> dict[str, Any]:
    try:
        raw = json.loads(GAME_PROFILES_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raw = {}
    except Exception as exc:
        print(f"Game profile load failed: {exc}")
        raw = {}
    return normalize_game_profiles(raw)


def save_game_profiles_sync(data: dict[str, Any]) -> None:
    normalized = normalize_game_profiles(data)
    GAME_PROFILES_FILE.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def add_unique_text(items: list[str], value: Optional[str], *, lower: bool = False, max_items: int = 20) -> None:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return
    if lower:
        text = text.lower()
    seen = {item.lower() for item in items}
    if text.lower() not in seen:
        items.append(text[:260])
    del items[max_items:]


def title_matches(title: str, needles: list[str]) -> Optional[str]:
    title_lower = title.lower()
    for needle in needles:
        clean = str(needle or "").strip()
        if len(clean) >= 3 and clean.lower() in title_lower:
            return clean
    return None


def path_matches(process_path: str, needles: list[str]) -> Optional[str]:
    path_lower = process_path.lower()
    for needle in needles:
        clean = str(needle or "").strip()
        if len(clean) >= 4 and clean.lower() in path_lower:
            return clean
    return None


def game_platform_marker(process_path: str) -> Optional[str]:
    path_lower = process_path.lower().replace("/", "\\")
    markers = [
        "\\steamapps\\common\\",
        "\\xboxgames\\",
        "\\epic games\\",
        "\\gog galaxy\\games\\",
        "\\riot games\\",
        "\\battle.net\\",
        "\\ubisoft\\",
        "\\ea games\\",
    ]
    return next((marker for marker in markers if marker in path_lower), None)


def candidate_from_profile(
    game_id: str,
    profile: dict[str, Any],
    *,
    confidence: float,
    source: str,
    match: str,
) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "name": profile.get("name") or game_display_name(game_id),
        "confidence": confidence,
        "source": source,
        "match": match,
        "learned": bool(profile.get("learned")),
    }


def resolve_game_from_window(window: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not window:
        if last_active_game_detection:
            stale = dict(last_active_game_detection)
            stale["stale"] = True
            stale["source"] = f"{stale.get('source', 'unknown')}_cached"
            return stale
        return {"active": False, "game_id": None, "name": "", "confidence": 0.0, "source": "no_window"}

    process_name = Path(str(window.get("process_name") or "")).name.lower()
    process_path = str(window.get("process_path") or "")
    title = str(window.get("title") or "")
    profiles_data = load_game_profiles_sync()
    profiles = dict(profiles_data.get("profiles") or {})

    for guide_game in list_guide_games_sync():
        profiles.setdefault(
            guide_game,
            {
                "name": game_display_name(guide_game),
                "aliases": [game_display_name(guide_game), guide_game],
                "processes": [],
                "window_titles": [],
                "process_paths": [],
                "learned": False,
            },
        )

    candidates: list[dict[str, Any]] = []
    mapped_game_id = (profiles_data.get("process_mappings") or {}).get(process_name)
    if mapped_game_id and mapped_game_id in profiles:
        candidates.append(
            candidate_from_profile(
                mapped_game_id,
                profiles[mapped_game_id],
                confidence=0.98,
                source="learned_process",
                match=process_name,
            )
        )

    for game_id, profile in profiles.items():
        processes = [Path(item).name.lower() for item in profile.get("processes") or []]
        if process_name and process_name in processes:
            candidates.append(
                candidate_from_profile(game_id, profile, confidence=0.96, source="process_exact", match=process_name)
            )
        path_match = path_matches(process_path, profile.get("process_paths") or [])
        if path_match:
            candidates.append(
                candidate_from_profile(game_id, profile, confidence=0.88, source="path_match", match=path_match)
            )
        title_match = title_matches(title, profile.get("window_titles") or [])
        if title_match:
            candidates.append(
                candidate_from_profile(game_id, profile, confidence=0.84, source="title_match", match=title_match)
            )
        alias_match = title_matches(title, [profile.get("name") or "", *(profile.get("aliases") or []), game_id])
        if alias_match:
            candidates.append(
                candidate_from_profile(game_id, profile, confidence=0.68, source="alias_title", match=alias_match)
            )

    if not candidates and title:
        fallback_id = normalize_game_id(title)
        if fallback_id:
            platform_marker = game_platform_marker(process_path)
            candidates.append(
                {
                    "game_id": fallback_id,
                    "name": title[:120],
                    "confidence": 0.58 if platform_marker else 0.42,
                    "source": "game_path_guess" if platform_marker else "window_title_guess",
                    "match": platform_marker or title[:120],
                    "learned": False,
                }
            )

    best = max(candidates, key=lambda item: float(item.get("confidence") or 0.0), default=None)
    if not best:
        best = {"game_id": None, "name": "", "confidence": 0.0, "source": "unknown", "match": ""}

    return {
        "active": bool(best.get("game_id")),
        **best,
        "process_name": process_name,
        "process_path": process_path,
        "window_title": title,
        "window": {
            "title": title,
            "process_name": process_name,
            "process_path": process_path,
            "width": int(window.get("width") or 0),
            "height": int(window.get("height") or 0),
        },
        "stale": False,
    }


def detect_active_game_sync() -> dict[str, Any]:
    global last_active_game_window, last_active_game_detection
    window = get_foreground_window_info()
    if window:
        last_active_game_window = window
    elif last_active_game_window:
        window = last_active_game_window
    detection = resolve_game_from_window(window)
    if detection.get("active"):
        last_active_game_detection = detection
    return detection


def learn_active_game_sync(request: GameProfileLearnRequest) -> dict[str, Any]:
    game_id = normalize_game_id(request.game_id)
    if not game_id:
        raise ValueError("game_id is required.")

    window = get_foreground_window_info() or last_active_game_window or {}
    process_name = Path(str(request.process_name or window.get("process_name") or "")).name.lower()
    process_path = str(request.process_path or window.get("process_path") or "").strip()
    window_title = str(request.window_title or window.get("title") or "").strip()

    data = load_game_profiles_sync()
    profiles = data.setdefault("profiles", {})
    profile = profiles.setdefault(
        game_id,
        {
            "name": request.name or game_display_name(game_id),
            "aliases": [],
            "processes": [],
            "window_titles": [],
            "process_paths": [],
            "learned": True,
            "updated_at": "",
        },
    )
    profile["name"] = str(request.name or profile.get("name") or game_display_name(game_id)).strip()[:120]
    profile["learned"] = True
    profile["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    add_unique_text(profile.setdefault("aliases", []), game_display_name(game_id))
    add_unique_text(profile.setdefault("processes", []), process_name, lower=True)
    if window_title and window_title.lower() not in {"game companion", "overlay-chat"}:
        add_unique_text(profile.setdefault("window_titles", []), window_title)
    if process_path:
        add_unique_text(profile.setdefault("process_paths", []), process_path)
    if process_name:
        data.setdefault("process_mappings", {})[process_name] = game_id

    save_game_profiles_sync(data)
    detection = resolve_game_from_window(window if window else None)
    return {"profile": normalize_game_profiles(data)["profiles"][game_id], "detection": detection}


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


def normalize_tags_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = re.split(r"[,#;\n]+", value)
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]
    tags: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        tag = re.sub(r"\s+", " ", str(item or "").strip())
        tag = tag.lstrip("#").strip()
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        tags.append(tag[:80])
    return tags[:12]


def tags_to_text(value: Any) -> str:
    return ",".join(normalize_tags_value(value))


def normalize_spoiler_level(value: str) -> str:
    clean = re.sub(r"[^A-Za-z]+", "", str(value or "low").strip().lower()) or "low"
    if clean in {"no", "none", "safe"}:
        return "none"
    if clean in {"minimal", "light"}:
        return "low"
    if clean in {"med", "mid"}:
        return "medium"
    if clean in {"all", "solution"}:
        return "full"
    return clean if clean in SPOILER_RANKS else "low"


def spoiler_rank(value: str) -> int:
    return SPOILER_RANKS.get(normalize_spoiler_level(value), 1)


def clean_gamepath_metadata_value(value: Any, max_len: int = 80) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = text.strip(" \t\r\n:：,，.。;；'\"「」『』()（）[]【】")
    return text[:max_len] if text else ""


def normalize_gamepath_entity_type(value: Any) -> str:
    clean = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "_", str(value or "").strip().lower()).strip("_-")
    if not clean:
        return ""
    alias = GAMEPATH_ENTITY_TYPE_ALIASES.get(clean, clean)
    chinese_aliases = {
        "\u7269\u54c1": "item",
        "\u9053\u5177": "item",
        "\u88dd\u5099": "item",
        "\u6b66\u5668": "item",
        "\u7d20\u6750": "material",
        "\u6750\u6599": "material",
        "\u4efb\u52d9": "quest",
        "\u652f\u7dda": "quest",
        "\u8001\u95c6": "boss",
        "\u9996\u9818": "boss",
        "\u5730\u5716": "map",
        "\u8def\u7dda": "route",
        "\u89e3\u8b0e": "puzzle",
        "\u6a5f\u5236": "mechanic",
        "\u4f4d\u7f6e": "location",
        "\u5340\u57df": "location",
        "\u5834\u666f": "location",
        "\u89d2\u8272": "character",
        "\u6575\u4eba": "enemy",
        "\u602a\u7269": "enemy",
    }
    alias = chinese_aliases.get(alias, alias)
    return alias if alias in GAMEPATH_ENTITY_TYPES else ""


def infer_gamepath_entity_type(*texts: str, tags: Any = None) -> str:
    for tag in normalize_tags_value(tags):
        tag_type = normalize_gamepath_entity_type(tag)
        if tag_type:
            return tag_type
    text = "\n".join(str(item or "") for item in texts).lower()
    checks = [
        ("boss", r"(boss|weakness|\u6253\u6cd5|\u600e\u9ebc\u6253|\u5f31\u9ede|\u8001\u95c6|\u9996\u9818)"),
        ("item", r"(item|usage|use|weapon|equipment|\u7269\u54c1|\u9053\u5177|\u7528\u9014|\u7528\u6cd5|\u6b66\u5668|\u88dd\u5099)"),
        ("material", r"(material|resource|craft|recipe|\u7d20\u6750|\u6750\u6599|\u914d\u65b9|\u88fd\u4f5c)"),
        ("quest", r"(quest|objective|\u4efb\u52d9|\u652f\u7dda|\u76ee\u6a19)"),
        ("mechanic", r"(mechanic|system|\u6a5f\u5236|\u7cfb\u7d71|\u64cd\u4f5c|\u4e92\u52d5|\u555f\u52d5|\u89f8\u767c)"),
        ("route", r"(route|path|where|walkthrough|\u8def\u7dda|\u6307\u8def|\u5f80\u54ea|\u600e\u9ebc\u8d70)"),
        ("map", r"(map|\u5730\u5716|\u6a19\u8a18|\u5730\u9ede)"),
        ("puzzle", r"(puzzle|solve|\u89e3\u8b0e|\u8b0e\u984c|\u6a5f\u95dc)"),
        ("npc", r"(npc|\u5546\u4eba|\u5c0d\u8a71)"),
        ("character", r"(character|\u89d2\u8272|\u8eab\u4efd|\u540d\u5b57|\u540d\u7a31)"),
        ("enemy", r"(enemy|monster|zombie|\u6575\u4eba|\u602a\u7269|\u50f5\u5c4d|\u5973\u58eb\u5c4d)"),
        ("location", r"(area|zone|location|chapter|stage|\u5340\u57df|\u5834\u666f|\u95dc\u5361|\u7ae0\u7bc0)"),
    ]
    for entity_type, pattern in checks:
        if re.search(pattern, text, re.IGNORECASE):
            return entity_type
    return ""


def infer_gamepath_version(*texts: str) -> str:
    text = " ".join(str(item or "") for item in texts)
    match = re.search(
        r"(?i)(?:version|patch|ver\.?|v)\s*[:#]?\s*([0-9]+(?:\.[0-9A-Za-z_-]+){0,3})",
        text,
    )
    if match:
        return clean_gamepath_metadata_value(match.group(1), 40)
    match = re.search(r"\b([0-9]+\.[0-9]+(?:\.[0-9A-Za-z_-]+){0,2})\b", text)
    return clean_gamepath_metadata_value(match.group(1), 40) if match else ""


def infer_gamepath_area(*texts: str, tags: Any = None) -> str:
    for tag in normalize_tags_value(tags):
        if re.search(r"(area|zone|chapter|stage|map|\u5340\u57df|\u5834\u666f|\u5730\u5716|\u95dc\u5361|\u7ae0)", tag, re.IGNORECASE):
            return clean_gamepath_metadata_value(tag, 80)
    text = " ".join(str(item or "") for item in texts)
    patterns = [
        r"(?i)(?:area|zone|chapter|stage|map)\s*[:：]?\s*([A-Za-z0-9_\- \u4e00-\u9fff]{2,40})",
        r"([\u4e00-\u9fffA-Za-z0-9_\- ]{1,24}(?:\u5340\u57df|\u5834\u666f|\u5730\u5716|\u95dc\u5361|\u7ae0|\u5c64))",
        r"([\u4e00-\u9fffA-Za-z0-9_\- ]{0,24}?(?:\u5165\u53e3|\u78bc\u982d|\u5eda\u623f|\u91ab\u9662|\u9152\u7a96|\u4e2d\u5ead|\u5927\u5ef3|\u8d70\u5eca|\u79ae\u62dc\u5802|\u8cc7\u6599\u5ba4|\u934b\u7210\u623f|\u9418\u5854))",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            area_candidate = re.sub(
                r"^(?:\u6211\u5728|\u76ee\u524d\u5728|\u73fe\u5728\u5728|\u5728|\u53bb|\u5230)",
                "",
                str(match.group(1) or "").strip(),
            )
            return clean_gamepath_metadata_value(area_candidate, 80)
    return ""


def infer_gamepath_entity_name(question: str, answer_summary: str = "", tags: Any = None) -> str:
    tag_candidates = [
        tag
        for tag in normalize_tags_value(tags)
        if tag.lower() not in GAMEPATH_GENERIC_TERMS
        and tag.lower() not in GAMEPATH_SOURCE_TAGS
        and not normalize_gamepath_entity_type(tag)
    ]
    if tag_candidates:
        return clean_gamepath_metadata_value(tag_candidates[0], 80)
    text = str(question or "")
    named_patterns = [
        r"([\u4e00-\u9fffA-Za-z0-9]{1,24}\s*NPC)",
        r"([\u4e00-\u9fffA-Za-z0-9]{1,24}(?:\u89d2\u8272|\u602a|\u9470\u5319|\u788e\u7247|\u6a5f\u95dc|\u95a5\u9580|\u8b77\u7b26|\u679d|\u9580))",
    ]
    for pattern in named_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return clean_gamepath_metadata_value(match.group(1), 80)
    quoted = re.search(r"[「『\"']([^」』\"']{2,40})[」』\"']", text)
    if quoted:
        return clean_gamepath_metadata_value(quoted.group(1), 80)
    english = re.search(r"\b([A-Z][A-Za-z0-9_'’.-]{2,}(?:\s+[A-Z][A-Za-z0-9_'’.-]{2,}){0,4})\b", text)
    if english:
        return clean_gamepath_metadata_value(english.group(1), 80)
    return ""


def clamp_source_quality(value: Any, *, source_type: str = "", agent_used: bool = False) -> float:
    try:
        quality = float(value)
    except Exception:
        source = str(source_type or "").lower()
        if source == "manual":
            quality = 0.72
        elif "vision" in source:
            quality = 0.55
        elif "web" in source:
            quality = 0.62
        elif agent_used:
            quality = 0.58
        else:
            quality = 0.5
    return max(0.0, min(1.0, quality))


def infer_gamepath_metadata(
    question: str,
    answer_summary: str = "",
    tags: Any = None,
    *,
    version: Any = None,
    area: Any = None,
    entity_type: Any = None,
    entity_name: Any = None,
    source_quality: Any = None,
    source_type: str = "",
    agent_used: bool = False,
) -> dict[str, Any]:
    inferred_type = (
        normalize_gamepath_entity_type(entity_type)
        or infer_gamepath_entity_type(question)
        or infer_gamepath_entity_type(answer_summary)
        or infer_gamepath_entity_type("", tags=tags)
    )
    return {
        "version": clean_gamepath_metadata_value(version, 40) or infer_gamepath_version(question, answer_summary),
        "area": clean_gamepath_metadata_value(area, 80) or infer_gamepath_area(question, tags=tags),
        "entity_type": inferred_type,
        "entity_name": clean_gamepath_metadata_value(entity_name, 80)
        or infer_gamepath_entity_name(question, "", tags),
        "source_quality": clamp_source_quality(source_quality, source_type=source_type, agent_used=agent_used),
    }


def slug_text(value: str, fallback: str = "entry") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", (value or "").strip()).strip(".-_")
    return (slug or fallback)[:80]


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def gamepath_content_hash(game_id: str, question: str) -> str:
    key = re.sub(r"\s+", " ", str(question or "").strip().lower())[:360]
    return hashlib.sha256(f"{game_id}|{key}".encode("utf-8")).hexdigest()


def should_fuzzy_dedupe_gamepath_write(source_type: str, agent_used: bool) -> bool:
    source = str(source_type or "").strip().lower()
    return bool(agent_used or source.startswith("hermes_agent") or source.startswith("auto"))


def can_update_fuzzy_gamepath_duplicate(existing: dict[str, Any], source_type: str, agent_used: bool) -> bool:
    # Fuzzy matches are intentionally conservative: use them to avoid duplicate
    # writes, not to overwrite a previously saved hint with another model pass.
    return False


def sqlite_row_to_gamepath_item(row: sqlite3.Row, *, status: str = "") -> dict[str, Any]:
    item = {key: row[key] for key in row.keys()}
    if "id" in item:
        item["id"] = int(item["id"])
    if "agent_used" in item:
        item["agent_used"] = bool(item["agent_used"])
    if "source_quality" in item:
        item["source_quality"] = float(item.get("source_quality") or 0.5)
    if "dispute_count" in item:
        item["dispute_count"] = int(item.get("dispute_count") or 0)
    if status:
        item["status"] = status
    return item


def find_fuzzy_gamepath_duplicate_for_write(
    question: str,
    answer_summary: str,
    game_id: str,
    tags: Any,
    spoiler_level: str,
    source_type: str,
    agent_used: bool,
) -> Optional[dict[str, Any]]:
    if not should_fuzzy_dedupe_gamepath_write(source_type, agent_used):
        return None
    try:
        results = search_gamepath_multi_query_sync(
            question,
            game_id,
            8,
            tags=tags or None,
            spoiler_level=spoiler_level,
        )
    except Exception as exc:
        print(f"GamePath fuzzy duplicate search failed: {exc}")
        return None
    if not results:
        return None

    question_terms = gamepath_core_terms(question, max_terms=16)
    answer_terms = gamepath_core_terms(answer_summary, max_terms=16)
    for item in results:
        haystack = "\n".join(
            str(item.get(key) or "")
            for key in ("title", "question", "answer_summary", "tags", "relevant_excerpt", "snippet")
        )
        coverage = gamepath_term_coverage(question, haystack)
        core_overlap = term_overlap_ratio(question_terms, haystack)
        answer_overlap = term_overlap_ratio(answer_terms, str(item.get("answer_summary") or ""))
        scoring = score_gamepath_result(question, game_id, item)
        retrieval_score = max(float(item.get("retrieval_score") or 0.0), float(scoring.get("score") or 0.0))
        core_overlap = max(core_overlap, float(scoring.get("core_overlap") or 0.0))
        coverage = max(coverage, float(scoring.get("coverage") or 0.0))
        same_game = normalize_game_id(item.get("game_id")) in {normalize_game_id(game_id), "global"}
        if not same_game:
            continue
        if (
            (
                coverage >= 0.52
                and core_overlap >= 0.34
                and (retrieval_score >= 0.68 or answer_overlap >= 0.38)
            )
            or (answer_overlap >= 0.72 and coverage >= 0.25)
        ):
            duplicate = dict(item)
            duplicate["duplicate_coverage"] = round(coverage, 3)
            duplicate["duplicate_core_overlap"] = round(core_overlap, 3)
            duplicate["duplicate_answer_overlap"] = round(answer_overlap, 3)
            return duplicate
    return None


def gamepath_query_text(query: str) -> str:
    text = str(query or "")
    extras: list[str] = []
    lowered = text.lower()
    if re.search(r"(item|usage|用途|用法|能幹嘛|道具|物品|材料|素材)", lowered, re.IGNORECASE):
        extras.append("item usage material recipe craft npc quest unlock")
        extras.append("用途 用法 材料 素材 配方 任務 NPC 解鎖")
    if re.search(r"(boss|weakness|打法|弱點|怎麼打)", lowered, re.IGNORECASE):
        extras.append("boss weakness strategy build phase attack")
        extras.append("打法 弱點 配裝 階段 招式")
    if re.search(r"(quest|任務|支線|路線|在哪|哪裡|location|route|map)", lowered, re.IGNORECASE):
        extras.append("quest route location map walkthrough objective")
        extras.append("任務 支線 路線 位置 地圖 目標")
    if re.search(r"(version|patch|版本|更新)", lowered, re.IGNORECASE):
        extras.append("version patch update change current")
        extras.append("版本 更新 改動")
    return " ".join([text, *extras]).strip()


def ensure_gamepath_db() -> None:
    GAMEPATH_DIR.mkdir(parents=True, exist_ok=True)
    GAMEPATH_NOTES_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(GAMEPATH_DB) as conn:
        db_user_version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gamepath_entries(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                title TEXT NOT NULL,
                question TEXT NOT NULL,
                answer_summary TEXT NOT NULL,
                markdown_path TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '',
                version TEXT NOT NULL DEFAULT '',
                area TEXT NOT NULL DEFAULT '',
                entity_type TEXT NOT NULL DEFAULT '',
                entity_name TEXT NOT NULL DEFAULT '',
                spoiler_level TEXT NOT NULL DEFAULT 'low',
                spoiler_rank INTEGER NOT NULL DEFAULT 1,
                source_type TEXT NOT NULL DEFAULT 'manual',
                source_quality REAL NOT NULL DEFAULT 0.5,
                agent_used INTEGER NOT NULL DEFAULT 0,
                trust_state TEXT NOT NULL DEFAULT 'unverified',
                dispute_count INTEGER NOT NULL DEFAULT 0,
                last_feedback TEXT NOT NULL DEFAULT '',
                last_feedback_at TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        existing_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(gamepath_entries)").fetchall()
        }
        migrations = {
            "trust_state": "ALTER TABLE gamepath_entries ADD COLUMN trust_state TEXT NOT NULL DEFAULT 'unverified'",
            "dispute_count": "ALTER TABLE gamepath_entries ADD COLUMN dispute_count INTEGER NOT NULL DEFAULT 0",
            "last_feedback": "ALTER TABLE gamepath_entries ADD COLUMN last_feedback TEXT NOT NULL DEFAULT ''",
            "last_feedback_at": "ALTER TABLE gamepath_entries ADD COLUMN last_feedback_at TEXT NOT NULL DEFAULT ''",
            "version": "ALTER TABLE gamepath_entries ADD COLUMN version TEXT NOT NULL DEFAULT ''",
            "area": "ALTER TABLE gamepath_entries ADD COLUMN area TEXT NOT NULL DEFAULT ''",
            "entity_type": "ALTER TABLE gamepath_entries ADD COLUMN entity_type TEXT NOT NULL DEFAULT ''",
            "entity_name": "ALTER TABLE gamepath_entries ADD COLUMN entity_name TEXT NOT NULL DEFAULT ''",
            "source_quality": "ALTER TABLE gamepath_entries ADD COLUMN source_quality REAL NOT NULL DEFAULT 0.5",
        }
        for column, statement in migrations.items():
            if column not in existing_columns:
                conn.execute(statement)
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS gamepath_fts USING fts5(
                entry_id UNINDEXED,
                game_id UNINDEXED,
                title,
                question,
                answer_summary,
                tags,
                spoiler_level UNINDEXED,
                search_text,
                tokenize='unicode61'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gamepath_chunks(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id INTEGER NOT NULL,
                game_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                heading TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                char_count INTEGER NOT NULL DEFAULT 0,
                token_estimate INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(entry_id, chunk_index)
            )
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS gamepath_chunk_fts USING fts5(
                chunk_id UNINDEXED,
                entry_id UNINDEXED,
                game_id UNINDEXED,
                heading,
                content,
                tags,
                spoiler_level UNINDEXED,
                search_text,
                tokenize='unicode61'
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gamepath_scope ON gamepath_entries(game_id, spoiler_rank, entity_type, entity_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gamepath_version ON gamepath_entries(version)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gamepath_area ON gamepath_entries(area)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gamepath_source_quality ON gamepath_entries(source_quality)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gamepath_chunks_entry_id ON gamepath_chunks(entry_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gamepath_chunks_game_id ON gamepath_chunks(game_id)")
        conn.row_factory = sqlite3.Row
        if db_user_version < 8:
            rows = conn.execute("SELECT * FROM gamepath_entries").fetchall()
            for row in rows:
                item = dict(row)
                metadata = infer_gamepath_metadata(
                    item.get("question") or "",
                    item.get("answer_summary") or "",
                    item.get("tags") or "",
                    version=item.get("version") or "",
                    area="",
                    entity_type="",
                    entity_name="",
                    source_quality=None,
                    source_type=item.get("source_type") or "",
                    agent_used=bool(item.get("agent_used")),
                )
                conn.execute(
                    """
                    UPDATE gamepath_entries
                    SET version = ?, area = ?, entity_type = ?, entity_name = ?, source_quality = ?
                    WHERE id = ?
                    """,
                    (
                        metadata["version"],
                        metadata["area"],
                        metadata["entity_type"],
                        metadata["entity_name"],
                        metadata["source_quality"],
                        int(item["id"]),
                    ),
                )
            conn.execute("DELETE FROM gamepath_fts")
            conn.execute("DELETE FROM gamepath_chunk_fts")
            conn.execute("DELETE FROM gamepath_chunks")
            missing_chunk_rows = conn.execute("SELECT e.* FROM gamepath_entries e").fetchall()
        else:
            missing_chunk_rows = conn.execute(
                """
                SELECT e.*
                FROM gamepath_entries e
                LEFT JOIN gamepath_chunks c ON c.entry_id = e.id
                WHERE c.id IS NULL
                LIMIT 200
                """
            ).fetchall()
        for row in missing_chunk_rows:
            write_gamepath_fts(conn, dict(row))
            write_gamepath_chunks(conn, dict(row))
        conn.execute("PRAGMA user_version = 8")
        conn.commit()


def render_gamepath_markdown(row: dict[str, Any]) -> str:
    tags = [tag for tag in str(row.get("tags") or "").split(",") if tag]
    tag_text = " ".join(f"#{tag}" for tag in tags)
    return (
        f"# {row.get('title') or 'GamePath Entry'}\n\n"
        f"- Game: {row.get('game_id') or 'global'}\n"
        f"- Source: {row.get('source_type') or 'manual'}\n"
        f"- Agent used: {'yes' if row.get('agent_used') else 'no'}\n"
        f"- Version: {row.get('version') or 'unknown'}\n"
        f"- Area: {row.get('area') or 'unknown'}\n"
        f"- Entity type: {row.get('entity_type') or 'unknown'}\n"
        f"- Entity name: {row.get('entity_name') or 'unknown'}\n"
        f"- Source quality: {row.get('source_quality') if row.get('source_quality') is not None else 0.5}\n"
        f"- Spoiler level: {row.get('spoiler_level') or 'low'}\n"
        f"- Trust state: {row.get('trust_state') or 'unverified'}\n"
        f"- Dispute count: {row.get('dispute_count') or 0}\n"
        f"- Tags: {tag_text or 'none'}\n"
        f"- Updated: {row.get('updated_at') or ''}\n\n"
        "## Player Feedback\n\n"
        f"{row.get('last_feedback') or 'none'}\n\n"
        "## Player Question\n\n"
        f"{row.get('question') or ''}\n\n"
        "## Condensed Hint\n\n"
        f"{row.get('answer_summary') or ''}\n"
    )


def gamepath_markdown_path(entry_id: int, game_id: str, title: str, existing: str = "") -> Path:
    if existing:
        path = (PROJECT_ROOT / existing).resolve() if not Path(existing).is_absolute() else Path(existing)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        except Exception:
            pass
    folder = GAMEPATH_NOTES_DIR / slug_text(game_id, "global")
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{entry_id:06d}-{slug_text(title)}.md"


def gamepath_markdown_file(markdown_path: str) -> Optional[Path]:
    raw_path = str(markdown_path or "").strip()
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(GAMEPATH_NOTES_DIR.resolve())
    except (OSError, ValueError):
        return None
    return resolved


def write_gamepath_fts(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute("DELETE FROM gamepath_fts WHERE entry_id = ?", (row["id"],))
    search_text = expand_search_text(
        row.get("game_id") or "",
        row.get("title") or "",
        row.get("question") or "",
        row.get("answer_summary") or "",
        row.get("tags") or "",
        row.get("version") or "",
        row.get("area") or "",
        row.get("entity_type") or "",
        row.get("entity_name") or "",
        gamepath_query_text(row.get("question") or ""),
    )
    conn.execute(
        """
        INSERT INTO gamepath_fts(entry_id, game_id, title, question, answer_summary, tags, spoiler_level, search_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["id"],
            row.get("game_id") or "global",
            row.get("title") or "",
            row.get("question") or "",
            row.get("answer_summary") or "",
            row.get("tags") or "",
            row.get("spoiler_level") or "low",
            search_text,
        ),
    )


def gamepath_markdown_index_content(row: dict[str, Any]) -> str:
    answer_text = str(row.get("answer_summary") or "").strip()
    markdown_text = ""
    note_path = gamepath_markdown_file(str(row.get("markdown_path") or ""))
    if note_path and note_path.exists():
        try:
            markdown_text = note_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            markdown_text = ""

    condensed_match = re.search(
        r"(?ms)^##\s+Condensed Hint\s*\n(?P<body>.*?)(?=^##\s+|\Z)",
        markdown_text,
    )
    if condensed_match:
        condensed = condensed_match.group("body").strip()
        if condensed and (not answer_text or len(condensed) >= max(80, int(len(answer_text) * 0.5))):
            return condensed
        return answer_text
    if markdown_text and len(markdown_text) > max(1200, len(answer_text) * 1.4):
        return markdown_text.strip()
    return answer_text


def gamepath_chunk_heading(content: str, fallback: str) -> str:
    for line in str(content or "").splitlines():
        match = re.match(r"^#{1,4}\s+(?P<title>.+)$", line.strip())
        if match:
            return re.sub(r"\s+", " ", match.group("title")).strip()[:120]
    return re.sub(r"\s+", " ", str(fallback or "GamePath")).strip()[:120]


def build_gamepath_chunk_documents(row: dict[str, Any]) -> list[dict[str, Any]]:
    source = gamepath_markdown_index_content(row)
    if not source:
        return []
    sections = split_gamepath_markdown_sections(source)
    if not sections:
        sections = split_gamepath_passage(source)
    title = str(row.get("title") or row.get("question") or "GamePath").strip()
    chunks: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for index, section in enumerate(sections[:200]):
        content = re.sub(r"\n{3,}", "\n\n", str(section or "").strip())
        if len(content) < 8:
            continue
        content_hash = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        chunks.append(
            {
                "chunk_index": len(chunks),
                "heading": gamepath_chunk_heading(content, title),
                "content": content,
                "content_hash": content_hash,
                "char_count": len(content),
                "token_estimate": max(1, len(content) // 3),
            }
        )
    return chunks


def write_gamepath_chunks(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    entry_id = int(row.get("id") or 0)
    if entry_id <= 0:
        return
    conn.execute("DELETE FROM gamepath_chunk_fts WHERE entry_id = ?", (entry_id,))
    conn.execute("DELETE FROM gamepath_chunks WHERE entry_id = ?", (entry_id,))
    chunks = build_gamepath_chunk_documents(row)
    if not chunks:
        return

    now = str(row.get("updated_at") or time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    created_at = str(row.get("created_at") or now)
    game_id = str(row.get("game_id") or "global")
    title = str(row.get("title") or "")
    question = str(row.get("question") or "")
    tags = str(row.get("tags") or "")
    version = str(row.get("version") or "")
    area = str(row.get("area") or "")
    entity_type = str(row.get("entity_type") or "")
    entity_name = str(row.get("entity_name") or "")
    spoiler_level = str(row.get("spoiler_level") or "low")

    for chunk in chunks:
        cursor = conn.execute(
            """
            INSERT INTO gamepath_chunks(
                entry_id, game_id, chunk_index, heading, content, content_hash,
                char_count, token_estimate, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id,
                game_id,
                int(chunk["chunk_index"]),
                chunk["heading"],
                chunk["content"],
                chunk["content_hash"],
                int(chunk["char_count"]),
                int(chunk["token_estimate"]),
                created_at,
                now,
            ),
        )
        chunk_id = int(cursor.lastrowid)
        search_text = expand_search_text(
            game_id,
            title,
            question,
            chunk["heading"],
            chunk["content"],
            tags,
            version,
            area,
            entity_type,
            entity_name,
            gamepath_query_text(question),
        )
        conn.execute(
            """
            INSERT INTO gamepath_chunk_fts(
                chunk_id, entry_id, game_id, heading, content, tags, spoiler_level, search_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                entry_id,
                game_id,
                chunk["heading"],
                chunk["content"],
                tags,
                spoiler_level,
                search_text,
            ),
        )


def add_gamepath_sync(
    question: str,
    answer_summary: str,
    game_id: Optional[str],
    *,
    title: Optional[str] = None,
    tags: Any = None,
    spoiler_level: str = "low",
    source_type: str = "manual",
    agent_used: bool = False,
    version: Any = None,
    area: Any = None,
    entity_type: Any = None,
    entity_name: Any = None,
    source_quality: Any = None,
) -> dict[str, Any]:
    ensure_gamepath_db()
    normalized_game_id = normalize_game_id(game_id) or "global"
    clean_question = re.sub(r"\s+", " ", str(question or "").strip())
    clean_answer = re.sub(r"\n{3,}", "\n\n", str(answer_summary or "").strip())
    clean_answer = re.sub(r"https?://\S+", "", clean_answer).strip()
    if not clean_question:
        raise ValueError("GamePath question is empty.")
    if len(clean_answer) < 12:
        raise ValueError("GamePath answer is too short.")
    clean_title = re.sub(r"\s+", " ", str(title or "").strip())
    if not clean_title:
        first_line = next((line.strip() for line in clean_answer.splitlines() if line.strip()), "")
        clean_title = first_line[:80] or clean_question[:80]
    safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source_type or "manual").strip().lower()).strip("._-")
    safe_source = safe_source or "manual"
    safe_spoiler = normalize_spoiler_level(spoiler_level)
    safe_tags = tags_to_text(tags)
    metadata = infer_gamepath_metadata(
        clean_question,
        clean_answer,
        safe_tags,
        version=version,
        area=area,
        entity_type=entity_type,
        entity_name=entity_name,
        source_quality=source_quality,
        source_type=safe_source,
        agent_used=agent_used,
    )
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    content_hash = gamepath_content_hash(normalized_game_id, clean_question)
    fuzzy_duplicate = find_fuzzy_gamepath_duplicate_for_write(
        clean_question,
        clean_answer,
        normalized_game_id,
        safe_tags,
        safe_spoiler,
        safe_source,
        agent_used,
    )

    with sqlite3.connect(GAMEPATH_DB) as conn:
        conn.row_factory = sqlite3.Row
        existing = conn.execute(
            "SELECT * FROM gamepath_entries WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        fuzzy_status = ""
        if not existing and fuzzy_duplicate:
            duplicate_id = int(fuzzy_duplicate.get("id") or 0)
            if duplicate_id > 0:
                duplicate_row = conn.execute(
                    "SELECT * FROM gamepath_entries WHERE id = ?",
                    (duplicate_id,),
                ).fetchone()
                if duplicate_row:
                    duplicate_item = sqlite_row_to_gamepath_item(duplicate_row)
                    if can_update_fuzzy_gamepath_duplicate(duplicate_item, safe_source, agent_used):
                        existing = duplicate_row
                        fuzzy_status = "updated_duplicate"
                    else:
                        duplicate_item["status"] = "duplicate_existing"
                        duplicate_item["duplicate_coverage"] = fuzzy_duplicate.get("duplicate_coverage")
                        duplicate_item["duplicate_core_overlap"] = fuzzy_duplicate.get("duplicate_core_overlap")
                        duplicate_item["duplicate_answer_overlap"] = fuzzy_duplicate.get("duplicate_answer_overlap")
                        return duplicate_item
        if existing:
            entry_id = int(existing["id"])
            markdown_path = existing["markdown_path"] or ""
            existing_dispute_count = int(existing["dispute_count"] or 0)
            conn.execute(
                """
                UPDATE gamepath_entries
                SET title = ?, question = ?, answer_summary = ?, tags = ?,
                    version = ?, area = ?, entity_type = ?, entity_name = ?,
                    spoiler_level = ?, spoiler_rank = ?, source_type = ?, source_quality = ?,
                    agent_used = ?, trust_state = ?,
                    last_feedback = '', last_feedback_at = '', updated_at = ?
                WHERE id = ?
                """,
                (
                    clean_title,
                    clean_question,
                    clean_answer,
                    safe_tags,
                    metadata["version"],
                    metadata["area"],
                    metadata["entity_type"],
                    metadata["entity_name"],
                    safe_spoiler,
                    spoiler_rank(safe_spoiler),
                    safe_source,
                    metadata["source_quality"],
                    1 if agent_used else 0,
                    "unverified",
                    now,
                    entry_id,
                ),
            )
            created_at = existing["created_at"]
            status = fuzzy_status or "updated"
        else:
            cursor = conn.execute(
                """
                INSERT INTO gamepath_entries(
                    game_id, title, question, answer_summary, markdown_path, tags,
                    version, area, entity_type, entity_name, spoiler_level,
                    spoiler_rank, source_type, source_quality, agent_used, trust_state, dispute_count, last_feedback,
                    last_feedback_at, content_hash, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'unverified', 0, '', '', ?, ?, ?)
                """,
                (
                    normalized_game_id,
                    clean_title,
                    clean_question,
                    clean_answer,
                    safe_tags,
                    metadata["version"],
                    metadata["area"],
                    metadata["entity_type"],
                    metadata["entity_name"],
                    safe_spoiler,
                    spoiler_rank(safe_spoiler),
                    safe_source,
                    metadata["source_quality"],
                    1 if agent_used else 0,
                    content_hash,
                    now,
                    now,
                ),
            )
            entry_id = int(cursor.lastrowid)
            markdown_path = ""
            created_at = now
            existing_dispute_count = 0
            status = "created"

        note_path = gamepath_markdown_path(entry_id, normalized_game_id, clean_title, markdown_path)
        row = {
            "id": entry_id,
            "game_id": normalized_game_id,
            "title": clean_title,
            "question": clean_question,
            "answer_summary": clean_answer,
            "markdown_path": relative_or_absolute(note_path),
            "tags": safe_tags,
            "version": metadata["version"],
            "area": metadata["area"],
            "entity_type": metadata["entity_type"],
            "entity_name": metadata["entity_name"],
            "spoiler_level": safe_spoiler,
            "source_type": safe_source,
            "source_quality": metadata["source_quality"],
            "agent_used": bool(agent_used),
            "trust_state": "unverified",
            "dispute_count": existing_dispute_count,
            "last_feedback": "",
            "last_feedback_at": "",
            "created_at": created_at,
            "updated_at": now,
        }
        note_path.write_text(render_gamepath_markdown(row), encoding="utf-8", newline="\n")
        conn.execute(
            "UPDATE gamepath_entries SET markdown_path = ? WHERE id = ?",
            (row["markdown_path"], entry_id),
        )
        write_gamepath_fts(conn, row)
        write_gamepath_chunks(conn, row)
        conn.commit()
    row["status"] = status
    return row


def delete_gamepath_sync(entry_id: int) -> Optional[dict[str, Any]]:
    ensure_gamepath_db()
    try:
        safe_entry_id = int(entry_id)
    except (TypeError, ValueError):
        raise ValueError("GamePath entry id is invalid.")
    if safe_entry_id <= 0:
        raise ValueError("GamePath entry id is invalid.")

    with sqlite3.connect(GAMEPATH_DB) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, game_id, title, question, markdown_path
            FROM gamepath_entries
            WHERE id = ?
            """,
            (safe_entry_id,),
        ).fetchone()
        if not row:
            return None
        item = {
            "id": int(row["id"]),
            "game_id": row["game_id"],
            "title": row["title"],
            "question": row["question"],
            "markdown_path": row["markdown_path"],
            "markdown_deleted": False,
        }
        conn.execute("DELETE FROM gamepath_chunk_fts WHERE entry_id = ?", (safe_entry_id,))
        conn.execute("DELETE FROM gamepath_chunks WHERE entry_id = ?", (safe_entry_id,))
        conn.execute("DELETE FROM gamepath_fts WHERE entry_id = ?", (safe_entry_id,))
        conn.execute("DELETE FROM gamepath_entries WHERE id = ?", (safe_entry_id,))
        conn.commit()

    note_path = gamepath_markdown_file(item.get("markdown_path") or "")
    if note_path and note_path.exists():
        try:
            note_path.unlink()
            item["markdown_deleted"] = True
        except OSError as exc:
            item["markdown_delete_error"] = str(exc)
    return item


def get_gamepath_entry_sync(entry_id: int) -> Optional[dict[str, Any]]:
    ensure_gamepath_db()
    with sqlite3.connect(GAMEPATH_DB) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM gamepath_entries WHERE id = ?",
            (int(entry_id),),
        ).fetchone()
    if not row:
        return None
    return {
        "id": int(row["id"]),
        "game_id": row["game_id"],
        "title": row["title"],
        "question": row["question"],
        "answer_summary": row["answer_summary"],
        "markdown_path": row["markdown_path"],
        "tags": row["tags"],
        "spoiler_level": row["spoiler_level"],
        "source_type": row["source_type"],
        "agent_used": bool(row["agent_used"]),
        "trust_state": row["trust_state"],
        "dispute_count": int(row["dispute_count"] or 0),
        "last_feedback": row["last_feedback"],
        "last_feedback_at": row["last_feedback_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def normalize_gamepath_trust_state(state: str) -> str:
    clean = re.sub(r"[^A-Za-z_]+", "_", str(state or "disputed").strip().lower()).strip("_")
    return clean if clean in GAMEPATH_TRUST_STATES else "disputed"


def update_gamepath_feedback_sync(
    entry_id: int,
    message: str,
    *,
    state: str = "disputed",
) -> Optional[dict[str, Any]]:
    ensure_gamepath_db()
    safe_state = normalize_gamepath_trust_state(state)
    clean_message = re.sub(r"\s+", " ", str(message or "").strip())[:1000]
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with sqlite3.connect(GAMEPATH_DB) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM gamepath_entries WHERE id = ?", (int(entry_id),)).fetchone()
        if not row:
            return None
        dispute_increment = 1 if safe_state in {"disputed", "needs_review"} else 0
        dispute_count = int(row["dispute_count"] or 0) + dispute_increment
        conn.execute(
            """
            UPDATE gamepath_entries
            SET trust_state = ?, dispute_count = ?, last_feedback = ?, last_feedback_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (safe_state, dispute_count, clean_message, now, now, int(entry_id)),
        )
        updated = conn.execute("SELECT * FROM gamepath_entries WHERE id = ?", (int(entry_id),)).fetchone()
        item = dict(updated) if updated else {}
        if item:
            item["agent_used"] = bool(item.get("agent_used"))
            item["dispute_count"] = int(item.get("dispute_count") or 0)
            item["id"] = int(item["id"])
            note_path = gamepath_markdown_file(str(item.get("markdown_path") or ""))
            if note_path:
                try:
                    note_path.write_text(render_gamepath_markdown(item), encoding="utf-8", newline="\n")
                except OSError as exc:
                    item["markdown_write_error"] = str(exc)
        conn.commit()
    return item or None


def detect_gamepath_dispute(prompt: str) -> bool:
    return bool(GAMEPATH_DISPUTE_RE.search(prompt or ""))


def remember_gamepath_reference(item: dict[str, Any], *, route: str) -> None:
    if not item.get("id"):
        return
    last_gamepath_reference.clear()
    last_gamepath_reference.update(
        {
            "entry_id": int(item["id"]),
            "title": item.get("title") or item.get("question") or "GamePath",
            "game_id": item.get("game_id") or "",
            "route": route,
            "at": time.time(),
        }
    )


def recent_gamepath_reference(game_id: Optional[str], max_age_seconds: int = 1800) -> Optional[dict[str, Any]]:
    if not last_gamepath_reference:
        return None
    if time.time() - float(last_gamepath_reference.get("at") or 0) > max_age_seconds:
        return None
    ref_game_id = normalize_game_id(last_gamepath_reference.get("game_id"))
    current_game_id = normalize_game_id(game_id)
    if current_game_id and ref_game_id and ref_game_id not in {current_game_id, "global"}:
        return None
    return dict(last_gamepath_reference)


def build_gamepath_dispute_message(item: dict[str, Any], feedback: str) -> str:
    title = str(item.get("title") or item.get("question") or "GamePath").strip()
    dispute_count = int(item.get("dispute_count") or 0)
    return (
        f"了解，這代表我上一個 GamePath 提示「{title}」可能不適用你目前的版本、場景或進度。"
        f"我已把它標成 disputed（回報次數 {dispute_count}），下次不會直接拿它 fast path 硬答。\n"
        "接下來建議你截圖目前畫面，或告訴我任務名稱/區域/版本；我會改用驗證模式重新查 GamePath 與 Hermes/Tavily。"
    )


def gamepath_entry_count_sync() -> int:
    if not GAMEPATH_DB.exists():
        return 0
    try:
        with sqlite3.connect(GAMEPATH_DB) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM gamepath_entries").fetchone()[0])
    except sqlite3.Error:
        return 0


def gamepath_chunk_count_sync() -> int:
    if not GAMEPATH_DB.exists():
        return 0
    try:
        ensure_gamepath_db()
        with sqlite3.connect(GAMEPATH_DB) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM gamepath_chunks").fetchone()[0])
    except sqlite3.Error:
        return 0


def gamepath_last_updated_at_sync() -> str:
    if not GAMEPATH_DB.exists():
        return ""
    try:
        with sqlite3.connect(GAMEPATH_DB) as conn:
            row = conn.execute("SELECT MAX(updated_at) FROM gamepath_entries").fetchone()
        return str(row[0] or "") if row else ""
    except sqlite3.Error:
        return ""


def gamepath_games_sync() -> set[str]:
    if not GAMEPATH_DB.exists():
        return set()
    try:
        with sqlite3.connect(GAMEPATH_DB) as conn:
            rows = conn.execute("SELECT DISTINCT game_id FROM gamepath_entries WHERE game_id != ''").fetchall()
        return {str(row[0]) for row in rows if row[0]}
    except sqlite3.Error:
        return set()


def gamepath_term_coverage(query: str, text: str) -> float:
    terms = [
        term
        for term in search_terms(query, max_terms=18)
        if term.lower() not in GAMEPATH_GENERIC_TERMS and term not in GAMEPATH_GENERIC_TERMS
    ]
    if not terms:
        return 0.0
    haystack = str(text or "").lower()
    matched = sum(1 for term in terms if term.lower() in haystack)
    return min(1.0, matched / max(1, min(len(terms), 8)))


def gamepath_text_is_large(text: str) -> bool:
    clean = str(text or "").strip()
    if len(clean) > GAMEPATH_DIRECT_MAX_CHARS:
        return True
    heading_count = len(re.findall(r"(?m)^#{1,4}\s+\S", clean))
    return heading_count >= 4


def split_gamepath_passage(passsage: str, max_chars: int = GAMEPATH_PASSAGE_MAX_CHARS) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", str(passsage or "").strip())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            sentences = re.split(r"(?<=[.!?。！？])\s+", block)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(sentence) > max_chars:
                    for start in range(0, len(sentence), max_chars):
                        chunks.append(sentence[start : start + max_chars].strip())
                elif len((current + "\n" + sentence).strip()) > max_chars:
                    if current:
                        chunks.append(current.strip())
                    current = sentence
                else:
                    current = (current + "\n" + sentence).strip()
            continue

        candidate = (current + "\n\n" + block).strip() if current else block
        if len(candidate) > max_chars:
            if current:
                chunks.append(current.strip())
            current = block
        else:
            current = candidate

    if current:
        chunks.append(current.strip())
    return chunks


def split_gamepath_markdown_sections(text: str) -> list[str]:
    clean = re.sub(r"\r\n?", "\n", str(text or "")).strip()
    if not clean:
        return []

    sections: list[str] = []
    current: list[str] = []
    for line in clean.splitlines():
        if re.match(r"^#{1,4}\s+\S", line) and current:
            sections.extend(split_gamepath_passage("\n".join(current)))
            current = [line.rstrip()]
        else:
            current.append(line.rstrip())
    if current:
        sections.extend(split_gamepath_passage("\n".join(current)))
    return [section for section in sections if section.strip()]


def score_gamepath_passage(query: str, passage: str) -> float:
    terms = gamepath_core_terms(query, max_terms=18)
    if not terms:
        terms = [
            term
            for term in search_terms(query, max_terms=18)
            if term.lower() not in GAMEPATH_GENERIC_TERMS and term not in GAMEPATH_GENERIC_TERMS
        ]
    if not terms:
        return 0.0

    lowered = str(passage or "").lower()
    matched = [term for term in terms if term.lower() in lowered]
    if not matched:
        return 0.0

    overlap = len(matched) / max(1, min(len(terms), 8))
    query_text = re.sub(r"\s+", " ", str(query or "").strip()).lower()
    phrase_bonus = 0.18 if len(query_text) >= 4 and query_text in lowered else 0.0
    first_line = str(passage or "").splitlines()[0].lower() if str(passage or "").splitlines() else ""
    heading_bonus = 0.08 if first_line.startswith("#") and any(term.lower() in first_line for term in matched) else 0.0
    density_bonus = min(0.1, len(matched) * 0.015)
    return overlap + phrase_bonus + heading_bonus + density_bonus


def build_gamepath_relevant_context(query: str, item: dict[str, Any]) -> dict[str, Any]:
    answer_text = str(item.get("answer_summary") or "").strip()
    markdown_text = ""
    note_path = gamepath_markdown_file(str(item.get("markdown_path") or ""))
    if note_path and note_path.exists():
        try:
            markdown_text = note_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            markdown_text = ""

    content = answer_text
    condensed_match = re.search(
        r"(?ms)^##\s+Condensed Hint\s*\n(?P<body>.*?)(?=^##\s+|\Z)",
        markdown_text,
    )
    if condensed_match:
        condensed = condensed_match.group("body").strip()
        if condensed and len(condensed) >= len(content):
            content = condensed
    elif markdown_text and len(markdown_text) > len(content) * 1.2:
        content = markdown_text

    source_char_count = len(content)
    sections = split_gamepath_markdown_sections(content)
    scored: list[tuple[float, str]] = [
        (score_gamepath_passage(query, section), section)
        for section in sections
        if section.strip()
    ]
    scored = [item for item in scored if item[0] > 0]
    scored.sort(key=lambda item: item[0], reverse=True)

    excerpts: list[str] = []
    total_chars = 0
    best_score = scored[0][0] if scored else 0.0
    term_count = len(gamepath_core_terms(query, max_terms=18))
    max_passages = 2 if term_count <= 2 else 3
    min_passage_score = max(0.22, best_score * 0.62) if best_score else 0.0
    for score, passage in scored[:5]:
        if excerpts and score < min_passage_score:
            continue
        snippet = make_snippet(passage, query, max_len=GAMEPATH_PASSAGE_MAX_CHARS)
        if not snippet:
            continue
        next_total = total_chars + len(snippet) + (4 if excerpts else 0)
        if excerpts and next_total > GAMEPATH_CONTEXT_MAX_CHARS:
            break
        excerpts.append(snippet)
        total_chars = next_total
        if len(excerpts) >= max_passages:
            break

    if not excerpts and content:
        excerpts.append(make_snippet(content, query, max_len=min(GAMEPATH_PASSAGE_MAX_CHARS, GAMEPATH_CONTEXT_MAX_CHARS)))

    relevant_excerpt = "\n\n---\n\n".join(excerpts).strip()
    return {
        "relevant_excerpt": relevant_excerpt,
        "context_char_count": len(relevant_excerpt),
        "source_char_count": source_char_count,
        "passage_count": len(sections),
        "large_entry": gamepath_text_is_large(content),
    }


def normalize_gamepath_scope(
    query: str,
    tags: Any = None,
    *,
    version: Any = None,
    area: Any = None,
    entity_type: Any = None,
    entity_name: Any = None,
) -> dict[str, str]:
    metadata = infer_gamepath_metadata(
        query,
        "",
        tags,
        version=version,
        area=area,
        entity_type=entity_type,
        entity_name=entity_name,
    )
    return {
        "version": str(metadata.get("version") or ""),
        "area": str(metadata.get("area") or ""),
        "entity_type": str(metadata.get("entity_type") or ""),
        "entity_name": str(metadata.get("entity_name") or ""),
        "tags": ",".join(normalize_tags_value(tags)),
    }


def append_gamepath_scope_filters(
    sql: str,
    params: list[Any],
    scope: dict[str, str],
    *,
    strict_metadata: bool,
) -> tuple[str, list[Any]]:
    for key in ("entity_type", "version", "area"):
        wanted = scope.get(key) or ""
        if not wanted:
            continue
        if strict_metadata:
            sql += f" AND e.{key} = ?"
            params.append(wanted)
        else:
            sql += f" AND (e.{key} = ? OR e.{key} = '')"
            params.append(wanted)
    entity_name = scope.get("entity_name") or ""
    if entity_name and strict_metadata:
        sql += " AND e.entity_name = ?"
        params.append(entity_name)
    return sql, params


def gamepath_metadata_match_score(item: dict[str, Any], scope: dict[str, str], query: str) -> float:
    score = 0.0
    for key, weight in (("entity_type", 0.22), ("version", 0.16), ("area", 0.16)):
        wanted = str(scope.get(key) or "").strip().lower()
        actual = str(item.get(key) or "").strip().lower()
        if not wanted:
            continue
        if actual == wanted:
            score += weight
        elif key == "area" and actual and (wanted in actual or actual in wanted):
            score += weight * 0.7
        elif actual:
            score -= weight * 0.45
    entity_name = str(scope.get("entity_name") or "").strip().lower()
    if entity_name:
        haystack = "\n".join(
            str(item.get(key) or "")
            for key in ("entity_name", "title", "question", "answer_summary", "tags", "relevant_excerpt", "snippet")
        ).lower()
        compact_entity_name = re.sub(r"\s+", "", entity_name)
        compact_haystack = re.sub(r"\s+", "", haystack)
        if entity_name in haystack or (compact_entity_name and compact_entity_name in compact_haystack):
            score += 0.18
        elif str(item.get("entity_name") or "").strip():
            score -= 0.08
    semantic_hints = gamepath_semantic_hint_text(query)
    if semantic_hints:
        haystack = "\n".join(
            str(item.get(key) or "")
            for key in ("entity_name", "title", "question", "answer_summary", "tags", "relevant_excerpt", "snippet")
        )
        hint_overlap = term_overlap_ratio(gamepath_core_terms(semantic_hints, max_terms=16), haystack)
        score += min(0.16, hint_overlap * 0.16)
    requested_tags = {tag.strip().lower() for tag in str(scope.get("tags") or "").split(",") if tag.strip()}
    if requested_tags:
        row_tags = {tag.strip().lower() for tag in str(item.get("tags") or "").split(",") if tag.strip()}
        tag_matches = requested_tags.intersection(row_tags)
        if tag_matches:
            score += min(0.12, 0.05 + len(tag_matches) * 0.025)
        else:
            score -= 0.025
    trust = str(item.get("trust_state") or "").lower()
    if trust == "verified":
        score += 0.1
    elif trust in {"disputed", "needs_review"}:
        score -= 0.16
    elif trust == "deprecated":
        score -= 0.3
    score += max(0.0, min(float(item.get("source_quality") or 0.5), 1.0)) * 0.22
    score -= min(int(item.get("dispute_count") or 0), 5) * 0.04
    if gamepath_core_terms(query) and float(item.get("match_coverage") or 0.0) <= 0.2:
        score -= 0.08
    return round(score, 4)


def rank_gamepath_results(results: list[dict[str, Any]], scope: dict[str, str], query: str) -> list[dict[str, Any]]:
    for item in results:
        item["metadata_match_score"] = gamepath_metadata_match_score(item, scope, query)
    results.sort(
        key=lambda item: (
            -float(item.get("match_coverage") or 0.0),
            -float(item.get("metadata_match_score") or 0.0),
            float(item.get("score") or 0.0),
            -float(item.get("source_quality") or 0.5),
        )
    )
    return results


def merge_gamepath_results(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    scope: dict[str, str],
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for item in [*primary, *secondary]:
        try:
            entry_id = int(item.get("id") or 0)
        except (TypeError, ValueError):
            continue
        if entry_id <= 0:
            continue
        existing = merged.get(entry_id)
        if existing is None:
            merged[entry_id] = item
            continue
        existing_coverage = float(existing.get("match_coverage") or 0.0)
        item_coverage = float(item.get("match_coverage") or 0.0)
        existing_context = int(existing.get("context_char_count") or 0)
        item_context = int(item.get("context_char_count") or 0)
        if (item_coverage, item_context) > (existing_coverage, existing_context):
            merged[entry_id] = item
    ranked = rank_gamepath_results(list(merged.values()), scope, query)
    return ranked[: clamp_limit(limit, upper=30)]


def gamepath_semantic_hint_text(text: str) -> str:
    source = str(text or "")
    hints: list[str] = []
    elder_dialog = bool(
        re.search(
            r"((?:\u78bc\u982d|\u93e1\u6e56).{0,18}(?:\u8001\u4eba|\u8001\u982d))|"
            r"((?:\u8001\u4eba|\u8001\u982d).{0,18}(?:\u78bc\u982d|\u93e1\u6e56|\u9418\u8072))",
            source,
            re.IGNORECASE,
        )
    )
    if elder_dialog:
        hints.append("\u93e1\u6e56\u78bc\u982d \u8001\u4eba NPC \u5c0d\u8a71 \u9418\u8072 \u56de\u7b54 \u5b89\u5168")
    hint_patterns = [
        (r"(拿刀|砍|追我|屠夫)", "屠夫 怪 boss 打法 弱點"),
        (r"(紅色光|紅光|擋路)", "紅光 機制 互動 電箱 紫外線"),
        (r"(三個轉盤|轉盤|三個閥門|閥門)", "三閥門 閥門 順序 解謎 鍋爐房"),
        (r"(鐘聲|四次鐘|四聲鐘|敲)", "鐘塔 鐘聲 四聲鐘 順序 解謎"),
        (r"(女殭屍|唱歌)", "唱歌女殭屍 敵人 走廊 處理"),
        (r"(黑帽)", "黑帽角色 酒窖 身份 角色"),
        (r"(白衣\s*NPC|白衣)", "白衣 NPC 資料室 對話"),
        (r"(怪門)", "怪門 廚房 路線 開門"),
        (r"(鐵手臂|鐵臂)", "鐵臂怪 鍋爐房 boss 打法"),
        (r"(暗門)", "暗門 入口 西翼大廳 位置"),
        (r"(月銀枝)", "月銀枝 素材 用途"),
        (r"(保險絲)", "保險絲 舊醫院 用途"),
        (r"(地圖碎片|碎片)", "地圖碎片 位置 地圖"),
        (r"(老人|老頭)", "老人 NPC 對話"),
        (r"(迷路|繞回|路標)", "路線 迷路 路標 下一步"),
        (r"(下一步|接下來|去哪)", "任務 路線 下一步 目標"),
    ]
    for pattern, hint in hint_patterns:
        if elder_dialog and "\u9418" in hint and "\u9806\u5e8f" in hint:
            continue
        if re.search(pattern, source, re.IGNORECASE):
            hints.append(hint)
    return " ".join(hints)


def build_gamepath_query_variants(
    query: str,
    *,
    tags: Any = None,
    query_variants: Any = None,
    max_variants: int = 8,
) -> list[str]:
    candidates: list[str] = []

    def add(value: Any) -> None:
        clean = re.sub(r"\s+", " ", str(value or "").strip())
        if not clean:
            return
        if clean not in candidates:
            candidates.append(clean)

    semantic_sources: list[str] = []

    def remember_source(value: Any) -> None:
        clean = re.sub(r"\s+", " ", str(value or "").strip())
        if clean and clean not in semantic_sources:
            semantic_sources.append(clean)

    remember_source(query)
    add(query)
    if isinstance(query_variants, str):
        remember_source(query_variants)
        add(query_variants)
    elif isinstance(query_variants, list):
        for item in query_variants:
            remember_source(item)
            add(item)

    semantic_source = " ".join(semantic_sources).strip() or query
    semantic_hints = gamepath_semantic_hint_text(semantic_source)
    if semantic_hints:
        add(f"{query} {semantic_hints}")
        cjk_focused_hints = re.sub(
            r"\b(?:boss|enemy|npc|item|route|map|quest|guide|walkthrough|mechanic|puzzle|location|character)\b",
            " ",
            semantic_hints,
            flags=re.IGNORECASE,
        )
        area_hint = infer_gamepath_area(semantic_source)
        hint_terms: list[str] = []
        for raw in re.findall(r"[\u4e00-\u9fff]{2,}", cjk_focused_hints):
            if raw not in hint_terms:
                hint_terms.append(raw)
        for term in gamepath_core_terms(cjk_focused_hints, max_terms=12):
            if has_cjk_text(term) and term not in hint_terms:
                hint_terms.append(term)
        if area_hint and hint_terms:
            add(" ".join([area_hint, *hint_terms[:4]]))
            for priority_term in (
                "\u4e0b\u4e00\u6b65",
                "\u76ee\u6a19",
                "\u4f4d\u7f6e",
                "\u7528\u9014",
                "\u6253\u6cd5",
                "\u5c0d\u8a71",
            ):
                if priority_term in cjk_focused_hints:
                    add(f"{area_hint} {priority_term}")
                    break
        add(cjk_focused_hints)
        add(semantic_hints)

    core_terms = gamepath_core_terms(query, max_terms=12)
    if core_terms:
        add(" ".join(core_terms))

    compact = re.sub(r"[^\w\u3400-\u9fff]+", " ", str(query or ""), flags=re.IGNORECASE)
    add(compact)
    return candidates[:max_variants]


def search_gamepath_multi_query_sync(
    query: str,
    game_id: Optional[str],
    limit: int = 5,
    *,
    tags: Any = None,
    spoiler_level: str = "low",
    query_variants: Any = None,
    version: Any = None,
    area: Any = None,
    entity_type: Any = None,
    entity_name: Any = None,
    strict_metadata: bool = False,
) -> list[dict[str, Any]]:
    query = str(query or "").strip()
    variants = build_gamepath_query_variants(query, tags=tags, query_variants=query_variants)
    if not variants:
        return []
    merged: dict[int, dict[str, Any]] = {}
    for variant in variants:
        results = search_gamepath_sync(
            variant,
            game_id,
            max(limit * 4, 12),
            tags=tags,
            spoiler_level=spoiler_level,
            version=version,
            area=area,
            entity_type=entity_type,
            entity_name=entity_name,
            strict_metadata=strict_metadata,
        )
        for item in results:
            try:
                entry_id = int(item.get("id") or 0)
            except (TypeError, ValueError):
                continue
            if entry_id <= 0:
                continue
            candidate = dict(item)
            candidate["matched_query"] = variant
            candidate["match_coverage"] = max(
                float(candidate.get("match_coverage") or 0.0),
                gamepath_term_coverage(query, "\n".join(str(candidate.get(key) or "") for key in ("title", "question", "answer_summary", "tags", "relevant_excerpt", "snippet"))),
            )
            existing = merged.get(entry_id)
            if existing is None:
                merged[entry_id] = candidate
                continue
            if (
                float(candidate.get("match_coverage") or 0.0),
                float(candidate.get("metadata_match_score") or 0.0),
                int(candidate.get("context_char_count") or 0),
            ) > (
                float(existing.get("match_coverage") or 0.0),
                float(existing.get("metadata_match_score") or 0.0),
                int(existing.get("context_char_count") or 0),
            ):
                merged[entry_id] = candidate
    scope_query = " ".join(item for item in [query, gamepath_semantic_hint_text(query)] if item).strip()
    scope = normalize_gamepath_scope(
        scope_query or query,
        tags,
        version=version,
        area=area,
        entity_type=entity_type,
        entity_name=entity_name,
    )
    ranked = rank_gamepath_results(list(merged.values()), scope, query)
    return ranked[: clamp_limit(limit, upper=30)]


def search_gamepath_chunks_sync(
    query: str,
    game_id: Optional[str],
    limit: int = 5,
    *,
    tags: Any = None,
    spoiler_level: str = "low",
    version: Any = None,
    area: Any = None,
    entity_type: Any = None,
    entity_name: Any = None,
    strict_metadata: bool = False,
    apply_scope_filters: bool = True,
) -> list[dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []
    match = fts_query(gamepath_query_text(query))
    if not match:
        return []

    normalized_game_id = normalize_game_id(game_id)
    scope = normalize_gamepath_scope(
        query,
        tags,
        version=version,
        area=area,
        entity_type=entity_type,
        entity_name=entity_name,
    )
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
    params: list[Any] = [match, spoiler_rank(spoiler_level)]
    if normalized_game_id:
        sql += " AND e.game_id IN (?, 'global')"
        params.append(normalized_game_id)
    if apply_scope_filters:
        sql, params = append_gamepath_scope_filters(
            sql,
            params,
            scope,
            strict_metadata=strict_metadata,
        )
    sql += " ORDER BY chunk_score LIMIT ?"
    row_limit = max(clamp_limit(limit, upper=30) * 8, 40)
    params.append(min(row_limit, 240))

    try:
        with sqlite3.connect(GAMEPATH_DB) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []

    grouped: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for row in rows:
        chunk_content = str(row["chunk_content"] or "")
        chunk_haystack = "\n".join(
            [
                chunk_content,
                str(row["heading"] or ""),
            ]
        )
        coverage = gamepath_term_coverage(query, chunk_haystack)
        if coverage <= 0.0:
            continue
        haystack = "\n".join(
            [
                str(row["title"] or ""),
                str(row["question"] or ""),
                chunk_content,
                str(row["tags"] or ""),
                str(row["heading"] or ""),
            ]
        )
        coverage = max(coverage, gamepath_term_coverage(query, haystack))

        entry_id = int(row["entry_id"])
        if entry_id not in grouped:
            grouped[entry_id] = {
                "id": entry_id,
                "game_id": row["game_id"],
                "title": row["title"],
                "question": row["question"],
                "answer_summary": row["answer_summary"],
                "snippet": make_snippet(chunk_content or row["answer_summary"], query),
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
                "match_coverage": round(coverage, 3),
                "rag_lite": True,
                "rag_chunk_hits": [],
            }
            order.append(entry_id)
        item = grouped[entry_id]
        item["match_coverage"] = round(max(float(item.get("match_coverage") or 0.0), coverage), 3)
        item["score"] = min(float(item.get("score") or row["chunk_score"]), float(row["chunk_score"]))
        item["rag_chunk_hits"].append(
            {
                "chunk_id": int(row["chunk_id"]),
                "chunk_index": int(row["chunk_index"]),
                "heading": row["heading"],
                "content": chunk_content,
                "char_count": int(row["char_count"] or len(chunk_content)),
                "score": float(row["chunk_score"]),
                "coverage": round(coverage, 3),
            }
        )

    results: list[dict[str, Any]] = []
    for entry_id in order:
        item = grouped[entry_id]
        chunks = list(item.pop("rag_chunk_hits", []))
        chunks.sort(key=lambda chunk: (-float(chunk.get("coverage") or 0.0), float(chunk.get("score") or 0.0)))
        excerpts: list[str] = []
        total_chars = 0
        best_coverage = float(chunks[0].get("coverage") or 0.0) if chunks else 0.0
        min_coverage = max(0.25, best_coverage * 0.62) if best_coverage else 0.0
        for chunk in chunks[:5]:
            if excerpts and float(chunk.get("coverage") or 0.0) < min_coverage:
                continue
            heading = str(chunk.get("heading") or "").strip()
            snippet = make_snippet(str(chunk.get("content") or ""), query, max_len=GAMEPATH_PASSAGE_MAX_CHARS)
            if heading and heading.lower() not in snippet[:160].lower():
                snippet = f"### {heading}\n{snippet}"
            next_total = total_chars + len(snippet) + (5 if excerpts else 0)
            if excerpts and next_total > GAMEPATH_CONTEXT_MAX_CHARS:
                break
            excerpts.append(snippet)
            total_chars = next_total
            if len(excerpts) >= 3:
                break
        relevant_excerpt = "\n\n---\n\n".join(excerpts).strip()
        answer_len = len(str(item.get("answer_summary") or ""))
        item["relevant_excerpt"] = relevant_excerpt
        item["context_char_count"] = len(relevant_excerpt)
        item["source_char_count"] = answer_len
        item["passage_count"] = max(len(chunks), 1)
        item["rag_chunk_count"] = len(chunks)
        item["rag_chunk_ids"] = [int(chunk["chunk_id"]) for chunk in chunks[:5]]
        item["large_entry"] = gamepath_text_is_large(str(item.get("answer_summary") or "")) or len(chunks) > 1
        results.append(item)
        if len(results) >= clamp_limit(limit, upper=30):
            break
    return rank_gamepath_results(results, scope, query)


def search_gamepath_sync(
    query: str,
    game_id: Optional[str],
    limit: int = 5,
    *,
    tags: Any = None,
    spoiler_level: str = "low",
    use_rag_lite: bool = True,
    version: Any = None,
    area: Any = None,
    entity_type: Any = None,
    entity_name: Any = None,
    strict_metadata: bool = False,
) -> list[dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []
    ensure_gamepath_db()
    scope = normalize_gamepath_scope(
        query,
        tags,
        version=version,
        area=area,
        entity_type=entity_type,
        entity_name=entity_name,
    )
    if use_rag_lite:
        chunk_results = search_gamepath_chunks_sync(
            query,
            game_id,
            limit,
            tags=tags,
            spoiler_level=spoiler_level,
            version=scope.get("version"),
            area=scope.get("area"),
            entity_type=scope.get("entity_type"),
            entity_name=scope.get("entity_name"),
            strict_metadata=strict_metadata,
        )
        if chunk_results and strict_metadata:
            return chunk_results
        if not strict_metadata:
            broad_chunk_results = search_gamepath_chunks_sync(
                query,
                game_id,
                limit,
                tags=tags,
                spoiler_level=spoiler_level,
                version=scope.get("version"),
                area=scope.get("area"),
                entity_type=scope.get("entity_type"),
                entity_name=scope.get("entity_name"),
                strict_metadata=False,
                apply_scope_filters=False,
            )
            if chunk_results or broad_chunk_results:
                return merge_gamepath_results(
                    chunk_results,
                    broad_chunk_results,
                    scope,
                    query,
                    limit,
                )
        if chunk_results:
            return chunk_results
    match = fts_query(gamepath_query_text(query))
    if not match:
        return []
    normalized_game_id = normalize_game_id(game_id)
    sql = (
        "SELECT e.*, bm25(gamepath_fts) AS score FROM gamepath_fts "
        "JOIN gamepath_entries e ON e.id = gamepath_fts.entry_id "
        "WHERE gamepath_fts MATCH ? AND e.spoiler_rank <= ?"
    )
    params: list[Any] = [match, spoiler_rank(spoiler_level)]
    if normalized_game_id:
        sql += " AND e.game_id IN (?, 'global')"
        params.append(normalized_game_id)
    sql, params = append_gamepath_scope_filters(
        sql,
        params,
        scope,
        strict_metadata=strict_metadata,
    )
    sql += " ORDER BY score LIMIT ?"
    params.append(clamp_limit(limit, upper=30))
    try:
        with sqlite3.connect(GAMEPATH_DB) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []

    results: list[dict[str, Any]] = []
    for row in rows:
        haystack = "\n".join(
            str(row[key] or "")
            for key in ("title", "question", "answer_summary", "tags")
        )
        coverage = gamepath_term_coverage(query, haystack)
        if coverage <= 0.0:
            continue
        item = {
            "id": int(row["id"]),
            "game_id": row["game_id"],
            "title": row["title"],
            "question": row["question"],
            "answer_summary": row["answer_summary"],
            "snippet": make_snippet(row["answer_summary"], query),
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
            "score": float(row["score"]),
            "match_coverage": round(coverage, 3),
        }
        item.update(build_gamepath_relevant_context(query, item))
        results.append(item)
    return rank_gamepath_results(results, scope, query)


def recent_gamepath_sync(game_id: Optional[str] = None, limit: int = 10) -> list[dict[str, Any]]:
    if not GAMEPATH_DB.exists():
        return []
    normalized_game_id = normalize_game_id(game_id)
    sql = (
        "SELECT id, game_id, title, question, answer_summary, markdown_path, tags, "
        "version, area, entity_type, entity_name, spoiler_level, source_type, source_quality, "
        "agent_used, trust_state, dispute_count, last_feedback, last_feedback_at, "
        "created_at, updated_at FROM gamepath_entries"
    )
    params: list[Any] = []
    if normalized_game_id:
        sql += " WHERE game_id IN (?, 'global')"
        params.append(normalized_game_id)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(clamp_limit(limit, upper=30))
    with sqlite3.connect(GAMEPATH_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    return [
        {
            "id": int(row["id"]),
            "game_id": row["game_id"],
            "title": row["title"],
            "question": row["question"],
            "answer_summary": row["answer_summary"],
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
        }
        for row in rows
    ]


def should_use_gamepath(prompt: str, guide_requested: bool) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    if GAMEPATH_NEGATED_GUIDE_RE.search(text):
        return False
    if GAMEPATH_SAVE_ONLY_RE.search(text):
        return False
    has_gamepath_intent = bool(guide_requested or GAMEPATH_STORE_INTENT_RE.search(text))
    if not has_gamepath_intent:
        return False
    if GAMEPATH_UI_SKIP_RE.search(text) and not GUIDE_INTENT_RE.search(text):
        return False
    return True


def backend_hard_skips_gamepath(prompt: str) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return True
    if GAMEPATH_WEB_INTENT_RE.search(text) or GAMEPATH_VERSION_COMPARE_RE.search(text):
        return False
    if GAMEPATH_NEGATED_GUIDE_RE.search(text):
        return True
    if GAMEPATH_SAVE_ONLY_RE.search(text):
        return True
    return bool(GAMEPATH_UI_SKIP_RE.search(text))


def local_router_cache_key(kind: str, game_id: Optional[str], text: str, extra: str = "") -> str:
    raw = "\n".join(
        [
            LOCAL_ROUTER_MODEL,
            kind,
            normalize_game_id(game_id) or "",
            str(text or "").strip(),
            extra,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


def local_router_cache_get(key: str) -> Optional[dict[str, Any]]:
    if LOCAL_ROUTER_DECISION_CACHE_TTL_SECONDS <= 0:
        return None
    cached = local_router_decision_cache.get(key)
    if not cached:
        return None
    created_at, value = cached
    if time.time() - created_at > LOCAL_ROUTER_DECISION_CACHE_TTL_SECONDS:
        local_router_decision_cache.pop(key, None)
        return None
    result = dict(value)
    result["cache_hit"] = True
    return result


def local_router_cache_set(key: str, value: dict[str, Any]) -> None:
    if LOCAL_ROUTER_DECISION_CACHE_TTL_SECONDS <= 0:
        return
    if len(local_router_decision_cache) > 256:
        oldest = sorted(local_router_decision_cache.items(), key=lambda item: item[1][0])[:64]
        for old_key, _ in oldest:
            local_router_decision_cache.pop(old_key, None)
    local_router_decision_cache[key] = (time.time(), dict(value))


def normalize_router_tags(value: Any) -> list[str]:
    if value is None:
        return []
    raw_items = value
    if isinstance(value, str):
        raw_items = re.split(r"[,，、\s]+", value)
    if not isinstance(raw_items, list):
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        tag = re.sub(r"[^0-9A-Za-z_\-\u3400-\u9fff]+", "", str(item or "").strip().lower())
        if not tag or len(tag) > 24 or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
        if len(tags) >= 6:
            break
    return tags


def normalize_router_spoiler(value: Any) -> str:
    clean = str(value or "low").strip().lower()
    if clean in SPOILER_RANKS and SPOILER_RANKS[clean] <= SPOILER_RANKS["medium"]:
        return clean
    return "low"


def gamepath_explicit_spoiler_fallback_level(prompt: str, query: str, current_level: str) -> str:
    current = normalize_spoiler_level(current_level)
    text = f"{prompt or ''} {query or ''}"
    if not text.strip():
        return current
    if re.search(r"(結局|真結局|壞結局|好結局|全結局|最終謎題|完整解謎|final|ending|true ending)", text, re.I):
        target = "high"
    elif re.search(r"(弱點|密碼|答案|解法|怎麼解|怎麼打|打法|保險箱|代碼|謎題|boss|weakness|password|code|solution)", text, re.I):
        target = "medium"
    else:
        return current
    return target if spoiler_rank(target) > spoiler_rank(current) else current


GAMEPATH_ADAPTIVE_FOLLOWUP_RE = re.compile(
    r"(還是|仍然|依然|可是|但是|不過|我覺得|我是說|換個|更簡單|更安全|保守|省資源|"
    r"很難|太難|有點難|有難度|打不過|打不動|一直死|不行|做不到|卡住|難打|不好打)",
    re.IGNORECASE,
)
GAMEPATH_COMBAT_FOLLOWUP_RE = re.compile(
    r"(怪物|敵人|喪屍|殭屍|boss|主廚|暴君|泰坦|舔食者|怎麼打|打法|弱點|打不過|難打|戰鬥)",
    re.IGNORECASE,
)
GAMEPATH_FOLLOWUP_ANCHOR_RE = re.compile(
    r"(廚房|餐廳|旅館|療養院|警局|方舟|地下|東側|西側|一樓|二樓|三樓|房|室|門|保險箱|"
    r"鑰匙|門禁卡|怪物|敵人|喪屍|殭屍|boss|主廚|暴君|泰坦|舔食者|謎題|密碼|弱點|答案|怎麼|哪)",
    re.IGNORECASE,
)


def is_gamepath_adaptive_followup(prompt: str) -> bool:
    return bool(GAMEPATH_ADAPTIVE_FOLLOWUP_RE.search(str(prompt or "")))


def is_gamepath_vague_adaptive_followup(prompt: str) -> bool:
    text = re.sub(r"\s+", "", str(prompt or ""))
    if not text or not is_gamepath_adaptive_followup(text):
        return False
    return len(text) <= 18 and not GAMEPATH_FOLLOWUP_ANCHOR_RE.search(text)


def is_gamepath_tactical_reframe(prompt: str) -> bool:
    text = str(prompt or "")
    if is_gamepath_adaptive_followup(text):
        return True
    return bool(
        GAMEPATH_COMBAT_FOLLOWUP_RE.search(text)
        and re.search(r"(怎麼打|打法|打不過|打不動|難打|有難度|戰鬥|更安全|保守|省資源)", text, re.IGNORECASE)
    )


def augment_adaptive_gamepath_query_variants(prompt: str, query: str, variants: list[str]) -> list[str]:
    text = f"{prompt or ''} {query or ''}"
    output = list(variants or [])
    extras: list[str] = []
    if "廚房" in text and GAMEPATH_COMBAT_FOLLOWUP_RE.search(text):
        extras.extend(
            [
                "主廚喪屍 廚房 怎麼打",
                "廚房 主廚喪屍 食品儲藏室鑰匙",
                "療養院 一樓西側 遊樂室 主廚喪屍 打法",
            ]
        )
    for item in [query, prompt, *extras]:
        clean = re.sub(r"\s+", " ", str(item or "").strip())
        if clean and clean not in output:
            output.append(clean)
    return output[:8]


def should_search_gamepath_after_router_failure(prompt: str, guide_requested: bool) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    if GAMEPATH_WEB_INTENT_RE.search(text) or GAMEPATH_VERSION_COMPARE_RE.search(text):
        return False
    if backend_hard_skips_gamepath(text):
        return False
    if should_use_gamepath(text, guide_requested):
        return True
    return bool(LOCAL_ROUTER_GAMEPATH_CANDIDATE_RE.search(text))


def fallback_gamepath_decision(
    prompt: str,
    game_id: Optional[str],
    guide_requested: bool,
    reason: str,
) -> dict[str, Any]:
    should_search = should_use_gamepath(prompt, guide_requested)
    if not should_search and str(reason or "").startswith("router_failed:"):
        should_search = should_search_gamepath_after_router_failure(prompt, guide_requested)
    return {
        "used": False,
        "search_gamepath": should_search,
        "query": str(prompt or "").strip(),
        "tags": [],
        "spoiler_level": "low",
        "confidence": "fallback" if should_search else "none",
        "reason": reason,
        "model": None,
        "intent_route": "gamepath_query" if should_search else "general_chat",
    }


def web_preferred_gamepath_decision(prompt: str, game_id: Optional[str]) -> dict[str, Any]:
    return {
        "used": False,
        "search_gamepath": False,
        "query": str(prompt or "").strip(),
        "tags": [],
        "spoiler_level": "low",
        "confidence": "web",
        "reason": "web_current_intent",
        "model": None,
        "prefer_hermes_agent": True,
    }


def should_ask_local_router_for_gamepath(prompt: str, guide_requested: bool) -> bool:
    if not LOCAL_ROUTER_ENABLED or not LOCAL_ROUTER_GAMEPATH_GATE:
        return False
    text = str(prompt or "").strip()
    if not text or len(text) > LOCAL_ROUTER_GAMEPATH_MAX_CHARS:
        return False
    if detect_fact_lookup_key(text):
        return False
    if backend_hard_skips_gamepath(text):
        return False
    if should_use_gamepath(text, guide_requested):
        return False
    return bool(LOCAL_ROUTER_GAMEPATH_CANDIDATE_RE.search(text))


LOCAL_ROUTER_INTENT_ROUTES = {
    "gamepath_query",
    "hermes_web",
    "general_chat",
    "ui_command",
    "task_memory",
    "screenshot_gamepath_query",
    "screenshot_visual",
    "screenshot_hud",
    "skip",
    "clarify",
}


def normalize_local_router_intent_route(value: Any, parsed: Optional[dict[str, Any]] = None) -> str:
    route = re.sub(r"[^a-z0-9_\-]+", "_", str(value or "").strip().lower()).strip("_-")
    aliases = {
        "gamepath": "gamepath_query",
        "guide": "gamepath_query",
        "guide_query": "gamepath_query",
        "local_guide": "gamepath_query",
        "web": "hermes_web",
        "web_search": "hermes_web",
        "hermes": "hermes_web",
        "hermes_agent": "hermes_web",
        "chat": "general_chat",
        "general": "general_chat",
        "casual": "general_chat",
        "ui": "ui_command",
        "system": "ui_command",
        "app_command": "ui_command",
        "memory": "task_memory",
        "task": "task_memory",
        "screenshot": "screenshot_visual",
        "screen": "screenshot_visual",
        "vision": "screenshot_visual",
        "visual": "screenshot_visual",
        "inspect": "screenshot_visual",
        "describe": "screenshot_visual",
        "screenshot_guide": "screenshot_gamepath_query",
        "screenshot_gamepath": "screenshot_gamepath_query",
        "screenshot_gamepath_query": "screenshot_gamepath_query",
        "hud": "screenshot_hud",
        "mark": "screenshot_hud",
        "circle": "screenshot_hud",
        "overlay": "screenshot_hud",
        "none": "skip",
    }
    route = aliases.get(route, route)
    if route in LOCAL_ROUTER_INTENT_ROUTES:
        return route
    if parsed:
        if bool(parsed.get("search_gamepath", parsed.get("s", parsed.get("search", False)))):
            return "gamepath_query"
        if bool(parsed.get("prefer_hermes_agent") or parsed.get("web")):
            return "hermes_web"
    return "general_chat"


def local_router_gamepath_decision(
    prompt: str,
    game_id: Optional[str],
    guide_requested: bool = False,
) -> dict[str, Any]:
    text = str(prompt or "").strip()
    if not LOCAL_ROUTER_ENABLED or not LOCAL_ROUTER_GAMEPATH_GATE:
        if GAMEPATH_WEB_INTENT_RE.search(prompt or "") or GAMEPATH_VERSION_COMPARE_RE.search(prompt or ""):
            return web_preferred_gamepath_decision(prompt, game_id)
        if backend_hard_skips_gamepath(prompt):
            return fallback_gamepath_decision(prompt, game_id, False, "backend_hard_skip")
        return fallback_gamepath_decision(prompt, game_id, guide_requested, "router_disabled")
    if not text or len(text) > LOCAL_ROUTER_GAMEPATH_MAX_CHARS:
        return fallback_gamepath_decision(prompt, game_id, guide_requested, "router_input_out_of_range")
    if not LOCAL_ROUTER_ALWAYS_ROUTE:
        if GAMEPATH_WEB_INTENT_RE.search(prompt or "") or GAMEPATH_VERSION_COMPARE_RE.search(prompt or ""):
            return web_preferred_gamepath_decision(prompt, game_id)
        if backend_hard_skips_gamepath(prompt):
            return fallback_gamepath_decision(prompt, game_id, False, "backend_hard_skip")
        if should_use_gamepath(prompt, guide_requested):
            return fallback_gamepath_decision(prompt, game_id, True, "explicit_guide_intent")
        if not should_ask_local_router_for_gamepath(prompt, guide_requested):
            return fallback_gamepath_decision(prompt, game_id, guide_requested, "router_unavailable")

    cache_kind = (
        f"user-intent-route:{LOCAL_ROUTER_INTENT_CACHE_VERSION}"
        if LOCAL_ROUTER_ALWAYS_ROUTE
        else f"gamepath-route:{LOCAL_ROUTER_INTENT_CACHE_VERSION}"
    )
    cache_key = local_router_cache_key(cache_kind, game_id, prompt)
    cached = local_router_cache_get(cache_key)
    if cached:
        return cached

    messages = [
        {
            "role": "system",
            "content": (
                "You are the Game Companion user-intent router. JSON only, no prose. "
                "Schema: {\"route\":\"gamepath_query|hermes_web|general_chat|ui_command|task_memory|screenshot_gamepath_query|screenshot_visual|screenshot_hud|skip|clarify\","
                "\"q\":\"short search query\",\"t\":[\"item|quest|boss|map|route|puzzle|mechanic|enemy|npc|character|material|location\"],"
                "\"sp\":\"none|low|medium\",\"c\":\"low|medium|high\",\"ui_action\":\"optional_action\",\"reason\":\"short\"}. "
                "Use gamepath_query for game guides: item use, quest, location, boss, puzzle, route, enemy/NPC/character names, stuck/next-step help. "
                "Use hermes_web for web/current/latest/patch/version difference/community/speedrun/meta or when user explicitly asks to search online. "
                "Use ui_command for app/window/system controls: open/close GamePath/Task/Game Search, opacity, voice mode, content protection, restart, screenshot screen selection, virtual cursor. "
                "Use task_memory for player objectives, inventory/task records, or remembering what the player has. "
                "Use screenshot_visual when the player asks to look at, describe, inspect, or understand the current screen/image without asking for visible marks. "
                "Use screenshot_gamepath_query when the player asks what to do next, how to pass, where to go, how to fight/solve/use something based on the current screen. "
                "If the player asks to look at the current screen/scene and decide the next action, choose screenshot_gamepath_query, not gamepath_query. "
                "Use screenshot_hud only when the player asks for visible marking/circling/pointing/arrow/HUD on the current screen. "
                "Use general_chat for normal conversation or technical explanation. Use skip for negated requests like 'do not search guide'. "
                "If player_message is Chinese, keep q in Chinese. Do not translate Chinese to English. Qwen only routes; backend executes allowlisted actions."
            ),
        },
        {
            "role": "user",
            "content": (
                f"game_id: {game_id or 'unknown'}\n"
                f"player_message: {text[:LOCAL_ROUTER_GAMEPATH_MAX_CHARS]}"
            ),
        },
    ]
    started = time.perf_counter()
    output = call_local_router_once(messages, max_tokens=48)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    try:
        parsed = extract_json_object(output)
    except Exception:
        lowered = output.strip().lower()
        parsed = {"route": "gamepath_query" if "gamepath" in lowered or ("true" in lowered and "false" not in lowered) else "general_chat"}
    route = normalize_local_router_intent_route(parsed.get("route", parsed.get("r")), parsed)
    if route == "gamepath_query" and live_state_prompt_needs_screen(prompt):
        route = "screenshot_gamepath_query"
        parsed["route"] = route
        parsed["reason"] = parsed.get("reason") or "current_screen_help_guard"
    should_search = route == "gamepath_query"
    prefer_hermes_agent = route == "hermes_web"
    query = str(parsed.get("query") or parsed.get("q") or prompt or "").strip()
    if not query:
        query = str(prompt or "").strip()
    query_variants = [str(prompt or "").strip()]
    if query and query not in query_variants:
        query_variants.append(query)
    if has_cjk_text(prompt) and query and not has_cjk_text(query):
        query = str(prompt or "").strip()
    result = {
        "used": True,
        "search_gamepath": should_search,
        "query": query[:LOCAL_ROUTER_GAMEPATH_MAX_CHARS],
        "query_variants": [item[:LOCAL_ROUTER_GAMEPATH_MAX_CHARS] for item in query_variants if item],
        "tags": normalize_router_tags(parsed.get("tags", parsed.get("t"))),
        "spoiler_level": normalize_router_spoiler(parsed.get("spoiler_level", parsed.get("sp"))),
        "confidence": str(parsed.get("confidence") or parsed.get("c") or ("medium" if should_search else "low")).strip().lower()[:16],
        "reason": "qwen_user_intent_route" if LOCAL_ROUTER_ALWAYS_ROUTE else "qwen_semantic_route",
        "intent_route": route,
        "ui_action": re.sub(r"[^a-z0-9_\-]+", "_", str(parsed.get("ui_action") or "").strip().lower()).strip("_-")[:48],
        "prefer_hermes_agent": prefer_hermes_agent,
        "raw_reason": str(parsed.get("reason") or "").strip()[:160],
        "model": LOCAL_ROUTER_MODEL,
        "latency_ms": latency_ms,
        "cache_hit": False,
    }
    local_router_cache_set(cache_key, result)
    return result


SCREENSHOT_INTENT_ROUTES = {
    "screenshot_gamepath_query",
    "screenshot_hud",
    "screenshot_visual",
    "general_chat",
    "hermes_web",
}


def normalize_screenshot_intent_route(value: Any, parsed: Optional[dict[str, Any]] = None) -> str:
    route = re.sub(r"[^a-z0-9_\-]+", "_", str(value or "").strip().lower()).strip("_-")
    aliases = {
        "gamepath": "screenshot_gamepath_query",
        "gamepath_query": "screenshot_gamepath_query",
        "guide": "screenshot_gamepath_query",
        "guide_query": "screenshot_gamepath_query",
        "local_guide": "screenshot_gamepath_query",
        "walkthrough": "screenshot_gamepath_query",
        "screenshot_guide": "screenshot_gamepath_query",
        "screenshot_guide_query": "screenshot_gamepath_query",
        "hud": "screenshot_hud",
        "hud_overlay": "screenshot_hud",
        "overlay": "screenshot_hud",
        "mark": "screenshot_hud",
        "circle": "screenshot_hud",
        "visual": "screenshot_visual",
        "vision": "screenshot_visual",
        "visual_inspect": "screenshot_visual",
        "inspect": "screenshot_visual",
        "describe": "screenshot_visual",
        "chat": "general_chat",
        "general": "general_chat",
        "web": "hermes_web",
        "web_search": "hermes_web",
        "hermes": "hermes_web",
        "hermes_agent": "hermes_web",
    }
    route = aliases.get(route, route)
    if route in SCREENSHOT_INTENT_ROUTES:
        return route
    if parsed:
        if bool(parsed.get("needs_hud") or parsed.get("hud")):
            return "screenshot_hud"
        if bool(parsed.get("search_gamepath") or parsed.get("guide")):
            return "screenshot_gamepath_query"
        if bool(parsed.get("prefer_hermes_agent") or parsed.get("web")):
            return "hermes_web"
    return "screenshot_visual"


def fallback_screenshot_intent_decision(
    prompt: str,
    game_id: Optional[str],
    guide_requested: bool,
    reason: str,
) -> dict[str, Any]:
    route = "screenshot_visual"
    if should_use_overlay(prompt, "image"):
        route = "screenshot_hud"
    elif should_use_gamepath(prompt, guide_requested):
        route = "screenshot_gamepath_query"
    query = str(prompt or "").strip()
    return {
        "used": False,
        "route_source": "vision",
        "intent_route": route,
        "search_gamepath": route == "screenshot_gamepath_query",
        "query": query,
        "query_variants": [query] if query else [],
        "tags": [],
        "spoiler_level": "low",
        "confidence": "fallback" if route == "screenshot_gamepath_query" else "low",
        "reason": reason,
        "raw_reason": reason,
        "model": None,
        "latency_ms": 0.0,
        "cache_hit": False,
    }


def screenshot_intent_messages(
    prompt: str,
    game_id: Optional[str],
    image_base64: str,
    ocr_text: str = "",
) -> list[dict[str, Any]]:
    user_text = (
        "Decide how this screenshot request should be handled. JSON only, no Markdown.\n"
        "Schema: {\"route\":\"screenshot_gamepath_query|screenshot_hud|screenshot_visual|general_chat|hermes_web\","
        "\"q\":\"short GamePath search query in the player's language\","
        "\"t\":[\"item|quest|boss|map|route|puzzle|mechanic|enemy|npc|character|material|location\"],"
        "\"sp\":\"none|low|medium\",\"c\":\"low|medium|high\","
        "\"scene\":\"visible area or level if recognizable\","
        "\"objects\":[\"important visible objects/enemies/items\"],"
        "\"reason\":\"short reason\"}.\n"
        "Infer intent from both player text and screenshot. Do not depend on fixed keywords.\n"
        "Use screenshot_gamepath_query when the player wants help deciding what to do, how to pass, solve, fight, use an item, open a door, find a route, understand an objective, or recover from being stuck.\n"
        "Use screenshot_hud only when the player wants visible marking/circling/pointing on the current screen.\n"
        "Use screenshot_visual when the player mainly asks what is visible or asks for plain image description.\n"
        "Use hermes_web when the player explicitly wants online/latest/community/version info.\n"
        "For q, combine visible scene/object clues with the player's question so SQLite can search local guides. Keep q concise and Traditional Chinese when the player uses Chinese.\n"
        f"game_id: {game_id or 'unknown'}\n"
        f"player_message: {status_text(prompt, 260)}"
    )
    if ocr_text:
        user_text += f"\nvisible_text_hint: {status_text(ocr_text, 600)}"
    return [
        {
            "role": "system",
            "content": (
                "You are a screenshot intent router for an in-game companion. "
                "You inspect the image and the player's message, then choose the route. "
                "Return exactly one JSON object."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_base64)}},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def screenshot_intent_decision(
    prompt: str,
    game_id: Optional[str],
    image_base64: str,
    guide_requested: bool = False,
    ocr_text: str = "",
) -> dict[str, Any]:
    if not image_base64:
        return fallback_screenshot_intent_decision(prompt, game_id, guide_requested, "no_image")
    messages = screenshot_intent_messages(prompt, game_id, image_base64, ocr_text)
    started = time.perf_counter()
    if CHAT_BACKEND == "hermes" and HERMES_USE_CONFIG_MODEL:
        output = call_hermes_messages(messages, image_file=LATEST_VISION_INPUT, max_tokens=220)
        model_name = "hermes_vision"
    else:
        output = call_llama_once(messages, max_tokens=220)
        model_name = MODEL_ALIAS
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    try:
        parsed = extract_json_object(output)
    except Exception:
        parsed = {"route": "screenshot_visual", "reason": "invalid_json"}

    route = normalize_screenshot_intent_route(parsed.get("route", parsed.get("r")), parsed)
    query = str(parsed.get("query") or parsed.get("q") or "").strip()
    scene = str(parsed.get("scene") or "").strip()
    objects_raw = parsed.get("objects") or parsed.get("visible_objects") or []
    objects: list[str] = []
    if isinstance(objects_raw, str):
        objects_raw = re.split(r"[,，、\n]+", objects_raw)
    if isinstance(objects_raw, list):
        for item in objects_raw:
            clean = re.sub(r"\s+", " ", str(item or "").strip())
            if clean and clean not in objects:
                objects.append(clean[:60])
            if len(objects) >= 6:
                break
    if not query:
        query = " ".join([scene, *objects[:4], str(prompt or "").strip()]).strip()
    if has_cjk_text(prompt) and query and not has_cjk_text(query):
        query = " ".join([scene, *objects[:3], str(prompt or "").strip()]).strip()
    query = status_text(query or prompt, LOCAL_ROUTER_GAMEPATH_MAX_CHARS)
    variants = [str(prompt or "").strip(), query, scene, " ".join(objects[:4])]
    variants = [status_text(item, LOCAL_ROUTER_GAMEPATH_MAX_CHARS) for item in variants if item]
    deduped_variants: list[str] = []
    for item in variants:
        if item and item not in deduped_variants:
            deduped_variants.append(item)

    return {
        "used": True,
        "route_source": "vision",
        "search_gamepath": route == "screenshot_gamepath_query",
        "query": query,
        "query_variants": deduped_variants,
        "tags": normalize_router_tags(parsed.get("tags", parsed.get("t"))),
        "spoiler_level": normalize_router_spoiler(parsed.get("spoiler_level", parsed.get("sp"))),
        "confidence": str(parsed.get("confidence") or parsed.get("c") or "medium").strip().lower()[:16],
        "reason": "vision_screenshot_intent_route",
        "intent_route": route,
        "prefer_hermes_agent": route == "hermes_web",
        "needs_hud": route == "screenshot_hud",
        "scene": status_text(scene, 120),
        "objects": objects,
        "raw_reason": str(parsed.get("reason") or "").strip()[:180],
        "model": model_name,
        "latency_ms": latency_ms,
        "cache_hit": False,
    }


def screenshot_gamepath_prompt(prompt: str, decision: Optional[dict[str, Any]]) -> str:
    if not decision:
        return prompt
    parts = [str(prompt or "").strip()]
    query = str(decision.get("query") or "").strip()
    if query and query not in parts[0]:
        parts.append(f"GamePath 搜尋意圖：{query}")
    return "\n".join(item for item in parts if item)


def gamepath_core_terms(text: str, max_terms: int = 14) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for term in search_terms(gamepath_query_text(text), max_terms=48):
        lowered = term.lower()
        if lowered in GAMEPATH_GENERIC_TERMS or term in GAMEPATH_GENERIC_TERMS:
            continue
        if len(term) < 2:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def term_overlap_ratio(terms: list[str], text: str) -> float:
    if not terms:
        return 0.0
    haystack = str(text or "").lower()
    matched = sum(1 for term in terms if term.lower() in haystack)
    return min(1.0, matched / max(1, min(len(terms), 8)))


def parse_gamepath_time(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", text)
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def gamepath_recency_score(updated_at: Any) -> float:
    parsed = parse_gamepath_time(updated_at)
    if not parsed:
        return 0.02
    age_days = max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 86400)
    if age_days <= 30:
        return 0.06
    if age_days <= 180:
        return 0.04
    if age_days <= 730:
        return 0.02
    return 0.0


def score_gamepath_result(query: str, game_id: Optional[str], result: dict[str, Any]) -> dict[str, Any]:
    normalized_game_id = normalize_game_id(game_id)
    result_game_id = normalize_game_id(result.get("game_id"))
    haystack = "\n".join(
        str(result.get(key) or "")
        for key in ("title", "question", "answer_summary", "relevant_excerpt", "tags")
    )
    core_terms = gamepath_core_terms(query)
    core_overlap = term_overlap_ratio(core_terms, haystack)
    coverage = max(0.0, min(1.0, float(result.get("match_coverage") or 0.0)))
    answer_len = len(str(result.get("answer_summary") or "").strip())
    context_len = len(str(result.get("relevant_excerpt") or result.get("snippet") or "").strip())
    large_entry = bool(result.get("large_entry")) or answer_len > GAMEPATH_DIRECT_MAX_CHARS
    result_spoiler_rank = spoiler_rank(str(result.get("spoiler_level") or "low"))

    game_score = 0.12
    game_reason = "no_selected_game"
    if normalized_game_id:
        if result_game_id == normalized_game_id:
            game_score = 0.22
            game_reason = "same_game"
        elif result_game_id == "global":
            game_score = 0.12
            game_reason = "global_entry"
        else:
            game_score = 0.0
            game_reason = "different_game"

    answer_score = 0.0
    if answer_len >= 180:
        answer_score = 0.08
    elif answer_len >= 80:
        answer_score = 0.07
    elif answer_len >= 40:
        answer_score = 0.05
    elif answer_len >= 20:
        answer_score = 0.03

    spoiler_score = 0.05 if result_spoiler_rank <= spoiler_rank("low") else 0.02
    source_quality = max(0.0, min(float(result.get("source_quality") or 0.5), 1.0))
    source_score = (0.03 if result.get("agent_used") else 0.02) + source_quality * 0.16
    recency_score = gamepath_recency_score(result.get("updated_at") or result.get("created_at"))
    trust_state = str(result.get("trust_state") or "unverified").strip().lower()
    dispute_count = int(result.get("dispute_count") or 0)
    trust_score = {
        "verified": 0.08,
        "unverified": 0.02,
        "needs_review": -0.18,
        "disputed": -0.26,
        "deprecated": -0.4,
    }.get(trust_state, 0.0)
    dispute_penalty = min(0.18, max(0, dispute_count) * 0.06)
    core_score = 0.24 * core_overlap
    coverage_score = 0.23 * coverage
    metadata_score = max(-0.12, min(0.16, float(result.get("metadata_match_score") or 0.0) * 0.25))
    semantic_hint_score = 0.0
    semantic_hints = gamepath_semantic_hint_text(query)
    if semantic_hints:
        semantic_terms = gamepath_core_terms(semantic_hints, max_terms=12)
        semantic_hint_score = 0.12 * term_overlap_ratio(semantic_terms, haystack)
    area_overlap_score = 0.0
    actual_area = str(result.get("area") or "").strip()
    if actual_area and has_cjk_text(actual_area):
        compact_query = re.sub(r"\s+", "", query)
        area_terms = [term for term in search_terms(actual_area, max_terms=12) if has_cjk_text(term)]
        if any(term and term in compact_query for term in area_terms):
            area_overlap_score = 0.08

    score = min(
        1.0,
        max(
            0.0,
            game_score
            + core_score
            + coverage_score
            + metadata_score
            + semantic_hint_score
            + area_overlap_score
            + answer_score
            + spoiler_score
            + source_score
            + recency_score
            + trust_score
            - dispute_penalty,
        ),
    )
    reasons = [
        game_reason,
        f"core_overlap:{core_overlap:.2f}",
        f"coverage:{coverage:.2f}",
        f"metadata:{metadata_score:.2f}",
        f"semantic_hint:{semantic_hint_score:.2f}",
        f"area_overlap:{area_overlap_score:.2f}",
        f"answer_len:{answer_len}",
        f"context_len:{context_len}",
        f"large_entry:{int(large_entry)}",
        f"spoiler:{result.get('spoiler_level') or 'low'}",
        f"source_quality:{source_quality:.2f}",
        f"trust:{trust_state}",
        f"disputes:{dispute_count}",
    ]
    return {
        "score": round(score, 3),
        "core_overlap": round(core_overlap, 3),
        "coverage": round(coverage, 3),
        "answer_len": answer_len,
        "context_len": context_len,
        "large_entry": large_entry,
        "reasons": reasons,
    }


def evaluate_gamepath_retrieval(
    query: str,
    game_id: Optional[str],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not results:
        return {
            "confidence": "miss",
            "score": 0.0,
            "gap": 0.0,
            "reason": "no_results",
            "results": [],
        }

    evaluated_results: list[dict[str, Any]] = []
    for item in results:
        scored = dict(item)
        evaluation = score_gamepath_result(query, game_id, scored)
        scored["retrieval_score"] = evaluation["score"]
        scored["retrieval_reasons"] = evaluation["reasons"]
        scored["core_overlap"] = evaluation["core_overlap"]
        scored["answer_len"] = evaluation["answer_len"]
        scored["context_len"] = evaluation["context_len"]
        scored["large_entry"] = evaluation["large_entry"]
        evaluated_results.append(scored)
    evaluated_results.sort(
        key=lambda item: (
            float(item.get("retrieval_score") or 0.0),
            float(item.get("core_overlap") or 0.0),
            float(item.get("match_coverage") or 0.0),
            float(item.get("metadata_match_score") or 0.0),
        ),
        reverse=True,
    )

    top = evaluated_results[0]
    top_score = float(top.get("retrieval_score") or 0.0)
    second_score = float(evaluated_results[1].get("retrieval_score") or 0.0) if len(evaluated_results) > 1 else 0.0
    gap = top_score - second_score
    answer_len = int(top.get("answer_len") or 0)
    context_len = int(top.get("context_len") or 0)
    large_entry = bool(top.get("large_entry")) or answer_len > GAMEPATH_DIRECT_MAX_CHARS
    if (
        top_score >= 0.74
        and answer_len >= 40
        and not large_entry
        and answer_len <= GAMEPATH_DIRECT_MAX_CHARS
        and (len(evaluated_results) == 1 or gap >= 0.1)
    ):
        confidence = "direct"
        reason = "high_score_clear_winner"
    elif top_score >= 0.46 and max(answer_len, context_len) >= 20:
        confidence = "summarize"
        reason = "large_entry_needs_model_extraction" if large_entry else "medium_score_needs_model_summary"
    else:
        confidence = "miss"
        reason = "low_score"

    if str(top.get("trust_state") or "unverified").lower() in {"disputed", "needs_review", "deprecated"}:
        if confidence == "direct":
            confidence = "summarize"
            reason = "trust_state_requires_review"
        elif top_score < 0.56:
            confidence = "miss"
            reason = "trust_state_low_score"

    return {
        "confidence": confidence,
        "score": round(top_score, 3),
        "gap": round(gap, 3),
        "reason": reason,
        "top_id": top.get("id"),
        "top_title": top.get("title"),
        "results": evaluated_results,
    }


def local_router_retrieval_decision(
    query: str,
    game_id: Optional[str],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    if not LOCAL_ROUTER_ENABLED or not LOCAL_ROUTER_RETRIEVAL_EVAL:
        return {"used": False, "reason": "disabled"}
    results = list(evaluation.get("results") or [])
    if not results:
        return {"used": False, "reason": "no_results"}

    candidate_lines: list[str] = []
    cache_parts: list[str] = []
    for index, item in enumerate(results[:3], 1):
        entry_id = int(item.get("id") or 0)
        title = str(item.get("title") or "").strip()[:100]
        question = str(item.get("question") or "").strip()[:120]
        excerpt = str(item.get("relevant_excerpt") or item.get("snippet") or item.get("answer_summary") or "")
        excerpt = re.sub(r"\s+", " ", excerpt).strip()[:320]
        score = item.get("retrieval_score", item.get("score", 0.0))
        candidate_lines.append(
            f"{index}. id={entry_id}; score={score}; title={title}; question={question}; excerpt={excerpt}"
        )
        cache_parts.append(f"{entry_id}:{item.get('updated_at') or ''}:{score}")

    cache_key = local_router_cache_key(
        "gamepath-retrieval-eval",
        game_id,
        query,
        "|".join(cache_parts),
    )
    cached = local_router_cache_get(cache_key)
    if cached:
        return cached

    messages = [
        {
            "role": "system",
            "content": (
                "Judge GamePath candidates. JSON only: "
                "{\"confidence\":\"direct|summarize|miss\",\"top_id\":number}. "
                "direct=exact short answer, summarize=relevant needs condensing, miss=not enough/wrong."
            ),
        },
        {
            "role": "user",
            "content": (
                f"game_id: {game_id or 'unknown'}\n"
                f"player_query: {query[:LOCAL_ROUTER_GAMEPATH_MAX_CHARS]}\n"
                "candidates:\n" + "\n".join(candidate_lines)
            ),
        },
    ]
    started = time.perf_counter()
    output = call_local_router_once(messages, max_tokens=128)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    parsed = extract_json_object(output)
    confidence = str(parsed.get("confidence") or "").strip().lower()
    if confidence not in {"direct", "summarize", "miss"}:
        confidence = "summarize" if str(parsed.get("route") or "").lower() == "hit" else "miss"
    try:
        top_id = int(parsed.get("top_id") or results[0].get("id") or 0)
    except Exception:
        top_id = int(results[0].get("id") or 0)
    result = {
        "used": True,
        "confidence": confidence,
        "top_id": top_id,
        "reason": str(parsed.get("reason") or "qwen_retrieval_eval").strip()[:120],
        "model": LOCAL_ROUTER_MODEL,
        "latency_ms": latency_ms,
        "cache_hit": False,
    }
    local_router_cache_set(cache_key, result)
    return result


def apply_local_router_retrieval_decision(
    evaluation: dict[str, Any],
    router_decision: dict[str, Any],
) -> dict[str, Any]:
    if not router_decision.get("used"):
        return evaluation
    confidence = str(router_decision.get("confidence") or "").strip().lower()
    if confidence not in {"direct", "summarize", "miss"}:
        return evaluation
    updated = dict(evaluation)
    results = list(updated.get("results") or [])
    top_id = int(router_decision.get("top_id") or 0)
    if top_id:
        results.sort(key=lambda item: 0 if int(item.get("id") or 0) == top_id else 1)
        updated["top_id"] = top_id
        if results:
            updated["top_title"] = results[0].get("title")
    updated["results"] = results
    updated["confidence"] = confidence
    updated["reason"] = f"local_router_retrieval:{router_decision.get('reason') or confidence}"
    updated["local_router_retrieval"] = router_decision
    return updated


def classify_gamepath_hit(results: list[dict[str, Any]]) -> str:
    if not results:
        return "miss"
    score = float(results[0].get("retrieval_score") or 0.0)
    answer_len = len(str(results[0].get("answer_summary") or ""))
    large_entry = bool(results[0].get("large_entry")) or answer_len > GAMEPATH_DIRECT_MAX_CHARS
    if score >= 0.74 and answer_len >= 40 and not large_entry:
        return "direct"
    if score >= 0.46 and answer_len >= 20:
        return "summarize"
    return "miss"


def confident_gamepath_hit(results: list[dict[str, Any]]) -> bool:
    return classify_gamepath_hit(results) == "direct"


def build_gamepath_answer(result: dict[str, Any]) -> str:
    title = str(result.get("title") or "GamePath").strip()
    summary = str(result.get("answer_summary") or "").strip()
    if len(summary) > GAMEPATH_DIRECT_MAX_CHARS:
        summary = str(result.get("relevant_excerpt") or result.get("snippet") or summary[:GAMEPATH_DIRECT_MAX_CHARS]).strip()
    return f"GamePath 已有紀錄：{title}\n{summary}"


def strip_model_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.IGNORECASE | re.DOTALL).strip()
    return cleaned


def build_gamepath_hint_answer(prompt: str, result: dict[str, Any]) -> str:
    fallback = build_gamepath_answer(result)
    if not LOCAL_ROUTER_ENABLED:
        return fallback
    title = status_text(result.get("title") or result.get("question") or "GamePath", 120)
    source_text = "\n".join(
        str(result.get(key) or "").strip()
        for key in ("relevant_excerpt", "snippet", "answer_summary")
        if str(result.get(key) or "").strip()
    )
    source_text = re.sub(r"\s+", " ", source_text).strip()[:1100]
    if not source_text:
        return fallback
    messages = [
        {
            "role": "system",
            "content": (
                "你是遊戲攻略提示整理器。只能使用提供的 GamePath 本地資料，不要新增未提供事實，"
                "不要列來源網址，不要貼原文全文。用繁體中文，輸出給玩家看的短回覆與詳細回覆。"
                "弱點、密碼、道具名稱、地點名稱必須沿用資料原詞；不要改寫成資料裡沒有的部位或名詞。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"玩家問題：{status_text(prompt, 180)}\n"
                f"GamePath 標題：{title}\n"
                f"GamePath 本地資料：{source_text}\n\n"
                "請只輸出下面 2 段，不要前言，不要來源，不要使用分級提示標籤：\n"
                "短回覆：<一句最可執行的教學提示，像玩家真的卡住時需要的下一步>\n"
                "詳細回覆：<較完整但仍精簡的教學，包含原因、路線/站位/操作，2 到 4 句>\n"
                "注意：如果是戰鬥問題，要給站位、迴避或省資源打法；如果是謎題/密碼，先提示再給明確答案。"
                "禁止使用資料中沒有出現的弱點部位，例如不要把「屁股」改成「關節、核心、腹部」。"
            ),
        },
    ]
    try:
        output = strip_model_thinking(call_local_router_once(messages, max_tokens=256, timeout_seconds=60))
    except Exception as exc:
        print(f"GamePath Qwen hint format failed: {exc}")
        return fallback
    output = condense_agent_answer(output, prompt).strip()
    if not output:
        return fallback
    return output[:1800]


def status_text(value: Any, max_chars: int = 140) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def gamepath_markdown_abs_path(markdown_path: Any) -> str:
    text = str(markdown_path or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        if path.parts and path.parts[0].lower() == GAMEPATH_DIR.name.lower():
            path = PROJECT_ROOT / path
        else:
            path = GAMEPATH_DIR / path
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def gamepath_status_candidate(item: dict[str, Any]) -> dict[str, Any]:
    markdown_path = str(item.get("markdown_path") or "")
    return {
        "id": item.get("id"),
        "game_id": item.get("game_id"),
        "title": status_text(item.get("title"), 90),
        "question": status_text(item.get("question"), 110),
        "matched_query": status_text(item.get("matched_query"), 90),
        "markdown_path": markdown_path,
        "markdown_abs_path": gamepath_markdown_abs_path(markdown_path),
        "retrieval_score": item.get("retrieval_score"),
        "match_coverage": item.get("match_coverage"),
        "metadata_match_score": item.get("metadata_match_score"),
        "bm25_score": item.get("score"),
        "trust_state": item.get("trust_state"),
        "dispute_count": item.get("dispute_count"),
        "spoiler_level": item.get("spoiler_level"),
        "source_type": item.get("source_type"),
        "source_quality": item.get("source_quality"),
        "agent_used": bool(item.get("agent_used")),
        "version": item.get("version"),
        "area": item.get("area"),
        "entity_type": item.get("entity_type"),
        "entity_name": item.get("entity_name"),
        "answer_len": item.get("answer_len"),
        "context_len": item.get("context_len"),
        "large_entry": bool(item.get("large_entry")),
        "rag_lite": bool(item.get("rag_lite")),
        "rag_chunk_count": item.get("rag_chunk_count"),
        "rag_chunk_ids": item.get("rag_chunk_ids"),
        "updated_at": item.get("updated_at"),
        "retrieval_reasons": list(item.get("retrieval_reasons") or [])[:8],
    }


def gamepath_lookup_status_details(
    game_id: Optional[str],
    evaluation: dict[str, Any],
    *,
    query_variants: Optional[list[str]] = None,
    search_elapsed_ms: Optional[float] = None,
) -> dict[str, Any]:
    results = list(evaluation.get("results") or [])
    if search_elapsed_ms is None:
        search_elapsed_ms = evaluation.get("search_elapsed_ms")
    return {
        "game_id": normalize_game_id(game_id) or "global",
        "gamepath_db_path": str(GAMEPATH_DB.resolve()),
        "gamepath_notes_path": str(GAMEPATH_NOTES_DIR.resolve()),
        "rag_backend": "gamepath_rag_lite_sqlite_fts5_chunks",
        "search_query": evaluation.get("search_query"),
        "query_variants": list(query_variants or evaluation.get("query_variants") or []),
        "search_tags": list(evaluation.get("search_tags") or []),
        "spoiler_level": evaluation.get("spoiler_level"),
        "search_scope": evaluation.get("search_scope") or {},
        "search_elapsed_ms": search_elapsed_ms,
        "candidate_count": len(results),
        "top_id": evaluation.get("top_id"),
        "top_title": evaluation.get("top_title"),
        "candidates": [gamepath_status_candidate(item) for item in results[:5]],
    }


def lookup_route_stage(
    *,
    game_id: Optional[str],
    hermes_agent_web_enabled: bool,
    gamepath_requested: bool,
    gamepath_raw_hits: int,
    gamepath_evaluation: Optional[dict[str, Any]],
    gamepath_results: list[dict[str, Any]],
    guide_results: list[dict[str, Any]],
    memory_results: list[dict[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    evaluation = gamepath_evaluation or {}
    router_info = evaluation.get("local_router")
    gamepath_details = gamepath_lookup_status_details(game_id, evaluation)
    if gamepath_results:
        return (
            "gamepath_summarizing",
            "GamePath 找到相關本地紀錄，正在交給模型濃縮成玩家提示。",
            {
                "source": "gamepath",
                "web_search": False,
                "fast_path": False,
                "gamepath_hits": len(gamepath_results),
                "retrieval_score": evaluation.get("score", 0.0),
                "retrieval_gap": evaluation.get("gap", 0.0),
                "retrieval_reason": evaluation.get("reason", ""),
                "local_router": evaluation.get("local_router"),
                "local_router_retrieval": evaluation.get("local_router_retrieval"),
                "search_query": evaluation.get("search_query"),
                **gamepath_details,
            },
        )
    if gamepath_requested:
        return (
            "gamepath_miss",
            "GamePath 沒有足夠高信心命中，交給 Hermes Agent 判斷是否需要 Tavily。",
            {
                "source": "gamepath",
                "web_search": "possible" if hermes_agent_web_enabled else False,
                "fast_path": False,
                "gamepath_hits": gamepath_raw_hits,
                "guide_hits": len(guide_results),
                "memory_hits": len(memory_results),
                "retrieval_score": evaluation.get("score", 0.0),
                "retrieval_gap": evaluation.get("gap", 0.0),
                "retrieval_reason": evaluation.get("reason", ""),
                "local_router": evaluation.get("local_router"),
                "local_router_retrieval": evaluation.get("local_router_retrieval"),
                "search_query": evaluation.get("search_query"),
                **gamepath_details,
            },
        )
    if guide_results:
        return (
            "guide_context",
            "本機攻略索引有命中，交給 Hermes 整理；必要時才可能查網路。",
            {
                "source": "guide_cache",
                "web_search": "possible" if hermes_agent_web_enabled else False,
                "fast_path": False,
                "guide_hits": len(guide_results),
                "local_router": router_info,
            },
        )
    if memory_results:
        return (
            "memory_context",
            "玩家記憶有命中，交給 Hermes 參考；必要時才可能查網路。",
            {
                "source": "memory_cache",
                "web_search": "possible" if hermes_agent_web_enabled else False,
                "fast_path": False,
                "memory_hits": len(memory_results),
                "local_router": router_info,
            },
        )
    if not guide_results and not memory_results:
        route = str((router_info or {}).get("intent_route") or "").strip()
        message = (
            f"Qwen 判斷意圖：{route}，已跳過 GamePath SQLite 查詢。"
            if route
            else "這不是攻略型問題，已跳過 GamePath SQLite 查詢。"
        )
        return (
            "gamepath_skipped",
            message,
            {
                "source": "chat",
                "web_search": "possible" if hermes_agent_web_enabled else False,
                "fast_path": True,
                "gamepath_checked": False,
                "local_router": router_info,
            },
        )
    if hermes_agent_web_enabled:
        return (
            "agent_may_search_web",
            "本地沒有命中，交給 Hermes Agent 判斷是否用 Tavily 查網路。",
            {
                "source": "hermes_tavily",
                "web_search": "possible",
                "fast_path": False,
                "gamepath_hits": 0,
                "local_router": router_info,
            },
        )
    return (
        "agent_no_tools",
        "本地沒有命中，交給 Hermes 無工具模式回答。",
        {
            "source": "hermes",
            "web_search": False,
            "fast_path": False,
            "gamepath_hits": 0,
            "local_router": router_info,
        },
    )


def gamepath_store_skip_reason(
    prompt: str,
    answer: str,
    game_id: Optional[str],
    agent_used: bool,
) -> str:
    if not agent_used:
        return "agent_not_used"
    clean_answer = str(answer or "").strip()
    if len(clean_answer) < 40:
        return "answer_too_short"
    if backend_hard_skips_gamepath(prompt):
        return "backend_hard_skip"
    if not should_use_gamepath(prompt, bool(GUIDE_INTENT_RE.search(prompt or ""))):
        return "not_guide_intent"
    uncertain_near_start = GAMEPATH_UNCERTAIN_RE.search(clean_answer[:220])
    has_actionable_hint = re.search(r"(Hint|提示|直接答案|建議|下一步|步驟|做法|打法)", clean_answer, re.IGNORECASE)
    if uncertain_near_start and not has_actionable_hint:
        return f"uncertain_answer:{uncertain_near_start.group(0)}"
    if re.search(
        r"(需要.*(道具名稱|哪個道具)|告訴我.*(道具名稱|物品描述)|直接回我.*道具名稱|你把.*道具名稱|"
        r"(道具名稱|物品描述|所在房間).{0,80}(貼|回|告訴|提供))",
        clean_answer[:780],
    ):
        return "needs_specific_item"
    if re.search(r"(這個道具|哪個道具|不知道.*道具|道具.*做什麼)", str(prompt or "")) and re.search(
        r"(道具名稱|物品描述|所在房間|貼給我|回我)",
        clean_answer[:780],
    ):
        return "needs_specific_item"
    if re.search(r"https?://", clean_answer):
        clean_answer = re.sub(r"https?://\S+", "", clean_answer)
    if not clean_answer:
        return "answer_empty_after_url_strip"
    return ""


def should_store_gamepath_answer(prompt: str, answer: str, game_id: Optional[str], agent_used: bool) -> bool:
    return not gamepath_store_skip_reason(prompt, answer, game_id, agent_used)


def resolve_gamepath_store_game_id(
    selected_game_id: Optional[str],
    active_game_context: Optional[dict[str, Any]] = None,
) -> str:
    detected_game_id = None
    if isinstance(active_game_context, dict):
        confidence = float(active_game_context.get("confidence") or 0.0)
        source = str(active_game_context.get("source") or "")
        if confidence >= 0.68 and source not in {"window_title_guess", "game_path_guess"}:
            detected_game_id = active_game_context.get("game_id")
    return normalize_game_id(selected_game_id) or normalize_game_id(detected_game_id) or "global"


def should_use_guides(prompt: str, explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return explicit
    if GAMEPATH_NEGATED_GUIDE_RE.search(prompt or ""):
        return False
    return bool(GUIDE_INTENT_RE.search(prompt or ""))


def should_use_overlay(prompt: str, image_base64: Optional[str]) -> bool:
    return bool(
        image_base64
        and (
            OVERLAY_INTENT_RE.search(prompt or "")
            or re.search(r"(圈出|圈起|框出|標記|標出|幫我圈|幫我標|指給我)", prompt or "", re.IGNORECASE)
        )
    )


def should_use_visual_scene(prompt: str) -> bool:
    return bool(
        VISUAL_SCENE_INTENT_RE.search(prompt or "")
        or re.search(r"(你看見|你看到|看見什麼|看到什麼|幫我看|看一下.*畫面|畫面.*什麼|螢幕.*什麼)", prompt or "", re.IGNORECASE)
    )


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


def detect_implicit_memory_fact(prompt: str) -> Optional[dict[str, str]]:
    text = re.sub(r"\s+", " ", (prompt or "").strip())
    if detect_fact_lookup_key(text):
        return None
    match = IMPLICIT_MEMORY_FACT_RE.search(text)
    if not match:
        return None
    if not USER_FACT_CONTEXT_RE.search(text):
        return None
    key = match.group("key").strip()
    value = match.group("value").strip(" ：:，,。.!?？；;「」'\"")
    if any(marker in value for marker in ("什麼", "多少", "哪個", "嗎", "?")):
        return None
    if not key or not value:
        return None
    kind = "preference" if key == "偏好" else "state"
    return {
        "content": f"玩家{key}是 {value}",
        "kind": kind,
        "key": key,
        "value": value,
    }


def detect_fact_lookup_key(prompt: str) -> Optional[str]:
    text = re.sub(r"\s+", " ", (prompt or "").strip())
    match = FACT_LOOKUP_RE.search(text)
    if match:
        key = match.group("key").strip()
        if GAME_ENTITY_FACT_CONTEXT_RE.search(text) and not USER_FACT_CONTEXT_RE.search(text):
            return None
        if not USER_FACT_CONTEXT_RE.search(text):
            return None
        return key
    if any(marker in text for marker in ("剛剛", "前面", "上一句", "我說的")):
        fact = IMPLICIT_MEMORY_FACT_RE.search(text)
        if fact:
            return fact.group("key").strip()
    return None


def memory_kinds_for_chat(prompt: str, *, has_image: bool = False) -> list[str]:
    kinds = ["state", "preference"]
    text = str(prompt or "")
    if has_image or MEMORY_TASK_CONTEXT_RE.search(text):
        kinds.append("task")
    if MEMORY_NOTE_CONTEXT_RE.search(text):
        kinds.append("note")
    if detect_fact_lookup_key(text):
        kinds.extend(["task", "note"])
    deduped: list[str] = []
    for kind in kinds:
        if kind not in deduped:
            deduped.append(kind)
    return deduped


def answer_fact_lookup(prompt: str, memory_results: list[dict[str, Any]]) -> Optional[str]:
    key = detect_fact_lookup_key(prompt)
    if not key:
        return None
    key_options = [key]
    if key == "代號":
        key_options.append("測試代號")
    for item in memory_results:
        content = str(item.get("content") or "")
        for option in key_options:
            if option not in content:
                continue
            match = re.search(rf"{re.escape(option)}是\s*([^。；;，,\n]+)", content)
            if match:
                value = match.group(1).strip()
                if any(marker in value for marker in ("什麼", "多少", "哪個", "嗎", "?")):
                    continue
                if value:
                    return f"你的{option}是 {value}。"
    return None


def format_rag_context(
    guide_results: list[dict[str, Any]],
    memory_results: list[dict[str, Any]],
    guide_was_requested: bool,
    gamepath_results: Optional[list[dict[str, Any]]] = None,
) -> str:
    lines: list[str] = []
    if memory_results:
        lines.append("Local player memory (CPU SQLite, use only as user-specific context):")
        for item in memory_results[:8]:
            lines.append(f"- [{item.get('kind')}] {item.get('content')}")
    if gamepath_results:
        lines.append(
            "GamePath extracted passages (stable local SQLite + Markdown). "
            "Use only the passages relevant to the player's question; never dump the whole document."
        )
        for index, item in enumerate(gamepath_results[:5], 1):
            title = item.get("title") or "GamePath"
            snippet = item.get("relevant_excerpt") or item.get("snippet") or make_snippet(item.get("answer_summary") or "", title)
            tags = item.get("tags") or ""
            path = item.get("markdown_path") or ""
            lines.append(f"{index}. {title} [{tags}] ({path}): {snippet}")
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
        "Extract the smallest useful answer from local guide text; do not paste unrelated sections or full guides.\n"
        "If local guide snippets are empty, say the local guide library has no matching entry.\n\n"
        f"{rag_context}"
    )


def build_overlay_messages(prompt: str, image_base64: str, rag_context: str) -> list[dict[str, Any]]:
    valid_cells = [
        f"{chr(ord('A') + col)}{row + 1}"
        for row in range(OVERLAY_GRID_ROWS)
        for col in range(OVERLAY_GRID_COLUMNS)
    ]
    valid_cells_text = ", ".join(valid_cells)
    grid_span = f"A1 through {chr(ord('A') + OVERLAY_GRID_COLUMNS - 1)}{OVERLAY_GRID_ROWS}"
    if wants_circle_marker(prompt):
        user_text = prompt
        if rag_context:
            user_text += "\n\nLocal context:\n" + rag_context
        user_text += (
            f"\n\nThe screenshot has a cyan planning grid with cells {grid_span}. "
            f"Valid cells are {valid_cells_text}. "
            "Pick the single cell that contains the requested visible object or the best visible target. "
            "If the object spans multiple cells, choose the cell containing its center. "
            "If the object is not visible or you are unsure, reply NONE. "
            "Reply with exactly one cell name like B2, or NONE. No Markdown, no JSON, no explanation."
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a visual grid selector. Output only one token: a valid grid cell or NONE."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_overlay_grid_data_url(image_base64)}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

    system = (
        "You are a game companion HUD planner. Return one compact ASCII JSON object only, no Markdown. "
        "Use Traditional Chinese for answer and labels. "
        f"The screenshot has a cyan planning grid with cells {grid_span}. The grid is not part of the game. "
        f"Valid cells are {valid_cells_text}. "
        "For every visual HUD item, prefer the cell key instead of numeric x/y. "
        "If you are not visually confident about the requested object or cell, return an empty overlay items array. "
        "Use exact JSON keys only: answer, overlay, duration_ms, items, type, cell, x, y, radius, label, color, from, to, points. "
        "Do not duplicate or merge key names. Do not output coordinates in answer. "
        "JSON shape: {\"answer\":\"...\",\"overlay\":{\"duration_ms\":6000,\"items\":[...]}}. "
        "The answer field must never mention coordinates, grid cells, x/y values, normalized positions, or JSON. "
        "Location data belongs only inside overlay.items for the HUD renderer. "
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
        "Choose the grid cell containing the requested visible object; if it spans cells, choose its center cell. "
        "For circle use type, cell, radius, color. For label/pin use type, cell, label. "
        "For arrow use from {cell} and to {cell}. For path use points [{cell}]. "
        "Use circle for objects to circle, arrow for where to go next, path for route lines, and pin for targets/objectives. "
        "Example: {\"answer\":\"我已標記。\",\"overlay\":{\"duration_ms\":6000,\"items\":[{\"type\":\"circle\",\"cell\":\"C2\",\"radius\":0.14,\"color\":\"#ff2d2d\"}]}}"
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_overlay_grid_data_url(image_base64)}},
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


def grid_cell_to_xy(cell: Any) -> Optional[tuple[float, float]]:
    if cell is None:
        return None
    match = re.search(r"\b([A-Za-z])\s*[-_ ]?\s*(\d{1,2})\b", str(cell))
    if not match:
        return None
    col = ord(match.group(1).upper()) - ord("A")
    row = int(match.group(2)) - 1
    if col < 0 or col >= OVERLAY_GRID_COLUMNS or row < 0 or row >= OVERLAY_GRID_ROWS:
        return None
    return ((col + 0.5) / OVERLAY_GRID_COLUMNS, (row + 0.5) / OVERLAY_GRID_ROWS)


def point_from_value(value: Any) -> Optional[dict[str, float]]:
    if isinstance(value, str):
        cell_xy = grid_cell_to_xy(value)
        if cell_xy:
            return {"x": cell_xy[0], "y": cell_xy[1]}
        return None
    if not isinstance(value, dict):
        return None
    cell_xy = grid_cell_to_xy(value.get("cell"))
    if cell_xy:
        return {"x": cell_xy[0], "y": cell_xy[1]}
    if "x" not in value or "y" not in value:
        return None
    return {"x": clamp_float(value.get("x")), "y": clamp_float(value.get("y"))}


def point_from_item(item: dict[str, Any]) -> Optional[dict[str, float]]:
    cell_xy = grid_cell_to_xy(item.get("cell"))
    if cell_xy:
        return {"x": cell_xy[0], "y": cell_xy[1]}
    if "x" not in item or "y" not in item:
        return None
    return {"x": clamp_float(item.get("x")), "y": clamp_float(item.get("y"))}


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
    color = "#ff2d2d" if item_type == "circle" else str(item.get("color") or "#ff2d2d")
    label = str(item.get("label") or "")[:80]
    cleaned: dict[str, Any] = {"type": item_type, "color": color}
    if label:
        cleaned["label"] = label
    if item_type == "circle":
        point = point_from_item(item)
        if point is None:
            return None
        cleaned.update(
            {
                "x": point["x"],
                "y": point["y"],
                "radius": max(0.01, min(parse_loose_float(str(item.get("radius") or "0.06"), 0.06), 0.35)),
            }
        )
    elif item_type == "arrow":
        src = point_from_value(item.get("from"))
        dst = point_from_value(item.get("to"))
        if src is None or dst is None:
            return None
        cleaned["from"] = src
        cleaned["to"] = dst
    elif item_type == "path":
        points = item.get("points") or []
        cleaned["points"] = [
            parsed
            for point in points
            for parsed in [point_from_value(point)]
            if parsed
        ][:8]
        if len(cleaned["points"]) < 2:
            return None
    else:
        point = point_from_item(item)
        if point is None:
            return None
        cleaned.update(point)
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
    try:
        duration = int(raw_overlay.get("duration_ms") or 6000)
    except Exception:
        duration = 6000
    return {"duration_ms": max(3000, min(duration, 8000)), "items": cleaned_items}


def add_pixel_point(point: dict[str, Any], image_width: int, image_height: int) -> None:
    if "x" in point and "y" in point:
        point["pixel_x"] = int(round(clamp_float(point.get("x")) * image_width))
        point["pixel_y"] = int(round(clamp_float(point.get("y")) * image_height))


def attach_overlay_image_space(overlay: Optional[dict[str, Any]], image_base64: str) -> Optional[dict[str, Any]]:
    if not overlay or not overlay.get("items"):
        return overlay
    try:
        image_width, image_height = decode_image_size(image_base64)
    except Exception:
        return overlay

    overlay["coordinate_space"] = {
        "type": "source_image_pixels",
        "image_width": image_width,
        "image_height": image_height,
    }
    min_edge = max(1, min(image_width, image_height))
    for item in overlay.get("items", []):
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").lower()
        if item_type in {"circle", "pin", "label"}:
            add_pixel_point(item, image_width, image_height)
            if item_type == "circle":
                item["radius_px"] = int(round(clamp_float(item.get("radius"), 0.06) * min_edge))
        elif item_type == "arrow":
            if isinstance(item.get("from"), dict):
                add_pixel_point(item["from"], image_width, image_height)
            if isinstance(item.get("to"), dict):
                add_pixel_point(item["to"], image_width, image_height)
        elif item_type == "path":
            for point in item.get("points") or []:
                if isinstance(point, dict):
                    add_pixel_point(point, image_width, image_height)
    return overlay


def parse_loose_float(value: str, default: float = 0.5) -> float:
    cleaned = re.sub(r"[^0-9.+-]", "", value or "")
    cleaned = cleaned.replace("..", ".")
    if cleaned.startswith("."):
        cleaned = "0" + cleaned
    if cleaned.count(".") > 1:
        first, rest = cleaned.split(".", 1)
        cleaned = first + "." + rest.replace(".", "")
    try:
        if re.fullmatch(r"[+-]?\d+", cleaned):
            whole = int(cleaned)
            if abs(whole) > 1:
                digits = len(str(abs(whole)))
                denominator = 1000 if digits >= 3 else 10**digits
                return clamp_float(whole / denominator, default)
        return clamp_float(float(cleaned), default)
    except Exception:
        return default


def find_loose_number_after(text: str, label: str) -> Optional[float]:
    if label in {"x", "y"}:
        key_pattern = rf"(?<![A-Za-z0-9_])['\"]?(?:\d*{label}+|{label}\w*)['\"]?"
    elif label == "radius":
        key_pattern = r"(?<![A-Za-z0-9_])['\"]?(?:radius|r)\w*['\"]?"
    else:
        key_pattern = rf"['\"]?{re.escape(label)}\w*['\"]?"

    pattern = rf"{key_pattern}\s*[:=]\s*['\"]?[^0-9.+-]{{0,8}}([0-9.+-]{{1,16}})"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return parse_loose_float(match.group(1))


def find_cell_candidate(fragment: str) -> Optional[str]:
    for match in re.finditer(r"([A-Za-z])\s*[-_ ]?(\d{1,2})", fragment or ""):
        cell = f"{match.group(1).upper()}{match.group(2)}"
        if grid_cell_to_xy(cell):
            return cell
    return None


def find_loose_cell(text: str) -> Optional[str]:
    text = text or ""
    for marker in re.finditer(r"(?:circle|pin|target|objective|標記|圈)", text, re.IGNORECASE):
        cell = find_cell_candidate(text[marker.start() : marker.start() + 180])
        if cell:
            return cell
    for marker in re.finditer(r"(?:cell|grid|格子|方格|區塊)", text, re.IGNORECASE):
        cell = find_cell_candidate(text[marker.end() : marker.end() + 80])
        if cell:
            return cell
    cell = find_cell_candidate(text)
    if cell:
        return cell
    return None


def extract_loose_overlay(text: str) -> Optional[dict[str, Any]]:
    lower = (text or "").lower()
    loose_cell = find_loose_cell(text)
    if not loose_cell and not any(token in lower for token in ("circle", "pin", "label", "cell", "overlay", "items")):
        return None
    cell_xy = grid_cell_to_xy(loose_cell)
    radius = find_loose_number_after(text, "radius")
    if cell_xy:
        return {
            "duration_ms": 6000,
            "items": [
                {
                    "type": "circle",
                    "x": cell_xy[0],
                    "y": cell_xy[1],
                    "radius": max(0.04, min(radius if radius is not None else 0.1, 0.35)),
                    "color": "#ff2d2d",
                }
            ],
        }
    x = find_loose_number_after(text, "x")
    y = find_loose_number_after(text, "y")
    if x is None or y is None:
        return None
    if (x <= 0.02 or x >= 0.98 or y <= 0.02 or y >= 0.98) and (
        radius is None or radius >= 0.3
    ):
        return None
    return {
        "duration_ms": 6000,
        "items": [
            {
                "type": "circle",
                "x": x,
                "y": y,
                "radius": max(0.04, min(radius if radius is not None else 0.1, 0.35)),
                "color": "#ff2d2d",
            }
        ],
    }


def wants_circle_marker(prompt: str) -> bool:
    return bool(re.search(r"(圈|圈出|圈選|框出|標記|標出|circle|mark|highlight)", prompt or "", re.IGNORECASE))


def has_position_hint(prompt: str) -> bool:
    return bool(
        re.search(
            r"(左上|右上|左下|右下|左邊|右邊|上方|下方|上面|下面|中間|中央|left|right|top|bottom|center)",
            prompt or "",
            re.IGNORECASE,
        )
    )


def wants_main_area_marker(prompt: str) -> bool:
    return bool(re.search(r"(主要區域|主要畫面|中間|中央|main visible area|main area|center)", prompt or "", re.IGNORECASE))


def fallback_circle_overlay() -> dict[str, Any]:
    return {
        "duration_ms": 6000,
        "items": [
            {
                "type": "circle",
                "x": 0.5,
                "y": 0.5,
                "radius": 0.16,
                "color": "#ff2d2d",
            }
        ],
    }


def cell_circle_overlay(cell: str) -> Optional[dict[str, Any]]:
    cell_xy = grid_cell_to_xy(cell)
    if not cell_xy:
        return None
    return {
        "duration_ms": 4500,
        "items": [
            {
                "type": "circle",
                "x": cell_xy[0],
                "y": cell_xy[1],
                "radius": 0.13,
                "color": "#ff2d2d",
            }
        ],
    }


def prompt_position_fallback_overlay(prompt: str) -> dict[str, Any]:
    text = (prompt or "").lower()
    x = 0.5
    y = 0.5
    if any(token in text for token in ("左", "left")):
        x = 0.24
    if any(token in text for token in ("右", "right")):
        x = 0.76
    if any(token in text for token in ("上", "top")):
        y = 0.24
    if any(token in text for token in ("下", "bottom")):
        y = 0.76
    overlay = fallback_circle_overlay()
    overlay["items"][0]["x"] = x
    overlay["items"][0]["y"] = y
    return overlay


def fallback_overlay_for_prompt(prompt: str) -> Optional[dict[str, Any]]:
    if has_position_hint(prompt) or wants_main_area_marker(prompt):
        return prompt_position_fallback_overlay(prompt)
    return None


def create_overlay_response(prompt: str, image_base64: str, rag_context: str) -> dict[str, Any]:
    if wants_circle_marker(prompt) and has_position_hint(prompt):
        overlay = attach_overlay_image_space(prompt_position_fallback_overlay(prompt), image_base64)
        return {"answer": "我已依照你指定的位置畫上紅圈。", "overlay": overlay}

    circle_marker_request = wants_circle_marker(prompt)
    overlay_messages = build_overlay_messages(prompt, image_base64, rag_context)
    overlay_tokens = 32 if circle_marker_request else int(os.environ.get("LLAMA_OVERLAY_RESPONSE_TOKENS", "128"))
    if CHAT_BACKEND == "hermes" and HERMES_USE_CONFIG_MODEL:
        output = call_hermes_messages(
            overlay_messages,
            image_file=LATEST_OVERLAY_GRID_INPUT,
            max_tokens=overlay_tokens,
        )
    else:
        output = call_llama_once(
            overlay_messages,
            max_tokens=overlay_tokens,
        )
    try:
        (LOG_DIR / "latest-overlay-raw.txt").write_text(output, encoding="utf-8")
    except Exception:
        pass
    if circle_marker_request:
        cell = find_cell_candidate(output)
        if cell:
            overlay = attach_overlay_image_space(cell_circle_overlay(cell), image_base64)
            return {"answer": "我已標記。", "overlay": overlay}
        fallback = fallback_overlay_for_prompt(prompt)
        if fallback:
            fallback = attach_overlay_image_space(fallback, image_base64)
            return {"answer": "我先用紅圈標出可能區域。", "overlay": fallback}
        return {"answer": "我沒有抓到可靠座標，所以先不畫錯位置。請指定方位或物件。", "overlay": None}
    try:
        parsed = extract_json_object(output)
    except Exception:
        overlay = extract_loose_overlay(output)
        if overlay:
            overlay = attach_overlay_image_space(overlay, image_base64)
            return {"answer": "我已把紅色標記畫在 HUD 上。", "overlay": overlay}
        fallback = fallback_overlay_for_prompt(prompt)
        if fallback:
            fallback = attach_overlay_image_space(fallback, image_base64)
            return {"answer": "我先用紅圈標出可能區域。", "overlay": fallback}
        return {"answer": "我沒有抓到可靠座標，所以先不畫錯位置。請指定要標記的物件或方位。", "overlay": None}
    answer = clean_vision_answer(str(parsed.get("answer") or "").strip())
    answer = re.sub(r"\(?\s*x\s*[:=]\s*0?\.\d+\s*,?\s*y\s*[:=]\s*0?\.\d+\s*\)?", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"座標\s*[:：]?\s*[0-9.,，\s]+", "", answer)
    answer = re.sub(r"(?:格子|方格|cell)\s*[:：]?\s*[A-Ha-h]\s*[-_ ]?\s*[1-6]", "", answer, flags=re.IGNORECASE)
    answer = clean_short_answer(answer)
    overlay = sanitize_overlay(parsed.get("overlay"))
    if not overlay:
        overlay = fallback_overlay_for_prompt(prompt)
        if overlay and (not answer or "無法" in answer):
            answer = "我先用紅圈標出可能區域。"
    overlay = attach_overlay_image_space(overlay, image_base64)
    if not answer:
        answer = "我沒有抓到可靠座標，所以先不畫錯位置。請指定要標記的物件或方位。"
    return {"answer": answer, "overlay": overlay}


@app.on_event("startup")
async def startup_event():
    if LLAMA_AUTO_START:
        await asyncio.to_thread(start_llama_server)
    else:
        print("llama auto-start disabled; backend will use configured non-llama chat route.")
    if live_state_enabled:
        ensure_live_state_task()


@app.on_event("shutdown")
async def shutdown_event():
    global live_state_enabled, live_state_task
    live_state_enabled = False
    if live_state_task and not live_state_task.done():
        live_state_task.cancel()
    live_state_task = None
    stop_llama_server()


@app.get("/health")
async def health():
    return {
        "status": "ok" if (not LLAMA_AUTO_START or llama_ready()) else "loading",
        "model": MODEL_ALIAS,
        "llama_url": llama_base_url(),
        "llama_auto_start": LLAMA_AUTO_START,
        "vulkan_device": VULKAN_DEVICE,
        "resource_policy": "game",
        "llama_ctx_size": LLAMA_CTX_SIZE,
        "llama_gpu_layers": LLAMA_GPU_LAYERS,
        "llama_flash_attn": LLAMA_FLASH_ATTN,
        "llama_skip_chat_parsing": LLAMA_SKIP_CHAT_PARSING,
        "llama_image_min_tokens": LLAMA_IMAGE_MIN_TOKENS or "default",
        "llama_image_max_tokens": LLAMA_IMAGE_MAX_TOKENS_SERVER or "default",
        "image_response_tokens": os.environ.get(
            "LLAMA_IMAGE_RESPONSE_TOKENS",
            os.environ.get("LLAMA_IMAGE_MAX_TOKENS", "64"),
        ),
        "overlay_grid_long_edge": bounded_int_env("LLAMA_OVERLAY_GRID_LONG_EDGE", 960, 384, 1280),
        "overlay_grid_columns": OVERLAY_GRID_COLUMNS,
        "overlay_grid_rows": OVERLAY_GRID_ROWS,
        "overlay_response_tokens": os.environ.get("LLAMA_OVERLAY_RESPONSE_TOKENS", "128"),
        "vision_long_edge": bounded_int_env("LLAMA_VISION_LONG_EDGE", 960, 512, 2560),
        "screenshot_long_edge": bounded_int_env("IGPU_SCREENSHOT_LONG_EDGE", 960, 512, 2560),
        "screenshot_format": os.environ.get("IGPU_SCREENSHOT_FORMAT", "JPEG"),
        "llama_parallel": LLAMA_PARALLEL or "auto",
        "llama_cache_ram_mib": LLAMA_CACHE_RAM or "default",
        "chat_backend": CHAT_BACKEND,
        "hermes_wsl_distro": HERMES_WSL_DISTRO if CHAT_BACKEND == "hermes" else None,
        "hermes_base_url": HERMES_BASE_URL if CHAT_BACKEND == "hermes" else None,
        "hermes_use_config_model": HERMES_USE_CONFIG_MODEL if CHAT_BACKEND == "hermes" else False,
        "hermes_agent_web_enabled": HERMES_AGENT_WEB_ENABLED if CHAT_BACKEND == "hermes" else False,
        "hermes_agent_toolsets": HERMES_AGENT_TOOLSETS if CHAT_BACKEND == "hermes" else None,
        "hermes_agent_max_tokens": HERMES_AGENT_MAX_TOKENS if CHAT_BACKEND == "hermes" else None,
        "cloud_vision": bool(CHAT_BACKEND == "hermes" and HERMES_USE_CONFIG_MODEL),
        "hermes_timeout_seconds": HERMES_TIMEOUT_SECONDS,
        "hermes_max_tokens": HERMES_MAX_TOKENS,
        "hermes_context_length": HERMES_CONTEXT_LENGTH,
        "openai_max_tokens_cap": OPENAI_MAX_TOKENS_CAP or None,
        "local_tools": ENABLE_LOCAL_TOOLS,
        "local_router_enabled": LOCAL_ROUTER_ENABLED,
        "local_router_url": LOCAL_ROUTER_URL if LOCAL_ROUTER_ENABLED else None,
        "local_router_model": LOCAL_ROUTER_MODEL if LOCAL_ROUTER_ENABLED else None,
        "local_router_role": LOCAL_ROUTER_ROLE if LOCAL_ROUTER_ENABLED else None,
        "local_router_ready": local_router_ready(),
        "local_router_timeout_seconds": LOCAL_ROUTER_TIMEOUT_SECONDS if LOCAL_ROUTER_ENABLED else None,
        "local_router_gamepath_gate": LOCAL_ROUTER_GAMEPATH_GATE if LOCAL_ROUTER_ENABLED else False,
        "local_router_always_route": LOCAL_ROUTER_ALWAYS_ROUTE if LOCAL_ROUTER_ENABLED else False,
        "local_router_gamepath_max_chars": LOCAL_ROUTER_GAMEPATH_MAX_CHARS if LOCAL_ROUTER_ENABLED else None,
        "local_router_retrieval_eval": LOCAL_ROUTER_RETRIEVAL_EVAL if LOCAL_ROUTER_ENABLED else False,
        "local_router_cache_ttl_seconds": LOCAL_ROUTER_DECISION_CACHE_TTL_SECONDS if LOCAL_ROUTER_ENABLED else None,
        "rag_backend": "gamepath_rag_lite_sqlite_fts5_chunks",
        "gamepath_enabled": True,
        "gamepath_db_exists": GAMEPATH_DB.exists(),
        "gamepath_entry_count": gamepath_entry_count_sync(),
        "gamepath_chunk_count": gamepath_chunk_count_sync(),
        "gamepath_last_updated_at": gamepath_last_updated_at_sync(),
        "live_state_enabled": bool(live_state_enabled),
        "live_state_running": bool(live_state_task and not live_state_task.done()),
        "live_state_last_updated_at": live_state_memory.get("updated_at") or "",
        "live_state_model": live_state_model_name(),
    }


@app.get("/guides/games")
async def guide_games():
    games = await asyncio.to_thread(list_guide_games_sync)
    return {"games": games}


@app.get("/game-profiles")
async def game_profiles():
    profiles = await asyncio.to_thread(load_game_profiles_sync)
    games = await asyncio.to_thread(list_guide_games_sync)
    return {
        "profiles": profiles.get("profiles", {}),
        "process_mappings": profiles.get("process_mappings", {}),
        "games": games,
    }


@app.get("/active-game")
async def active_game():
    return await asyncio.to_thread(detect_active_game_sync)


@app.post("/game-profiles/learn")
async def learn_game_profile(request: GameProfileLearnRequest):
    try:
        result = await asyncio.to_thread(learn_active_game_sync, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", **result}


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


@app.post("/gamepath/add")
async def gamepath_add(request: GamePathAddRequest):
    try:
        item = await asyncio.to_thread(
            add_gamepath_sync,
            request.question,
            request.answer_summary,
            request.game_id,
            title=request.title,
            tags=request.tags,
            spoiler_level=request.spoiler_level,
            source_type=request.source_type,
            agent_used=request.agent_used,
            version=request.version,
            area=request.area,
            entity_type=request.entity_type,
            entity_name=request.entity_name,
            source_quality=request.source_quality,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok", "item": item}


@app.post("/gamepath/search")
async def gamepath_search(request: GamePathSearchRequest):
    results = await asyncio.to_thread(
        search_gamepath_sync,
        request.query,
        request.game_id,
        request.limit,
        tags=request.tags,
        spoiler_level=request.spoiler_level,
        version=request.version,
        area=request.area,
        entity_type=request.entity_type,
        entity_name=request.entity_name,
        strict_metadata=request.strict_metadata,
    )
    evaluation = await asyncio.to_thread(
        evaluate_gamepath_retrieval,
        request.query,
        request.game_id,
        results,
    )
    return {
        "game_id": normalize_game_id(request.game_id),
        "query": request.query,
        "search_scope": normalize_gamepath_scope(
            request.query,
            request.tags,
            version=request.version,
            area=request.area,
            entity_type=request.entity_type,
            entity_name=request.entity_name,
        ),
        "evaluation": {key: value for key, value in evaluation.items() if key != "results"},
        "results": evaluation.get("results") or results,
    }


@app.get("/gamepath/recent")
async def gamepath_recent(game_id: Optional[str] = None, limit: int = 10):
    results = await asyncio.to_thread(recent_gamepath_sync, game_id, limit)
    return {"game_id": normalize_game_id(game_id), "results": results}


@app.post("/gamepath/feedback")
async def gamepath_feedback(request: GamePathFeedbackRequest):
    entry_id = request.entry_id
    if not entry_id:
        ref = recent_gamepath_reference(request.game_id)
        entry_id = int(ref["entry_id"]) if ref else None
    if not entry_id:
        raise HTTPException(status_code=404, detail="No recent GamePath entry to mark.")
    item = await asyncio.to_thread(
        update_gamepath_feedback_sync,
        int(entry_id),
        request.message,
        state=request.state,
    )
    if not item:
        raise HTTPException(status_code=404, detail="GamePath entry not found.")
    return {"status": "ok", "item": item}


@app.delete("/gamepath/{entry_id}")
async def gamepath_delete(entry_id: int):
    try:
        item = await asyncio.to_thread(delete_gamepath_sync, entry_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not item:
        raise HTTPException(status_code=404, detail="GamePath entry not found.")
    return {"status": "ok", "item": item}


@app.post("/tasks/analyze")
async def task_analyze(request: TaskAnalyzeRequest):
    async with generate_lock:
        result = await asyncio.to_thread(analyze_task_sync, request)
    return {"status": "ok", "task": result}


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
    body.setdefault("max_tokens", int(os.environ.get("LLAMA_MAX_TOKENS", "512")))
    if OPENAI_MAX_TOKENS_CAP > 0:
        try:
            body["max_tokens"] = min(int(body.get("max_tokens") or OPENAI_MAX_TOKENS_CAP), OPENAI_MAX_TOKENS_CAP)
        except (TypeError, ValueError):
            body["max_tokens"] = OPENAI_MAX_TOKENS_CAP
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
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload.")
    if len(audio_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio upload is too large.")

    suffix = Path(file.filename or "voice.webm").suffix.lower()
    if suffix not in {".webm", ".wav", ".mp3", ".m4a", ".ogg"}:
        suffix = ".webm"

    voice_dir = LOG_DIR / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)
    audio_path = voice_dir / f"voice-{int(time.time() * 1000)}{suffix}"
    audio_path.write_bytes(audio_bytes)

    try:
        return await asyncio.to_thread(transcribe_audio_path, audio_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Voice transcription failed: {exc}")


class WinRect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


DWMWA_EXTENDED_FRAME_BOUNDS = 9


def rect_to_dict(rect: Any) -> dict[str, int]:
    left = int(rect.left)
    top = int(rect.top)
    right = int(rect.right)
    bottom = int(rect.bottom)
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": max(0, right - left),
        "height": max(0, bottom - top),
    }


def get_dwm_extended_frame_bounds(hwnd: int) -> Optional[dict[str, int]]:
    if os.name != "nt" or not hwnd:
        return None
    try:
        dwmapi = ctypes.windll.dwmapi
        rect = WinRect()
        result = dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(int(hwnd)),
            ctypes.c_uint(DWMWA_EXTENDED_FRAME_BOUNDS),
            ctypes.byref(rect),
            ctypes.sizeof(rect),
        )
        if result != 0:
            return None
        bounds = rect_to_dict(rect)
        if bounds["width"] <= 0 or bounds["height"] <= 0:
            return None
        return bounds
    except Exception:
        return None


def get_foreground_window_info() -> Optional[dict[str, Any]]:
    if os.name != "nt":
        return None
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None

        info = get_window_info(user32, hwnd)
        if info:
            return info

        time.sleep(0.12)
        return get_top_window_info(user32)
    except Exception:
        return None


def get_window_process_path(user32: Any, hwnd: int) -> str:
    if os.name != "nt":
        return ""

    try:
        kernel32 = ctypes.windll.kernel32
        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(wintypes.HWND(int(hwnd)), ctypes.byref(process_id))
        if not process_id.value:
            return ""

        process_query_limited_information = 0x1000
        handle = kernel32.OpenProcess(process_query_limited_information, False, process_id.value)
        if not handle:
            return ""

        try:
            path_buffer = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(len(path_buffer))
            if kernel32.QueryFullProcessImageNameW(handle, 0, path_buffer, ctypes.byref(size)):
                return path_buffer.value
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return ""

    return ""


def get_window_info(
    user32: Any,
    hwnd: int,
    *,
    allow_ignored: bool = False,
) -> Optional[dict[str, Any]]:
    if not hwnd:
        return None

    visible = bool(user32.IsWindowVisible(hwnd))
    title_buffer = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, title_buffer, 512)
    title = title_buffer.value.strip()
    process_path = get_window_process_path(user32, int(hwnd))
    process_name = Path(process_path).name.lower() if process_path else ""
    title_lower = title.lower()
    ignored_window = any(ignored.lower() in title_lower for ignored in IGNORED_CAPTURE_TITLES) or (
        process_name in IGNORED_CAPTURE_PROCESSES
    )
    if not visible and not (allow_ignored and ignored_window):
        return None
    if ignored_window and not allow_ignored:
        return None

    rect = WinRect()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None

    window_rect = rect_to_dict(rect)
    width = window_rect["width"]
    height = window_rect["height"]
    if width <= 0 or height <= 0:
        return None
    if not allow_ignored and (width < 320 or height < 240):
        return None

    return {
        "hwnd": int(hwnd),
        "title": title,
        "visible": visible,
        "left": window_rect["left"],
        "top": window_rect["top"],
        "right": window_rect["right"],
        "bottom": window_rect["bottom"],
        "width": width,
        "height": height,
        "window_rect": window_rect,
        "dwm_extended_frame_bounds": get_dwm_extended_frame_bounds(int(hwnd)),
        "process_name": process_name,
        "process_path": process_path,
        "ignored": ignored_window,
    }


def get_top_window_info(user32: Any) -> Optional[dict[str, Any]]:
    hwnd = user32.GetTopWindow(0)
    checked = 0
    while hwnd and checked < 80:
        info = get_window_info(user32, hwnd)
        if info:
            return info
        hwnd = user32.GetWindow(hwnd, 2)
        checked += 1
    return None


def get_ignored_window_infos() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []

    try:
        user32 = ctypes.windll.user32
        ignored: list[dict[str, Any]] = []

        enum_proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @enum_proc_type
        def enum_window(hwnd: int, _lparam: int) -> bool:
            info = get_window_info(user32, int(hwnd), allow_ignored=True)
            if info and info.get("ignored"):
                ignored.append(info)

            return True

        user32.EnumWindows(enum_window, 0)
        return ignored
    except Exception:
        return []


def hide_ignored_windows_for_capture(enabled: bool) -> list[dict[str, Any]]:
    if not enabled or os.name != "nt":
        return []

    try:
        user32 = ctypes.windll.user32
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.ShowWindow.restype = wintypes.BOOL
        user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        user32.SetWindowPos.restype = wintypes.BOOL
        hidden: list[dict[str, Any]] = []
        swp_no_size = 0x0001
        swp_no_zorder = 0x0004
        swp_no_activate = 0x0010
        move_flags = swp_no_size | swp_no_zorder | swp_no_activate
        for window in get_ignored_window_infos():
            hwnd_value = int(window.get("hwnd") or 0)
            if not hwnd_value:
                continue
            hwnd = wintypes.HWND(hwnd_value)
            hidden.append(
                {
                    "hwnd": hwnd_value,
                    "visible": bool(window.get("visible")),
                    "left": int(window.get("left") or 0),
                    "top": int(window.get("top") or 0),
                }
            )
            user32.SetWindowPos(hwnd, wintypes.HWND(0), -32000, -32000, 0, 0, move_flags)
            if window.get("visible"):
                user32.ShowWindow(hwnd, 0)  # SW_HIDE
        if hidden:
            try:
                ctypes.windll.dwmapi.DwmFlush()
            except Exception:
                pass
            time.sleep(0.18)
        return hidden
    except Exception as exc:
        print(f"Could not hide companion windows before capture: {exc}")
        return []


def capture_hide_fallback_enabled() -> bool:
    return os.environ.get("IGPU_CAPTURE_HIDE_FALLBACK", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def restore_hidden_windows_after_capture(hidden_windows: list[dict[str, Any]]) -> None:
    if not hidden_windows or os.name != "nt":
        return

    try:
        user32 = ctypes.windll.user32
        user32.IsWindow.argtypes = [wintypes.HWND]
        user32.IsWindow.restype = wintypes.BOOL
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.ShowWindow.restype = wintypes.BOOL
        user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        user32.SetWindowPos.restype = wintypes.BOOL
        swp_no_size = 0x0001
        swp_no_zorder = 0x0004
        swp_no_activate = 0x0010
        move_flags = swp_no_size | swp_no_zorder | swp_no_activate
        for window in reversed(hidden_windows):
            hwnd = wintypes.HWND(int(window.get("hwnd") or 0))
            if user32.IsWindow(hwnd):
                if window.get("visible"):
                    user32.SetWindowPos(
                        hwnd,
                        wintypes.HWND(0),
                        int(window.get("left") or 0),
                        int(window.get("top") or 0),
                        0,
                        0,
                        move_flags,
                    )
                    user32.ShowWindow(hwnd, 4)  # SW_SHOWNOACTIVATE
        try:
            ctypes.windll.dwmapi.DwmFlush()
        except Exception:
            pass
        time.sleep(0.04)
    except Exception as exc:
        print(f"Could not restore companion windows after capture: {exc}")


def crop_to_foreground_window(
    img: Image.Image,
    monitor: dict[str, int],
    window: dict[str, Any],
) -> Optional[Image.Image]:
    monitor_left = int(monitor.get("left", 0))
    monitor_top = int(monitor.get("top", 0))
    monitor_right = monitor_left + int(monitor["width"])
    monitor_bottom = monitor_top + int(monitor["height"])

    left = max(int(window["left"]), monitor_left)
    top = max(int(window["top"]), monitor_top)
    right = min(int(window["right"]), monitor_right)
    bottom = min(int(window["bottom"]), monitor_bottom)

    if right - left < 320 or bottom - top < 240:
        return None

    return img.crop(
        (
            left - monitor_left,
            top - monitor_top,
            right - monitor_left,
            bottom - monitor_top,
        )
    )


def get_capture_bounds(
    monitor: dict[str, int],
    window: Optional[dict[str, Any]] = None,
) -> tuple[int, int, int, int]:
    monitor_left = int(monitor.get("left", 0))
    monitor_top = int(monitor.get("top", 0))
    monitor_right = monitor_left + int(monitor["width"])
    monitor_bottom = monitor_top + int(monitor["height"])

    if not window:
        return monitor_left, monitor_top, monitor_right, monitor_bottom

    return (
        max(int(window["left"]), monitor_left),
        max(int(window["top"]), monitor_top),
        min(int(window["right"]), monitor_right),
        min(int(window["bottom"]), monitor_bottom),
    )


def intersection_area(
    a_left: int,
    a_top: int,
    a_right: int,
    a_bottom: int,
    b_left: int,
    b_top: int,
    b_right: int,
    b_bottom: int,
) -> int:
    width = max(0, min(a_right, b_right) - max(a_left, b_left))
    height = max(0, min(a_bottom, b_bottom) - max(a_top, b_top))
    return width * height


def select_capture_monitor(
    monitors: list[dict[str, int]],
    window: Optional[dict[str, Any]] = None,
    preferred_monitor: Optional[int] = None,
) -> tuple[dict[str, int], int]:
    if not monitors:
        raise RuntimeError("No monitor found for screenshot capture.")

    if preferred_monitor is not None:
        if 0 <= preferred_monitor < len(monitors):
            return monitors[preferred_monitor], preferred_monitor
        raise RuntimeError(
            f"Requested monitor {preferred_monitor} is unavailable; found {max(0, len(monitors) - 1)} display(s)."
        )

    if not window:
        index = 1 if len(monitors) > 1 else 0
        return monitors[index], index

    best_index = 1 if len(monitors) > 1 else 0
    best_area = -1
    win_left = int(window["left"])
    win_top = int(window["top"])
    win_right = int(window["right"])
    win_bottom = int(window["bottom"])

    for index, monitor in enumerate(monitors[1:] or monitors):
        actual_index = index + 1 if len(monitors) > 1 else index
        mon_left = int(monitor.get("left", 0))
        mon_top = int(monitor.get("top", 0))
        mon_right = mon_left + int(monitor["width"])
        mon_bottom = mon_top + int(monitor["height"])
        area = intersection_area(
            win_left,
            win_top,
            win_right,
            win_bottom,
            mon_left,
            mon_top,
            mon_right,
            mon_bottom,
        )
        if area > best_area:
            best_area = area
            best_index = actual_index

    return monitors[best_index], best_index


@app.get("/monitors")
async def monitors_endpoint():
    try:
        with mss.mss() as sct:
            monitors = []
            for index, monitor in enumerate(sct.monitors):
                width = int(monitor["width"])
                height = int(monitor["height"])
                left = int(monitor.get("left", 0))
                top = int(monitor.get("top", 0))
                monitors.append(
                    {
                        "index": index,
                        "label": (
                            f"All displays {width}x{height}"
                            if index == 0
                            else f"Screen {index} {width}x{height}"
                        ),
                        "left": left,
                        "top": top,
                        "width": width,
                        "height": height,
                        "aggregate": index == 0,
                    }
                )
            return {"monitors": monitors}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def redact_ignored_windows(
    img: Image.Image,
    capture_bounds: tuple[int, int, int, int],
    *,
    enabled: Optional[bool] = None,
) -> tuple[Image.Image, int]:
    if enabled is None:
        enabled = os.environ.get("IGPU_REDACT_IGNORED_WINDOWS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    if not enabled:
        return img, 0

    ignored_windows = get_ignored_window_infos()
    if not ignored_windows:
        return img, 0

    capture_left, capture_top, capture_right, capture_bottom = capture_bounds
    redacted = img.copy()
    draw = ImageDraw.Draw(redacted)
    count = 0

    for window in ignored_windows:
        left = max(int(window["left"]), capture_left)
        top = max(int(window["top"]), capture_top)
        right = min(int(window["right"]), capture_right)
        bottom = min(int(window["bottom"]), capture_bottom)
        if right - left < 8 or bottom - top < 8:
            continue

        draw.rectangle(
            (
                left - capture_left,
                top - capture_top,
                right - capture_left,
                bottom - capture_top,
            ),
            fill=(8, 8, 10),
        )
        count += 1

    return redacted, count


def is_blank_capture(img: Image.Image) -> bool:
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    low, high = stat.extrema[0]
    return stat.mean[0] < 8 and high - low < 4


def capture_window_with_print_window(window: dict[str, Any]) -> Optional[Image.Image]:
    if os.name != "nt":
        return None

    hwnd_value = int(window.get("hwnd") or 0)
    width = int(window.get("width") or 0)
    height = int(window.get("height") or 0)
    if not hwnd_value or width < 320 or height < 240:
        return None

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    hwnd = wintypes.HWND(hwnd_value)

    user32.GetWindowDC.argtypes = [wintypes.HWND]
    user32.GetWindowDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    user32.PrintWindow.restype = wintypes.BOOL

    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.GetBitmapBits.argtypes = [wintypes.HBITMAP, ctypes.c_long, ctypes.c_void_p]
    gdi32.GetBitmapBits.restype = ctypes.c_long
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    gdi32.DeleteDC.restype = wintypes.BOOL

    hdc_window = user32.GetWindowDC(hwnd)
    if not hdc_window:
        return None

    hdc_mem = gdi32.CreateCompatibleDC(hdc_window)
    if not hdc_mem:
        user32.ReleaseDC(hwnd, hdc_window)
        return None

    bitmap = gdi32.CreateCompatibleBitmap(hdc_window, width, height)
    if not bitmap:
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(hwnd, hdc_window)
        return None

    old_bitmap = gdi32.SelectObject(hdc_mem, bitmap)
    try:
        ok = user32.PrintWindow(hwnd, hdc_mem, 0x00000002)
        if not ok:
            ok = user32.PrintWindow(hwnd, hdc_mem, 0)
        if not ok:
            return None

        buffer_size = width * height * 4
        buffer = ctypes.create_string_buffer(buffer_size)
        copied = gdi32.GetBitmapBits(bitmap, buffer_size, buffer)
        if copied <= 0:
            return None

        img = Image.frombuffer("RGB", (width, height), buffer, "raw", "BGRX", 0, 1).copy()
        if is_blank_capture(img):
            print(f"PrintWindow returned a blank image for hwnd={hwnd_value}")
            return None
        return img
    except Exception as exc:
        print(f"PrintWindow capture failed for hwnd={hwnd_value}: {exc}")
        return None
    finally:
        if old_bitmap:
            gdi32.SelectObject(hdc_mem, old_bitmap)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(hwnd, hdc_window)


def capture_window_image(window: dict[str, Any]) -> Optional[tuple[Image.Image, str]]:
    if os.name != "nt":
        return None

    hwnd = int(window.get("hwnd") or 0)
    if not hwnd:
        return None

    img = capture_window_with_print_window(window)
    if img is not None:
        return img, "print_window"

    try:
        img = ImageGrab.grab(window=hwnd)
    except Exception as exc:
        print(f"Window capture failed for hwnd={hwnd}: {exc}")
        return None

    if img.width < 320 or img.height < 240:
        return None
    img = img.convert("RGB")
    if is_blank_capture(img):
        print(f"Window capture returned a blank image for hwnd={hwnd}")
        return None
    return img, "image_grab_window"


def make_capture_source(
    *,
    mode: str,
    capture_method: str,
    img: Image.Image,
    capture_left: int,
    capture_top: int,
    window: Optional[dict[str, Any]] = None,
    monitor: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    capture_width = int(img.width)
    capture_height = int(img.height)
    source: dict[str, Any] = {
        "mode": mode,
        "capture_method": capture_method,
        "window_title": str(window.get("title") or "") if window else "",
        "bitmap_width": capture_width,
        "bitmap_height": capture_height,
        "capture_left": int(capture_left),
        "capture_top": int(capture_top),
        "capture_width": capture_width,
        "capture_height": capture_height,
        "capture_right": int(capture_left) + capture_width,
        "capture_bottom": int(capture_top) + capture_height,
        # Legacy aliases kept for older frontend builds.
        "left": int(capture_left),
        "top": int(capture_top),
        "width": capture_width,
        "height": capture_height,
    }
    if window:
        source["hwnd"] = int(window.get("hwnd") or 0)
        source["window_rect"] = window.get("window_rect") or {
            "left": int(window["left"]),
            "top": int(window["top"]),
            "right": int(window["right"]),
            "bottom": int(window["bottom"]),
            "width": int(window["width"]),
            "height": int(window["height"]),
        }
        source["dwm_extended_frame_bounds"] = window.get("dwm_extended_frame_bounds")
    if monitor:
        source["monitor"] = int(monitor.get("index") or 1)
        source["monitor_rect"] = {
            "left": int(monitor.get("left", 0)),
            "top": int(monitor.get("top", 0)),
            "width": int(monitor["width"]),
            "height": int(monitor["height"]),
        }
    return source


def screenshot_profile_settings(profile: str) -> tuple[int, int, str]:
    profile_name = (profile or "fast").strip().lower()
    if profile_name == "full":
        return 2560, 90, "PNG"
    if profile_name == "balanced":
        return 1280, 76, "JPEG"
    if profile_name == "turbo":
        return 768, 60, "JPEG"
    if profile_name == "state":
        return (
            bounded_int_env("IGPU_LIVE_STATE_MAX_LONG_EDGE", 640, 320, 1280),
            bounded_int_env("IGPU_LIVE_STATE_JPEG_QUALITY", 55, 35, 80),
            "JPEG",
        )
    return (
        bounded_int_env("IGPU_SCREENSHOT_LONG_EDGE", 960, 512, 2560),
        bounded_int_env("IGPU_SCREENSHOT_QUALITY", 65, 35, 95),
        os.environ.get("IGPU_SCREENSHOT_FORMAT", "JPEG"),
    )


def make_screenshot_response(img: Image.Image, source: dict[str, Any], profile: str) -> dict[str, Any]:
    long_edge, quality, image_format = screenshot_profile_settings(profile)
    model_img = resize_to_long_edge(img, long_edge)
    encoded, mime_type = encode_image_base64(
        model_img,
        image_format=image_format,
        quality=quality,
        save_path=LATEST_SCREENSHOT,
    )

    source = dict(source)
    source["bitmap_width"] = int(model_img.width)
    source["bitmap_height"] = int(model_img.height)
    source["model_width"] = int(model_img.width)
    source["model_height"] = int(model_img.height)
    source["model_long_edge"] = int(long_edge)
    source["model_mime_type"] = mime_type
    source["model_scale_x"] = float(model_img.width / max(1, int(source.get("capture_width") or img.width)))
    source["model_scale_y"] = float(model_img.height / max(1, int(source.get("capture_height") or img.height)))

    return {
        "image_base64": encoded,
        "mime_type": mime_type,
        "width": model_img.width,
        "height": model_img.height,
        "source": source,
        "debug_path": str(LATEST_SCREENSHOT),
        "profile": (profile or "fast").strip().lower(),
        "original_width": int(img.width),
        "original_height": int(img.height),
    }


def capture_screenshot_sync(
    mode: str = "foreground",
    redact: Optional[bool] = None,
    profile: str = "fast",
    monitor: Optional[int] = None,
) -> dict[str, Any]:
    mode_name = (mode or "foreground").lower()
    window = None
    if mode_name in {"foreground", "window"}:
        window = get_foreground_window_info()
        if not window:
            window = get_foreground_window_info()

    protection_enabled = (
        redact
        if redact is not None
        else os.environ.get("IGPU_REDACT_IGNORED_WINDOWS", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    hide_fallback = bool(protection_enabled) and capture_hide_fallback_enabled()
    hidden_hwnds = hide_ignored_windows_for_capture(hide_fallback)
    capture_protection_mode = (
        "hide_restore"
        if hidden_hwnds
        else "display_affinity"
        if protection_enabled
        else "off"
    )
    try:
        if mode_name == "window":
            if window:
                captured = capture_window_image(window)
                if captured is not None:
                    img, capture_method = captured
                    source = make_capture_source(
                        mode="window",
                        capture_method=capture_method,
                        img=img,
                        capture_left=int(window["left"]),
                        capture_top=int(window["top"]),
                        window=window,
                    )
                    source["ignored_overlay_windows"] = len(hidden_hwnds)
                    source["redacted_overlay_windows"] = 0
                    source["capture_protection"] = capture_protection_mode
                    return make_screenshot_response(img, source, profile)
                print("Direct window capture failed; falling back to screen crop.")
            else:
                print("No target window found; falling back to monitor capture.")

        with mss.mss() as sct:
            selected_monitor, monitor_index = select_capture_monitor(sct.monitors, window, monitor)
            sct_img = sct.grab(selected_monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            capture_bounds = get_capture_bounds(selected_monitor)
            source: dict[str, Any] = make_capture_source(
                mode="screen",
                capture_method="mss_monitor",
                img=img,
                capture_left=int(capture_bounds[0]),
                capture_top=int(capture_bounds[1]),
                monitor={**selected_monitor, "index": monitor_index},
            )

            if mode_name in {"foreground", "window"}:
                if window:
                    cropped = crop_to_foreground_window(img, selected_monitor, window)
                    if cropped:
                        img = cropped
                        capture_bounds = get_capture_bounds(selected_monitor, window)
                        source = make_capture_source(
                            mode=mode_name,
                            capture_method=(
                                "screen_crop_fallback" if mode_name == "window" else "screen_crop"
                            ),
                            img=img,
                            capture_left=int(capture_bounds[0]),
                            capture_top=int(capture_bounds[1]),
                            window=window,
                            monitor={**selected_monitor, "index": monitor_index},
                        )
                        if mode_name == "window":
                            source["direct_capture_failed"] = True

            source["ignored_overlay_windows"] = len(hidden_hwnds)
            source["redacted_overlay_windows"] = 0
            source["capture_protection"] = capture_protection_mode
            return make_screenshot_response(img, source, profile)
    finally:
        restore_hidden_windows_after_capture(hidden_hwnds)


@app.get("/screenshot")
async def screenshot_endpoint(
    mode: str = "foreground",
    redact: Optional[bool] = None,
    profile: str = "fast",
    monitor: Optional[int] = None,
):
    try:
        return await asyncio.to_thread(capture_screenshot_sync, mode, redact, profile, monitor)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def live_state_config() -> dict[str, Any]:
    return {
        "interval_ms": bounded_int_env("IGPU_LIVE_STATE_INTERVAL_MS", 3000, 1000, 60000),
        "min_change_score": max(
            0.0,
            min(1.0, float(os.environ.get("IGPU_LIVE_STATE_MIN_CHANGE_SCORE", "0.18") or 0.18)),
        ),
        "min_analyze_gap_ms": bounded_int_env(
            "IGPU_LIVE_STATE_MIN_ANALYZE_GAP_MS",
            8000,
            1000,
            120000,
        ),
        "fresh_ms": bounded_int_env("IGPU_LIVE_STATE_FRESH_MS", 30000, 5000, 300000),
        "timeout_seconds": bounded_int_env("IGPU_LIVE_STATE_TIMEOUT_SECONDS", 12, 4, 60),
    }


def live_state_model_name() -> str:
    if LOCAL_ROUTER_ENABLED:
        return LOCAL_ROUTER_MODEL
    return MODEL_ALIAS


def live_state_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def live_state_write(payload: dict[str, Any]) -> dict[str, Any]:
    global live_state_memory
    LIVE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    live_state_memory = dict(payload)
    LIVE_STATE_FILE.write_text(
        json.dumps(live_state_memory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return live_state_memory


def live_state_update_status(status: str, **extra: Any) -> dict[str, Any]:
    payload = dict(live_state_memory)
    payload.update(
        {
            "status": status,
            "enabled": live_state_enabled,
            "running": bool(live_state_task and not live_state_task.done()),
            "model": live_state_model_name(),
            "status_updated_at": live_state_now(),
        }
    )
    payload.update(extra)
    return live_state_write(payload)


def live_state_status_snapshot() -> dict[str, Any]:
    running = bool(live_state_task and not live_state_task.done())
    status = live_state_memory.get("status") or ("Watching" if live_state_enabled else "Off")
    if not live_state_enabled:
        status = "Off"
    elif live_state_busy:
        status = "Thinking"
    elif running and status in {"Off", "stopped"}:
        status = "Watching"
    return {
        "enabled": bool(live_state_enabled),
        "running": running,
        "status": status,
        "last_updated_at": live_state_memory.get("updated_at") or "",
        "last_analyze_at": live_state_memory.get("last_analyze_at") or "",
        "last_error": live_state_memory.get("last_error") or "",
        "last_skip_reason": live_state_memory.get("last_skip_reason") or "",
        "confidence": live_state_memory.get("confidence"),
        "scene": live_state_memory.get("scene") or "",
        "player_status": live_state_memory.get("player_status") or "",
        "possible_intent": live_state_memory.get("possible_intent") or "",
        "model": live_state_model_name(),
        "state_path": str(LIVE_STATE_FILE),
        "mode": live_state_mode,
        "monitor": live_state_monitor,
        "config": live_state_config(),
    }


def live_state_image_signature(image_base64: str) -> list[float]:
    raw = image_base64.split(",", 1)[-1]
    img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("L")
    img.thumbnail((32, 32), Image.Resampling.BILINEAR)
    histogram = img.histogram()
    bins = [sum(histogram[index : index + 16]) for index in range(0, 256, 16)]
    total = float(sum(bins) or 1.0)
    return [value / total for value in bins]


def live_state_change_score(signature: list[float]) -> float:
    if live_state_last_signature is None:
        return 1.0
    return min(1.0, sum(abs(a - b) for a, b in zip(signature, live_state_last_signature)) / 2.0)


def live_state_clean_list(value: Any, max_items: int = 8) -> list[str]:
    if isinstance(value, str):
        value = re.split(r"[,，、\n]+", value)
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "").strip())
        if text and text not in cleaned:
            cleaned.append(status_text(text, 48))
        if len(cleaned) >= max_items:
            break
    return cleaned


def live_state_enum(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else default


def normalize_live_state_payload(raw: dict[str, Any], source: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
    confidence = parse_loose_float(str(raw.get("confidence", 0.0)), 0.0)
    confidence = max(0.0, min(1.0, confidence))
    return {
        "status": "Watching",
        "enabled": bool(live_state_enabled),
        "running": bool(live_state_task and not live_state_task.done()),
        "model": live_state_model_name(),
        "scene": status_text(raw.get("scene") or "未知", 120),
        "visible_ui": live_state_clean_list(raw.get("visible_ui")),
        "visible_objects": live_state_clean_list(raw.get("visible_objects")),
        "player_status": live_state_enum(
            raw.get("player_status"),
            {"探索中", "戰鬥中", "解謎中", "未知"},
            "未知",
        ),
        "possible_intent": live_state_enum(
            raw.get("possible_intent"),
            {"找路", "戰鬥", "解謎", "道具確認", "未知"},
            "未知",
        ),
        "risk": live_state_enum(raw.get("risk"), {"none", "low", "medium", "high"}, "none"),
        "confidence": confidence,
        "updated_at": live_state_now(),
        "last_analyze_at": live_state_now(),
        "latency_ms": round(elapsed_ms, 1),
        "source": source,
        "last_error": "",
        "last_skip_reason": "",
    }


def live_state_messages(image_base64: str, mime_type: str) -> list[dict[str, Any]]:
    data_url = f"data:{mime_type or 'image/jpeg'};base64,{image_base64}"
    schema = (
        "Return exactly one minified JSON object. No markdown, no comments, no trailing text. "
        "Use Traditional Chinese values with this schema: "
        '{"scene":"...","visible_ui":["..."],"visible_objects":["..."],'
        '"player_status":"探索中|戰鬥中|解謎中|未知",'
        '"possible_intent":"找路|戰鬥|解謎|道具確認|未知",'
        '"risk":"none|low|medium|high","confidence":0.0}. '
        "Do not answer the player. Do not provide a guide. If uncertain, use 未知 and low confidence."
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a low-frequency local game-state observer for an in-game companion. "
                "You summarize the current screenshot as short structured state only."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": schema},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]


def call_live_state_qwen(messages: list[dict[str, Any]]) -> str:
    timeout_seconds = live_state_config()["timeout_seconds"]
    if not LOCAL_ROUTER_ENABLED:
        raise RuntimeError("local_qwen_router_unavailable")
    return call_local_router_once(messages, max_tokens=96, timeout_seconds=timeout_seconds, temperature=0.0)


def live_state_exception_detail(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        return f"HTTPError {exc.code}: {detail or exc.reason or exc}"
    return f"{type(exc).__name__}: {exc}"


def analyze_live_state_sync(
    force: bool = False,
    monitor: Optional[int] = None,
    mode: str = "foreground",
    reason: str = "manual",
) -> dict[str, Any]:
    global live_state_busy, live_state_last_signature, live_state_last_analyze_at, live_state_last_capture_at
    global live_state_error_count
    if live_state_busy:
        return live_state_update_status("Paused", last_skip_reason="analysis_already_running")
    if not force and generate_lock.locked():
        return live_state_update_status("Paused", last_skip_reason="backend_busy")

    config = live_state_config()
    now = time.perf_counter()
    if not force and live_state_last_analyze_at:
        elapsed_gap_ms = (now - live_state_last_analyze_at) * 1000
        if elapsed_gap_ms < config["min_analyze_gap_ms"]:
            return live_state_update_status(
                "Watching",
                last_skip_reason="cooldown",
                cooldown_remaining_ms=round(config["min_analyze_gap_ms"] - elapsed_gap_ms, 1),
            )

    live_state_busy = True
    live_state_update_status("Thinking", last_skip_reason="", trigger=reason)
    started = time.perf_counter()
    try:
        shot = capture_screenshot_sync(mode=mode, redact=True, profile="state", monitor=monitor)
        live_state_last_capture_at = time.perf_counter()
        signature = live_state_image_signature(shot["image_base64"])
        change_score = live_state_change_score(signature)
        if not force and change_score < config["min_change_score"]:
            live_state_last_signature = signature
            return live_state_update_status(
                "Watching",
                last_skip_reason="unchanged_frame",
                change_score=round(change_score, 4),
                source=shot.get("source", {}),
            )

        messages = live_state_messages(shot["image_base64"], shot.get("mime_type") or "image/jpeg")
        output = call_live_state_qwen(messages)
        try:
            parsed = extract_json_object(output)
        except Exception as parse_exc:
            live_state_last_signature = signature
            live_state_last_analyze_at = time.perf_counter()
            parse_error = f"parse_failed: {live_state_exception_detail(parse_exc)}"
            if live_state_memory.get("scene"):
                retained = dict(live_state_memory)
                retained.update(
                    {
                        "status": "uncertain",
                        "enabled": bool(live_state_enabled),
                        "running": bool(live_state_task and not live_state_task.done()),
                        "last_error": parse_error,
                        "last_skip_reason": "parse_failed_retained_previous",
                        "raw_model_output": status_text(output, 500),
                        "status_updated_at": live_state_now(),
                        "last_analyze_at": live_state_now(),
                        "trigger": reason,
                        "change_score": round(change_score, 4),
                        "consecutive_errors": 0,
                    }
                )
                live_state_error_count = 0
                return live_state_write(retained)
            raise RuntimeError(f"{parse_error}; raw={status_text(output, 240)}") from parse_exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = normalize_live_state_payload(parsed, shot.get("source", {}), elapsed_ms)
        payload["change_score"] = round(change_score, 4)
        payload["trigger"] = reason
        payload["raw_model_output"] = status_text(output, 500)
        live_state_error_count = 0

        previous_confidence = parse_loose_float(str(live_state_memory.get("confidence", 0.0)), 0.0)
        if payload["confidence"] < 0.25 and previous_confidence >= 0.55:
            retained = dict(live_state_memory)
            retained.update(
                {
                    "status": "uncertain",
                    "uncertain_observation": payload,
                    "last_error": "low_confidence_observation",
                    "last_skip_reason": "",
                    "status_updated_at": live_state_now(),
                    "trigger": reason,
                }
            )
            live_state_last_signature = signature
            live_state_last_analyze_at = time.perf_counter()
            return live_state_write(retained)

        live_state_last_signature = signature
        live_state_last_analyze_at = time.perf_counter()
        return live_state_write(payload)
    except Exception as exc:
        live_state_last_analyze_at = time.perf_counter()
        live_state_error_count += 1
        error_detail = live_state_exception_detail(exc)
        lower_detail = error_detail.lower()
        unsupported_image = "image input is not supported" in lower_detail or "mmproj" in lower_detail
        if unsupported_image:
            live_state_error_count = max(live_state_error_count, 3)
        status = (
            "capture_failed"
            if "screenshot" in lower_detail
            else "Paused"
            if unsupported_image
            else "Error"
        )
        return live_state_update_status(
            status,
            last_error=error_detail,
            consecutive_errors=live_state_error_count,
            trigger=reason,
            last_analyze_at=live_state_now(),
            last_skip_reason="image_input_unsupported" if unsupported_image else "",
        )
    finally:
        live_state_busy = False


async def live_state_worker() -> None:
    global live_state_enabled
    while live_state_enabled:
        if live_state_error_count >= 3:
            live_state_update_status("Paused", last_skip_reason="too_many_live_state_errors")
            await asyncio.sleep(live_state_config()["interval_ms"] / 1000)
            continue
        try:
            await asyncio.to_thread(
                analyze_live_state_sync,
                False,
                live_state_monitor,
                live_state_mode,
                "scheduled",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            live_state_update_status("Error", last_error=f"{type(exc).__name__}: {exc}")
        await asyncio.sleep(live_state_config()["interval_ms"] / 1000)


def ensure_live_state_task() -> None:
    global live_state_task
    if live_state_task and not live_state_task.done():
        return
    live_state_task = asyncio.create_task(live_state_worker())


def live_state_is_fresh(max_age_ms: Optional[int] = None) -> bool:
    updated_at = live_state_memory.get("updated_at")
    if not updated_at:
        return False
    try:
        dt = datetime.fromisoformat(str(updated_at))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    age_ms = (datetime.now(timezone.utc) - dt).total_seconds() * 1000
    return age_ms <= (max_age_ms or live_state_config()["fresh_ms"])


def live_state_prompt_needs_screen(prompt: str) -> bool:
    text = str(prompt or "").strip().lower()
    if not text:
        return False
    patterns = (
        "現在在哪",
        "現在該",
        "我在哪",
        "這畫面",
        "看畫面",
        "幫我看",
        "該怎麼做",
        "what am i looking at",
        "where am i",
        "what should i do",
    )
    return any(pattern in text for pattern in patterns)


def live_state_context_for_chat(prompt: str, enabled: bool) -> str:
    if not enabled or not live_state_enabled or not live_state_is_fresh():
        return ""
    state = live_state_memory
    lines = [
        "Live State context (background local Qwen observation; use only as supplemental current-screen context):",
        f"- scene: {state.get('scene') or '未知'}",
        f"- player_status: {state.get('player_status') or '未知'}",
        f"- possible_intent: {state.get('possible_intent') or '未知'}",
        f"- risk: {state.get('risk') or 'none'}",
        f"- confidence: {state.get('confidence')}",
    ]
    objects = state.get("visible_objects") or []
    visible_ui = state.get("visible_ui") or []
    if objects:
        lines.append(f"- visible_objects: {', '.join(objects[:8])}")
    if visible_ui:
        lines.append(f"- visible_ui: {', '.join(visible_ui[:6])}")
    if not live_state_prompt_needs_screen(prompt):
        lines.append("- note: the player did not explicitly ask about the current screen, so do not overuse this context.")
    return "\n".join(lines)


@app.post("/live-state/start")
async def live_state_start(request: LiveStateStartRequest = LiveStateStartRequest()):
    global live_state_enabled, live_state_monitor, live_state_mode, live_state_error_count
    live_state_enabled = True
    live_state_monitor = request.monitor
    live_state_mode = request.mode or "foreground"
    live_state_error_count = 0
    ensure_live_state_task()
    live_state_update_status("Watching", last_error="", last_skip_reason="", mode=request.mode, monitor=request.monitor)
    return live_state_status_snapshot()


@app.post("/live-state/stop")
async def live_state_stop():
    global live_state_enabled, live_state_task, live_state_error_count
    live_state_enabled = False
    live_state_error_count = 0
    if live_state_task and not live_state_task.done():
        live_state_task.cancel()
    live_state_task = None
    live_state_update_status("Off", last_skip_reason="stopped", last_error="", consecutive_errors=0)
    return live_state_status_snapshot()


@app.get("/live-state/status")
async def live_state_status():
    return live_state_status_snapshot()


@app.get("/live-state/current")
async def live_state_current():
    return {
        "status": live_state_status_snapshot(),
        "state": live_state_memory,
        "fresh": live_state_is_fresh(),
    }


@app.post("/live-state/analyze-now")
async def live_state_analyze_now(request: LiveStateAnalyzeRequest = LiveStateAnalyzeRequest()):
    result = await asyncio.to_thread(
        analyze_live_state_sync,
        bool(request.force),
        request.monitor,
        request.mode,
        "manual",
    )
    return {"status": live_state_status_snapshot(), "state": result, "fresh": live_state_is_fresh()}


@app.post("/intent/route")
async def intent_route_endpoint(request: IntentRouteRequest):
    prompt = request.message or ""
    game_id = normalize_game_id(request.game_id)
    guide_was_requested = should_use_guides(prompt, request.use_guides)
    started = time.perf_counter()
    try:
        decision = await asyncio.to_thread(
            local_router_gamepath_decision,
            prompt,
            game_id,
            guide_was_requested,
        )
    except Exception as exc:
        decision = fallback_gamepath_decision(
            prompt,
            game_id,
            guide_was_requested,
            f"router_failed:{type(exc).__name__}",
        )
        print(f"Local router intent endpoint failed: {exc}")
    route = str(decision.get("intent_route") or "").strip()
    if not route:
        if decision.get("prefer_hermes_agent"):
            route = "hermes_web"
        elif decision.get("search_gamepath"):
            route = "gamepath_query"
        else:
            route = "general_chat"
    return {
        "status": "ok",
        "route": route,
        "decision": decision,
        "game_id": game_id,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


@app.post("/chat")
async def chat_endpoint(fastapi_request: Request, chat_request: ChatRequest):
    prompt = chat_request.message or "請分析這張截圖。"
    game_id = normalize_game_id(chat_request.game_id)
    guide_was_requested = should_use_guides(prompt, chat_request.use_guides)
    use_live_state_context = bool(chat_request.use_live_state)
    if (
        use_live_state_context
        and live_state_enabled
        and not chat_request.image_base64
        and live_state_prompt_needs_screen(prompt)
        and not live_state_is_fresh()
    ):
        await asyncio.to_thread(
            analyze_live_state_sync,
            True,
            live_state_monitor,
            live_state_mode,
            "chat_request",
        )
    preflight_router_gamepath_result: Optional[dict[str, Any]] = None
    screenshot_intent_result: Optional[dict[str, Any]] = None
    if not chat_request.image_base64:
        try:
            preflight_router_gamepath_result = await asyncio.to_thread(
                local_router_gamepath_decision,
                prompt,
                game_id,
                guide_was_requested,
            )
        except Exception as exc:
            preflight_router_gamepath_result = fallback_gamepath_decision(
                prompt,
                game_id,
                guide_was_requested,
                f"router_failed:{type(exc).__name__}",
            )
            print(f"Local router user intent preflight failed: {exc}")
    else:
        if should_use_overlay(prompt, chat_request.image_base64):
            screenshot_intent_result = fallback_screenshot_intent_decision(
                prompt,
                game_id,
                guide_was_requested,
                "explicit_hud_overlay",
            )
        elif should_use_visual_scene(prompt) and not should_use_gamepath(prompt, guide_was_requested):
            screenshot_intent_result = fallback_screenshot_intent_decision(
                prompt,
                game_id,
                guide_was_requested,
                "explicit_visual_scene",
            )
        else:
            try:
                screenshot_intent_result = await asyncio.to_thread(
                    screenshot_intent_decision,
                    prompt,
                    game_id,
                    chat_request.image_base64,
                    guide_was_requested,
                )
            except Exception as exc:
                screenshot_intent_result = fallback_screenshot_intent_decision(
                    prompt,
                    game_id,
                    guide_was_requested,
                    f"vision_router_failed:{type(exc).__name__}",
                )
                print(f"Screenshot intent route failed: {exc}")

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

    implicit_memory = detect_implicit_memory_fact(prompt)
    if implicit_memory and not chat_request.image_base64:
        try:
            await asyncio.to_thread(
                add_memory_sync,
                implicit_memory["content"],
                game_id,
                implicit_memory["kind"],
                "auto",
                4,
            )
        except Exception as exc:
            print(f"Implicit memory write failed: {exc}")

    if detect_gamepath_dispute(prompt) and not chat_request.image_base64:
        ref = recent_gamepath_reference(game_id)
        if ref:
            async def gamepath_dispute_event_generator():
                item = await asyncio.to_thread(
                    update_gamepath_feedback_sync,
                    int(ref["entry_id"]),
                    prompt,
                    state="disputed",
                )
                if not item:
                    message = "我收到你的回報，但找不到上一筆 GamePath 條目可標記；請截圖目前畫面，我會重新判斷。"
                    yield lookup_status_event(
                        "gamepath_feedback_missing",
                        "找不到可標記的上一筆 GamePath 條目。",
                        source="gamepath",
                        fast_path=False,
                    )
                else:
                    message = build_gamepath_dispute_message(item, prompt)
                    yield lookup_status_event(
                        "gamepath_disputed",
                        "玩家回報上一個 GamePath 提示不符合，已降權並切換驗證模式。",
                        source="gamepath",
                        fast_path=False,
                        entry_id=item.get("id"),
                        trust_state=item.get("trust_state"),
                        dispute_count=item.get("dispute_count"),
                    )
                append_history("user", prompt)
                append_history("assistant", message)
                yield sse_data({"content": message})

            return StreamingResponse(gamepath_dispute_event_generator(), media_type="text/event-stream")

    local_router_gamepath_result: dict[str, Any] = {
        "used": False,
        "search_gamepath": False,
        "query": str(prompt or "").strip(),
        "tags": [],
        "spoiler_level": "low",
        "confidence": "none",
        "reason": "not_needed",
    }
    if preflight_router_gamepath_result is not None:
        local_router_gamepath_result = preflight_router_gamepath_result
    elif not chat_request.image_base64:
        try:
            local_router_gamepath_result = await asyncio.to_thread(
                local_router_gamepath_decision,
                prompt,
                game_id,
                guide_was_requested,
            )
        except Exception as exc:
            local_router_gamepath_result = fallback_gamepath_decision(
                prompt,
                game_id,
                guide_was_requested,
                f"router_failed:{type(exc).__name__}",
            )
            print(f"Local router GamePath gate failed: {exc}")
    else:
        local_router_gamepath_result = screenshot_intent_result or fallback_screenshot_intent_decision(
            prompt,
            game_id,
            guide_was_requested,
            "image_route_skipped",
        )
    gamepath_was_requested = bool(local_router_gamepath_result.get("search_gamepath"))
    gamepath_search_query = str(local_router_gamepath_result.get("query") or prompt or "").strip()
    gamepath_query_variants = list(local_router_gamepath_result.get("query_variants") or [])
    gamepath_answer_prompt = (
        screenshot_gamepath_prompt(prompt, screenshot_intent_result)
        if chat_request.image_base64 and screenshot_intent_result
        else prompt
    )
    adaptive_gamepath_followup = is_gamepath_adaptive_followup(prompt)
    tactical_gamepath_reframe = is_gamepath_tactical_reframe(prompt)
    vague_gamepath_followup = is_gamepath_vague_adaptive_followup(prompt)
    if tactical_gamepath_reframe:
        gamepath_query_variants = augment_adaptive_gamepath_query_variants(
            prompt,
            gamepath_search_query,
            gamepath_query_variants,
        )
    gamepath_search_tags = normalize_router_tags(local_router_gamepath_result.get("tags"))
    gamepath_spoiler_level = normalize_router_spoiler(local_router_gamepath_result.get("spoiler_level"))
    if gamepath_was_requested and gamepath_spoiler_level == "none":
        gamepath_spoiler_level = "low"
        local_router_gamepath_result["spoiler_search_floor"] = "none->low"
    if gamepath_was_requested:
        guide_was_requested = True
    gamepath_search_elapsed_ms: Optional[float] = None
    memory_results: list[dict[str, Any]] = []
    guide_results: list[dict[str, Any]] = []
    gamepath_results: list[dict[str, Any]] = []
    if chat_request.use_memory:
        memory_kinds = memory_kinds_for_chat(prompt, has_image=bool(chat_request.image_base64))
        memory_results = await asyncio.to_thread(search_memory_sync, prompt, game_id, memory_kinds, 8)
    gamepath_evaluation: dict[str, Any] = {
        "confidence": "skipped",
        "score": 0.0,
        "gap": 0.0,
        "reason": "intent_gate_skipped",
        "results": [],
        "local_router": local_router_gamepath_result,
        "search_query": gamepath_search_query,
        "query_variants": gamepath_query_variants,
        "search_tags": gamepath_search_tags,
        "spoiler_level": gamepath_spoiler_level,
        "game_id": game_id,
    }
    if gamepath_was_requested:
        gamepath_search_started = time.perf_counter()
        recent_ref = recent_gamepath_reference(game_id) if vague_gamepath_followup else None
        if recent_ref:
            recent_item = await asyncio.to_thread(get_gamepath_entry_sync, int(recent_ref.get("entry_id") or 0))
            if recent_item:
                recent_item.update(build_gamepath_relevant_context(gamepath_search_query or prompt, recent_item))
                recent_item["matched_query"] = "recent_followup"
                recent_item["match_coverage"] = max(float(recent_item.get("match_coverage") or 0.0), 0.35)
                gamepath_results = [recent_item]
                local_router_gamepath_result["recent_followup_entry_id"] = recent_ref.get("entry_id")
                local_router_gamepath_result["recent_followup"] = True
        if not gamepath_results:
            gamepath_results = await asyncio.to_thread(
                search_gamepath_multi_query_sync,
                gamepath_search_query,
                game_id,
                5,
                tags=gamepath_search_tags or None,
                spoiler_level=gamepath_spoiler_level,
                query_variants=gamepath_query_variants,
            )
        if not gamepath_results and gamepath_search_tags:
            gamepath_results = await asyncio.to_thread(
                search_gamepath_multi_query_sync,
                gamepath_search_query,
                game_id,
                5,
                spoiler_level=gamepath_spoiler_level,
                query_variants=gamepath_query_variants,
            )
            local_router_gamepath_result["tag_filter_fallback"] = True
        relaxed_spoiler_level = gamepath_explicit_spoiler_fallback_level(
            prompt,
            gamepath_search_query,
            gamepath_spoiler_level,
        )
        if not gamepath_results and relaxed_spoiler_level != gamepath_spoiler_level:
            gamepath_results = await asyncio.to_thread(
                search_gamepath_multi_query_sync,
                gamepath_search_query,
                game_id,
                5,
                tags=gamepath_search_tags or None,
                spoiler_level=relaxed_spoiler_level,
                query_variants=gamepath_query_variants,
            )
            if not gamepath_results and gamepath_search_tags:
                gamepath_results = await asyncio.to_thread(
                    search_gamepath_multi_query_sync,
                    gamepath_search_query,
                    game_id,
                    5,
                    spoiler_level=relaxed_spoiler_level,
                    query_variants=gamepath_query_variants,
                )
            if gamepath_results:
                local_router_gamepath_result["spoiler_fallback"] = f"{gamepath_spoiler_level}->{relaxed_spoiler_level}"
                gamepath_spoiler_level = relaxed_spoiler_level
        gamepath_search_elapsed_ms = round((time.perf_counter() - gamepath_search_started) * 1000, 1)
        gamepath_evaluation = evaluate_gamepath_retrieval(gamepath_search_query, game_id, gamepath_results)
        gamepath_results = list(gamepath_evaluation.get("results") or gamepath_results)
        gamepath_evaluation["local_router"] = local_router_gamepath_result
        gamepath_evaluation["search_query"] = gamepath_search_query
        gamepath_evaluation["query_variants"] = gamepath_query_variants
        gamepath_evaluation["search_tags"] = gamepath_search_tags
        gamepath_evaluation["spoiler_level"] = gamepath_spoiler_level
        gamepath_evaluation["game_id"] = game_id
        gamepath_evaluation["search_elapsed_ms"] = gamepath_search_elapsed_ms
        gamepath_evaluation["search_scope"] = normalize_gamepath_scope(
            gamepath_search_query,
            gamepath_search_tags,
        )
        if (
            gamepath_results
            and LOCAL_ROUTER_RETRIEVAL_EVAL
            and not chat_request.image_base64
            and str(gamepath_evaluation.get("confidence") or "") != "direct"
        ):
            try:
                retrieval_decision = await asyncio.to_thread(
                    local_router_retrieval_decision,
                    gamepath_search_query,
                    game_id,
                    gamepath_evaluation,
                )
                gamepath_evaluation = apply_local_router_retrieval_decision(
                    gamepath_evaluation,
                    retrieval_decision,
                )
                gamepath_results = list(gamepath_evaluation.get("results") or gamepath_results)
            except Exception as exc:
                gamepath_evaluation["local_router_retrieval"] = {
                    "used": True,
                    "reason": f"retrieval_router_failed:{type(exc).__name__}",
                }
                print(f"Local router GamePath retrieval eval failed: {exc}")
        if tactical_gamepath_reframe and gamepath_results:
            gamepath_evaluation["confidence"] = "summarize"
            gamepath_evaluation["reason"] = "tactical_followup_requires_model_reframe"
            gamepath_evaluation["score"] = max(0.5, float(gamepath_evaluation.get("score") or 0.0))
            local_router_gamepath_result["adaptive_followup"] = adaptive_gamepath_followup
            local_router_gamepath_result["tactical_reframe"] = True
    gamepath_route = str(gamepath_evaluation.get("confidence") or "skipped") if gamepath_was_requested else "skipped"
    gamepath_context_results = gamepath_results if gamepath_route in {"direct", "summarize"} else []
    if guide_was_requested:
        guide_results = await asyncio.to_thread(search_guides_sync, prompt, game_id, 5)

    fact_answer = answer_fact_lookup(prompt, memory_results)
    if fact_answer and not chat_request.image_base64:
        async def fact_answer_event_generator():
            append_history("user", prompt)
            append_history("assistant", fact_answer)
            yield f"data: {json.dumps({'content': fact_answer}, ensure_ascii=False)}\n\n"
            yield response_format_event(fact_answer)

        return StreamingResponse(fact_answer_event_generator(), media_type="text/event-stream")

    if gamepath_route == "direct":
        async def gamepath_answer_event_generator():
            top_item = gamepath_results[0]
            remember_gamepath_reference(top_item, route="direct")
            yield lookup_status_event(
                "gamepath_hit",
                "GamePath 命中，正在用地端 Qwen 整理成提示。",
                source="gamepath",
                web_search=False,
                fast_path=True,
                hits=len(gamepath_results),
                retrieval_score=gamepath_evaluation.get("score", 0.0),
                retrieval_gap=gamepath_evaluation.get("gap", 0.0),
                retrieval_reason=gamepath_evaluation.get("reason", ""),
                local_router=gamepath_evaluation.get("local_router"),
                local_router_retrieval=gamepath_evaluation.get("local_router_retrieval"),
                **gamepath_lookup_status_details(game_id, gamepath_evaluation),
            )
            hint_started = time.perf_counter()
            raw_answer = await asyncio.to_thread(build_gamepath_hint_answer, gamepath_answer_prompt, top_item)
            display_format = build_response_format(raw_answer)
            answer = display_format["long"]
            hint_elapsed_ms = round((time.perf_counter() - hint_started) * 1000, 1)
            yield lookup_status_event(
                "gamepath_hit",
                "GamePath 已用地端 Qwen 整理成提示。",
                source="gamepath",
                web_search=False,
                fast_path=True,
                hits=len(gamepath_results),
                retrieval_score=gamepath_evaluation.get("score", 0.0),
                retrieval_gap=gamepath_evaluation.get("gap", 0.0),
                retrieval_reason=gamepath_evaluation.get("reason", ""),
                gamepath_hint_ms=hint_elapsed_ms,
                local_router=gamepath_evaluation.get("local_router"),
                local_router_retrieval=gamepath_evaluation.get("local_router_retrieval"),
                **gamepath_lookup_status_details(game_id, gamepath_evaluation),
            )
            append_history("user", prompt)
            append_history("assistant", answer)
            yield sse_data({"content": answer})
            yield sse_data({"response_format": display_format})

        return StreamingResponse(gamepath_answer_event_generator(), media_type="text/event-stream")

    if (
        guide_was_requested
        and not guide_results
        and not gamepath_context_results
        and not chat_request.image_base64
        and not (CHAT_BACKEND == "hermes" and HERMES_AGENT_WEB_ENABLED)
    ):
        async def no_guide_event_generator():
            game_label = game_id or "目前遊戲"
            message = f"本機攻略庫沒有找到「{game_label}」相關條目。你可以把攻略 .md/.txt/.html 放進 game_guides/{game_label}/ 後重建索引。"
            if memory_results:
                message += "\n我有找到一些玩家記憶，但沒有本機攻略片段，所以不會硬編攻略。"
            yield f"data: {json.dumps({'content': message}, ensure_ascii=False)}\n\n"
            yield response_format_event(message)

        return StreamingResponse(no_guide_event_generator(), media_type="text/event-stream")

    rag_context = format_rag_context(guide_results, memory_results, guide_was_requested, gamepath_context_results)
    live_state_context = live_state_context_for_chat(prompt, use_live_state_context)
    if live_state_context:
        rag_context = f"{live_state_context}\n\n{rag_context}" if rag_context else live_state_context
    if gamepath_route == "summarize" and rag_context:
        if tactical_gamepath_reframe:
            rag_context = (
                "GamePath tactical follow-up: the player needs a different playable tactic, not a repeated route summary. "
                "Reframe the local passage into an easier strategy: safer positioning, "
                "resource-saving method, when to avoid fighting, and one concrete next action. If the local entries do not "
                "contain combat details, say so briefly and offer a practical fallback.\n"
                f"{rag_context}"
            )
        else:
            rag_context = (
                "GamePath route: matching local GamePath entries were found. Summarize these entries first. "
                "Do not use Tavily unless the local entries are clearly insufficient for the player's question.\n"
                f"{rag_context}"
            )
    model_prompt = gamepath_answer_prompt if chat_request.image_base64 and gamepath_was_requested else prompt
    augmented_prompt = build_augmented_prompt(model_prompt, rag_context)

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
        active_game_context = None
        if HERMES_AGENT_WEB_ENABLED:
            try:
                active_game_context = await asyncio.to_thread(detect_active_game_sync)
            except Exception as exc:
                print(f"Active game context for Hermes web agent failed: {exc}")
        hermes_prompt = (
            build_hermes_agent_web_prompt(
                prompt,
                game_id=game_id,
                rag_context=rag_context,
                active_game=active_game_context,
            )
            if HERMES_AGENT_WEB_ENABLED
            else build_hermes_prompt(augmented_prompt)
        )
        hermes_call = call_hermes_web_agent if HERMES_AGENT_WEB_ENABLED else call_hermes_no_tools

        async def hermes_event_generator():
            collected = ""
            stage, status_message, status_extra = lookup_route_stage(
                game_id=game_id,
                hermes_agent_web_enabled=HERMES_AGENT_WEB_ENABLED,
                gamepath_requested=gamepath_was_requested,
                gamepath_raw_hits=len(gamepath_results),
                gamepath_evaluation=gamepath_evaluation,
                gamepath_results=gamepath_context_results,
                guide_results=guide_results,
                memory_results=memory_results,
            )
            if gamepath_context_results:
                remember_gamepath_reference(gamepath_context_results[0], route="summarize")
            yield lookup_status_event(stage, status_message, **status_extra)
            async with generate_lock:
                try:
                    collected = await asyncio.to_thread(hermes_call, hermes_prompt)
                except subprocess.TimeoutExpired:
                    error = "Hermes 回應逾時。請稍後再試，或把 IGPU_CHAT_BACKEND 改成 llama 先走直接模型。"
                    yield f"data: {json.dumps({'content': error}, ensure_ascii=False)}\n\n"
                    return
                except Exception as exc:
                    error = f"Hermes 連線失敗：{exc}"
                    yield f"data: {json.dumps({'content': error}, ensure_ascii=False)}\n\n"
                    return

            if collected:
                if HERMES_AGENT_WEB_ENABLED:
                    collected = condense_agent_answer(collected, prompt)
                display_format = build_response_format(collected)
                collected = display_format["long"]
                store_game_id = resolve_gamepath_store_game_id(game_id, active_game_context)
                store_skip_reason = (
                    "tactical_followup_not_reusable"
                    if tactical_gamepath_reframe
                    else gamepath_store_skip_reason(
                        prompt,
                        collected,
                        store_game_id,
                        HERMES_AGENT_WEB_ENABLED,
                    )
                )
                if not store_skip_reason:
                    try:
                        stored_item = await asyncio.to_thread(
                            add_gamepath_sync,
                            prompt,
                            collected,
                            store_game_id,
                            title=prompt[:80],
                            tags=["auto", "hermes", "guide"],
                            spoiler_level="low",
                            source_type="hermes_agent_web",
                            agent_used=True,
                        )
                        yield gamepath_store_lookup_status_event(stored_item, store_game_id)
                    except Exception as exc:
                        print(f"GamePath auto-store failed: {exc}")
                elif gamepath_was_requested and HERMES_AGENT_WEB_ENABLED:
                    yield lookup_status_event(
                        "gamepath_not_stored",
                        "這次回答沒有符合可重用攻略條件，所以沒有寫入 GamePath。",
                        source="gamepath",
                        web_search=False,
                        fast_path=False,
                        game_id=store_game_id,
                        reason=store_skip_reason,
                    )
                append_history("user", prompt)
                append_history("assistant", collected)
                for chunk in chunk_text(collected):
                    if await fastapi_request.is_disconnected():
                        break
                    yield sse_data({"content": chunk})
                    await asyncio.sleep(0)
                yield sse_data({"response_format": display_format})

        return StreamingResponse(hermes_event_generator(), media_type="text/event-stream")

    if should_use_overlay(prompt, chat_request.image_base64) or (
        bool(chat_request.image_base64)
        and str((local_router_gamepath_result or {}).get("intent_route") or "") == "screenshot_hud"
    ):
        async def overlay_event_generator():
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
                display_format = build_response_format(answer)
                answer = display_format["long"]
                append_history("user", prompt)
                append_history("assistant", answer)
                for chunk in chunk_text(answer):
                    if await fastapi_request.is_disconnected():
                        return
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0)
                yield sse_data({"response_format": display_format})
            if overlay:
                yield f"data: {json.dumps({'overlay': overlay}, ensure_ascii=False)}\n\n"

        return StreamingResponse(overlay_event_generator(), media_type="text/event-stream")

    if chat_request.image_base64:
        async def image_event_generator():
            await asyncio.sleep(0)
            ocr_text = ""
            if ENABLE_OCR_CONTEXT:
                ocr_text = await asyncio.to_thread(extract_ocr_text, chat_request.image_base64)
            if gamepath_was_requested or gamepath_context_results:
                stage, status_message, status_extra = lookup_route_stage(
                    game_id=game_id,
                    hermes_agent_web_enabled=HERMES_AGENT_WEB_ENABLED,
                    gamepath_requested=gamepath_was_requested,
                    gamepath_raw_hits=len(gamepath_results),
                    gamepath_evaluation=gamepath_evaluation,
                    gamepath_results=gamepath_context_results,
                    guide_results=guide_results,
                    memory_results=memory_results,
                )
                if gamepath_context_results:
                    remember_gamepath_reference(gamepath_context_results[0], route="summarize")
                yield lookup_status_event(stage, status_message, **status_extra)
            if CHAT_BACKEND == "hermes" and HERMES_AGENT_WEB_ENABLED:
                try:
                    await asyncio.to_thread(image_to_data_url, chat_request.image_base64)
                    active_game_context = await asyncio.to_thread(detect_active_game_sync)
                    agent_prompt = build_hermes_agent_web_prompt(
                        model_prompt,
                        game_id=game_id,
                        rag_context=rag_context,
                        active_game=active_game_context,
                    )
                    answer = await asyncio.to_thread(
                        call_hermes_web_agent,
                        agent_prompt,
                        image_file=LATEST_VISION_INPUT,
                        max_tokens=HERMES_AGENT_MAX_TOKENS,
                    )
                except Exception as exc:
                    yield f"data: {json.dumps({'content': f'Agent vision/search failed: {exc}'}, ensure_ascii=False)}\n\n"
                    return

                answer = condense_agent_answer(answer, prompt)
                store_game_id = resolve_gamepath_store_game_id(game_id, active_game_context)
                store_question = gamepath_answer_prompt if gamepath_was_requested else prompt
                if should_store_gamepath_answer(store_question, answer, store_game_id, True):
                    try:
                        stored_item = await asyncio.to_thread(
                            add_gamepath_sync,
                            store_question,
                            answer,
                            store_game_id,
                            title=(gamepath_search_query or prompt)[:80],
                            tags=["auto", "vision", "hermes", "guide"],
                            spoiler_level="low",
                            source_type="hermes_agent_vision",
                            agent_used=True,
                        )
                        yield gamepath_store_lookup_status_event(stored_item, store_game_id)
                    except Exception as exc:
                        print(f"GamePath vision auto-store failed: {exc}")
                if not answer:
                    answer = "這次沒有產生可用回覆；請換個問法，或指定要看的畫面位置。"
                display_format = build_response_format(answer)
                answer = display_format["long"]
                append_history("user", prompt)
                append_history("assistant", answer)
                yield f"data: {json.dumps({'content': answer}, ensure_ascii=False)}\n\n"
                yield sse_data({"response_format": display_format})
                return

            visual_scene_request = should_use_visual_scene(prompt)
            if visual_scene_request:
                visual_prompt = (
                    f"{augmented_prompt}\n\n"
                    "這次請優先看圖片像素本身，不要只讀畫面文字。"
                    "請列出畫面中可見的主要物件、人物、UI 元素、場景或區域；"
                    "如果不確定，就說你能辨識到的形狀、位置或大概類型。"
                )
                messages = build_messages(visual_prompt, chat_request.image_base64, ocr_text)
            else:
                messages = build_messages(augmented_prompt, chat_request.image_base64, ocr_text)
            async with generate_lock:
                try:
                    image_tokens = int(
                        os.environ.get(
                            "LLAMA_IMAGE_RESPONSE_TOKENS",
                            os.environ.get("LLAMA_IMAGE_MAX_TOKENS", "64"),
                        )
                    )
                    if CHAT_BACKEND == "hermes" and HERMES_USE_CONFIG_MODEL:
                        answer = await asyncio.to_thread(
                            call_hermes_messages,
                            messages,
                            image_file=LATEST_VISION_INPUT,
                            max_tokens=max(image_tokens, 96),
                        )
                    else:
                        answer = await asyncio.to_thread(
                            call_llama_once,
                            messages,
                            image_tokens,
                        )
                except Exception as exc:
                    yield f"data: {json.dumps({'content': f'截圖分析失敗：{exc}'}, ensure_ascii=False)}\n\n"
                    return

            answer = clean_vision_answer(answer)
            if not answer:
                answer = "這張截圖我看不出可靠重點；你可以直接指定要我看哪個位置或 UI。"
            display_format = build_response_format(answer)
            answer = display_format["long"]
            append_history("user", prompt)
            append_history("assistant", answer)
            yield f"data: {json.dumps({'content': answer}, ensure_ascii=False)}\n\n"
            yield sse_data({"response_format": display_format})

        return StreamingResponse(image_event_generator(), media_type="text/event-stream")

    messages = build_messages(augmented_prompt, None)

    payload = {
        "model": MODEL_ALIAS,
        "messages": messages,
        "stream": True,
        "max_tokens": int(
            os.environ.get(
                "LLAMA_IMAGE_MAX_TOKENS" if chat_request.image_base64 else "LLAMA_MAX_TOKENS",
                "160" if chat_request.image_base64 else "128",
            )
        ),
        "temperature": 0.55,
        "top_p": 0.85,
        "top_k": 20,
        "repeat_penalty": 1.0,
    }

    async def event_generator():
        collected = ""
        if chat_request.image_base64:
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
                yield response_format_event(collected)
            elif not await fastapi_request.is_disconnected():
                fallback = "這次沒有產生可用回覆；請換個問法，或指定要看的畫面位置。"
                append_history("user", prompt)
                append_history("assistant", fallback)
                yield f"data: {json.dumps({'content': fallback}, ensure_ascii=False)}\n\n"
                yield response_format_event(fallback)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
