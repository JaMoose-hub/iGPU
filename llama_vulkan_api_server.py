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

llama_process: Optional[subprocess.Popen] = None
history: list[dict[str, Any]] = []
last_gamepath_reference: dict[str, Any] = {}
last_active_game_window: Optional[dict[str, Any]] = None
last_active_game_detection: Optional[dict[str, Any]] = None
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


class GamePathSearchRequest(BaseModel):
    query: str
    game_id: Optional[str] = None
    tags: Any = None
    spoiler_level: str = "low"
    limit: int = 5


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
        "for the full solution. Prefer a hint ladder: Hint 1, Hint 2, Hint 3. If the player asks for the "
        "answer directly, give a clear solution but still avoid unnecessary story spoilers. When you do "
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


def lookup_status_event(stage: str, message: str, **extra: Any) -> str:
    payload = {"stage": stage, "message": message}
    payload.update(extra)
    return sse_data({"lookup_status": payload})


def llama_ready() -> bool:
    try:
        get_json(f"{llama_base_url()}/v1/models", timeout=2)
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

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]
    return json.loads(stripped)


def call_llama_chat_payload(payload: dict[str, Any]) -> str:
    with post_json(f"{llama_base_url()}/v1/chat/completions", payload, 600) as response:
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
GAMEPATH_TRUST_STATES = {"unverified", "verified", "disputed", "needs_review", "deprecated"}
SPOILER_RANKS = {"none": 0, "low": 1, "medium": 2, "high": 3, "full": 4}
GAMEPATH_DIRECT_MAX_CHARS = 900
GAMEPATH_CONTEXT_MAX_CHARS = 1800
GAMEPATH_PASSAGE_MAX_CHARS = 900
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
                spoiler_level TEXT NOT NULL DEFAULT 'low',
                spoiler_rank INTEGER NOT NULL DEFAULT 1,
                source_type TEXT NOT NULL DEFAULT 'manual',
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
        conn.execute("PRAGMA user_version = 2")
        conn.commit()


def render_gamepath_markdown(row: dict[str, Any]) -> str:
    tags = [tag for tag in str(row.get("tags") or "").split(",") if tag]
    tag_text = " ".join(f"#{tag}" for tag in tags)
    return (
        f"# {row.get('title') or 'GamePath Entry'}\n\n"
        f"- Game: {row.get('game_id') or 'global'}\n"
        f"- Source: {row.get('source_type') or 'manual'}\n"
        f"- Agent used: {'yes' if row.get('agent_used') else 'no'}\n"
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
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    content_hash = gamepath_content_hash(normalized_game_id, clean_question)

    with sqlite3.connect(GAMEPATH_DB) as conn:
        conn.row_factory = sqlite3.Row
        existing = conn.execute(
            "SELECT * FROM gamepath_entries WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        if existing:
            entry_id = int(existing["id"])
            markdown_path = existing["markdown_path"] or ""
            existing_dispute_count = int(existing["dispute_count"] or 0)
            conn.execute(
                """
                UPDATE gamepath_entries
                SET title = ?, question = ?, answer_summary = ?, tags = ?, spoiler_level = ?,
                    spoiler_rank = ?, source_type = ?, agent_used = ?, trust_state = ?,
                    last_feedback = '', last_feedback_at = '', updated_at = ?
                WHERE id = ?
                """,
                (
                    clean_title,
                    clean_question,
                    clean_answer,
                    safe_tags,
                    safe_spoiler,
                    spoiler_rank(safe_spoiler),
                    safe_source,
                    1 if agent_used else 0,
                    "unverified",
                    now,
                    entry_id,
                ),
            )
            created_at = existing["created_at"]
            status = "updated"
        else:
            cursor = conn.execute(
                """
                INSERT INTO gamepath_entries(
                    game_id, title, question, answer_summary, markdown_path, tags, spoiler_level,
                    spoiler_rank, source_type, agent_used, trust_state, dispute_count, last_feedback,
                    last_feedback_at, content_hash, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, '', ?, ?, ?, ?, ?, 'unverified', 0, '', '', ?, ?, ?)
                """,
                (
                    normalized_game_id,
                    clean_title,
                    clean_question,
                    clean_answer,
                    safe_tags,
                    safe_spoiler,
                    spoiler_rank(safe_spoiler),
                    safe_source,
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
            "spoiler_level": safe_spoiler,
            "source_type": safe_source,
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
    return matched / max(1, min(len(terms), 8))


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


def search_gamepath_sync(
    query: str,
    game_id: Optional[str],
    limit: int = 5,
    *,
    tags: Any = None,
    spoiler_level: str = "low",
) -> list[dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []
    ensure_gamepath_db()
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
    sql += " ORDER BY score LIMIT ?"
    params.append(clamp_limit(limit, upper=10))
    try:
        with sqlite3.connect(GAMEPATH_DB) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []

    requested_tags = {tag.lower() for tag in normalize_tags_value(tags)}
    results: list[dict[str, Any]] = []
    for row in rows:
        row_tags = {tag.strip().lower() for tag in str(row["tags"] or "").split(",") if tag.strip()}
        if requested_tags and not requested_tags.intersection(row_tags):
            continue
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
            "spoiler_level": row["spoiler_level"],
            "source_type": row["source_type"],
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
    return results


def recent_gamepath_sync(game_id: Optional[str] = None, limit: int = 10) -> list[dict[str, Any]]:
    if not GAMEPATH_DB.exists():
        return []
    normalized_game_id = normalize_game_id(game_id)
    sql = (
        "SELECT id, game_id, title, question, answer_summary, markdown_path, tags, spoiler_level, "
        "source_type, agent_used, trust_state, dispute_count, last_feedback, last_feedback_at, "
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
        for row in rows
    ]


def should_use_gamepath(prompt: str, guide_requested: bool) -> bool:
    text = str(prompt or "").strip()
    if not text:
        return False
    has_gamepath_intent = bool(guide_requested or GAMEPATH_STORE_INTENT_RE.search(text))
    if not has_gamepath_intent:
        return False
    if GAMEPATH_UI_SKIP_RE.search(text) and not GUIDE_INTENT_RE.search(text):
        return False
    return True


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
    return matched / max(1, min(len(terms), 8))


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
        for key in ("title", "question", "answer_summary", "tags")
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
        answer_score = 0.12
    elif answer_len >= 80:
        answer_score = 0.09
    elif answer_len >= 40:
        answer_score = 0.06
    elif answer_len >= 20:
        answer_score = 0.03

    spoiler_score = 0.05 if result_spoiler_rank <= spoiler_rank("low") else 0.02
    source_score = 0.04 if result.get("agent_used") else 0.03
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
    core_score = 0.28 * core_overlap
    coverage_score = 0.23 * coverage

    score = min(
        1.0,
        max(
            0.0,
            game_score
            + core_score
            + coverage_score
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
        f"answer_len:{answer_len}",
        f"context_len:{context_len}",
        f"large_entry:{int(large_entry)}",
        f"spoiler:{result.get('spoiler_level') or 'low'}",
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
            float(item.get("match_coverage") or 0.0),
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


def lookup_route_stage(
    *,
    hermes_agent_web_enabled: bool,
    gamepath_requested: bool,
    gamepath_raw_hits: int,
    gamepath_evaluation: Optional[dict[str, Any]],
    gamepath_results: list[dict[str, Any]],
    guide_results: list[dict[str, Any]],
    memory_results: list[dict[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    if gamepath_results:
        evaluation = gamepath_evaluation or {}
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
            },
        )
    if gamepath_requested:
        evaluation = gamepath_evaluation or {}
        return (
            "gamepath_miss",
            "GamePath 沒有足夠高信心命中，交給 Hermes Agent 判斷是否需要 Tavily。",
            {
                "source": "gamepath",
                "web_search": "possible" if hermes_agent_web_enabled else False,
                "fast_path": False,
                "gamepath_hits": gamepath_raw_hits,
                "retrieval_score": evaluation.get("score", 0.0),
                "retrieval_gap": evaluation.get("gap", 0.0),
                "retrieval_reason": evaluation.get("reason", ""),
            },
        )
    if not guide_results and not memory_results:
        return (
            "gamepath_skipped",
            "這不是攻略型問題，已跳過 GamePath SQLite 查詢。",
            {
                "source": "chat",
                "web_search": "possible" if hermes_agent_web_enabled else False,
                "fast_path": True,
                "gamepath_checked": False,
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
    if not should_use_gamepath(prompt, bool(GUIDE_INTENT_RE.search(prompt or ""))):
        return "not_guide_intent"
    uncertain_near_start = GAMEPATH_UNCERTAIN_RE.search(clean_answer[:220])
    has_actionable_hint = re.search(r"(Hint|提示|直接答案|建議|下一步|步驟|做法|打法)", clean_answer, re.IGNORECASE)
    if uncertain_near_start and not has_actionable_hint:
        return f"uncertain_answer:{uncertain_near_start.group(0)}"
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
        return match.group("key").strip()
    if any(marker in text for marker in ("剛剛", "前面", "上一句", "我說的")):
        fact = IMPLICIT_MEMORY_FACT_RE.search(text)
        if fact:
            return fact.group("key").strip()
    return None


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


@app.on_event("shutdown")
async def shutdown_event():
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
        "rag_backend": "cpu_sqlite_fts5",
        "gamepath_enabled": True,
        "gamepath_db_exists": GAMEPATH_DB.exists(),
        "gamepath_entry_count": gamepath_entry_count_sync(),
        "gamepath_last_updated_at": gamepath_last_updated_at_sync(),
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


@app.get("/screenshot")
async def screenshot_endpoint(
    mode: str = "foreground",
    redact: Optional[bool] = None,
    profile: str = "fast",
    monitor: Optional[int] = None,
):
    try:
        mode_name = mode.lower()
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
        hidden_hwnds = hide_ignored_windows_for_capture(bool(protection_enabled))
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
                        source["capture_protection"] = "hide_restore" if protection_enabled else "off"
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
                source["capture_protection"] = "hide_restore" if protection_enabled else "off"
                return make_screenshot_response(img, source, profile)
        finally:
            restore_hidden_windows_after_capture(hidden_hwnds)
    except HTTPException:
        raise
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

    guide_was_requested = should_use_guides(prompt, chat_request.use_guides)
    gamepath_was_requested = should_use_gamepath(prompt, guide_was_requested)
    memory_results: list[dict[str, Any]] = []
    guide_results: list[dict[str, Any]] = []
    gamepath_results: list[dict[str, Any]] = []
    if chat_request.use_memory:
        memory_results = await asyncio.to_thread(search_memory_sync, prompt, game_id, None, 8)
    gamepath_evaluation: dict[str, Any] = {
        "confidence": "skipped",
        "score": 0.0,
        "gap": 0.0,
        "reason": "intent_gate_skipped",
        "results": [],
    }
    if gamepath_was_requested:
        gamepath_results = await asyncio.to_thread(search_gamepath_sync, prompt, game_id, 5)
        gamepath_evaluation = evaluate_gamepath_retrieval(prompt, game_id, gamepath_results)
        gamepath_results = list(gamepath_evaluation.get("results") or gamepath_results)
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

        return StreamingResponse(fact_answer_event_generator(), media_type="text/event-stream")

    if gamepath_route == "direct" and not chat_request.image_base64:
        async def gamepath_answer_event_generator():
            top_item = gamepath_results[0]
            answer = build_gamepath_answer(top_item)
            remember_gamepath_reference(top_item, route="direct")
            yield lookup_status_event(
                "gamepath_hit",
                "GamePath 命中，使用本地攻略紀錄。",
                source="gamepath",
                web_search=False,
                fast_path=True,
                hits=len(gamepath_results),
                retrieval_score=gamepath_evaluation.get("score", 0.0),
                retrieval_gap=gamepath_evaluation.get("gap", 0.0),
                retrieval_reason=gamepath_evaluation.get("reason", ""),
            )
            append_history("user", prompt)
            append_history("assistant", answer)
            yield sse_data({"content": answer})

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

        return StreamingResponse(no_guide_event_generator(), media_type="text/event-stream")

    rag_context = format_rag_context(guide_results, memory_results, guide_was_requested, gamepath_context_results)
    if gamepath_route == "summarize" and rag_context:
        rag_context = (
            "GamePath route: matching local GamePath entries were found. Summarize these entries first. "
            "Do not use Tavily unless the local entries are clearly insufficient for the player's question.\n"
            f"{rag_context}"
        )
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
                store_game_id = resolve_gamepath_store_game_id(game_id, active_game_context)
                store_skip_reason = gamepath_store_skip_reason(
                    prompt,
                    collected,
                    store_game_id,
                    HERMES_AGENT_WEB_ENABLED,
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
                        yield lookup_status_event(
                            "gamepath_stored",
                            "已濃縮並存入 GamePath，下次同類問題會走本地快取。",
                            source="gamepath",
                            web_search=False,
                            fast_path=False,
                            entry_id=stored_item.get("id"),
                            game_id=store_game_id,
                        )
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

        return StreamingResponse(hermes_event_generator(), media_type="text/event-stream")

    if should_use_overlay(prompt, chat_request.image_base64):
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

    if chat_request.image_base64:
        async def image_event_generator():
            await asyncio.sleep(0)
            ocr_text = ""
            if ENABLE_OCR_CONTEXT:
                ocr_text = await asyncio.to_thread(extract_ocr_text, chat_request.image_base64)
            if CHAT_BACKEND == "hermes" and HERMES_AGENT_WEB_ENABLED:
                try:
                    await asyncio.to_thread(image_to_data_url, chat_request.image_base64)
                    active_game_context = await asyncio.to_thread(detect_active_game_sync)
                    agent_prompt = build_hermes_agent_web_prompt(
                        prompt,
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
                if should_store_gamepath_answer(prompt, answer, store_game_id, True):
                    try:
                        stored_item = await asyncio.to_thread(
                            add_gamepath_sync,
                            prompt,
                            answer,
                            store_game_id,
                            title=prompt[:80],
                            tags=["auto", "vision", "hermes", "guide"],
                            spoiler_level="low",
                            source_type="hermes_agent_vision",
                            agent_used=True,
                        )
                        yield lookup_status_event(
                            "gamepath_stored",
                            "已濃縮並存入 GamePath，下次同類問題會走本地快取。",
                            source="gamepath",
                            web_search=False,
                            fast_path=False,
                            entry_id=stored_item.get("id"),
                            game_id=store_game_id,
                        )
                    except Exception as exc:
                        print(f"GamePath vision auto-store failed: {exc}")
                if not answer:
                    answer = "這次沒有產生可用回覆；請換個問法，或指定要看的畫面位置。"
                append_history("user", prompt)
                append_history("assistant", answer)
                yield f"data: {json.dumps({'content': answer}, ensure_ascii=False)}\n\n"
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
            append_history("user", prompt)
            append_history("assistant", answer)
            yield f"data: {json.dumps({'content': answer}, ensure_ascii=False)}\n\n"

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
            elif not await fastapi_request.is_disconnected():
                fallback = "這次沒有產生可用回覆；請換個問法，或指定要看的畫面位置。"
                append_history("user", prompt)
                append_history("assistant", fallback)
                yield f"data: {json.dumps({'content': fallback}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
