import asyncio
import base64
import ctypes
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
VULKAN_DEVICE = os.environ.get("GGML_VK_VISIBLE_DEVICES", "0")
LLAMA_CTX_SIZE = os.environ.get("LLAMA_CTX_SIZE", "8192")
LLAMA_GPU_LAYERS = os.environ.get("LLAMA_GPU_LAYERS", "1")
LLAMA_PARALLEL = os.environ.get("LLAMA_PARALLEL", "").strip()
LLAMA_CACHE_RAM = os.environ.get("LLAMA_CACHE_RAM", "").strip()
CHAT_BACKEND = os.environ.get("IGPU_CHAT_BACKEND", "llama").strip().lower()
HISTORY_CONTEXT_MESSAGES = int(os.environ.get("IGPU_HISTORY_CONTEXT_MESSAGES", "30"))
IMAGE_HISTORY_CONTEXT_MESSAGES = int(os.environ.get("IGPU_IMAGE_HISTORY_CONTEXT_MESSAGES", "12"))
HISTORY_STORE_MESSAGES = int(os.environ.get("IGPU_HISTORY_STORE_MESSAGES", "80"))
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
GAME_GUIDES_DIR = PROJECT_ROOT / "game_guides"
GUIDE_CACHE_DIR = PROJECT_ROOT / "guide_cache"
GUIDE_DB = GUIDE_CACHE_DIR / "guide.sqlite"
MEMORY_CACHE_DIR = PROJECT_ROOT / "memory_cache"
MEMORY_DB = MEMORY_CACHE_DIR / "memory.sqlite"

llama_process: Optional[subprocess.Popen] = None
history: list[dict[str, Any]] = []
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


class TaskAnalyzeRequest(BaseModel):
    message: str = ""
    image_base64: Optional[str] = None
    game_id: Optional[str] = None
    source_title: Optional[str] = None


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
        "on",
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
    line_fill = (255, 45, 45, 210)
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
        output = call_llama_once(messages, int(os.environ.get("LLAMA_TASK_RESPONSE_TOKENS", "360")))
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
            f"\n\nThe screenshot has a red planning grid with cells {grid_span}. "
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
        f"The screenshot has a red planning grid with cells {grid_span}. The grid is not part of the game. "
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
    output = call_llama_once(
        build_overlay_messages(prompt, image_base64, rag_context),
        max_tokens=8 if circle_marker_request else int(os.environ.get("LLAMA_OVERLAY_RESPONSE_TOKENS", "128")),
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
    await asyncio.to_thread(start_llama_server)


@app.on_event("shutdown")
async def shutdown_event():
    stop_llama_server()


@app.get("/health")
async def health():
    return {
        "status": "ok" if llama_ready() else "loading",
        "model": MODEL_ALIAS,
        "llama_url": llama_base_url(),
        "vulkan_device": VULKAN_DEVICE,
        "resource_policy": "game",
        "llama_ctx_size": LLAMA_CTX_SIZE,
        "llama_gpu_layers": LLAMA_GPU_LAYERS,
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
) -> tuple[dict[str, int], int]:
    if not monitors:
        raise RuntimeError("No monitor found for screenshot capture.")

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
async def screenshot_endpoint(mode: str = "foreground", redact: Optional[bool] = None, profile: str = "fast"):
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
                monitor, monitor_index = select_capture_monitor(sct.monitors, window)
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                capture_bounds = get_capture_bounds(monitor)
                source: dict[str, Any] = make_capture_source(
                    mode="screen",
                    capture_method="mss_monitor",
                    img=img,
                    capture_left=int(capture_bounds[0]),
                    capture_top=int(capture_bounds[1]),
                    monitor={**monitor, "index": monitor_index},
                )

                if mode_name in {"foreground", "window"}:
                    if window:
                        cropped = crop_to_foreground_window(img, monitor, window)
                        if cropped:
                            img = cropped
                            capture_bounds = get_capture_bounds(monitor, window)
                            source = make_capture_source(
                                mode=mode_name,
                                capture_method=(
                                    "screen_crop_fallback" if mode_name == "window" else "screen_crop"
                                ),
                                img=img,
                                capture_left=int(capture_bounds[0]),
                                capture_top=int(capture_bounds[1]),
                                window=window,
                                monitor={**monitor, "index": monitor_index},
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

    guide_was_requested = should_use_guides(prompt, chat_request.use_guides)
    memory_results: list[dict[str, Any]] = []
    guide_results: list[dict[str, Any]] = []
    if chat_request.use_memory:
        memory_results = await asyncio.to_thread(search_memory_sync, prompt, game_id, None, 8)
    if guide_was_requested:
        guide_results = await asyncio.to_thread(search_guides_sync, prompt, game_id, 5)

    fact_answer = answer_fact_lookup(prompt, memory_results)
    if fact_answer and not chat_request.image_base64:
        async def fact_answer_event_generator():
            append_history("user", prompt)
            append_history("assistant", fact_answer)
            yield f"data: {json.dumps({'content': fact_answer}, ensure_ascii=False)}\n\n"

        return StreamingResponse(fact_answer_event_generator(), media_type="text/event-stream")

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
                    answer = await asyncio.to_thread(
                        call_llama_once,
                        messages,
                        int(
                            os.environ.get(
                                "LLAMA_IMAGE_RESPONSE_TOKENS",
                                os.environ.get("LLAMA_IMAGE_MAX_TOKENS", "64"),
                            )
                        ),
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
