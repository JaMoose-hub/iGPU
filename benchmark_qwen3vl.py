# -*- coding: utf-8 -*-
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
benchmark_qwen3vl.py
====================
Qwen3-VL 8B OpenVINO iGPU Benchmark Suite

測試指標：
  - TTFT  (Time To First Token)          — 第一個 token 出現的延遲
  - TPS   (Tokens Per Second)             — 穩態 throughput
  - E2E   (End-to-End Latency)            — 整體生成時間
  - 模型載入時間
  - 記憶體使用 (RSS)

測試場景：
  1. Pure Text     — 純文字問答 (短/中/長 prompt)
  2. Vision (VQA)  — 圖片理解 (帶圖問答)
  3. Stress        — 壓力測試 (連續多輪生成)

用法：
  python benchmark_qwen3vl.py [--device GPU|CPU] [--max-tokens 200] [--rounds 5]
"""

import os
import sys
import gc
import time
import json
import argparse
import psutil
import io
import base64
from threading import Thread
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ────────────────────────────────────────────────────────────
# 顏色輸出工具
# ────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    RED    = "\033[91m"
    GRAY   = "\033[90m"

def p(msg, color=C.RESET):
    print(f"{color}{msg}{C.RESET}", flush=True)

def header(msg):
    line = "─" * 60
    p(f"\n{line}", C.CYAN)
    p(f"  {msg}", C.BOLD)
    p(f"{line}", C.CYAN)

# ────────────────────────────────────────────────────────────
# 路徑設定
# ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODELS_ROOT = BASE_DIR / "models"
MODEL_PATH  = MODELS_ROOT / "Qwen3-VL-8B-openvino-int4"

# ────────────────────────────────────────────────────────────
# 結果資料結構
# ────────────────────────────────────────────────────────────
@dataclass
class RunResult:
    scenario:     str
    prompt_desc:  str
    prompt_tokens: int = 0
    gen_tokens:   int  = 0
    ttft_ms:      float = 0.0   # 第一個 token 延遲 (ms)
    e2e_ms:       float = 0.0   # 整體生成時間 (ms)
    tps:          float = 0.0   # tokens/sec
    success:      bool  = True
    error:        str   = ""

@dataclass
class BenchmarkReport:
    device:       str
    model:        str
    load_time_s:  float
    peak_rss_mb:  float
    results:      list = field(default_factory=list)

# ────────────────────────────────────────────────────────────
# 測試 Prompt 集
# ────────────────────────────────────────────────────────────
TEXT_SCENARIOS = [
    {
        "desc": "Short Text (1 sentence)",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]}],
    },
    {
        "desc": "Medium Text (paragraph)",
        "messages": [{"role": "user", "content": [{"type": "text", "text": (
            "Explain the key differences between CPU and GPU architectures "
            "in terms of core count, memory bandwidth, and workload suitability. "
            "Be concise but thorough."
        )}]}],
    },
    {
        "desc": "Long Text (detailed analysis)",
        "messages": [{"role": "user", "content": [{"type": "text", "text": (
            "You are a performance analyst. A user reports their Intel integrated GPU "
            "runs LLM inference at 5 tokens/sec with INT4 quantized models. "
            "List 8 concrete, actionable tips to improve their throughput, covering: "
            "model selection, quantization strategy, batch settings, memory allocation, "
            "driver configuration, OpenVINO config hints, power plan, and thermal management. "
            "Format as a numbered list with short explanations."
        )}]}],
    },
    {
        "desc": "Chinese QA",
        "messages": [{"role": "user", "content": [{"type": "text", "text": (
            "請用繁體中文解釋，在 Intel iGPU 上跑大語言模型推理時，"
            "為什麼使用 INT4 量化比 FP16 更節省顯示記憶體，同時對精度影響不大？"
        )}]}],
    },
]

def make_test_image(width=512, height=512) -> Image.Image:
    """生成一張帶有文字標籤的測試圖片（不需要外部圖片）"""
    img = Image.new("RGB", (width, height), color=(30, 40, 70))
    draw = ImageDraw.Draw(img)
    # 畫格線
    for x in range(0, width, 64):
        draw.line([(x, 0), (x, height)], fill=(50, 60, 90), width=1)
    for y in range(0, height, 64):
        draw.line([(0, y), (width, y)], fill=(50, 60, 90), width=1)
    # 中央文字
    draw.rectangle([150, 200, 362, 312], fill=(80, 100, 180), outline=(120, 140, 220), width=2)
    draw.text((165, 220), "BENCHMARK", fill=(255, 255, 255))
    draw.text((165, 248), "TEST IMAGE", fill=(200, 220, 255))
    draw.text((165, 276), f"{width}x{height}", fill=(150, 180, 255))
    return img

VISION_SCENARIOS = [
    {
        "desc": "VQA: Describe image",
        "question": "What do you see in this image? Describe it in detail.",
    },
    {
        "desc": "VQA: Technical analysis",
        "question": (
            "This image is a benchmark test pattern. "
            "What colors, shapes, and text can you identify? "
            "How many grid cells are visible approximately?"
        ),
    },
]

# ────────────────────────────────────────────────────────────
# 核心計時工具
# ────────────────────────────────────────────────────────────
def get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def run_single_inference(
    model,
    processor,
    tokenizer,
    messages: list,
    max_new_tokens: int,
) -> dict:
    """
    執行單次推理並計時。
    回傳: { ttft_ms, e2e_ms, gen_tokens, prompt_tokens, text }
    """
    from transformers import TextIteratorStreamer

    # ── 準備輸入 ──────────────────────────────────────────────
    all_images = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "image" and part.get("image"):
                    all_images.append(part["image"])
        if msg.get("images"):
            all_images.extend(msg["images"])

    full_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    target_images = None
    if all_images:
        target_images = all_images[0] if len(all_images) == 1 else all_images

    try:
        inputs = processor(text=full_prompt, images=target_images, return_tensors="pt")
    except Exception as e:
        return {"error": str(e)}

    prompt_tokens = inputs["input_ids"].shape[-1]

    # ── Streaming 生成 + 計時 ──────────────────────────────────
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=False)

    first_token_time = [None]
    start_time = time.perf_counter()

    def _generate():
        try:
            model.generate(**gen_kwargs)
        except Exception as e:
            print(f"  [Gen Error] {e}")

    thread = Thread(target=_generate)
    thread.start()

    collected = ""
    token_count = 0

    for token in streamer:
        if first_token_time[0] is None:
            first_token_time[0] = time.perf_counter()
        collected += token
        token_count += 1

    end_time = time.perf_counter()
    thread.join()

    ttft_ms = (first_token_time[0] - start_time) * 1000 if first_token_time[0] else 0
    e2e_ms  = (end_time - start_time) * 1000
    tps     = token_count / (e2e_ms / 1000) if e2e_ms > 0 else 0

    return {
        "ttft_ms":      ttft_ms,
        "e2e_ms":       e2e_ms,
        "gen_tokens":   token_count,
        "prompt_tokens": prompt_tokens,
        "tps":          tps,
        "text":         collected,
    }

# ────────────────────────────────────────────────────────────
# Benchmark 主體
# ────────────────────────────────────────────────────────────
def run_benchmark(device: str, max_tokens: int, rounds: int, skip_vision: bool):
    import openvino as ov
    from optimum.intel.openvino import OVModelForVisualCausalLM
    from transformers import (
        AutoProcessor, AutoTokenizer, AutoConfig,
        Qwen2VLImageProcessorFast, Qwen2VLProcessor, Qwen2VLImageProcessor
    )
    import transformers.models.auto.video_processing_auto as video_auto

    # Transformers 補丁（照你們既有的方式）
    try:
        original_func = video_auto.video_processor_class_from_name
        def safe_video_processor_class_from_name(class_name):
            if getattr(video_auto, "VIDEO_IMAGE_PROCESSOR_MAPPING", None) is None:
                video_auto.VIDEO_IMAGE_PROCESSOR_MAPPING = {}
            try:
                return original_func(class_name)
            except Exception:
                return None
        video_auto.video_processor_class_from_name = safe_video_processor_class_from_name
    except Exception:
        pass

    # ── 模型路徑確認 ───────────────────────────────────────────
    model_path = str(MODEL_PATH)
    if not MODEL_PATH.exists():
        p(f"[ERROR] Model not found: {model_path}", C.RED)
        p(f"  Please check model path: {MODEL_PATH}", C.YELLOW)
        sys.exit(1)

    p(f"[Path]   {model_path}", C.GRAY)
    p(f"[Device] {device}", C.GRAY)
    p(f"[Tokens] {max_tokens}", C.GRAY)
    p(f"[Rounds] {rounds}", C.GRAY)

    # ── 載入 Processor ─────────────────────────────────────────
    header("載入 Processor & Tokenizer")
    tokenizer  = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        p(f"  AutoProcessor 失敗 ({e})，嘗試手動建構...", C.YELLOW)
        try:
            img_proc = Qwen2VLImageProcessorFast.from_pretrained(model_path)
        except Exception:
            img_proc = Qwen2VLImageProcessor.from_pretrained(model_path)
        processor = Qwen2VLProcessor(img_proc, tokenizer)
    p("  [OK] Processor loaded.", C.GREEN)

    # ── 載入模型並計時 ─────────────────────────────────────────
    header("載入 OV 模型")
    rss_before = get_rss_mb()
    load_start = time.perf_counter()

    ov_config = {
        "INFERENCE_PRECISION_HINT": "f16",
        "CACHE_DIR": str(BASE_DIR / "model_cache"),
    }
    config = AutoConfig.from_pretrained(model_path)

    try:
        model = OVModelForVisualCausalLM.from_pretrained(
            model_path,
            device=device,
            config=config,
            ov_config=ov_config,
            trust_remote_code=True,
        )
        actual_device = device
        p(f"  [OK] Model loaded on {device}.", C.GREEN)
    except Exception as e:
        p(f"  [WARN] {device} failed: {e}", C.YELLOW)
        p("  [WARN] Falling back to CPU...", C.YELLOW)
        model = OVModelForVisualCausalLM.from_pretrained(
            model_path, device="CPU", config=config, trust_remote_code=True
        )
        actual_device = "CPU"
        p("  [OK] Model loaded on CPU.", C.GREEN)

    load_end = time.perf_counter()
    load_time_s = load_end - load_start
    rss_after   = get_rss_mb()

    p(f"  Load time : {load_time_s:.2f} sec", C.CYAN)
    p(f"  RSS delta : {rss_after - rss_before:.0f} MB  (total {rss_after:.0f} MB)", C.CYAN)

    report = BenchmarkReport(
        device=actual_device,
        model="Qwen3-VL-8B-INT4",
        load_time_s=load_time_s,
        peak_rss_mb=rss_after,
    )

    # ═══════════════════════════════════════════════════════
    # 場景 1：Warm-up
    # ═══════════════════════════════════════════════════════
    header("🔥 Warm-up (不計入結果)")
    warmup_msg = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    warmup_res = run_single_inference(model, processor, tokenizer, warmup_msg, 30)
    if "error" not in warmup_res:
        p(f"  Warm-up done  ({warmup_res['e2e_ms']:.0f} ms, "
          f"{warmup_res['gen_tokens']} tokens)", C.GRAY)
    else:
        p(f"  Warm-up FAILED: {warmup_res['error']}", C.RED)
    gc.collect()

    # ═══════════════════════════════════════════════════════
    # 場景 2：Pure Text Benchmark
    # ═══════════════════════════════════════════════════════
    header("📝 Pure Text Benchmark")
    for scenario in TEXT_SCENARIOS:
        p(f"\n  [{scenario['desc']}]", C.BOLD)
        run_results = []
        for r in range(rounds):
            res = run_single_inference(
                model, processor, tokenizer,
                scenario["messages"], max_tokens
            )
            if "error" in res:
                p(f"    Round {r+1}: ❌ {res['error']}", C.RED)
                continue
            run_results.append(res)
            p(
                f"    Round {r+1}: "
                f"TTFT={res['ttft_ms']:6.0f}ms  "
                f"E2E={res['e2e_ms']/1000:5.1f}s  "
                f"TPS={res['tps']:5.1f}  "
                f"Tokens={res['gen_tokens']}",
                C.GRAY,
            )
            gc.collect()

        if run_results:
            avg_ttft = sum(r["ttft_ms"]   for r in run_results) / len(run_results)
            avg_e2e  = sum(r["e2e_ms"]    for r in run_results) / len(run_results)
            avg_tps  = sum(r["tps"]       for r in run_results) / len(run_results)
            avg_gen  = sum(r["gen_tokens"] for r in run_results) / len(run_results)

            p(f"  ─── AVG: TTFT={avg_ttft:.0f}ms  E2E={avg_e2e/1000:.1f}s  "
              f"TPS={avg_tps:.1f}  GenTokens={avg_gen:.0f}", C.GREEN)

            report.results.append(RunResult(
                scenario="text",
                prompt_desc=scenario["desc"],
                prompt_tokens=run_results[0]["prompt_tokens"],
                gen_tokens=int(avg_gen),
                ttft_ms=avg_ttft,
                e2e_ms=avg_e2e,
                tps=avg_tps,
            ))

    # ═══════════════════════════════════════════════════════
    # 場景 3：Vision (VQA) Benchmark
    # ═══════════════════════════════════════════════════════
    if not skip_vision:
        header("🖼️  Vision (VQA) Benchmark")
        test_img = make_test_image(512, 512)

        for scenario in VISION_SCENARIOS:
            p(f"\n  [{scenario['desc']}]", C.BOLD)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": test_img},
                        {"type": "text",  "text": scenario["question"]},
                    ],
                    "images": [test_img],
                }
            ]

            run_results = []
            for r in range(rounds):
                res = run_single_inference(
                    model, processor, tokenizer, messages, max_tokens
                )
                if "error" in res:
                    p(f"    Round {r+1}: ❌ {res['error']}", C.RED)
                    continue
                run_results.append(res)
                p(
                    f"    Round {r+1}: "
                    f"TTFT={res['ttft_ms']:6.0f}ms  "
                    f"E2E={res['e2e_ms']/1000:5.1f}s  "
                    f"TPS={res['tps']:5.1f}  "
                    f"Tokens={res['gen_tokens']}",
                    C.GRAY,
                )
                gc.collect()

            if run_results:
                avg_ttft = sum(r["ttft_ms"]   for r in run_results) / len(run_results)
                avg_e2e  = sum(r["e2e_ms"]    for r in run_results) / len(run_results)
                avg_tps  = sum(r["tps"]       for r in run_results) / len(run_results)
                avg_gen  = sum(r["gen_tokens"] for r in run_results) / len(run_results)

                p(f"  ─── AVG: TTFT={avg_ttft:.0f}ms  E2E={avg_e2e/1000:.1f}s  "
                  f"TPS={avg_tps:.1f}  GenTokens={avg_gen:.0f}", C.GREEN)

                report.results.append(RunResult(
                    scenario="vision",
                    prompt_desc=scenario["desc"],
                    prompt_tokens=run_results[0]["prompt_tokens"],
                    gen_tokens=int(avg_gen),
                    ttft_ms=avg_ttft,
                    e2e_ms=avg_e2e,
                    tps=avg_tps,
                ))

    # ═══════════════════════════════════════════════════════
    # 場景 4：Throughput Stress Test
    # ═══════════════════════════════════════════════════════
    header("⚡ Throughput Stress Test (連續 10 輪短推理)")
    stress_msg  = [{"role": "user", "content": [{"type": "text",
                    "text": "List 5 famous programming languages."}]}]
    stress_reps = 10
    stress_results = []

    for r in range(stress_reps):
        res = run_single_inference(model, processor, tokenizer, stress_msg, 100)
        if "error" not in res:
            stress_results.append(res)
            p(f"  Rep {r+1:02d}: TPS={res['tps']:5.1f}  "
              f"E2E={res['e2e_ms']/1000:4.1f}s  Tokens={res['gen_tokens']}", C.GRAY)
        gc.collect()

    if stress_results:
        avg_tps = sum(r["tps"] for r in stress_results) / len(stress_results)
        total_tokens = sum(r["gen_tokens"] for r in stress_results)
        total_time   = sum(r["e2e_ms"]    for r in stress_results) / 1000
        p(f"\n  --- Stress AVG TPS : {avg_tps:.1f} tokens/sec", C.GREEN)
        p(f"  --- Total Tokens   : {total_tokens}", C.GREEN)
        p(f"  --- Total Time     : {total_time:.1f} sec", C.GREEN)

        report.results.append(RunResult(
            scenario="stress",
            prompt_desc="Throughput Stress (10 rounds)",
            gen_tokens=total_tokens,
            tps=avg_tps,
            e2e_ms=total_time * 1000,
        ))

    # =======================================================
    # 最終報告
    # =======================================================
    header("Benchmark Summary")
    print(f"\n{'Scenario':<30} {'TTFT(ms)':>10} {'E2E(s)':>8} {'TPS':>8} {'GenTok':>8}")
    print("-" * 70)
    for r in report.results:
        print(
            f"{r.prompt_desc:<30} "
            f"{r.ttft_ms:>10.0f} "
            f"{r.e2e_ms/1000:>8.1f} "
            f"{r.tps:>8.1f} "
            f"{r.gen_tokens:>8}"
        )
    print("-" * 70)
    p(f"\n  Device    : {report.device}", C.CYAN)
    p(f"  Model     : {report.model}", C.CYAN)
    p(f"  Load Time : {report.load_time_s:.2f} sec", C.CYAN)
    p(f"  Peak RSS  : {report.peak_rss_mb:.0f} MB", C.CYAN)

    # -- 結果存檔 --------------------------------------------
    report_path = BASE_DIR / f"benchmark_results_{actual_device}_{int(time.time())}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        data = asdict(report)
        data["results"] = [asdict(r) for r in report.results]
        json.dump(data, f, indent=2, ensure_ascii=False)
    p(f"\n  [SAVED] {report_path}", C.GREEN)

    return report

# ────────────────────────────────────────────────────────────
# Entry Point
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Qwen3-VL 8B OpenVINO iGPU Benchmark")
    parser.add_argument(
        "--device", type=str, default="GPU",
        choices=["GPU", "CPU", "AUTO"],
        help="推理裝置 (default: GPU)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=200,
        help="每輪最多生成 token 數 (default: 200)"
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="每個場景跑幾輪取平均 (default: 3)"
    )
    parser.add_argument(
        "--skip-vision", action="store_true",
        help="跳過 Vision (VQA) 場景"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="覆寫預設模型路徑"
    )
    args = parser.parse_args()

    if args.model_path:
        MODEL_PATH = Path(args.model_path)

    p("\n=== Qwen3-VL 8B -- OpenVINO iGPU Benchmark ===", C.BOLD + C.CYAN)
    p(f"    Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", C.GRAY)

    try:
        run_benchmark(
            device=args.device,
            max_tokens=args.max_tokens,
            rounds=args.rounds,
            skip_vision=args.skip_vision,
        )
    except KeyboardInterrupt:
        p("\n[STOP] Benchmark interrupted by user.", C.YELLOW)
    except Exception as e:
        p(f"\n[ERROR] Benchmark failed: {e}", C.RED)
        import traceback
        traceback.print_exc()
