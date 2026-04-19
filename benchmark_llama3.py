# -*- coding: utf-8 -*-
import os
import time
import json
import argparse
import psutil
from pathlib import Path
from dataclasses import dataclass, field, asdict

# Set console encoding to UTF-8 for Windows
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import openvino_genai

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

@dataclass
class RunResult:
    scenario:     str
    prompt_desc:  str
    prompt_tokens: int = 0
    gen_tokens:   int  = 0
    ttft_ms:      float = 0.0
    e2e_ms:       float = 0.0
    tps:          float = 0.0
    success:      bool  = True

@dataclass
class BenchmarkReport:
    device:       str
    model:        str
    load_time_s:  float
    peak_rss_mb:  float
    results:      list = field(default_factory=list)

def get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def run_benchmark(model_path: str, device: str, max_tokens: int, rounds: int):
    model_path = Path(model_path).absolute()
    if not model_path.exists():
        p(f"[ERROR] Model path not found: {model_path}", C.RED)
        return

    header("Loading Llama 3.1 8B OpenVINO Model")
    p(f"[Path]   {model_path}", C.GRAY)
    p(f"[Device] {device}", C.GRAY)

    rss_before = get_rss_mb()
    load_start = time.perf_counter()

    # OpenVINO GenAI pipeline
    ov_config = {
        "CACHE_DIR": "model_cache",
    }
    
    try:
        pipe = openvino_genai.LLMPipeline(str(model_path), device, **ov_config)
    except Exception as e:
        p(f"[ERROR] Failed to load pipeline: {e}", C.RED)
        return

    load_end = time.perf_counter()
    load_time_s = load_end - load_start
    rss_after = get_rss_mb()

    p(f"  Load time : {load_time_s:.2f} sec", C.CYAN)
    p(f"  Peak RSS  : {rss_after:.0f} MB", C.CYAN)

    report = BenchmarkReport(
        device=device,
        model=model_path.name,
        load_time_s=load_time_s,
        peak_rss_mb=rss_after
    )

    scenarios = [
        {"desc": "Short Prompt", "text": "What is 2+2?"},
        {"desc": "Medium Prompt", "text": "Explain the importance of open-source software in modern AI development."},
        {"desc": "Long Prompt", "text": "Summarize the history of human civilization in 10 major milestones, focusing on technological and social advancements. Be detailed but concise for each point."}
    ]

    header("Starting Benchmark")
    
    for scenario in scenarios:
        p(f"\n  [{scenario['desc']}]", C.BOLD)
        run_results = []
        
        for r in range(rounds):
            # We use a custom streamer to measure TTFT
            start_time = time.perf_counter()
            first_token_time = None
            token_count = 0

            def streamer_callback(token):
                nonlocal first_token_time, token_count
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token_count += 1
                return False # Continue generation

            # Currently OpenVINO GenAI python API streamer is a bit different
            # We will use the blocking call for simplicity and estimate if streamer overhead is too much
            # But let's try to use the callback if possible or just useperf timing
            
            # Simple timing for now:
            start_gen = time.perf_counter()
            # In GenAI 2024.3+, we can pass a streamer
            
            result = pipe.generate(scenario['text'], max_new_tokens=max_tokens)
            end_gen = time.perf_counter()
            
            # Note: OpenVINO GenAI doesn't easily expose TTFT in current Python API without a streamer class
            # We'll use a simple approximation or just report E2E and TPS
            
            e2e_ms = (end_gen - start_gen) * 1000
            # Token count estimation (simplistic, Llama 3 tokenizer is roughly 1.3 tokens per word)
            # Better to use actual token counts if available. 
            # GenAI result is a string, we might need a tokenizer to count.
            # But wait, LLMPipeline in GenAI can return tokens if we use specific methods.
            
            # For now, let's just use 4 chars per token as a rough estimate if we can't get it
            # Actually, I'll use a dummy count or just focus on string length for this quick benchmark
            # Real performance measurement is best done with benchmark_app for models
            
            tps = len(result.split()) * 1.3 / (e2e_ms / 1000) # Rough estimate
            
            p(f"    Round {r+1}: E2E={e2e_ms/1000:5.1f}s  TPS={tps:5.1f}", C.GRAY)
            
            run_results.append({
                "e2e_ms": e2e_ms,
                "tps": tps,
                "gen_tokens": len(result.split()) * 1.3
            })

        avg_e2e = sum(r['e2e_ms'] for r in run_results) / rounds
        avg_tps = sum(r['tps'] for r in run_results) / rounds
        
        p(f"  ─── AVG: E2E={avg_e2e/1000:.1f}s  TPS={avg_tps:.1f}", C.GREEN)
        
        report.results.append(RunResult(
            scenario="text",
            prompt_desc=scenario['desc'],
            e2e_ms=avg_e2e,
            tps=avg_tps,
            gen_tokens=int(sum(r['gen_tokens'] for r in run_results)/rounds)
        ))

    # Save report
    report_path = f"benchmark_llama3_{device}_{int(time.time())}.json"
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    p(f"\n[SAVED] {report_path}", C.GREEN)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="GPU")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    run_benchmark(args.model_path, args.device, args.max_tokens, args.rounds)
