import os
import gc

import base64
import json
import io
import asyncio
import numpy as np
import sys
import multiprocessing
import argparse
from typing import Optional, Any, Union
from threading import Thread

from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import time
import uuid
import mss
from PIL import Image
import librosa

# OpenVINO 基礎庫
import openvino as ov
import openvino_genai as ov_genai
from openvino import opset12 as opset

# Gemma-4 / Qwen3-VL 視覺庫 (via Optimum Intel)
from optimum.intel.openvino import OVModelForVisualCausalLM, OVModelForCausalLM
from transformers import AutoProcessor, TextIteratorStreamer, AutoConfig, GenerationConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.gemma.configuration_gemma import GemmaConfig

# --- 路徑處理 (支援打包環境) ---
def get_base_path():
    if getattr(sys, 'frozen', False):
        # 打包環境：.exe 所在的目錄
        return os.path.dirname(sys.executable)
    # 開發環境
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_path()
MODELS_ROOT = os.path.join(BASE_DIR, "models")

# --- 命令列參數解析 ---
parser = argparse.ArgumentParser(description="iGPU VLM Backend")
parser.add_argument("model_pos", type=str, nargs="?", help="Select model: gemma4 or qwen3 (positional)")
parser.add_argument("--model", type=str, help="Select model: gemma4 or qwen3 (flag)")
args, unknown = parser.parse_known_args()

# 優先順序：--model 標籤 > 位置參數 > 預設 gemma4
SELECTED_MODEL = args.model or args.model_pos or "gemma4"
IS_VISION_MODEL = "vl" in SELECTED_MODEL.lower() or "gemma" in SELECTED_MODEL.lower()
DEVICE = "GPU" # 如果 GPU 跑不動，會自動回退到 CPU

# --- 修復與註冊專區 ---
def patch_model_for_gpu_precision(model_path, xml_filename):
    """
    手動修復 OpenVINO 模型圖中的精度不匹配問題 (f16 vs f32)。
    """
    core = ov.Core()
    xml_path = os.path.join(model_path, xml_filename)
    if not os.path.exists(xml_path):
        return None
    
    print(f"Applying aggressive FP16 precision patch to {xml_filename}...")
    model = core.read_model(xml_path)
    
    count = 0
    # 1. 統一所有變數 (Variables / KV Cache) 的類型為 f16
    try:
        for var in model.get_variables():
            if var.get_info().element_type == ov.Type.f32:
                var.get_info().element_type = ov.Type.f16
                count += 1
        if count > 0:
            print(f"Global Patch: Forced {count} variables to f16.")
    except Exception: pass
    
    # 2. 全圖 FP16 強制對齊
    op_count = 0
    for op in model.get_ops():
        for i in range(op.get_input_size()):
            try:
                input_node = op.input(i)
                if input_node.get_element_type() == ov.Type.f32:
                    source_output = input_node.get_source_output()
                    if source_output.get_node().get_type_name() != "Convert":
                        convert_node = opset.convert(source_output, ov.Type.f16)
                        input_node.replace_source_output(convert_node.output(0))
                        op_count += 1
            except: continue
                
    if op_count > 0 or count > 0:
        try:
            model.validate_nodes_and_infer_types()
            print(f"Global Patch: Successfully unified {op_count} nodes to FP16.")
        except Exception: pass
            
        patched_dir = os.path.join(model_path, "patched_gpu")
        os.makedirs(patched_dir, exist_ok=True)
        import shutil
        for item in os.listdir(model_path):
            s = os.path.join(model_path, item)
            d = os.path.join(patched_dir, item)
            if os.path.isfile(s) and not item.endswith(('.xml', '.bin')):
                shutil.copy2(s, d)
        ov.save_model(model, os.path.join(patched_dir, xml_filename), compress_to_fp16=False)
        return patched_dir
    return None

def apply_gemma4_hacks():
    """僅在選用 Gemma-4 時執行的架構與配置修補"""
    if "gemma4" not in CONFIG_MAPPING:
        CONFIG_MAPPING.register("gemma4", GemmaConfig)
        print("Registered 'gemma4' architecture in Transformers.")

    from optimum.intel.openvino.modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING
    if "gemma4" not in MODEL_TYPE_TO_CLS_MAPPING:
        MODEL_TYPE_TO_CLS_MAPPING["gemma4"] = OVModelForVisualCausalLM
        print("Registered 'gemma4' architecture in Optimum-Intel.")

    # Monkey Patch GenerationConfig
    original_from_model_config = GenerationConfig.from_model_config
    def patched_from_model_config(model_config, **kwargs):
        if isinstance(model_config, GenerationConfig): return model_config
        try:
            if hasattr(model_config, "to_dict"): clean_dict = model_config.to_dict()
            elif isinstance(model_config, (dict, list)): clean_dict = dict(model_config)
            else: clean_dict = vars(model_config)
            valid_keys = GenerationConfig().to_dict().keys()
            final_params = {k: v for k, v in clean_dict.items() if k in valid_keys}
            return GenerationConfig(**final_params)
        except Exception:
            try: return original_from_model_config(model_config, **kwargs)
            except: return GenerationConfig()
    GenerationConfig.from_model_config = patched_from_model_config
    print("Applied ROBUST GenerationConfig monkey patch for Gemma-4.")

def apply_qwen3_hacks():
    """僅在選用 Qwen3 時執行的架構註冊"""
    from optimum.intel.openvino.modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING, _OVQwen2_5_VLForCausalLM
    if "qwen3_vl" not in MODEL_TYPE_TO_CLS_MAPPING:
        # 使用 Qwen2.5-VL 的具體實作類別來處理 Qwen3-VL
        MODEL_TYPE_TO_CLS_MAPPING["qwen3_vl"] = _OVQwen2_5_VLForCausalLM
        print("Registered 'qwen3_vl' architecture as Qwen2.5-VL compatible in Optimum-Intel.")

def apply_transformers_video_patch():
    """修復 Transformers 庫在處理 Qwen3-VL 影像處理器時的 NoneType 迭代錯誤"""
    try:
        import transformers.models.auto.video_processing_auto as video_auto
        original_func = video_auto.video_processor_class_from_name
        
        def safe_video_processor_class_from_name(class_name):
            # 如果影片處理器對應表尚未初始化，手動給它一個空的字典避免崩潰
            if getattr(video_auto, "VIDEO_IMAGE_PROCESSOR_MAPPING", None) is None:
                video_auto.VIDEO_IMAGE_PROCESSOR_MAPPING = {}
            try:
                return original_func(class_name)
            except Exception:
                return None
                
        video_auto.video_processor_class_from_name = safe_video_processor_class_from_name
        print("Applied Transformers library fix for Qwen3-VL video processor.")
    except Exception as e:
        print(f"Warning: Failed to apply Transformers patch: {e}")

def get_model_paths(target_model):
    if target_model == "gemma4":
        vlm_path = os.path.join(MODELS_ROOT, "gemma-4-E4B-ov")
    elif target_model == "qwen2.5":
        vlm_path = os.path.join(MODELS_ROOT, "Qwen2.5-7B-Instruct-openvino-int4")
    else:
        vlm_path = os.path.join(MODELS_ROOT, "Qwen3-VL-8B-openvino-int4")
        
    whisper_path = os.path.join(MODELS_ROOT, "whisper-base-ov")
    return vlm_path, whisper_path

# --- 初始化執行 ---
apply_transformers_video_patch()

if SELECTED_MODEL == "gemma4":
    apply_gemma4_hacks()
elif SELECTED_MODEL == "qwen3":
    apply_qwen3_hacks()

VLM_MODEL_PATH, WHISPER_MODEL_PATH = get_model_paths(SELECTED_MODEL)
if not os.path.exists(VLM_MODEL_PATH):
    print(f"❌ Error: Model path not found: {VLM_MODEL_PATH}")
    # 回退嘗試
    ALT_MODEL = "qwen3" if SELECTED_MODEL == "gemma4" else "gemma4"
    VLM_MODEL_PATH, _ = get_model_paths(ALT_MODEL)
    if os.path.exists(VLM_MODEL_PATH):
        print(f"Trying fallback model: {ALT_MODEL}")
        SELECTED_MODEL = ALT_MODEL
        if SELECTED_MODEL == "gemma4": apply_gemma4_hacks()
    else:
        print("❌ Fatal: No valid VLM models found.")

VLM_MODEL_PATH, WHISPER_MODEL_PATH = get_model_paths(SELECTED_MODEL)

app = FastAPI(title="iGPU Multi-Model VLM Backend")

# 允許跨域請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 模型與狀態初始化 ---
# 全域狀態
model = None          # 用於儲存 Optimum 模型 (Gemma-4) 或 GenAI Pipeline (Qwen3)
processor = None
tokenizer = None

# --- OpenVINO GenAI 專用串流器 ---
class GenAIStreamer(ov_genai.StreamerBase):
    """將 ov_genai 的串流 Token 轉導至非同步隊列"""
    def __init__(self, queue, event_loop):
        super().__init__()
        self.queue = queue
        self.loop = event_loop

    def write(self, token: str) -> bool:
        # ov_genai 會從背景 thread 呼叫此方法
        self.loop.call_soon_threadsafe(self.queue.put_nowait, token)
        return False # False 表示繼續生成

def pil_to_ov_tensor(pil_image):
    """將 PIL 圖片轉換為 OpenVINO GenAI 接受的 Tensor 格式"""
    import numpy as np
    # GenAI 預期的是 RGB 格式的 numpy array [H, W, C]
    img_array = np.array(pil_image)
    return ov_genai.Tensor(img_array)

def load_vlm_components(path, model_type):
    """
    強韌的組件載入：嘗試多種方式獲取 Processor 與 Tokenizer
    """
    from transformers import AutoProcessor, AutoTokenizer, AutoConfig
    local_tokenizer = None
    local_processor = None
    
    # 1. 嘗試載入 Tokenizer (通常最穩定)
    try:
        local_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Failed to load tokenizer: {e}")
        
    # 2. 嘗試載入 Processor (僅視覺模型需要)
    if IS_VISION_MODEL:
        try:
            # 嘗試標準方式
            local_processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        except Exception as e:
            print(f"AutoProcessor failed: {e}. Attempting manual construction...")
            # 針對 Qwen 家族的手動修復
            try:
                from transformers import Qwen2VLImageProcessorFast, Qwen2VLProcessor, Qwen2VLImageProcessor
                print("Attempting to load Qwen2VLImageProcessorFast...")
                try:
                    img_proc = Qwen2VLImageProcessorFast.from_pretrained(path)
                except Exception:
                    img_proc = Qwen2VLImageProcessor.from_pretrained(path)
                
                local_processor = Qwen2VLProcessor(img_proc, local_tokenizer)
            except Exception as inner_e:
                print(f"Manual processor construction failed: {inner_e}")
    else:
        # 純文字模型，Processor 其實就是 Tokenizer 的包裝
        local_processor = local_tokenizer
        
    if not local_processor:
        raise RuntimeError("Could not load a valid VLM/LLM processor. Please check your environment.")
        
    return local_processor, local_tokenizer

print(f"Loading VLM components for {SELECTED_MODEL}...")
try:
    if SELECTED_MODEL == "qwen3":
        # --- Qwen3-VL: 使用專屬的 GenAI VLMPipeline (最穩定) ---
        print(f"Initializing OpenVINO GenAI VLMPipeline on {DEVICE}...")
        model = ov_genai.VLMPipeline(VLM_MODEL_PATH, DEVICE)
        # GenAI Pipeline 內建了 Tokenizer 與 Processor 功能
        processor, tokenizer = load_vlm_components(VLM_MODEL_PATH, SELECTED_MODEL)
    else:
        # --- 其他模型 (如 Gemma-4): 使用 Optimum-Intel 載入 ---
        processor, tokenizer = load_vlm_components(VLM_MODEL_PATH, SELECTED_MODEL)
        
        # 建立 Config 與 載入
        config = (GemmaConfig.from_pretrained(VLM_MODEL_PATH) 
                 if SELECTED_MODEL == "gemma4" else AutoConfig.from_pretrained(VLM_MODEL_PATH))
        
        target_path = VLM_MODEL_PATH
        if SELECTED_MODEL == "gemma4" and DEVICE == "GPU":
            patched_path = patch_model_for_gpu_precision(VLM_MODEL_PATH, "openvino_language_model.xml")
            if patched_path: target_path = patched_path

        ov_config = {"INFERENCE_PRECISION_HINT": "f16", "CACHE_DIR": "model_cache"}
        ModelClass = OVModelForVisualCausalLM if IS_VISION_MODEL else OVModelForCausalLM
        
        model = ModelClass.from_pretrained(
            target_path, 
            device=DEVICE,
            config=config,
            ov_config=ov_config,
            trust_remote_code=True,
            file_name="openvino_language_model.xml" if IS_VISION_MODEL else None
        )
    print(f"Model {SELECTED_MODEL} loaded successfully on {DEVICE}!")
except Exception as e:
    print(f"Failed to load VLM on {DEVICE}: {e}. Retrying on CPU...")
    # 清理資源
    if 'model' in locals() and model is not None: del model
    gc.collect() 
    
    if SELECTED_MODEL == "qwen3":
        model = ov_genai.VLMPipeline(VLM_MODEL_PATH, "CPU")
        if not processor: processor, tokenizer = load_vlm_components(VLM_MODEL_PATH, SELECTED_MODEL)
    else:
        if not processor: processor, tokenizer = load_vlm_components(VLM_MODEL_PATH, SELECTED_MODEL)
        config = (GemmaConfig.from_pretrained(VLM_MODEL_PATH) 
                 if SELECTED_MODEL == "gemma4" else AutoConfig.from_pretrained(VLM_MODEL_PATH))
        ModelClass = OVModelForVisualCausalLM if IS_VISION_MODEL else OVModelForCausalLM
        model = ModelClass.from_pretrained(
            VLM_MODEL_PATH, device="CPU", config=config, trust_remote_code=True,
            file_name="openvino_language_model.xml" if IS_VISION_MODEL else None
        )
    print(f"Model {SELECTED_MODEL} loaded on CPU!")

# --- Whisper 初始化 (用於語音輸入) ---
whisper_pipe = None
if os.path.exists(WHISPER_MODEL_PATH):
    print(f"Loading Whisper model ({WHISPER_MODEL_PATH})...")
    try:
        whisper_pipe = ov_genai.WhisperPipeline(WHISPER_MODEL_PATH, "CPU")
        print("Whisper model loaded!")
    except Exception as e:
        print(f"Failed to load Whisper: {e}")
else:
    print(f"⚠️ Whisper model not found at {WHISPER_MODEL_PATH}. Voice input disabled.")

# 全域對話歷史與鎖
history = [] 
generate_lock = asyncio.Lock()

def get_system_prompt():
    return (
        "你是『🎮 AI 遊戲陪伴也是電腦系統效能分析師』，一個充滿活力、說話幽默且親切的遊戲高手。 "
        "你會用繁體中文跟使用者聊天，分析截圖時會給出專業但好懂的建議。 "
        "偶爾可以開點小玩笑，不要太死板！敘述可以不用回太冗長。 "
        "詢問電腦相關系統效能問題 也可以專業的回答我。"
    )

class ChatRequest(BaseModel):
    message: str
    image_base64: Optional[str] = None

# --- OpenAI Compatibility Models ---
class OpenAIMessage(BaseModel):
    role: str
    content: Union[str, list[Any]] # OpenAI compat: text or list of blocks

class OpenAICompletionRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 8192
    temperature: Optional[float] = 0.7

# --- API 端點 ---
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": SELECTED_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "igpu-local"
            }
        ]
    }

@app.post("/clear")
async def clear_history():
    global history
    history = []
    print("Chat history cleared.")
    return {"status": "success", "message": "History cleared"}

@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    if whisper_pipe is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")
    try:
        audio_content = await file.read()
        audio_stream = io.BytesIO(audio_content)
        raw_speech, sr = librosa.load(audio_stream, sr=16000)
        input_data = raw_speech.tolist()
        result = await asyncio.to_thread(whisper_pipe.generate, input_data)
        text = str(result).strip()
        print(f"Transcribed: {text}")
        return {"status": "success", "text": text}
    except Exception as e:
        print(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(fastapi_request: Request, chat_request: ChatRequest):
    global history
    
    if not history:
        history.append({"role": "system", "content": get_system_prompt()})
    
    prompt_text = chat_request.message or "請分析這張截圖。"
    image = None
    if chat_request.image_base64 and IS_VISION_MODEL:
        try:
            if "," in chat_request.image_base64:
                chat_request.image_base64 = chat_request.image_base64.split(",")[1]
            img_data = base64.b64decode(chat_request.image_base64)
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
            image.thumbnail((768, 768), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Image decode error: {e}")
    elif chat_request.image_base64 and not IS_VISION_MODEL:
        print("💡 Current model is text-only. Ignoring image.")

    # 加入使用者訊息
    if SELECTED_MODEL == "qwen3":
        # Qwen3-VL/Qwen2-VL 建議使用 List of Dictionaries
        content_list = []
        if image:
            content_list.append({"type": "image", "image": image})
        content_list.append({"type": "text", "text": prompt_text})
        user_msg = {"role": "user", "content": content_list}
        if image: user_msg["images"] = [image]
    else:
        # Gemma-4 與其他模型使用帶有標籤的字串
        text_content = prompt_text
        if image:
            if "<image>" not in text_content:
                text_content = "<image>\n" + text_content
        user_msg = {"role": "user", "content": text_content}
        if image: user_msg["images"] = [image]
    
    history.append(user_msg)
    
    # --- 記憶體優化 (Image LRU): 只保留最近 2 張圖 ---
    images_found = 0
    for msg in reversed(history):
        if "images" in msg and msg["images"]:
            images_found += 1
            if images_found > 2:
                # 較舊的訊息：清除圖像對象以釋放內存
                msg["images"] = []
                # 如果是 qwen3 且 content 是 list，也清理裡面的 image 對象
                if isinstance(msg["content"], list):
                    msg["content"] = [c for c in msg["content"] if c["type"] == "text"]
                    msg["content"].insert(0, {"type": "text", "text": "[歷史影像已從內存釋放]"})
    
    # 保持歷史紀錄在合理範圍
    if len(history) > 21:
        history = [history[0]] + history[-20:]

    async def event_generator():
        if SELECTED_MODEL == "qwen3":
            # --- 使用 OpenVINO GenAI VLMPipeline 生成 ---
            queue = asyncio.Queue()
            loop = asyncio.get_event_loop()
            streamer = GenAIStreamer(queue, loop)
            
            # 1. 準備輸入影像
            ov_images = []
            for msg in history:
                if "images" in msg and msg["images"]:
                    ov_images.extend([pil_to_ov_tensor(img) for img in msg["images"]])

            # 2. 準備輸入文字
            full_prompt = processor.apply_chat_template(history, add_generation_prompt=True, tokenize=False)

            def generate_in_thread():
                try:
                    # 修正：GenAI 不接受 images=None。如果沒圖就完全不傳該參數。
                    if ov_images:
                        model.generate(full_prompt, images=ov_images, streamer=streamer, max_new_tokens=8192)
                    else:
                        model.generate(full_prompt, streamer=streamer, max_new_tokens=8192)
                except Exception as e:
                    print(f"GenAI Thread Error: {e}")
                finally:
                    # 無論成功失敗都必須發送 None 終止隊列
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            async with generate_lock:
                thread = Thread(target=generate_in_thread)
                thread.start()
                
                collected_response = ""
                try:
                    while True:
                        if await fastapi_request.is_disconnected(): break
                        token = await queue.get()
                        if token is None: break
                        collected_response += token
                        yield f"data: {json.dumps({'content': token})}\n\n"
                    
                    if collected_response:
                        history.append({"role": "assistant", "content": collected_response})
                finally:
                    thread.join()
                    gc.collect()
        else:
            # --- 使用原本的 Optimum-Intel 邏輯 (Gemma-4 等) ---
            all_images = []
            for msg in history:
                if "images" in msg and msg["images"]:
                    all_images.extend(msg["images"])
            
            full_prompt = processor.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
            target_images = all_images if all_images else None
            if target_images and len(target_images) == 1: target_images = target_images[0]
                
            try:
                if IS_VISION_MODEL:
                    inputs = processor(text=full_prompt, images=target_images, return_tensors="pt")
                else:
                    inputs = tokenizer(full_prompt, return_tensors="pt")
            except Exception as e:
                print(f"Processor/Tokenizer Error: {e}")
                yield f"data: {json.dumps({'content': f'❌ 處理錯誤: {e}.'})}\n\n"
                return

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=8192, do_sample=False)

            def generate_in_thread():
                try: model.generate(**generation_kwargs)
                except Exception as e: print(f"Generation Thread Error: {e}")

            async with generate_lock:
                thread = Thread(target=generate_in_thread)
                thread.start()
                collected_response = ""
                try:
                    while True:
                        if await fastapi_request.is_disconnected(): break
                        token = await asyncio.to_thread(lambda: next(streamer, None))
                        if token is None: break
                        collected_response += token
                        yield f"data: {json.dumps({'content': token})}\n\n"
                    if collected_response: history.append({"role": "assistant", "content": collected_response})
                finally:
                    thread.join()
                    gc.collect()

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/screenshot")
async def screenshot_endpoint():
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return {"image_base64": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def openai_chat_endpoint(fastapi_request: Request, request: OpenAICompletionRequest):
    global history
    
    # 1. 映射 OpenAI 歷史到內部格式
    new_history = [{"role": "system", "content": get_system_prompt()}]
    latest_image = None
    
    for msg in request.messages:
        role = msg.role
        content = msg.content
        text_parts = []
        
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if "base64," in url:
                            base64_data = url.split("base64,")[1]
                            try:
                                img_data = base64.b64decode(base64_data)
                                latest_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                                latest_image.thumbnail((768, 768), Image.Resampling.LANCZOS)
                            except Exception as e:
                                print(f"OpenAI Image decode error: {e}")
        else:
            text_parts.append(str(content))
            
        full_text = " ".join(text_parts)
        
        # 構造內部消息
        if role == "user":
            if SELECTED_MODEL == "qwen3":
                int_content = []
                if latest_image:
                    int_content.append({"type": "image", "image": latest_image})
                int_content.append({"type": "text", "text": full_text})
                new_msg = {"role": "user", "content": int_content, "images": [latest_image] if latest_image else []}
            else:
                final_text = full_text
                if IS_VISION_MODEL and latest_image and "<image>" not in final_text:
                    final_text = "<image>\n" + final_text
                new_msg = {"role": "user", "content": final_text, "images": [latest_image] if (latest_image and IS_VISION_MODEL) else []}
            new_history.append(new_msg)
        else:
            new_history.append({"role": role, "content": full_text})

    # 更新全局歷史 (可選，這裡我們先使用臨時歷史以保證相容性)
    history = new_history
    
    # 2. 調用現有的推理邏輯 (重用 event_generator 邏輯但改進輸出格式)
    async def openai_event_generator():
        all_images = []
        for msg in history:
            if "images" in msg and msg["images"]:
                all_images.extend(msg["images"])
        
        full_prompt = processor.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
        target_images = all_images if len(all_images) > 0 else None
        if target_images and len(target_images) == 1: target_images = target_images[0]
        
        try:
            if IS_VISION_MODEL:
                inputs = processor(text=full_prompt, images=target_images, return_tensors="pt")
            else:
                inputs = tokenizer(full_prompt, return_tensors="pt")
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=request.max_tokens, do_sample=False)
        
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        def generate_in_thread():
            try: model.generate(**generation_kwargs)
            except Exception as e: print(f"OpenAI Gen Error: {e}")

        async with generate_lock:
            thread = Thread(target=generate_in_thread); thread.start()
            try:
                while True:
                    token = await asyncio.to_thread(lambda: next(streamer, None))
                    if token is None: break
                    
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": SELECTED_MODEL,
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # 發送結束標記
                yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                thread.join(); gc.collect()

    if request.stream:
        return StreamingResponse(openai_event_generator(), media_type="text/event-stream")
    else:
        # 非串流模式（簡單實現）
        full_response = ""
        async for chunk in openai_event_generator():
            if "content" in chunk:
                data = json.loads(chunk.replace("data: ", ""))
                if "choices" in data and "content" in data["choices"][0]["delta"]:
                    full_response += data["choices"][0]["delta"]["content"]
        
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": SELECTED_MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": full_response}, "finish_reason": "stop"}]
        }

if __name__ == "__main__":
    import uvicorn
    multiprocessing.freeze_support()
    uvicorn.run(app, host="127.0.0.1", port=8000)
