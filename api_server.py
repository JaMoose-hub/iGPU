import os
import base64
import json
import io
import asyncio
import numpy as np
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mss
from PIL import Image
import openvino as ov
import openvino_genai as ov_genai
import sys
import multiprocessing

def get_model_path():
    # 優先尋找與執行檔同層的 models 資料夾
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    # 預設路徑
    default_path = os.path.join(base_path, "models", "Qwen3-VL-8B-openvino-int4")
    
    # 如果預設路徑不存在，也可以考慮開發時的絕對路徑作為備援
    if not os.path.exists(default_path):
        dev_path = r"C:\Project\iGPU\Qwen3-VL-8B-openvino-int4"
        return dev_path if os.path.exists(dev_path) else default_path
        
    return default_path

# --- 配置 ---
MODEL_PATH = get_model_path()
DEVICE = "GPU" # 如果 GPU 跑不動，可以手動改為 "CPU"

app = FastAPI(title="iGPU Qwen3-VL Backend")

# 允許跨域請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 模型與狀態初始化 ---
print(f"Loading VLM model Qwen3-VL ({DEVICE})... Please wait...")
try:
    pipe = ov_genai.VLMPipeline(MODEL_PATH, DEVICE)
    print("VLM model loaded successfully!")
except Exception as e:
    print(f"Failed to load model on {DEVICE}: {e}")
    pipe = ov_genai.VLMPipeline(MODEL_PATH, "CPU")
    print("VLM model loaded on CPU!")

# 全域對話歷史
history = ov_genai.ChatHistory()

def get_system_prompt():
    return (
        "你是『🎮 AI 遊戲陪伴』，一個充滿活力、說話幽默且親切的遊戲高手。 "
        "你會用繁體中文跟使用者聊天，分析截圖時會給出專業但好懂的建議。 "
        "偶爾可以開點小玩笑，不要太死板！"
    )

# --- 定義資料結構 ---
class ChatRequest(BaseModel):
    message: str
    image_base64: Optional[str] = None

# --- 輔助函式 ---
def base64_to_ov_tensor(base64_str: str):
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(img_data)).convert("RGB")
    
    max_size = 768
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    return ov.Tensor(np.array(image))

# --- API 端點 ---

@app.post("/clear")
async def clear_history():
    """清除對話歷史"""
    global history
    history.clear()
    print("Chat history cleared.")
    return {"status": "success", "message": "History cleared"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global history
    
    # 1. 檢查是否需要初始化 System Prompt
    if len(history) == 0:
        history.append({"role": "system", "content": get_system_prompt()})
    
    # 2. 處理使用者訊息與影像
    prompt = request.message or "請分析這張截圖。"
    image_tensor = None
    if request.image_base64:
        try:
            image_tensor = base64_to_ov_tensor(request.image_base64)
        except Exception as e:
            print(f"Image decode error: {e}")

    # 3. 準備生成器
    def event_generator():
        tokens_queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        full_response = []

        def streamer_callback(token: str):
            loop.call_soon_threadsafe(tokens_queue.put_nowait, token)
            full_response.append(token)
            return False

        # 在背景執行緒中執行耗時的推論，避免卡死 event loop
        async def run_generate():
            # 注入「性格」：親切、幽默、一點點專業
            if len(history) == 0:
                history.append({"role": "system", "content": get_system_prompt()})
            
            # 將使用者訊息加入歷史
            history.append({"role": "user", "content": prompt})

            try:
                # 呼叫帶有歷史紀錄的生成
                # Positional Argument 1: history
                if image_tensor is not None:
                    await asyncio.to_thread(
                        pipe.generate, history, image=image_tensor, streamer=streamer_callback, max_new_tokens=2048
                    )
                else:
                    await asyncio.to_thread(
                        pipe.generate, history, streamer=streamer_callback, max_new_tokens=2048
                    )
            except Exception as e:
                loop.call_soon_threadsafe(tokens_queue.put_nowait, f"❌ 推論錯誤: {str(e)}")
            finally:
                loop.call_soon_threadsafe(tokens_queue.put_nowait, None) # 結束標記

        asyncio.create_task(run_generate())

        async def stream():
            while True:
                token = await tokens_queue.get()
                if token is None: break
                yield f"data: {json.dumps({'content': token})}\n\n"

        return stream()

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

if __name__ == "__main__":
    import uvicorn
    # 修復 PyInstaller 多進程問題
    multiprocessing.freeze_support()
    # 啟動在 8000 埠
    uvicorn.run(app, host="127.0.0.1", port=8000)
