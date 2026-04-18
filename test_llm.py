import os
import sys
import codecs
import openvino_genai as ov_genai
from huggingface_hub import snapshot_download

# Fix Windows console encoding issue (force UTF-8)
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# 這是一個專門為 OpenVINO 準備過的小巧模型 (大約佔據 1GB 空間，跑起來很快)
model_id = "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"
model_path = "./TinyLlama-1.1B-int4"

print("步驟 1：正在幫你下載並準備小小 AI 模型，這大概需要幾分鐘時間喔...")
if not os.path.exists(model_path):
    # 如果還沒有下載過，就把它下載到同一個資料夾底下
    snapshot_download(repo_id=model_id, local_dir=model_path)
    print("太棒了！模型下載完成！")
else:
    print("模型之前已經下載過了，我們直接開始囉！")

print("步驟 2：正在叫醒小 AI，把它放進大腦裡準備回答問題...")
# 設定使用 CPU 來跑模型（這是我幫你預設的，也可以改成 GPU）
pipe = ov_genai.LLMPipeline(model_path, "CPU")

# 這是特別設計給這個小 AI 的溝通格式，它才能聽得懂
prompt = "<|system|>\nYou are a helpful assistant. (你是一個有用的助手)<|user|>\nWhat is a Large Language Model (LLM)? Explain it to a 12-year-old child.<|assistant|>\n"

print("\n--- AI 開始說話囉 ---")
answer = pipe.generate(prompt, max_new_tokens=100)
print(answer)

print("\n--- 結束 ---")
print("\n測試大成功！這就是目前在你電腦上活生生跑起來的 AI 喔！")
