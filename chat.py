import os
import sys
import codecs
import openvino_genai as ov_genai
from huggingface_hub import snapshot_download

# 修正 Windows 終端機亂碼問題
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

model_id = "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"
model_path = "./TinyLlama-1.1B-int4"

print("正在準備 AI 小幫手，請稍候...")

# 如果模型沒有下載過，就先下載
if not os.path.exists(model_path):
    snapshot_download(repo_id=model_id, local_dir=model_path)

# 設定給 CPU 跑
pipe = ov_genai.LLMPipeline(model_path, "CPU")

# 這裡是一些設定，像是它一次最多可以講多少單字
config = ov_genai.GenerationConfig()
config.max_new_tokens = 200

print("\n=============================================")
print("  AI 小幫手已經準備好囉！快來跟我聊天吧！  ")
print("  (如果你想結束，只要輸入 '離開' 或 'exit' 就可以關閉)  ")
print("=============================================\n")

# 用一個無窮迴圈 (while True) 讓你可以一直問一直問！
pipe.start_chat() # 告訴它我們要開始連續對話了！

while True:
    # 這行會在螢幕上顯示「你：」，並等你打字然後按下 Enter
    user_input = input("你：")
    
    # 如果使用者輸入離開，就跳出迴圈結束程式
    if user_input.strip() == "離開" or user_input.strip().lower() == "exit":
        print("\nAI：掰掰！下次見囉！")
        break
        
    print("AI 正在思考中...\n")
    
    # pipe 會記住前面的對話，並且回答新的問題！
    answer = pipe.generate(user_input, config)
    
    # 印出 AI 的回答
    print("AI：" + answer.strip() + "\n")
    print("-" * 50)
    
pipe.finish_chat()
