# iGPU AI 專案

這是一個專為 Intel iGPU 優化的 AI 推論服務，支援 Qwen2.5-VL 與 Gemma-4 等最新模型。

## 🐍 開發環境

本專案建議在 **Conda** 環境中執行，以確保 OpenVINO 與 Intel GPU 驅動程式的相容性。

- **Conda 環境名稱**: `igpu`
- **Python 版本**: `3.12+`

### 切換環境
```bash
conda activate igpu
```

## 🚀 快速啟動

在 `igpu` 環境中，你可以使用以下指令啟動不同的模型服務：

### 啟動 Qwen3-VL (預設模式)
針對 Qwen2.5-VL 模型，我們使用 `openvino-genai` 引擎以獲得最高穩定性。
```bash
python api_server.py qwen3
```

### 啟動 Gemma-4
```bash
python api_server.py gemma4
```

## 🛠 關鍵組件
- **後端引擎**: FastAPI
- **推論引擎**: 
  - `openvino-genai` (適用於 Qwen3-VL)
  - `optimum-intel` (適用於傳統 Transformers 模型)
- **硬體加速**: Intel OpenVINO (GPU 加速)

## 📁 模型路徑
模型檔案應放置於 `./models/` 目錄下：
- Qwen3: `C:\Project\iGPU\models\Qwen3-VL-8B-openvino-int4`
- Gemma: `C:\Project\iGPU\models\gemma-4-E4B-ov`
