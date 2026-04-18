import subprocess
import os
import sys

def download_and_export_whisper():
    model_id = "openai/whisper-base"
    export_dir = "models/whisper-base-ov"
    
    if os.path.exists(export_dir):
        print(f"Model directory {export_dir} already exists. Skipping download.")
        return

    print(f"Exporting {model_id} to {export_dir} using OpenVINO... This may take a few minutes.")
    
    try:
        # 使用 optimum-cli 進行導出
        # 需確保已安裝 optimum[openvino]
        cmd = [
            "optimum-cli", "export", "openvino",
            "--model", model_id,
            "--task", "automatic-speech-recognition-with-past",
            export_dir
        ]
        
        subprocess.run(cmd, check=True)
        print(f"\n✅ 成功導出模型至 {export_dir}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 導出失敗: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ 找不到 optimum-cli。請確保已安裝 optimum[openvino] (pip install optimum[openvino])")
        sys.exit(1)

if __name__ == "__main__":
    download_and_export_whisper()
