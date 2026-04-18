document.addEventListener("DOMContentLoaded", async () => {
  // ── Tauri v2 API 初始化 ──
  const { getCurrentWindow } = window.__TAURI__.window;
  const { register, unregisterAll } = window.__TAURI__.globalShortcut;

  const closeBtn = document.getElementById("closeBtn");
  const sendBtn = document.getElementById("sendBtn");
  const stopBtn = document.getElementById("stopBtn");
  const messageInput = document.getElementById("messageInput");
  const chatWindow = document.getElementById("chatWindow");
  const dragBar = document.querySelector(".drag-bar");
  const screenshotBtn = document.getElementById("screenshotBtn");
  const opacitySlider = document.getElementById("opacitySlider");
  const perfBtn = document.getElementById("perfBtn");
  const voiceBtn = document.getElementById("voiceBtn");
  
  const imagePreviewArea = document.getElementById("imagePreviewArea");
  const previewImg = document.getElementById("previewImg");
  const removeImgBtn = document.getElementById("removeImgBtn");

  const appWindow = getCurrentWindow();
  
  // ── 透明度初始化與監聽 ──
  const savedOpacity = localStorage.getItem("ui-opacity") || "0.85";
  document.documentElement.style.setProperty("--glass-opacity", savedOpacity);
  if (opacitySlider) opacitySlider.value = savedOpacity;

  if (opacitySlider) {
    opacitySlider.addEventListener("input", (e) => {
      const val = e.target.value;
      document.documentElement.style.setProperty("--glass-opacity", val);
      localStorage.setItem("ui-opacity", val);
    });
  }
  
  // ── 效能模式初始化與監聽 ──
  const isPerfMode = localStorage.getItem("perf-mode") === "true";
  if (isPerfMode) document.body.classList.add("perf-mode");

  if (perfBtn) {
    perfBtn.addEventListener("click", () => {
      const enabled = document.body.classList.toggle("perf-mode");
      localStorage.setItem("perf-mode", enabled);
    });
  }

  // ── 語音輸入 (OpenVINO Whisper) ──
  const { listen } = window.__TAURI__.event;
  let mediaRecorder = null;
  let audioChunks = [];

  // 輔助函式：將 AudioBuffer 轉換為 WAV 格式
  function audioBufferToWav(buffer) {
    const numOfChan = buffer.numberOfChannels,
          length = buffer.length * numOfChan * 2 + 44,
          bufferArr = new ArrayBuffer(length),
          view = new DataView(bufferArr),
          channels = [];
    let sample, offset = 0, pos = 0;

    const setUint16 = (data) => { view.setUint16(pos, data, true); pos += 2; };
    const setUint32 = (data) => { view.setUint32(pos, data, true); pos += 4; };

    setUint32(0x46464952); // "RIFF"
    setUint32(length - 8);
    setUint32(0x45564157); // "WAVE"
    setUint32(0x20746d66); // "fmt "
    setUint32(16);
    setUint16(1); // PCM
    setUint16(numOfChan);
    setUint32(buffer.sampleRate);
    setUint32(buffer.sampleRate * 2 * numOfChan);
    setUint16(numOfChan * 2);
    setUint16(16);
    setUint32(0x61746164); // "data"
    setUint32(length - pos - 4);

    for (let i = 0; i < numOfChan; i++) channels.push(buffer.getChannelData(i));
    while (pos < length) {
      for (let i = 0; i < numOfChan; i++) {
        sample = Math.max(-1, Math.min(1, channels[i][offset]));
        sample = (sample < 0 ? sample * 0x8000 : sample * 0x7FFF) | 0;
        view.setInt16(pos, sample, true);
        pos += 2;
      }
      offset++;
    }
    return new Blob([bufferArr], { type: 'audio/wav' });
  }

  const handleRecordingStop = async () => {
    const webmBlob = new Blob(audioChunks, { type: 'audio/webm' });
    const originalIcon = voiceBtn ? voiceBtn.innerText : "🎙️";
    if (voiceBtn) voiceBtn.innerText = "⌛"; // 辨識中
    
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const arrayBuffer = await webmBlob.arrayBuffer();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      
      const offlineCtx = new OfflineAudioContext(1, Math.ceil(audioBuffer.duration * 16000), 16000);
      const source = offlineCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(offlineCtx.destination);
      source.start();
      const resampledBuffer = await offlineCtx.startRendering();
      
      const wavBlob = audioBufferToWav(resampledBuffer);
      const formData = new FormData();
      formData.append('file', wavBlob, 'record.wav');

      const resp = await fetch("http://127.0.0.1:8000/transcribe", {
        method: "POST",
        body: formData
      });
      const data = await resp.json();
      if (data.status === "success" && data.text) {
        messageInput.value = data.text;
        messageInput.focus();
        // 自動送出
        if (sendBtn.style.display !== "none") {
          sendMessage();
        }
      }
    } catch (err) {
      console.error("STT Failed:", err);
    } finally {
      if (voiceBtn) voiceBtn.innerText = originalIcon;
    }
  };

  const startRecording = async () => {
    if (mediaRecorder && mediaRecorder.state === "recording") return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];
      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
      mediaRecorder.onstop = handleRecordingStop;
      mediaRecorder.start();
      if (voiceBtn) voiceBtn.classList.add("recording");
    } catch (err) {
      console.error("Mic Error:", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
      if (voiceBtn) voiceBtn.classList.remove("recording");
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
  };

  // 監聽全域快捷鍵
  listen("voice-hotkey-start", startRecording);
  listen("voice-hotkey-stop", stopRecording);

  if (voiceBtn) {
    voiceBtn.addEventListener("mousedown", startRecording);
    voiceBtn.addEventListener("mouseup", stopRecording);
    voiceBtn.addEventListener("mouseleave", stopRecording);
  }

  let pendingImageBase64 = null;
  let abortController = null;

  // ── 頂部可拖曳區域 ──
  if (dragBar) {
    dragBar.addEventListener("mousedown", async (e) => {
      if (e.target.id === "closeBtn" || e.target.closest(".drag-bar-actions button")) return;
      await appWindow.startDragging();
    });
  }

  // ── 關閉按鈕 ──
  if (closeBtn) {
    closeBtn.addEventListener("click", async () => {
      try { await unregisterAll(); } catch (_) {}
      await appWindow.close();
    });
  }

  // ── 圖片預覽管理 ──
  const showImagePreview = (base64) => {
    pendingImageBase64 = base64;
    previewImg.src = `data:image/png;base64,${base64}`;
    imagePreviewArea.style.display = "flex";
  };

  const clearImagePreview = () => {
    pendingImageBase64 = null;
    previewImg.src = "";
    imagePreviewArea.style.display = "none";
  };

  removeImgBtn.addEventListener("click", clearImagePreview);

  // ── 訊息顯示 ──
  const appendMessage = (text, sender) => {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", sender === "user" ? "user-message" : "bot-message");
    msgDiv.innerText = text;
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return msgDiv;
  };

  // ── 傳送訊息給 AI ──
  const sendToAI = async (text, imageBase64 = null) => {
    // 顯示「停止」按鈕，隱藏「發送」按鈕
    sendBtn.style.display = "none";
    stopBtn.style.display = "block";
    
    const botMsgDiv = appendMessage("💭 思考中...", "bot");
    abortController = new AbortController();

    try {
      const body = { message: text };
      if (imageBase64) body.image_base64 = imageBase64;

      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: abortController.signal
      });

      botMsgDiv.innerText = "";

      if (!response.body) throw new Error("No response body");
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let buffer = "";
      let rafId = null;

      const updateUI = () => {
        if (buffer) {
          botMsgDiv.innerText += buffer;
          buffer = "";
          chatWindow.scrollTop = chatWindow.scrollHeight;
        }
        rafId = null;
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");
        for (let line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.replace("data: ", "").trim();
            if (!dataStr) continue;
            try {
              const dataObj = JSON.parse(dataStr);
              const content = dataObj.content || "";
              if (content) {
                buffer += content;
                // 使用 requestAnimationFrame 進行節流更新
                if (!rafId) {
                  rafId = requestAnimationFrame(updateUI);
                }
              }
            } catch (_) {}
          }
        }
      }
      // 確保最後剩餘的內容也被渲染
      if (rafId) cancelAnimationFrame(rafId);
      updateUI();
    } catch (error) {
      if (error.name === "AbortError") {
        botMsgDiv.innerText += "\n\n⚠️ 中斷回覆。";
      } else {
        console.error(error);
        botMsgDiv.innerText = "❌ 出現錯誤，請確認後端是否啟動。";
      }
    } finally {
      sendBtn.style.display = "block";
      stopBtn.style.display = "none";
      abortController = null;
    }
  };

  // ── 停止按鈕功能 ──
  stopBtn.addEventListener("click", () => {
    if (abortController) {
      abortController.abort();
    }
  });

  // ── 截圖核心函式 ──
  const takeScreenshot = async () => {
    try {
      screenshotBtn.textContent = "⏳";
      screenshotBtn.disabled = true;

      const res = await fetch("http://127.0.0.1:8000/screenshot");
      const data = await res.json();

      screenshotBtn.textContent = "📷";
      screenshotBtn.disabled = false;

      if (data.image_base64) {
        showImagePreview(data.image_base64);
      }
    } catch (err) {
      screenshotBtn.textContent = "📷";
      screenshotBtn.disabled = false;
      console.error("截圖失敗", err);
    }
  };

  screenshotBtn.addEventListener("click", takeScreenshot);

  // ── 全局快捷鍵 F9 ──
  try {
    await register("F9", () => {
      takeScreenshot();
    });
    console.log("✅ 全局快捷鍵 F9 已啟用");
  } catch (e) {
    console.warn("⚠️ 全局快捷鍵 F9 註冊失敗:", e);
  }

  // ── 發送文字訊息 ──
  const sendMessage = async () => {
    const text = messageInput.value.trim();
    const hasImage = !!pendingImageBase64;
    
    if (!text && !hasImage) return;

    // 清空輸入
    messageInput.value = "";
    
    // 顯示使用者發送的訊息內容
    const userDiv = document.createElement("div");
    userDiv.classList.add("message", "user-message");
    
    let userContent = text || (hasImage ? "📷 [影像已發送]" : "");
    userDiv.innerText = userContent;
    
    if (hasImage) {
      const img = document.createElement("img");
      img.src = `data:image/png;base64,${pendingImageBase64}`;
      userDiv.appendChild(document.createElement("br"));
      userDiv.appendChild(img);
    }
    
    chatWindow.appendChild(userDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;

    const currentImage = pendingImageBase64;
    clearImagePreview();

    await sendToAI(text || "請分析這張截圖。", currentImage);
  };

  sendBtn.addEventListener("click", sendMessage);
  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      if (sendBtn.style.display !== "none") {
        sendMessage();
      }
    }
  });
});
