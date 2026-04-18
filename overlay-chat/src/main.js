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
  
  const imagePreviewArea = document.getElementById("imagePreviewArea");
  const previewImg = document.getElementById("previewImg");
  const removeImgBtn = document.getElementById("removeImgBtn");

  const appWindow = getCurrentWindow();
  
  // ── 透明度初始化與監聽 ──
  const savedOpacity = localStorage.getItem("ui-opacity") || "0.75";
  document.documentElement.style.setProperty("--glass-opacity", savedOpacity);
  opacitySlider.value = savedOpacity;

  opacitySlider.addEventListener("input", (e) => {
    const val = e.target.value;
    document.documentElement.style.setProperty("--glass-opacity", val);
    localStorage.setItem("ui-opacity", val);
  });

  let pendingImageBase64 = null;
  let abortController = null;

  // ── 頂部可拖曳區域 ──
  dragBar.addEventListener("mousedown", async (e) => {
    if (e.target.id === "closeBtn" || e.target.closest(".drag-bar-actions button")) return;
    await appWindow.startDragging();
  });

  // ── 關閉按鈕 ──
  closeBtn.addEventListener("click", async () => {
    try { await unregisterAll(); } catch (_) {}
    await appWindow.close();
  });

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
              botMsgDiv.innerText += dataObj.content || "";
              chatWindow.scrollTop = chatWindow.scrollHeight;
            } catch (_) {}
          }
        }
      }
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
