document.addEventListener("DOMContentLoaded", async () => {
  // ── Tauri v2 API 初始化 ──
  const { getCurrentWindow } = window.__TAURI__.window;
  const { WebviewWindow } = window.__TAURI__.webviewWindow;
  const { register, unregisterAll } = window.__TAURI__.globalShortcut;
  const { invoke } = window.__TAURI__.core;
  const { listen, emitTo } = window.__TAURI__.event;

  const closeBtn = document.getElementById("closeBtn");
  const minBtn = document.getElementById("minBtn");
  const maxBtn = document.getElementById("maxBtn");
  const sendBtn = document.getElementById("sendBtn");
  const stopBtn = document.getElementById("stopBtn");
  const messageInput = document.getElementById("messageInput");
  const chatWindow = document.getElementById("chatWindow");
  const dragBar = document.querySelector(".drag-bar");
  const screenshotBtn = document.getElementById("screenshotBtn");
  const opacitySlider = document.getElementById("opacitySlider");
  const perfBtn = document.getElementById("perfBtn");
  const voiceBtn = document.getElementById("voiceBtn");
  const gameSelect = document.getElementById("gameSelect");
  const hudBtn = document.getElementById("hudBtn");
  const resizeGrip = document.getElementById("resizeGrip");
  
  const imagePreviewArea = document.getElementById("imagePreviewArea");
  const previewImg = document.getElementById("previewImg");
  const removeImgBtn = document.getElementById("removeImgBtn");

  const appWindow = getCurrentWindow();
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  let hudWindowPromise = null;
  let hudReady = false;
  let hudReadyPromise = null;
  
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
  const guideIntentRe = /(攻略|怎麼打|怎麼走|在哪|哪裡|弱點|任務|素材|材料|路線|指路|地圖|boss|npc|quest|guide|route|map|weakness)/i;
  let hudEnabled = localStorage.getItem("hud-enabled") !== "false";
  if (hudBtn) hudBtn.classList.toggle("active", hudEnabled);

  const waitForHudReady = async () => {
    if (hudReady) return;
    if (hudReadyPromise) return hudReadyPromise;

    hudReadyPromise = new Promise(async (resolve) => {
      let settled = false;
      let unlisten = null;
      const finish = async () => {
        if (settled) return;
        settled = true;
        hudReady = true;
        if (unlisten) await unlisten().catch(() => {});
        resolve();
      };

      const timeout = setTimeout(finish, 4000);
      unlisten = await listen("hud:ready", async (event) => {
        if (event?.payload?.label === "hud") {
          clearTimeout(timeout);
          await finish();
        }
      }).catch(() => null);
    });

    return hudReadyPromise;
  };

  waitForHudReady();

  const ensureHudWindow = async () => {
    if (hudWindowPromise) return hudWindowPromise;

    hudWindowPromise = (async () => {
      const existing = await WebviewWindow.getByLabel("hud").catch(() => null);
      if (existing) {
        await waitForHudReady();
        await invoke("configure_hud_window").catch((err) => console.warn("HUD configure failed:", err));
        return existing;
      }

      const readyPromise = waitForHudReady();
      const hudWidth = Math.max(Number(window.screen?.width) || 1280, 1);
      const hudHeight = Math.max(Number(window.screen?.height) || 720, 1);
      const hudWindow = new WebviewWindow("hud", {
        url: "/hud.html",
        title: "game-guidance-hud",
        x: 0,
        y: 0,
        width: hudWidth,
        height: hudHeight,
        transparent: true,
        decorations: false,
        alwaysOnTop: true,
        skipTaskbar: true,
        focusable: false,
        resizable: false,
        visible: false,
        shadow: false
      });

      await new Promise((resolve, reject) => {
        const timeout = setTimeout(resolve, 2000);
        hudWindow.once("tauri://created", () => {
          clearTimeout(timeout);
          resolve();
        });
        hudWindow.once("tauri://error", (event) => {
          clearTimeout(timeout);
          hudWindowPromise = null;
          reject(event.payload || event);
        });
      });

      await readyPromise;
      await hudWindow.hide().catch(() => {});
      await invoke("configure_hud_window").catch((err) => console.warn("HUD configure failed:", err));
      return hudWindow;
    })();

    return hudWindowPromise;
  };

  const showHudOverlay = async (overlay) => {
    if (!hudEnabled || !overlay || !overlay.items || !overlay.items.length) return;
    try {
      await ensureHudWindow();
      await invoke("show_hud_window", {
        width: Math.max(Number(window.screen?.width) || 1280, 1),
        height: Math.max(Number(window.screen?.height) || 720, 1)
      });
      await emitTo("hud", "hud:show", overlay);
    } catch (err) {
      console.warn("HUD show failed:", err);
    }
  };

  const clearHudOverlay = async () => {
    try {
      if (!hudWindowPromise) return;
      await ensureHudWindow();
      await emitTo("hud", "hud:clear");
      await invoke("hide_hud_window").catch(() => {});
    } catch (err) {
      console.warn("HUD clear failed:", err);
    }
  };

  if (hudBtn) {
    hudBtn.addEventListener("click", async () => {
      hudEnabled = !hudEnabled;
      localStorage.setItem("hud-enabled", hudEnabled ? "true" : "false");
      hudBtn.classList.toggle("active", hudEnabled);
      if (!hudEnabled) await clearHudOverlay();
    });
  }

  const loadGuideGames = async () => {
    if (!gameSelect) return;
    const savedGameId = localStorage.getItem("currentGameId") || "";
    try {
      const resp = await fetch("http://127.0.0.1:8000/guides/games");
      const data = await resp.json();
      const games = Array.isArray(data.games) ? data.games : [];
      gameSelect.innerHTML = '<option value="">遊戲</option>';
      for (const game of games) {
        const option = document.createElement("option");
        option.value = game;
        option.textContent = game;
        gameSelect.appendChild(option);
      }
      gameSelect.value = savedGameId;
    } catch (err) {
      console.warn("Guide game list unavailable:", err);
    }
  };

  if (gameSelect) {
    gameSelect.value = localStorage.getItem("currentGameId") || "";
    gameSelect.addEventListener("change", () => {
      localStorage.setItem("currentGameId", gameSelect.value);
    });
    loadGuideGames();
  }

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
      if (e.target.closest(".drag-bar-actions")) return;
      await appWindow.startDragging();
    });
  }

  // ── 關閉按鈕 ──
  const toggleMaximize = async () => {
    try {
      await appWindow.toggleMaximize();
    } catch (err) {
      console.warn("Toggle maximize failed:", err);
    }
  };

  if (dragBar) {
    dragBar.addEventListener("dblclick", async (e) => {
      if (e.target.closest(".drag-bar-actions")) return;
      await toggleMaximize();
    });
  }

  if (minBtn) {
    minBtn.addEventListener("click", async () => {
      try {
        await appWindow.minimize();
      } catch (err) {
        console.warn("Minimize failed:", err);
      }
    });
  }

  if (maxBtn) {
    maxBtn.addEventListener("click", toggleMaximize);
  }

  if (resizeGrip) {
    resizeGrip.addEventListener("mousedown", async (e) => {
      e.preventDefault();
      e.stopPropagation();
      try {
        await appWindow.startResizeDragging("SouthEast");
      } catch (err) {
        console.warn("Resize drag failed:", err);
      }
    });
  }

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
      const currentGameId = gameSelect ? gameSelect.value : (localStorage.getItem("currentGameId") || "");
      const body = {
        message: text,
        game_id: currentGameId || null,
        use_guides: guideIntentRe.test(text || ""),
        use_memory: true
      };
      if (imageBase64) body.image_base64 = imageBase64;

      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: abortController.signal
      });

      if (!response.body) throw new Error("No response body");
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let buffer = "";
      let sseBuffer = "";
      let rafId = null;
      let hasStreamContent = false;

      const updateUI = () => {
        if (buffer) {
          if (!hasStreamContent) {
            botMsgDiv.innerText = "";
            hasStreamContent = true;
          }
          botMsgDiv.innerText += buffer;
          buffer = "";
          chatWindow.scrollTop = chatWindow.scrollHeight;
        }
        rafId = null;
      };

      const handleSseLine = (line) => {
        if (!line.startsWith("data: ")) return;
        const dataStr = line.replace("data: ", "").trim();
        if (!dataStr) return;
        try {
          const dataObj = JSON.parse(dataStr);
          const content = dataObj.content || "";
          if (dataObj.overlay) {
            showHudOverlay(dataObj.overlay);
          }
          if (content) {
            buffer += content;
            if (!rafId) {
              rafId = requestAnimationFrame(updateUI);
            }
          }
        } catch (err) {
          console.warn("SSE parse failed:", err);
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        sseBuffer += decoder.decode(value, { stream: true });
        let lineEnd = sseBuffer.indexOf("\n");
        while (lineEnd >= 0) {
          const line = sseBuffer.slice(0, lineEnd).trimEnd();
          sseBuffer = sseBuffer.slice(lineEnd + 1);
          handleSseLine(line);
          lineEnd = sseBuffer.indexOf("\n");
        }
      }
      if (sseBuffer.trim()) handleSseLine(sseBuffer.trimEnd());
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
    if (screenshotBtn.disabled) return;

    try {
      screenshotBtn.textContent = "⏳";
      screenshotBtn.disabled = true;

      await clearHudOverlay();
      await appWindow.hide();
      await sleep(280);

      const res = await fetch("http://127.0.0.1:8000/screenshot");
      if (!res.ok) throw new Error(`Screenshot API failed: ${res.status}`);
      const data = await res.json();

      if (data.image_base64) {
        showImagePreview(data.image_base64);
      }
    } catch (err) {
      console.error("截圖失敗", err);
    } finally {
      await appWindow.show().catch(() => {});
      screenshotBtn.textContent = "📷";
      screenshotBtn.disabled = false;
    }
  };

  screenshotBtn.addEventListener("click", takeScreenshot);
  await listen("capture-hotkey", () => {
    takeScreenshot();
  });
  await listen("clear-hud-hotkey", () => {
    clearHudOverlay();
  });

  // ── 全局快捷鍵 F9 ──
  try {
    await register("F9", () => {
      takeScreenshot();
    });
    console.log("✅ 全局快捷鍵 F9 已啟用");
  } catch (e) {
    console.warn("⚠️ 全局快捷鍵 F9 註冊失敗:", e);
  }

  try {
    await register("F10", () => {
      clearHudOverlay();
    });
    console.log("✅ 全局快捷鍵 F10 已啟用");
  } catch (e) {
    console.warn("⚠️ 全局快捷鍵 F10 註冊失敗:", e);
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
