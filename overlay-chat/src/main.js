document.addEventListener("DOMContentLoaded", async () => {
  const tauri = window.__TAURI__ || {};
  const appWindow = tauri.window?.getCurrentWindow?.();
  const globalShortcut = tauri.globalShortcut || {};
  const events = tauri.event || {};
  const invoke = tauri.core?.invoke;

  const API_BASE = "http://127.0.0.1:8000";

  const closeBtn = document.getElementById("closeBtn");
  const minBtn = document.getElementById("minBtn");
  const maxBtn = document.getElementById("maxBtn");
  const sendBtn = document.getElementById("sendBtn");
  const stopBtn = document.getElementById("stopBtn");
  const messageInput = document.getElementById("messageInput");
  const chatWindow = document.getElementById("chatWindow");
  const dragBar = document.querySelector(".drag-bar");
  const screenshotBtn = document.getElementById("screenshotBtn");
  const perfBtn = document.getElementById("perfBtn");
  const voiceBtn = document.getElementById("voiceBtn");
  const gameSelect = document.getElementById("gameSelect");
  const hudBtn = document.getElementById("hudBtn");
  const hudTestBtn = document.getElementById("hudTestBtn");
  const protectBtn = document.getElementById("protectBtn");
  const resizeGrip = document.getElementById("resizeGrip");
  const imagePreviewArea = document.getElementById("imagePreviewArea");
  const previewImg = document.getElementById("previewImg");
  const removeImgBtn = document.getElementById("removeImgBtn");

  let pendingImageBase64 = null;
  let pendingImageMimeType = "image/jpeg";
  let pendingCaptureSource = null;
  let abortController = null;
  let isSending = false;
  let isVoiceRecording = false;
  let isVoiceBusy = false;
  let mediaRecorder = null;
  let mediaStream = null;
  let voiceChunks = [];
  let voiceStatusMessage = null;
  let isVoiceStarting = false;
  let voiceStopRequested = false;
  let lastHudError = "";
  localStorage.setItem("protect-mode", "off");
  let captureProtectionEnabled = false;

  localStorage.removeItem("hud-overlay");
  await invoke?.("hide_hud_window").catch((err) => {
    console.warn("Could not hide stale HUD window:", err);
  });
  document.documentElement.style.setProperty("--glass-opacity", "1");
  localStorage.removeItem("ui-opacity");
  await appWindow?.setContentProtected?.(false).catch((err) => {
    console.warn("Could not clear main-window content protection:", err);
  });
  await invoke?.("set_main_capture_exclusion", { excluded: captureProtectionEnabled }).catch((err) => {
    console.warn("Could not set window capture exclusion:", err);
  });

  if (localStorage.getItem("perf-mode") === "true") {
    document.body.classList.add("perf-mode");
  }

  const updateProtectButton = () => {
    protectBtn?.classList.toggle("active", captureProtectionEnabled);
    if (protectBtn) {
      protectBtn.textContent = captureProtectionEnabled ? "Protect" : "NoProt";
      protectBtn.title = captureProtectionEnabled
        ? "Windows capture protection on: companion windows are excluded from screenshots and screen sharing"
        : "Windows capture protection off: companion windows can appear in screenshots and screen sharing";
    }
    localStorage.setItem("protect-mode", captureProtectionEnabled ? "os-exclude" : "off");
  };

  updateProtectButton();

  protectBtn?.addEventListener("click", async () => {
    captureProtectionEnabled = !captureProtectionEnabled;
    updateProtectButton();
    await invoke?.("set_main_capture_exclusion", { excluded: captureProtectionEnabled }).catch((err) => {
      appendMessage(`Protect command failed: ${err?.message || err}`, "bot");
    });
    appendMessage(
      captureProtectionEnabled
        ? "Windows capture protection on."
        : "Screenshot protection off.",
      "bot"
    );
  });

  perfBtn?.addEventListener("click", () => {
    const enabled = document.body.classList.toggle("perf-mode");
    localStorage.setItem("perf-mode", enabled ? "true" : "false");
  });

  const clearHudOverlay = async () => {
    localStorage.removeItem("hud-overlay");
    if (invoke) {
      await invoke("clear_hud_overlay").catch((err) => console.warn("HUD clear command failed:", err));
      return;
    }
    if (events.emit) {
      await events.emit("hud:clear").catch((err) => console.warn("HUD clear failed:", err));
    }
  };

  const hudTargetFromSource = (source) => {
    const fallback = {
      width: window.screen?.width || window.innerWidth || 1920,
      height: window.screen?.height || window.innerHeight || 1080,
      x: 0,
      y: 0
    };
    const sourceWidth = Number(source?.capture_width ?? source?.width);
    const sourceHeight = Number(source?.capture_height ?? source?.height);
    const sourceLeft = Number(source?.capture_left ?? source?.left);
    const sourceTop = Number(source?.capture_top ?? source?.top);
    if (
      Number.isFinite(sourceWidth) &&
      Number.isFinite(sourceHeight) &&
      sourceWidth > 32 &&
      sourceHeight > 32 &&
      Number.isFinite(sourceLeft) &&
      Number.isFinite(sourceTop)
    ) {
      return {
        width: sourceWidth,
        height: sourceHeight,
        x: sourceLeft,
        y: sourceTop
      };
    }
    return fallback;
  };

  const makeTestOverlay = (target) => {
    const imageWidth = Number(target?.imageWidth || target?.width || window.screen?.width || 1920);
    const imageHeight = Number(target?.imageHeight || target?.height || window.screen?.height || 1080);
    const minEdge = Math.min(imageWidth, imageHeight);
    return {
      duration_ms: 6500,
      coordinate_space: {
        type: "source_image_pixels",
        image_width: imageWidth,
        image_height: imageHeight
      },
      items: [
        {
          type: "circle",
          x: 0.5,
          y: 0.5,
          pixel_x: Math.round(imageWidth * 0.5),
          pixel_y: Math.round(imageHeight * 0.5),
          radius: 0.12,
          radius_px: Math.round(minEdge * 0.12),
          color: "#ff2d2d",
          label: "HUD"
        },
        {
          type: "arrow",
          from: {
            x: 0.2,
            y: 0.78,
            pixel_x: Math.round(imageWidth * 0.2),
            pixel_y: Math.round(imageHeight * 0.78)
          },
          to: {
            x: 0.5,
            y: 0.5,
            pixel_x: Math.round(imageWidth * 0.5),
            pixel_y: Math.round(imageHeight * 0.5)
          },
          color: "#ff2d2d",
          label: "Here"
        }
      ]
    };
  };

  const showHudOverlay = async (overlay, captureSource = null) => {
    if (!overlay?.items?.length) return;
    lastHudError = "";
    const target = hudTargetFromSource(captureSource);
    const space = overlay.coordinate_space || {};
    if (Number.isFinite(Number(space.image_width)) && Number.isFinite(Number(space.image_height))) {
      target.imageWidth = Number(space.image_width);
      target.imageHeight = Number(space.image_height);
    }
    const overlayForHud = {
      ...overlay,
      render_target: target
    };
    const payload = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      overlay: overlayForHud,
      target
    };
    if (invoke) {
      try {
        const args = { overlay: overlayForHud, width: target.width, height: target.height };
        if (Number.isFinite(target.x) && Number.isFinite(target.y)) {
          args.x = target.x;
          args.y = target.y;
        }
        await invoke("show_hud_overlay", args);
        return true;
      } catch (err) {
        lastHudError = `HUD command failed: ${err?.message || err}`;
        console.warn(lastHudError);
      }
    }
    if (events.emit) {
      try {
        await events.emit("hud:show", overlayForHud);
        return true;
      } catch (err) {
        lastHudError = `HUD event failed: ${err?.message || err}`;
        console.warn(lastHudError);
      }
    }
    localStorage.setItem("hud-overlay", JSON.stringify(payload));
    return false;
  };

  hudBtn?.addEventListener("click", async () => {
    await clearHudOverlay();
    appendMessage("HUD cleared.", "bot");
  });

  hudTestBtn?.addEventListener("click", async () => {
    const testSource = pendingCaptureSource || {
      capture_left: 0,
      capture_top: 0,
      capture_width: window.screen?.width || window.innerWidth || 1920,
      capture_height: window.screen?.height || window.innerHeight || 1080,
      bitmap_width: window.screen?.width || window.innerWidth || 1920,
      bitmap_height: window.screen?.height || window.innerHeight || 1080
    };
    const target = hudTargetFromSource(testSource);
    target.imageWidth = Number(testSource.bitmap_width || target.width);
    target.imageHeight = Number(testSource.bitmap_height || target.height);
    const ok = await showHudOverlay(makeTestOverlay(target), testSource);
    appendMessage(ok ? "HUD test sent." : `HUD test failed. ${lastHudError}`, "bot");
  });

  const appendMessage = (text, sender) => {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", sender === "user" ? "user-message" : "bot-message");
    msgDiv.textContent = text;
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return msgDiv;
  };

  const showImagePreview = (base64, source = null, mimeType = "image/jpeg") => {
    pendingImageBase64 = base64;
    pendingImageMimeType = mimeType || "image/jpeg";
    pendingCaptureSource = source;
    previewImg.src = `data:${pendingImageMimeType};base64,${base64}`;
    imagePreviewArea.style.display = "flex";
  };

  const clearImagePreview = () => {
    pendingImageBase64 = null;
    pendingImageMimeType = "image/jpeg";
    pendingCaptureSource = null;
    previewImg.src = "";
    imagePreviewArea.style.display = "none";
  };

  removeImgBtn?.addEventListener("click", clearImagePreview);

  const AUTO_CAPTURE_RE =
    /(你看見|你看到|看見什麼|看到什麼|幫我看|看一下|畫面|螢幕|截圖|圈出|圈起|框出|標記|標出|指給|指引|箭頭|在哪|哪裡|what.*see|what.*screen|describe.*screen|look.*screen|circle|mark|highlight|arrow|where|target|objective|hud)/i;

  const shouldAutoCapture = (text) => AUTO_CAPTURE_RE.test(text || "");

  const captureScreen = async (profile = "turbo") => {
    const res = await fetch(`${API_BASE}/screenshot?mode=screen&redact=0&profile=${profile}`);
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`Screenshot failed: ${res.status}${detail ? ` ${detail}` : ""}`);
    }
    const data = await res.json();
    if (!data.image_base64) throw new Error("Screenshot API returned no image.");
    return data;
  };

  const appendUserMessage = (text, imageBase64, mimeType = "image/jpeg") => {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", "user-message");
    msgDiv.textContent = text || "[screenshot]";
    if (imageBase64) {
      const img = document.createElement("img");
      img.src = `data:${mimeType || "image/jpeg"};base64,${imageBase64}`;
      msgDiv.appendChild(document.createElement("br"));
      msgDiv.appendChild(img);
    }
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  };

  const loadGuideGames = async () => {
    if (!gameSelect) return;
    try {
      const resp = await fetch(`${API_BASE}/guides/games`);
      const data = await resp.json();
      const savedGameId = localStorage.getItem("currentGameId") || "";
      gameSelect.innerHTML = '<option value="">Game</option>';
      for (const game of data.games || []) {
        const option = document.createElement("option");
        option.value = game;
        option.textContent = game;
        gameSelect.appendChild(option);
      }
      gameSelect.value = savedGameId;
    } catch (err) {
      console.warn("Guide list unavailable:", err);
    }
  };

  gameSelect?.addEventListener("change", () => {
    localStorage.setItem("currentGameId", gameSelect.value || "");
  });
  await loadGuideGames();

  const setBusy = (busy) => {
    isSending = busy;
    sendBtn.style.display = busy ? "none" : "block";
    stopBtn.style.display = busy ? "block" : "none";
  };

  const setVoiceRecording = (recording) => {
    isVoiceRecording = recording;
    voiceBtn?.classList.toggle("recording", recording);
    if (voiceBtn) {
      voiceBtn.textContent = recording ? "Rec" : "Mic";
      voiceBtn.title = recording ? "Recording voice input" : "Voice input";
      voiceBtn.disabled = isVoiceBusy && !recording;
    }
  };

  const setVoiceBusy = (busy) => {
    isVoiceBusy = busy;
    if (voiceBtn && !isVoiceRecording) {
      voiceBtn.disabled = busy;
      voiceBtn.textContent = busy ? "..." : "Mic";
    }
  };

  const stopVoiceTracks = () => {
    mediaStream?.getTracks?.().forEach((track) => track.stop());
    mediaStream = null;
  };

  const transcribeVoiceBlob = async (blob, statusMessage = null) => {
    if (blob.size < 800) {
      if (statusMessage) statusMessage.textContent = "No voice was recorded.";
      return;
    }

    setVoiceBusy(true);
    if (statusMessage) statusMessage.textContent = "Transcribing voice...";

    try {
      const formData = new FormData();
      formData.append("file", blob, "voice.webm");
      const response = await fetch(`${API_BASE}/transcribe`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const detail = await response.text().catch(() => "");
        throw new Error(`Transcription failed: ${response.status}${detail ? ` ${detail}` : ""}`);
      }

      const data = await response.json();
      const transcript = (data.text || "").trim();
      if (!transcript) {
        if (statusMessage) statusMessage.textContent = "I could not hear clear speech.";
        return;
      }

      const typedText = messageInput.value.trim();
      const combinedText = typedText ? `${typedText} ${transcript}` : transcript;
      if (statusMessage) statusMessage.textContent = `Heard: ${transcript}`;
      await sendMessage(combinedText);
    } catch (err) {
      if (statusMessage) {
        statusMessage.textContent = `Voice failed: ${err.message || err}`;
      } else {
        appendMessage(`Voice failed: ${err.message || err}`, "bot");
      }
    } finally {
      setVoiceBusy(false);
    }
  };

  const startVoiceRecording = async () => {
    if (isVoiceStarting || isVoiceRecording || isVoiceBusy || isSending) return;
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      appendMessage("Voice input is not available in this WebView.", "bot");
      return;
    }

    isVoiceStarting = true;
    voiceStopRequested = false;
    try {
      voiceChunks = [];
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      const preferredMime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "";
      mediaRecorder = new MediaRecorder(
        mediaStream,
        preferredMime ? { mimeType: preferredMime } : undefined
      );

      mediaRecorder.addEventListener("dataavailable", (event) => {
        if (event.data?.size > 0) voiceChunks.push(event.data);
      });

      mediaRecorder.addEventListener("stop", async () => {
        const mimeType = mediaRecorder?.mimeType || preferredMime || "audio/webm";
        const blob = new Blob(voiceChunks, { type: mimeType });
        mediaRecorder = null;
        stopVoiceTracks();
        setVoiceRecording(false);
        await transcribeVoiceBlob(blob, voiceStatusMessage);
        voiceStatusMessage = null;
      });

      mediaRecorder.addEventListener("error", (event) => {
        stopVoiceTracks();
        setVoiceRecording(false);
        const message = event.error?.message || "Recording error";
        if (voiceStatusMessage) voiceStatusMessage.textContent = `Voice failed: ${message}`;
        voiceStatusMessage = null;
      });

      mediaRecorder.start(250);
      setVoiceRecording(true);
      voiceStatusMessage = appendMessage("Listening...", "bot");
      if (voiceStopRequested) stopVoiceRecording();
    } catch (err) {
      stopVoiceTracks();
      setVoiceRecording(false);
      appendMessage(`Mic permission or recording failed: ${err.message || err}`, "bot");
    } finally {
      isVoiceStarting = false;
    }
  };

  const stopVoiceRecording = () => {
    if (isVoiceStarting) {
      voiceStopRequested = true;
      return;
    }
    if (!isVoiceRecording || !mediaRecorder) return;
    if (voiceStatusMessage) voiceStatusMessage.textContent = "Stopping recording...";
    try {
      mediaRecorder.stop();
    } catch (err) {
      stopVoiceTracks();
      setVoiceRecording(false);
      appendMessage(`Voice stop failed: ${err.message || err}`, "bot");
    }
  };

  const sendToAI = async (text, imageBase64 = null, captureSource = null) => {
    if (isSending) return;
    setBusy(true);
    abortController = new AbortController();

    const botMsgDiv = appendMessage(
      imageBase64 ? "Reading compressed screenshot..." : "Reading response...",
      "bot"
    );
    let receivedText = false;

    try {
      const body = {
        message: text,
        game_id: gameSelect?.value || null,
        use_guides: false,
        use_memory: true
      };
      if (imageBase64) body.image_base64 = imageBase64;

      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: abortController.signal
      });

      if (!response.ok) {
        const detail = await response.text().catch(() => "");
        throw new Error(`Backend returned ${response.status}: ${detail}`);
      }
      const rawResponse = await response.text();
      const lines = rawResponse.split(/\r?\n/);
      let collected = "";
      let showedOverlay = false;

      const handleSseLine = async (line) => {
        if (!line.startsWith("data: ")) return;
        const dataStr = line.slice(6).trim();
        if (!dataStr || dataStr === "[DONE]") return;

        try {
          const dataObj = JSON.parse(dataStr);
          if (dataObj.overlay) {
            const hudShown = await showHudOverlay(dataObj.overlay, captureSource);
            showedOverlay = showedOverlay || hudShown;
            if (!hudShown) {
              collected += `\nHUD 顯示失敗。${lastHudError}`;
              receivedText = true;
            }
          }
          const content = dataObj.content || "";
          if (!content) return;
          collected += content;
          receivedText = true;
        } catch (err) {
          console.warn("Could not parse SSE line:", line, err);
        }
      };

      for (const line of lines) await handleSseLine(line.trimEnd());
      botMsgDiv.textContent = collected.trim() || (
        showedOverlay
          ? "HUD 已標記。"
          : "這次沒有產生可用回覆；請換個問法，或指定要看的畫面位置。"
      );
      chatWindow.scrollTop = chatWindow.scrollHeight;
    } catch (error) {
      if (error.name === "AbortError") {
        botMsgDiv.textContent += "\n\nStopped.";
      } else {
        console.error(error);
        botMsgDiv.textContent = `Connection failed: ${error.message || error}`;
      }
    } finally {
      abortController = null;
      setBusy(false);
    }
  };

  const sendMessage = async (overrideText = null) => {
    const text = (overrideText ?? messageInput.value).trim();
    let imageBase64 = pendingImageBase64;
    let imageMimeType = pendingImageMimeType;
    let captureSource = pendingCaptureSource;
    if (!text && !imageBase64) return;

    if (!imageBase64 && shouldAutoCapture(text)) {
      const status = appendMessage("Auto-capturing screen...", "bot");
      try {
        screenshotBtn.disabled = true;
        screenshotBtn.textContent = "Shot...";
        const data = await captureScreen("turbo");
        imageBase64 = data.image_base64;
        imageMimeType = data.mime_type || "image/jpeg";
        captureSource = data.source || null;
        const sourceWidth = data.source?.capture_width || data.original_width || data.width;
        const sourceHeight = data.source?.capture_height || data.original_height || data.height;
        status.textContent = `Auto-captured ${data.width}x${data.height} from source ${sourceWidth}x${sourceHeight}.`;
      } catch (err) {
        status.textContent = `Auto screenshot failed: ${err.message || err}`;
        screenshotBtn.textContent = "Shot";
        screenshotBtn.disabled = false;
        return;
      } finally {
        screenshotBtn.textContent = "Shot";
        screenshotBtn.disabled = false;
      }
    }

    messageInput.value = "";
    clearImagePreview();
    appendUserMessage(text || "Analyze this screenshot.", imageBase64, imageMimeType);
    const fixedSourceHint = captureSource?.window_title
      ? `\n\nScreenshot source window title: ${captureSource.window_title}`
      : "";
    await sendToAI(
      (text || "Analyze this screenshot and give one useful next step.") + fixedSourceHint,
      imageBase64,
      captureSource
    );
  };

  const takeScreenshot = async () => {
    if (screenshotBtn.disabled || isSending) return;
    screenshotBtn.disabled = true;
    screenshotBtn.textContent = "Shot...";

    try {
      const data = await captureScreen("turbo");
      showImagePreview(data.image_base64, data.source || null, data.mime_type || "image/jpeg");
      const title = data.source?.window_title || "full screen";
      const sourceWidth = data.source?.capture_width || data.original_width || data.width;
      const sourceHeight = data.source?.capture_height || data.original_height || data.height;
      appendMessage(
        `Captured ${data.width}x${data.height} turbo view from ${title} (source ${sourceWidth}x${sourceHeight}). Add your prompt, then press Send.`,
        "bot"
      );
      messageInput?.focus();
    } catch (err) {
      appendMessage(`Screenshot failed: ${err.message || err}`, "bot");
    } finally {
      screenshotBtn.textContent = "Shot";
      screenshotBtn.disabled = false;
    }
  };

  sendBtn?.addEventListener("click", () => sendMessage());
  messageInput?.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !isSending) {
      event.preventDefault();
      sendMessage();
    }
  });

  stopBtn?.addEventListener("click", () => {
    abortController?.abort();
  });

  screenshotBtn?.addEventListener("click", takeScreenshot);

  voiceBtn?.addEventListener("click", () => {
    if (isVoiceRecording) {
      stopVoiceRecording();
    } else {
      startVoiceRecording();
    }
  });

  dragBar?.addEventListener("mousedown", async (event) => {
    if (event.target.closest(".drag-bar-actions")) return;
    await appWindow?.startDragging?.().catch(() => {});
  });

  dragBar?.addEventListener("dblclick", async (event) => {
    if (event.target.closest(".drag-bar-actions")) return;
    await appWindow?.toggleMaximize?.().catch(() => {});
  });

  minBtn?.addEventListener("click", async () => {
    await appWindow?.minimize?.().catch(() => {});
  });

  maxBtn?.addEventListener("click", async () => {
    await appWindow?.toggleMaximize?.().catch(() => {});
  });

  closeBtn?.addEventListener("click", async () => {
    await globalShortcut.unregisterAll?.().catch(() => {});
    await appWindow?.close?.();
  });

  resizeGrip?.addEventListener("mousedown", async (event) => {
    event.preventDefault();
    await appWindow?.startResizeDragging?.("SouthEast").catch(() => {});
  });

  await events.listen?.("capture-hotkey", takeScreenshot).catch(() => {});
  await events.listen?.("voice-hotkey-start", startVoiceRecording).catch(() => {});
  await events.listen?.("voice-hotkey-stop", stopVoiceRecording).catch(() => {});
  await events.listen?.("clear-hud-hotkey", async () => {
    await clearHudOverlay();
    appendMessage("HUD cleared.", "bot");
  }).catch(() => {});

  await globalShortcut.register?.("F9", takeScreenshot).catch((err) => {
    console.warn("F9 registration failed:", err);
  });

  await globalShortcut.register?.("F10", () => {
    clearHudOverlay();
    appendMessage("HUD cleared.", "bot");
  }).catch(() => {});
});
