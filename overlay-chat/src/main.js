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
  const tasksBtn = document.getElementById("tasksBtn");
  const sendBtn = document.getElementById("sendBtn");
  const stopBtn = document.getElementById("stopBtn");
  const messageInput = document.getElementById("messageInput");
  const chatWindow = document.getElementById("chatWindow");
  const dragBar = document.querySelector(".drag-bar");
  const screenshotBtn = document.getElementById("screenshotBtn");
  const taskCaptureBtn = document.getElementById("taskCaptureBtn");
  const perfBtn = document.getElementById("perfBtn");
  const voiceBtn = document.getElementById("voiceBtn");
  const gameSelect = document.getElementById("gameSelect");
  const hudBtn = document.getElementById("hudBtn");
  const hudTestBtn = document.getElementById("hudTestBtn");
  const protectBtn = document.getElementById("protectBtn");
  const resizeGrip = document.getElementById("resizeGrip");
  const opacitySlider = document.getElementById("opacitySlider");
  const opacityValue = document.getElementById("opacityValue");
  const imagePreviewArea = document.getElementById("imagePreviewArea");
  const previewImg = document.getElementById("previewImg");
  const removeImgBtn = document.getElementById("removeImgBtn");

  const iconSvg = {
    arrow: '<path d="M5 12h14"/><path d="m13 6 6 6-6 6"/>',
    blend: '<circle cx="9" cy="9" r="7"/><circle cx="15" cy="15" r="7"/>',
    camera: '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/>',
    crosshair: '<circle cx="12" cy="12" r="10"/><path d="M22 12h-4"/><path d="M6 12H2"/><path d="M12 6V2"/><path d="M12 22v-4"/>',
    flag: '<path d="M4 22V4"/><path d="M4 4h12l-1 4 1 4H4"/>',
    gauge: '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    grip: '<circle cx="9" cy="6" r="1"/><circle cx="15" cy="6" r="1"/><circle cx="9" cy="12" r="1"/><circle cx="15" cy="12" r="1"/><circle cx="9" cy="18" r="1"/><circle cx="15" cy="18" r="1"/>',
    list: '<path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
    loader: '<path d="M21 12a9 9 0 1 1-6.2-8.56"/>',
    mic: '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><path d="M12 19v3"/>',
    minus: '<path d="M5 12h14"/>',
    radio: '<path d="M4.9 19.1C1 15.2 1 8.8 4.9 4.9"/><path d="M7.8 16.2a6 6 0 0 1 0-8.5"/><circle cx="12" cy="12" r="2"/><path d="M16.2 7.8a6 6 0 0 1 0 8.5"/><path d="M19.1 4.9c3.9 3.9 3.9 10.3 0 14.1"/>',
    send: '<path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>',
    shield: '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.5 3.8 17 5 19 5a1 1 0 0 1 1 1Z"/>',
    "shield-off": '<path d="M2 2 22 22"/><path d="M18.7 18.7A13 13 0 0 1 12.34 22a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c1.2 0 2.6-.43 3.9-1.08"/><path d="M11.24 2.28a1.17 1.17 0 0 1 1.52 0C14.5 3.8 17 5 19 5a1 1 0 0 1 1 1v7a8.7 8.7 0 0 1-.56 3.14"/>',
    square: '<rect width="14" height="14" x="5" y="5" rx="2"/>',
    target: '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>'
  };

  const renderIcon = (target, name) => {
    if (!target || !iconSvg[name]) return;
    target.dataset.icon = name;
    target.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${iconSvg[name]}</svg>`;
  };

  const hydrateIcons = () => {
    document.querySelectorAll(".icon[data-icon]").forEach((icon) => renderIcon(icon, icon.dataset.icon));
  };

  const setButtonContent = (button, iconName, label) => {
    if (!button) return;
    const icon = button.querySelector(".icon");
    const labelNode = button.querySelector(".button-label");
    renderIcon(icon, iconName);
    if (labelNode) labelNode.textContent = label;
  };

  hydrateIcons();

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
  let speechRecognition = null;
  let liveSpeechRestartTimer = null;
  let liveSpeechSilenceTimer = null;
  let liveSpeechBlocked = false;
  let liveVoiceBaseText = "";
  let liveVoiceFinalText = "";
  let liveVoiceInterimText = "";
  let liveVoiceSent = false;
  let liveVoiceStopRequested = false;
  let skipVoiceBlobTranscription = false;
  let lastHudError = "";
  localStorage.setItem("protect-mode", "off");
  let captureProtectionEnabled = false;

  localStorage.removeItem("hud-overlay");
  await invoke?.("hide_hud_window").catch((err) => {
    console.warn("Could not hide stale HUD window:", err);
  });
  const applyOpacity = (value) => {
    const numeric = Math.max(10, Math.min(100, Number(value) || 100));
    const scale = numeric / 100;
    document.documentElement.style.setProperty("--ui-opacity", scale.toFixed(2));
    document.documentElement.style.setProperty("--ui-bg-alpha", (scale * 0.96).toFixed(3));
    if (opacitySlider) opacitySlider.value = String(numeric);
    if (opacityValue) opacityValue.textContent = String(numeric);
    localStorage.setItem("ui-opacity", String(numeric));
    events.emit?.("ui-opacity-updated", { opacity: numeric }).catch(() => {});
  };
  applyOpacity(localStorage.getItem("ui-opacity") || "100");
  opacitySlider?.addEventListener("input", (event) => applyOpacity(event.target.value));
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
      setButtonContent(protectBtn, captureProtectionEnabled ? "shield" : "shield-off", captureProtectionEnabled ? "Protect" : "NoProt");
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

  const TASK_INTENT_RE =
    /(任務紀錄|記錄任務|加入任務|新增任務|列為任務|列為目標|任務目標|目標清單|追蹤目標|我拿到|我取得|我獲得|拿到這個|取得這個|獲得這個|這個物品|這個道具|這個材料|用途|用在哪|能用在哪|不知道.*用|task|quest log|objective|goal)/i;

  const VOICE_TASK_INTENT_RE =
    /(任務記錄|任務紀錄|任務記一下|記一下任務|幫我記任務|幫我記錄任務|幫我建立任務|建立任務|加入任務|新增任務|列為任務|列為目標|作為目標|當成目標|追蹤這個|幫我追蹤|目標清單|待辦|我拿到|我取得|我獲得|拿到這個|取得這個|獲得這個|這個物品|這個道具|這個材料|這個能幹嘛|這個能做什麼|用途|用在哪|能用在哪|不知道.*用|task|quest log|objective|goal|todo|log this|track this)/i;

  const shouldAutoCapture = (text) => AUTO_CAPTURE_RE.test(text || "");
  const shouldTaskIntent = (text) => TASK_INTENT_RE.test(text || "") || VOICE_TASK_INTENT_RE.test(text || "");

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

  const TASK_STORAGE_KEY = "igpu-task-log-v1";

  const loadTasks = () => {
    try {
      const parsed = JSON.parse(localStorage.getItem(TASK_STORAGE_KEY) || "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  };

  const saveTasks = (tasks) => {
    const trimmed = tasks.slice(0, 80);
    localStorage.setItem(TASK_STORAGE_KEY, JSON.stringify(trimmed));
    events.emit?.("tasks:updated", { tasks: trimmed }).catch(() => {});
  };

  const openTasksWindow = async () => {
    if (invoke) {
      await invoke("show_tasks_window").catch((err) => {
        appendMessage(`Task window failed: ${err?.message || err}`, "bot");
      });
    }
    await events.emit?.("tasks:updated", { tasks: loadTasks() }).catch(() => {});
  };

  const addTaskFromAnalysis = (analysis, note, captureSource = null) => {
    const now = new Date().toISOString();
    const sourceWidth = captureSource?.capture_width || captureSource?.bitmap_width || null;
    const sourceHeight = captureSource?.capture_height || captureSource?.bitmap_height || null;
    const task = {
      id: `task-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      status: "active",
      title: analysis?.title || note || "調查目前取得的物品",
      category: analysis?.category || "unknown",
      itemName: analysis?.item_name || "",
      objective: analysis?.objective || "",
      why: analysis?.why || "",
      nextSteps: Array.isArray(analysis?.next_steps) ? analysis.next_steps.slice(0, 4) : [],
      tags: Array.isArray(analysis?.tags) ? analysis.tags.slice(0, 6) : [],
      confidence: Number.isFinite(Number(analysis?.confidence)) ? Number(analysis.confidence) : 0.4,
      summary: analysis?.summary || "",
      note: note || "",
      gameId: analysis?.game_id || gameSelect?.value || "global",
      sourceTitle: captureSource?.window_title || "",
      sourceSize: sourceWidth && sourceHeight ? `${sourceWidth}x${sourceHeight}` : "",
      memoryId: analysis?.memory_item?.id || null,
      createdAt: now,
      updatedAt: now
    };
    const tasks = loadTasks();
    tasks.unshift(task);
    saveTasks(tasks);
    return task;
  };

  const formatTaskAddedMessage = (task) => {
    const parts = [`已加入任務：${task.title}`];
    if (task.itemName) parts.push(`物品/指標：${task.itemName}`);
    if (task.objective) parts.push(`目標：${task.objective}`);
    if (task.nextSteps?.length) parts.push(`下一步：${task.nextSteps[0]}`);
    return parts.join("\n");
  };

  const createTaskFromContext = async (note = "", context = {}) => {
    if (isSending) return;
    const originalNote = (note || "").trim();
    let imageBase64 = context.imageBase64 ?? pendingImageBase64;
    let imageMimeType = context.imageMimeType ?? pendingImageMimeType;
    let captureSource = context.captureSource ?? pendingCaptureSource;
    const status = appendMessage("Building task from screen...", "bot");
    setBusy(true);
    if (taskCaptureBtn) {
      taskCaptureBtn.disabled = true;
      setButtonContent(taskCaptureBtn, "loader", "Task...");
    }

    try {
      if (!imageBase64 && context.capture !== false) {
        screenshotBtn.disabled = true;
        setButtonContent(screenshotBtn, "camera", "Shot...");
        const data = await captureScreen("turbo");
        imageBase64 = data.image_base64;
        imageMimeType = data.mime_type || "image/jpeg";
        captureSource = data.source || null;
      }

      appendUserMessage(originalNote || "Create a task from the current screen.", imageBase64, imageMimeType);
      const response = await fetch(`${API_BASE}/tasks/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: originalNote,
          image_base64: imageBase64,
          game_id: gameSelect?.value || null,
          source_title: captureSource?.window_title || ""
        })
      });

      if (!response.ok) {
        const detail = await response.text().catch(() => "");
        throw new Error(`Task analysis failed: ${response.status}${detail ? ` ${detail}` : ""}`);
      }

      const data = await response.json();
      const task = addTaskFromAnalysis(data.task || {}, originalNote, captureSource);
      status.textContent = formatTaskAddedMessage(task);
      clearImagePreview();
      if (messageInput) messageInput.value = "";
      await openTasksWindow();
    } catch (err) {
      status.textContent = `Task logging failed: ${err.message || err}`;
    } finally {
      setBusy(false);
      if (taskCaptureBtn) {
        taskCaptureBtn.disabled = false;
        setButtonContent(taskCaptureBtn, "flag", "Task");
      }
      setButtonContent(screenshotBtn, "camera", "Shot");
      screenshotBtn.disabled = false;
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }
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
      setButtonContent(voiceBtn, recording ? "radio" : "mic", recording ? "Rec" : "Mic");
      voiceBtn.title = recording ? "Recording voice input" : "Voice input";
      voiceBtn.disabled = isVoiceBusy && !recording;
    }
  };

  const setVoiceBusy = (busy) => {
    isVoiceBusy = busy;
    if (voiceBtn && !isVoiceRecording) {
      voiceBtn.disabled = busy;
      setButtonContent(voiceBtn, busy ? "loader" : "mic", busy ? "..." : "Mic");
    }
  };

  const normalizeSpeechText = (text) => (text || "").replace(/\s+/g, " ").trim();

  const liveVoiceSpokenText = (includeInterim = true) => normalizeSpeechText(
    `${liveVoiceFinalText} ${includeInterim ? liveVoiceInterimText : ""}`
  );

  const composeLiveVoiceText = (includeInterim = true) => {
    const spoken = liveVoiceSpokenText(includeInterim);
    return normalizeSpeechText([liveVoiceBaseText, spoken].filter(Boolean).join(" "));
  };

  const updateLiveVoiceInput = () => {
    if (!messageInput || liveVoiceSent) return;
    messageInput.value = composeLiveVoiceText(true);
    messageInput.focus();
    const cursor = messageInput.value.length;
    messageInput.setSelectionRange?.(cursor, cursor);
  };

  const clearLiveVoiceDraft = ({ clearInput = false } = {}) => {
    liveVoiceBaseText = "";
    liveVoiceFinalText = "";
    liveVoiceInterimText = "";
    if (clearInput && messageInput) {
      messageInput.value = "";
    }
  };

  const clearLiveSpeechRestart = () => {
    if (liveSpeechRestartTimer) {
      clearTimeout(liveSpeechRestartTimer);
      liveSpeechRestartTimer = null;
    }
  };

  const clearLiveSpeechSilence = () => {
    if (liveSpeechSilenceTimer) {
      clearTimeout(liveSpeechSilenceTimer);
      liveSpeechSilenceTimer = null;
    }
  };

  const stopRecorderAfterLiveSpeech = () => {
    if (mediaRecorder?.state && mediaRecorder.state !== "inactive") {
      try {
        mediaRecorder.stop();
      } catch (err) {
        console.warn("Recorder stop after live speech failed:", err);
        stopVoiceTracks();
        setVoiceRecording(false);
      }
    } else {
      stopVoiceTracks();
      setVoiceRecording(false);
    }
  };

  const sendLiveVoiceTranscript = ({ includeInterim = false } = {}) => {
    if (liveVoiceSent) return false;
    const spoken = liveVoiceSpokenText(includeInterim);
    if (!spoken) return false;
    const messageText = composeLiveVoiceText(includeInterim);

    clearLiveSpeechSilence();
    liveVoiceSent = true;
    skipVoiceBlobTranscription = true;
    clearLiveVoiceDraft({ clearInput: true });
    if (voiceStatusMessage) voiceStatusMessage.textContent = `Heard: ${spoken}`;
    sendMessage(messageText).catch((err) => {
      appendMessage(`Voice send failed: ${err.message || err}`, "bot");
    });
    return true;
  };

  const scheduleLiveSpeechAutoSend = () => {
    clearLiveSpeechSilence();
    if (!liveVoiceSpokenText(true) || liveVoiceSent) return;

    liveSpeechSilenceTimer = window.setTimeout(() => {
      if (!isVoiceRecording || liveVoiceSent || !liveVoiceSpokenText(true)) return;
      if (sendLiveVoiceTranscript({ includeInterim: true })) {
        stopLiveSpeechRecognition();
        stopRecorderAfterLiveSpeech();
        voiceStatusMessage = null;
      }
    }, 1400);
  };

  const stopLiveSpeechRecognition = (abort = false) => {
    liveVoiceStopRequested = true;
    clearLiveSpeechRestart();
    clearLiveSpeechSilence();
    if (!speechRecognition) return;

    try {
      if (abort) {
        speechRecognition.abort?.();
      } else {
        speechRecognition.stop?.();
      }
    } catch (err) {
      console.warn("Live speech stop failed:", err);
    }
  };

  const startLiveSpeechRecognition = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return false;

    clearLiveSpeechRestart();
    liveSpeechBlocked = false;
    liveVoiceStopRequested = false;

    try {
      const recognition = new SpeechRecognition();
      recognition.lang = localStorage.getItem("speech-lang") || "zh-TW";
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;

      recognition.addEventListener("start", () => {
        if (voiceStatusMessage) {
          voiceStatusMessage.textContent = "Listening live... speech will appear in the input box.";
        }
      });

      recognition.addEventListener("result", (event) => {
        let finalText = "";
        let interimText = "";

        for (let index = event.resultIndex; index < event.results.length; index += 1) {
          const result = event.results[index];
          const transcript = result?.[0]?.transcript || "";
          if (result.isFinal) {
            finalText = `${finalText} ${transcript}`;
          } else {
            interimText = `${interimText} ${transcript}`;
          }
        }

        if (finalText) {
          liveVoiceFinalText = normalizeSpeechText(`${liveVoiceFinalText} ${finalText}`);
        }
        liveVoiceInterimText = normalizeSpeechText(interimText);
        updateLiveVoiceInput();
        scheduleLiveSpeechAutoSend();
      });

      recognition.addEventListener("error", (event) => {
        const error = event.error || "speech-recognition";
        if (["not-allowed", "service-not-allowed", "audio-capture"].includes(error)) {
          liveSpeechBlocked = true;
        }
        if (error !== "aborted" && voiceStatusMessage) {
          voiceStatusMessage.textContent = "Live speech unavailable; recording fallback is still running.";
        }
      });

      recognition.addEventListener("end", () => {
        if (speechRecognition !== recognition) return;
        speechRecognition = null;
        if (liveVoiceSent) return;
        liveVoiceInterimText = "";
        updateLiveVoiceInput();

        if (sendLiveVoiceTranscript({ includeInterim: false })) {
          stopRecorderAfterLiveSpeech();
          voiceStatusMessage = null;
          return;
        }

        if (!liveVoiceStopRequested && !liveSpeechBlocked && isVoiceRecording) {
          liveSpeechRestartTimer = window.setTimeout(() => {
            if (liveVoiceStopRequested || liveSpeechBlocked || !isVoiceRecording || liveVoiceSent) return;
            try {
              recognition.start();
              speechRecognition = recognition;
            } catch (err) {
              console.warn("Live speech restart failed:", err);
            }
          }, 150);
        }
      });

      recognition.start();
      speechRecognition = recognition;
      return true;
    } catch (err) {
      console.warn("Live speech start failed:", err);
      return false;
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
    liveVoiceBaseText = messageInput.value.trim();
    liveVoiceFinalText = "";
    liveVoiceInterimText = "";
    liveVoiceSent = false;
    liveVoiceStopRequested = false;
    skipVoiceBlobTranscription = false;
    liveSpeechBlocked = false;
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
        stopLiveSpeechRecognition(true);
        if (skipVoiceBlobTranscription || liveVoiceSent || liveVoiceSpokenText(true)) {
          if (!liveVoiceSent) {
            sendLiveVoiceTranscript({ includeInterim: true });
          }
          voiceStatusMessage = null;
          return;
        }
        await transcribeVoiceBlob(blob, voiceStatusMessage);
        voiceStatusMessage = null;
      });

      mediaRecorder.addEventListener("error", (event) => {
        stopLiveSpeechRecognition(true);
        stopVoiceTracks();
        setVoiceRecording(false);
        const message = event.error?.message || "Recording error";
        if (voiceStatusMessage) voiceStatusMessage.textContent = `Voice failed: ${message}`;
        voiceStatusMessage = null;
      });

      mediaRecorder.start(250);
      setVoiceRecording(true);
      voiceStatusMessage = appendMessage("Listening...", "bot");
      const liveStarted = startLiveSpeechRecognition();
      if (!liveStarted && voiceStatusMessage) {
        voiceStatusMessage.textContent = "Listening... release F8 or tap Mic again to send.";
      }
      if (voiceStopRequested) stopVoiceRecording();
    } catch (err) {
      stopLiveSpeechRecognition(true);
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
    stopLiveSpeechRecognition();
    const sentLive = sendLiveVoiceTranscript({ includeInterim: true });
    if (voiceStatusMessage && !sentLive) voiceStatusMessage.textContent = "Stopping recording...";
    try {
      mediaRecorder.stop();
    } catch (err) {
      stopLiveSpeechRecognition(true);
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

    if (text && shouldTaskIntent(text)) {
      await createTaskFromContext(text, {
        imageBase64,
        imageMimeType,
        captureSource,
        capture: true
      });
      return;
    }

    if (!imageBase64 && shouldAutoCapture(text)) {
      const status = appendMessage("Auto-capturing screen...", "bot");
      try {
        screenshotBtn.disabled = true;
        setButtonContent(screenshotBtn, "camera", "Shot...");
        const data = await captureScreen("turbo");
        imageBase64 = data.image_base64;
        imageMimeType = data.mime_type || "image/jpeg";
        captureSource = data.source || null;
        const sourceWidth = data.source?.capture_width || data.original_width || data.width;
        const sourceHeight = data.source?.capture_height || data.original_height || data.height;
        status.textContent = `Auto-captured ${data.width}x${data.height} from source ${sourceWidth}x${sourceHeight}.`;
      } catch (err) {
        status.textContent = `Auto screenshot failed: ${err.message || err}`;
        setButtonContent(screenshotBtn, "camera", "Shot");
        screenshotBtn.disabled = false;
        return;
      } finally {
        setButtonContent(screenshotBtn, "camera", "Shot");
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
    setButtonContent(screenshotBtn, "camera", "Shot...");

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
      setButtonContent(screenshotBtn, "camera", "Shot");
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
  taskCaptureBtn?.addEventListener("click", () => {
    createTaskFromContext(messageInput?.value || "", { capture: true });
  });
  tasksBtn?.addEventListener("click", openTasksWindow);

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
  await events.listen?.("task-hotkey", () => {
    createTaskFromContext(messageInput?.value || "", { capture: true });
  }).catch(() => {});
  await events.listen?.("voice-hotkey-start", startVoiceRecording).catch(() => {});
  await events.listen?.("voice-hotkey-stop", stopVoiceRecording).catch(() => {});
  await events.listen?.("clear-hud-hotkey", async () => {
    await clearHudOverlay();
    appendMessage("HUD cleared.", "bot");
  }).catch(() => {});
  await events.listen?.("tasks:ask", async (event) => {
    const task = event.payload || {};
    const title = task.title || "目前任務";
    messageInput.value = `根據任務「${title}」，請告訴我下一步怎麼做`;
    await appWindow?.show?.().catch(() => {});
    await appWindow?.setFocus?.().catch(() => {});
    messageInput?.focus();
  }).catch(() => {});

  await globalShortcut.register?.("F9", takeScreenshot).catch((err) => {
    console.warn("F9 registration failed:", err);
  });

  await globalShortcut.register?.("F10", () => {
    clearHudOverlay();
    appendMessage("HUD cleared.", "bot");
  }).catch(() => {});
});
