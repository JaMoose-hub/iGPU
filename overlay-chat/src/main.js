import lottie from "lottie-web";
import cosmosAnimation from "./assets/cosmos-lottie.json";
import { initVirtualCursor } from "./virtualCursor.js";

const runWhenDomReady = (callback) => {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", callback, { once: true });
  } else {
    callback();
  }
};

runWhenDomReady(async () => {
  const tauri = window.__TAURI__ || {};
  const appWindow = tauri.window?.getCurrentWindow?.();
  const globalShortcut = tauri.globalShortcut || {};
  const events = tauri.event || {};
  const invoke = tauri.core?.invoke;
  const cosmosLottie = document.getElementById("cosmosLottie");
  if (cosmosLottie) {
    try {
      const animation = lottie.loadAnimation({
        container: cosmosLottie,
        renderer: "svg",
        loop: true,
        autoplay: true,
        animationData: cosmosAnimation,
        rendererSettings: {
          preserveAspectRatio: "xMidYMid meet"
        }
      });
      window.addEventListener("beforeunload", () => animation.destroy(), { once: true });
    } catch (err) {
      console.warn("Could not start Cosmos lottie icon:", err);
    }
  }

  const API_BASE = "http://127.0.0.1:8000";

  const closeBtn = document.getElementById("closeBtn");
  const minBtn = document.getElementById("minBtn");
  const maxBtn = document.getElementById("maxBtn");
  const toolsBtn = document.getElementById("toolsBtn");
  const standbyBtn = document.getElementById("standbyBtn");
  const searchBtn = document.getElementById("searchBtn");
  const tasksBtn = document.getElementById("tasksBtn");
  const gamepathBtn = document.getElementById("gamepathBtn");
  const sendBtn = document.getElementById("sendBtn");
  const stopBtn = document.getElementById("stopBtn");
  const messageInput = document.getElementById("messageInput");
  const chatWindow = document.getElementById("chatWindow");
  const dragBar = document.querySelector(".drag-bar");
  const screenshotBtn = document.getElementById("screenshotBtn");
  const captureDisplaySelect = document.getElementById("captureDisplaySelect");
  const taskCaptureBtn = document.getElementById("taskCaptureBtn");
  const perfBtn = document.getElementById("perfBtn");
  const voiceBtn = document.getElementById("voiceBtn");
  const gameSelect = document.getElementById("gameSelect");
  const gameAutoBtn = document.getElementById("gameAutoBtn");
  const liveStateBtn = document.getElementById("liveStateBtn");
  const liveStateAnalyzeBtn = document.getElementById("liveStateAnalyzeBtn");
  const hudBtn = document.getElementById("hudBtn");
  const hudTestBtn = document.getElementById("hudTestBtn");
  const protectBtn = document.getElementById("protectBtn");
  const virtualCursorBtn = document.getElementById("virtualCursorBtn");
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
    search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    send: '<path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>',
    shield: '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.5 3.8 17 5 19 5a1 1 0 0 1 1 1Z"/>',
    "shield-off": '<path d="M2 2 22 22"/><path d="M18.7 18.7A13 13 0 0 1 12.34 22a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c1.2 0 2.6-.43 3.9-1.08"/><path d="M11.24 2.28a1.17 1.17 0 0 1 1.52 0C14.5 3.8 17 5 19 5a1 1 0 0 1 1 1v7a8.7 8.7 0 0 1-.56 3.14"/>',
    square: '<rect width="14" height="14" x="5" y="5" rx="2"/>',
    cursor: '<path d="m4 4 7.07 16.97 2.51-7.39 7.39-2.51Z"/><path d="m13.58 13.58 5.84 5.84"/>',
    database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
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
  document.querySelectorAll(".drag-bar-actions button").forEach((button) => {
    button.addEventListener("mousedown", (event) => event.stopPropagation());
    button.addEventListener("dblclick", (event) => event.stopPropagation());
  });

  let pendingImageBase64 = null;
  let pendingImageMimeType = "image/jpeg";
  let pendingCaptureSource = null;
  let abortController = null;
  let isSending = false;
  let isSendInFlight = false;
  let isScreenshotInFlight = false;
  let isVoiceRecording = false;
  let isVoiceModeEnabled = false;
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
  let gameDetectMode = localStorage.getItem("game-detect-mode") === "manual" ? "manual" : "auto";
  let lastDetectedGameKey = "";
  let skipVoiceBlobTranscription = false;
  let lastHudError = "";
  const voiceSendQueue = [];
  let isVoiceQueueRunning = false;
  let lastVoiceHotkeyToggleAt = 0;
  let lastVoiceToggleAt = 0;
  let lastSentVoiceText = "";
  let lastSentVoiceAt = 0;
  let captureProtectionEnabled = false;
  let liveStateEnabled = localStorage.getItem("live-state") === "on";
  let liveStatePollTimer = null;
  let latestLiveStateStatus = null;
  let selectedGameId = localStorage.getItem("currentGameId") || "";
  let selectedGameName = "";
  let gameCatalog = [{ id: "", name: "Game" }];
  let toolPanelStateFrame = 0;
  let virtualCursorEnabled = false;
  let standbyWindowMode = "collapsed";
  let standbyDetailedConversation = false;
  localStorage.setItem("protect-mode", "off");

  const normalizeStandbyMode = (value) => {
    const normalized = String(value || "").toLowerCase();
    if (normalized === "collapsed" || normalized === "typein" || normalized === "thinking" || normalized === "response" || normalized === "detail") {
      return normalized;
    }
    return null;
  };

  const buildToolPanelState = () => ({
    games: gameCatalog,
    selectedGameId,
    selectedGameName: selectedGameName || selectedGameId,
    gameDetectMode,
    autoMode: gameDetectMode !== "manual",
    liveStateEnabled,
    liveStateStatus: latestLiveStateStatus || {},
    captureProtectionEnabled,
    virtualCursorEnabled,
    perfEnabled: document.body.classList.contains("perf-mode"),
    opacity: Number(localStorage.getItem("ui-opacity") || "100")
  });

  const syncToolPanelState = (immediate = false) => {
    const emitState = () => {
      toolPanelStateFrame = 0;
      events.emit?.("tool-panel-state", buildToolPanelState()).catch(() => {});
    };
    if (immediate) {
      emitState();
      return;
    }
    if (toolPanelStateFrame) return;
    toolPanelStateFrame = window.requestAnimationFrame(emitState);
  };

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
    syncToolPanelState();
  };
  applyOpacity(localStorage.getItem("ui-opacity") || "100");
  opacitySlider?.addEventListener("input", (event) => applyOpacity(event.target.value));
  await appWindow?.setContentProtected?.(false).catch((err) => {
    console.warn("Could not clear main-window content protection:", err);
  });
  await invoke?.("set_main_capture_exclusion", { excluded: false }).catch((err) => {
    console.warn("Could not clear window capture exclusion:", err);
  });
  await invoke?.("set_app_capture_exclusion", { excluded: false }).catch((err) => {
    console.warn("Could not clear app capture exclusion:", err);
  });

  if (localStorage.getItem("perf-mode") === "true") {
    document.body.classList.add("perf-mode");
  }

  const updateProtectButton = () => {
    protectBtn?.classList.toggle("active", captureProtectionEnabled);
    if (protectBtn) {
      setButtonContent(protectBtn, captureProtectionEnabled ? "shield" : "shield-off", captureProtectionEnabled ? "Protect" : "NoProt");
      protectBtn.title = captureProtectionEnabled
        ? "Content protection on; all overlay windows are excluded from capture"
        : "Screenshot protection off";
    }
    localStorage.setItem("protect-mode", captureProtectionEnabled ? "software-redact" : "off");
    syncToolPanelState();
  };

  updateProtectButton();

  const virtualCursor = initVirtualCursor({
    events,
    invoke,
    windowLabel: "main",
    controller: true,
    onStateChange: (enabled) => {
      virtualCursorEnabled = Boolean(enabled);
      virtualCursorBtn?.classList.toggle("active", enabled);
      if (virtualCursorBtn) {
        setButtonContent(virtualCursorBtn, "cursor", enabled ? "Cursor" : "Cursor");
        virtualCursorBtn.title = enabled
          ? "Companion virtual cursor on (F11). WASD/arrows move, Tab targets, Enter clicks, Esc exits."
          : "Companion virtual cursor (F11)";
      }
      syncToolPanelState();
    }
  });
  virtualCursorBtn?.addEventListener("click", () => virtualCursor.toggle());

  const setVirtualCursorActiveWindow = async (label) => {
    await virtualCursor.setActiveWindow(label);
    localStorage.setItem("igpu-virtual-cursor-active-window", label);
    await events.emit?.("virtual-cursor-active-window", { window: label, source: "main" }).catch(() => {});
  };

  const exitVirtualCursorTextEntry = async (source = "main") => {
    await events.emit?.("virtual-cursor-exit-text-entry", { source }).catch(() => {});
  };

  const toggleCaptureProtection = async () => {
    captureProtectionEnabled = !captureProtectionEnabled;
    updateProtectButton();
    await appWindow?.setContentProtected?.(false).catch((err) => {
      console.warn("Could not clear main-window content protection:", err);
    });
    await invoke?.("set_main_capture_exclusion", { excluded: captureProtectionEnabled }).catch((err) => {
      appendMessage(`Protect command failed: ${err?.message || err}`, "bot");
    });
    if (!captureProtectionEnabled && liveStateEnabled) {
      await setAppCaptureExclusion(true);
    }
    appendMessage(
      captureProtectionEnabled
        ? "Content protection on. Main, Task, Game Search, and HUD are controlled by this lock; app screenshots still ignore them cleanly."
        : "Screenshot protection off.",
      "bot"
    );
  };

  protectBtn?.addEventListener("click", toggleCaptureProtection);

  perfBtn?.addEventListener("click", () => {
    const enabled = document.body.classList.toggle("perf-mode");
    localStorage.setItem("perf-mode", enabled ? "true" : "false");
    syncToolPanelState();
  });

  const setAppCaptureExclusion = async (excluded, { quiet = true } = {}) => {
    if (!invoke) return false;
    try {
      await invoke("set_app_capture_exclusion", { excluded });
      if (excluded) await new Promise((resolve) => window.setTimeout(resolve, 70));
      return true;
    } catch (err) {
      if (!quiet) appendMessage(`Capture exclusion failed: ${err?.message || err}`, "bot");
      console.warn("App capture exclusion unavailable:", err);
      return false;
    }
  };

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
      fullscreen: true
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

  const handleHudClear = async () => {
    await clearHudOverlay();
    appendMessage("HUD cleared.", "bot");
  };

  const handleHudTest = async () => {
    const target = hudTargetFromSource(null);
    target.imageWidth = Number(target.width);
    target.imageHeight = Number(target.height);
    const ok = await showHudOverlay(makeTestOverlay(target), null);
    appendMessage(ok ? "HUD test sent." : `HUD test failed. ${lastHudError}`, "bot");
  };

  hudBtn?.addEventListener("click", handleHudClear);
  hudTestBtn?.addEventListener("click", handleHudTest);

  let chatScrollFrame = 0;
  const scrollChatToBottom = () => {
    if (!chatWindow) return;
    chatWindow.scrollTop = chatWindow.scrollHeight;
    if (chatScrollFrame) window.cancelAnimationFrame(chatScrollFrame);
    chatScrollFrame = window.requestAnimationFrame(() => {
      chatScrollFrame = 0;
      chatWindow.scrollTop = chatWindow.scrollHeight;
    });
  };

  if (chatWindow && typeof MutationObserver !== "undefined") {
    const observer = new MutationObserver(scrollChatToBottom);
    observer.observe(chatWindow, { childList: true, subtree: true, characterData: true });
  }

  const clearNode = (node) => {
    while (node?.firstChild) node.removeChild(node.firstChild);
  };

  const cleanResponseText = (text) => String(text || "")
    .replace(/\r\n/g, "\n")
    .replace(/\n[ \t]+/g, "\n")
    .replace(/[ \t]+$/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  const splitInlineHintLines = (text) => {
    const source = String(text || "");
    return source.replace(
      /\s+(?=(?:Hint\s*\d+|提示\s*\d+|下一步|建議|注意|理由|答案|無暴雷提示|風險|操作)[:：])/gi,
      "\n"
    );
  };

  const parseMarkerLine = (line) => {
    const trimmed = String(line || "").trim();
    const hintMatch = trimmed.match(/^(Hint\s*\d+|提示\s*\d+|下一步|建議|注意|理由|答案|無暴雷提示|風險|操作)[:：]\s*(.*)$/i);
    if (hintMatch) {
      return { type: "hint", marker: hintMatch[1], text: hintMatch[2] || "" };
    }
    const numberedMatch = trimmed.match(/^(\d+[.)])\s+(.*)$/);
    if (numberedMatch) {
      return { type: "line", marker: numberedMatch[1], text: numberedMatch[2] || "" };
    }
    const bulletMatch = trimmed.match(/^([-*])\s+(.*)$/);
    if (bulletMatch) {
      return { type: "line", marker: bulletMatch[1], text: bulletMatch[2] || "" };
    }
    return { type: "paragraph", text: trimmed };
  };

  const appendFormattedText = (node, text) => {
    node.appendChild(document.createTextNode(text || ""));
  };

  const renderFormattedMessage = (target, text) => {
    if (!target) return;
    const normalized = cleanResponseText(splitInlineHintLines(text));
    clearNode(target);
    target.classList.add("formatted-response");
    if (!normalized) return;

    const paragraphs = normalized.split(/\n{2,}/);
    paragraphs.forEach((paragraph) => {
      const lines = paragraph.split("\n").map((line) => line.trim()).filter(Boolean);
      if (!lines.length) return;

      lines.forEach((line) => {
        const parsed = parseMarkerLine(line);
        const row = document.createElement("div");
        row.className = parsed.type === "hint" ? "response-line response-hint" : "response-line";

        if (parsed.type === "paragraph") {
          row.className = "response-paragraph";
          appendFormattedText(row, parsed.text);
          target.appendChild(row);
          return;
        }

        const marker = document.createElement("span");
        marker.className = parsed.type === "hint" ? "response-label" : "response-marker";
        marker.textContent = parsed.type === "hint" ? `${parsed.marker}:` : parsed.marker;

        const body = document.createElement("span");
        body.className = "response-body";
        appendFormattedText(body, parsed.text);
        row.append(marker, body);
        target.appendChild(row);
      });
    });
  };

  const appendMessage = (text, sender) => {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", sender === "user" ? "user-message" : "bot-message");
    if (sender === "bot" && text) {
      renderFormattedMessage(msgDiv, text);
    } else {
      msgDiv.textContent = text;
    }
    chatWindow.appendChild(msgDiv);
    scrollChatToBottom();
    return msgDiv;
  };

  const updateBotMessage = (msgDiv, text) => {
    if (!msgDiv) return;
    msgDiv.textContent = "";
    renderFormattedMessage(msgDiv, text || "");
    scrollChatToBottom();
  };

  const createBotResponseMessage = (statusText) => {
    const msgDiv = appendMessage("", "bot");
    msgDiv.classList.add("lookup-message");
    const statusDiv = document.createElement("div");
    statusDiv.className = "lookup-status";
    statusDiv.textContent = statusText || "Checking GamePath...";
    const routeLogDiv = document.createElement("div");
    routeLogDiv.className = "lookup-route-log";
    routeLogDiv.hidden = true;
    const contentDiv = document.createElement("div");
    contentDiv.className = "lookup-content";
    msgDiv.append(statusDiv, routeLogDiv, contentDiv);
    return { msgDiv, statusDiv, routeLogDiv, routeTrace: [], contentDiv };
  };

  const formatLookupStatus = (status) => {
    const stage = String(status?.stage || "");
    const accuracyText = lookupAccuracyParts(status).join(" / ");
    const timingText = lookupTimingParts(status).join(" / ");
    const scoreText = [
      accuracyText,
      timingText ? `耗時 ${timingText}` : ""
    ].filter(Boolean).join(" | ");
    const suffix = scoreText ? ` | ${scoreText}` : "";
    const isVisionRoute = status?.local_router?.route_source === "vision"
      || String(status?.local_router?.intent_route || "").startsWith("screenshot_");
    const routePrefix = status?.local_router?.used
      ? [isVisionRoute ? "Vision" : "地端 Qwen"]
      : status?.local_router?.intent_route
        ? [isVisionRoute ? "Vision補救" : "後端補救"]
        : [];
    const evalPrefix = status?.local_router_retrieval?.used ? ["地端 Qwen 評估"] : routePrefix;
    const path = (segments) => `搜尋路徑：${segments.filter(Boolean).join(" → ")}${suffix}`;
    if (stage === "gamepath_hit") return path([...evalPrefix, "GamePath", "本地回答"]);
    if (stage === "gamepath_summarizing") return path([...evalPrefix, "GamePath", "Hermes 整理"]);
    if (stage === "gamepath_miss") return path([...routePrefix, "GamePath 未命中", "Hermes/Tavily"]);
    if (stage === "gamepath_skipped") return path(["略過 GamePath", "一般聊天"]);
    if (stage === "gamepath_disputed") return "驗證：上一個 GamePath 提示已降權";
    if (stage === "gamepath_feedback_missing") return "驗證：找不到上一筆 GamePath 紀錄";
    if (stage === "gamepath_context") return path(["GamePath", "Hermes 整理"]);
    if (stage === "guide_context") return path(["本地攻略", "Hermes 整理"]);
    if (stage === "memory_context") return path(["玩家記憶", "Hermes 整理"]);
    if (stage === "agent_may_search_web") return path(["本地未命中", "Hermes/Tavily"]);
    if (stage === "agent_web_search") return path(["本地未命中", "Hermes/Tavily"]);
    if (stage === "agent_no_tools") return path(["GamePath 未命中", "Hermes 回答"]);
    if (stage === "gamepath_stored") return "保存：已寫入 GamePath";
    if (stage === "gamepath_not_stored") return "保存：未寫入 GamePath";
    return status?.message || "Checking guide source...";
  };

  const compactLookupValue = (value, maxLen = 90) => {
    if (value == null) return "";
    const text = Array.isArray(value)
      ? value.filter(Boolean).join(",")
      : typeof value === "object"
        ? JSON.stringify(value)
        : String(value);
    const compact = text.replace(/\s+/g, " ").trim();
    if (!compact) return "";
    return compact.length > maxLen ? `${compact.slice(0, Math.max(0, maxLen - 1))}…` : compact;
  };

  const formatLookupPercent = (value) => {
    const numberValue = Number(value);
    if (!Number.isFinite(numberValue)) return "";
    const clamped = Math.max(0, Math.min(1, numberValue));
    return `${Math.round(clamped * 100)}%`;
  };

  const formatLookupDuration = (value) => {
    const ms = Number(value);
    if (!Number.isFinite(ms) || ms < 0) return "";
    if (ms < 1000) return `${Math.round(ms)}ms`;
    if (ms < 10000) return `${(ms / 1000).toFixed(1)}s`;
    return `${Math.round(ms / 1000)}s`;
  };

  const intentRouteLabel = (route) => {
    switch (String(route || "")) {
      case "gamepath_query":
        return "攻略查詢";
      case "hermes_web":
        return "網路查詢";
      case "general_chat":
        return "一般聊天";
      case "ui_command":
        return "介面指令";
      case "task_memory":
        return "任務記錄";
      case "screenshot_gamepath_query":
        return "截圖攻略";
      case "screenshot_visual":
        return "截圖看圖";
      case "screenshot_hud":
        return "截圖/HUD";
      case "clarify":
        return "需要釐清";
      case "skip":
        return "略過";
      default:
        return compactLookupValue(route, 28);
    }
  };

  const retrievalLabel = (value) => {
    switch (String(value || "")) {
      case "direct":
        return "可直接用";
      case "summarize":
        return "需整理";
      case "miss":
        return "未命中";
      case "skipped":
        return "未查詢";
      default:
        return compactLookupValue(value, 28);
    }
  };

  const lookupAccuracyParts = (status) => {
    const parts = [];
    const retrievalRouter = status?.local_router_retrieval || null;
    const topCandidate = Array.isArray(status?.candidates) ? status.candidates[0] : null;
    const scoreText = formatLookupPercent(status?.retrieval_score);
    const coverageText = formatLookupPercent(topCandidate?.match_coverage);
    const routerScoreText = formatLookupPercent(retrievalRouter?.score);
    const retrievalConf = retrievalLabel(retrievalRouter?.confidence || status?.retrieval_confidence);

    if (scoreText) parts.push(`準確度 ${scoreText}`);
    else if (routerScoreText) parts.push(`準確度 ${routerScoreText}`);
    if (coverageText) parts.push(`覆蓋 ${coverageText}`);
    if (retrievalConf) parts.push(`評估 ${retrievalConf}`);
    return parts;
  };

  const lookupTimingParts = (status) => {
    const parts = [];
    const stage = String(status?.stage || "");
    const router = status?.local_router || null;
    const retrievalRouter = status?.local_router_retrieval || null;
    const intentMs = formatLookupDuration(router?.latency_ms ?? status?.intent_latency_ms);
    const sqliteMs = formatLookupDuration(status?.search_elapsed_ms);
    const retrievalMs = formatLookupDuration(retrievalRouter?.latency_ms);
    const hintMs = formatLookupDuration(status?.gamepath_hint_ms);

    const isVisionRoute = router?.route_source === "vision"
      || String(router?.intent_route || "").startsWith("screenshot_");
    if (intentMs) parts.push(`${isVisionRoute ? "Vision判斷" : "Qwen意圖"} ${intentMs}`);
    if (sqliteMs) parts.push(`SQLite ${sqliteMs}`);
    if (retrievalMs) parts.push(`Qwen評估 ${retrievalMs}`);
    if (hintMs) parts.push(`Qwen提示 ${hintMs}`);

    if (stage === "gamepath_miss") {
      parts.push("Hermes/Tavily 待開始");
    } else if (stage === "agent_may_search_web" || stage === "agent_web_search") {
      parts.push("Hermes/Tavily 進行中");
    } else if (stage === "agent_no_tools") {
      parts.push("Hermes 回答中");
    }

    return parts;
  };

  const lookupRouteSegments = (status) => {
    const stage = String(status?.stage || "");
    const router = status?.local_router || null;
    const retrievalRouter = status?.local_router_retrieval || null;
    const segments = [];

    if (router?.used || router?.intent_route) {
      const route = intentRouteLabel(router.intent_route || (router.search_gamepath ? "gamepath_query" : "general_chat"));
      const isVisionRoute = router?.route_source === "vision"
        || String(router?.intent_route || "").startsWith("screenshot_");
      const label = router?.used
        ? (isVisionRoute ? "Vision判斷" : "Qwen判斷")
        : (isVisionRoute ? "Vision補救" : "後端補救");
      segments.push(route ? `${label}:${route}` : label);
    }
    const pushRetrievalSegment = () => {
      if (!retrievalRouter?.used) return;
      const retrievalRoute = retrievalLabel(retrievalRouter.confidence || retrievalRouter.route || retrievalRouter.reason);
      segments.push(retrievalRoute ? `Qwen評估:${retrievalRoute}` : "Qwen評估");
    };

    if (stage === "gamepath_hit") {
      segments.push("GamePath本地命中");
      pushRetrievalSegment();
      segments.push("本地回答");
    } else if (stage === "gamepath_summarizing" || stage === "gamepath_context") {
      segments.push("GamePath本地命中");
      pushRetrievalSegment();
      segments.push("模型整理");
    } else if (stage === "gamepath_miss") {
      segments.push("GamePath未命中");
      pushRetrievalSegment();
      segments.push("Hermes/Tavily候選");
    }
    else if (stage === "gamepath_skipped") segments.push("略過GamePath", "一般聊天");
    else if (stage === "guide_context") segments.push("本地攻略快取", "模型整理");
    else if (stage === "memory_context") segments.push("玩家記憶", "模型整理");
    else if (stage === "agent_may_search_web" || stage === "agent_web_search") segments.push("本地未命中", "Hermes/Tavily");
    else if (stage === "agent_no_tools") segments.push("本地未命中", "Hermes回答");
    else if (stage === "gamepath_stored") segments.push("GamePath寫入", "SQLite+Markdown");
    else if (stage === "gamepath_not_stored") segments.push("GamePath未寫入");
    else if (stage === "gamepath_disputed") segments.push("玩家回報", "GamePath降權");
    else if (stage === "gamepath_feedback_missing") segments.push("玩家回報", "找不到紀錄");
    else if (stage === "error") segments.push("錯誤");
    else if (status?.source) segments.push(compactLookupValue(status.source, 40));

    return segments.filter(Boolean);
  };

  const formatLookupTrace = (status) => {
    const segments = lookupRouteSegments(status);
    const parts = [];
    const accuracyParts = lookupAccuracyParts(status);
    const gamepathHits = status?.gamepath_hits ?? status?.hits;
    const topCandidate = Array.isArray(status?.candidates) ? status.candidates[0] : null;
    const topId = status?.top_id ?? topCandidate?.id;
    const topTitle = status?.top_title || topCandidate?.title || topCandidate?.question;

    if (segments.length) parts.push(`路徑：${segments.join(" → ")}`);
    if (accuracyParts.length) parts.push(accuracyParts.join(" / "));
    const timingParts = lookupTimingParts(status);
    if (timingParts.length) parts.push(`耗時：${timingParts.join(" / ")}`);
    if (gamepathHits != null) parts.push(`本地候選 ${gamepathHits}`);
    if (topTitle) parts.push(`最佳候選 #${topId ?? "?"} ${compactLookupValue(topTitle, 46)}`);
    if (status?.entry_id != null) parts.push(`寫入 #${status.entry_id}`);
    if (status?.reason) parts.push(`原因：${compactLookupValue(status.reason, 46)}`);

    return parts.join(" | ");
  };

  const recordLookupStatus = (status, statusDiv, routeLogDiv, routeTrace) => {
    if (!status || !statusDiv) return;
    const summary = formatLookupStatus(status);
    const detail = formatLookupTrace(status);
    const stage = status?.stage || "";
    const previous = routeTrace[routeTrace.length - 1];

    statusDiv.textContent = summary;
    statusDiv.dataset.stage = stage;

    if (!previous || previous.detail !== detail) {
      routeTrace.push({
        stage,
        summary,
        detail,
        time: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit"
        })
      });
    }

    if (!routeLogDiv) return;
    routeLogDiv.replaceChildren();
    routeTrace.forEach((item, index) => {
      const row = document.createElement("div");
      row.className = "lookup-route-line";
      row.dataset.stage = item.stage || "";
      row.textContent = `${index + 1}. ${item.time} ${item.detail}`;
      routeLogDiv.appendChild(row);
    });
    routeLogDiv.hidden = routeTrace.length === 0;
    statusDiv.title = routeTrace.map((item, index) => `${index + 1}. ${item.time} ${item.detail}`).join("\n");
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
  const isScreenshotIntentRoute = (route) => String(route || "").trim().startsWith("screenshot_");

  const routeUserIntent = async (text, signal = null) => {
    const trimmed = String(text || "").trim();
    if (!trimmed) return null;
    try {
      const response = await fetch(`${API_BASE}/intent/route`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: trimmed,
          game_id: selectedGameId || null
        }),
        signal
      });
      if (!response.ok) {
        throw new Error(`Intent route returned ${response.status}`);
      }
      const data = await response.json();
      const decision = data?.decision || {};
      return {
        route: String(data?.route || decision.intent_route || "").trim() || (
          decision.prefer_hermes_agent
            ? "hermes_web"
            : decision.search_gamepath
              ? "gamepath_query"
              : "general_chat"
        ),
        decision,
        elapsedMs: data?.elapsed_ms
      };
    } catch (err) {
      if (err?.name === "AbortError") throw err;
      console.warn("Qwen intent route failed; continuing as chat:", err);
      return {
        route: "general_chat",
        decision: {
          used: false,
          intent_route: "general_chat",
          reason: `route_failed:${err?.message || err}`
        },
        elapsedMs: null
      };
    }
  };

  const formatIntentRouteStatus = (intent, fallbackRoute = "") => {
    const route = String(intent?.route || fallbackRoute || "unknown").trim();
    const decision = intent?.decision || {};
    const reason = decision.raw_reason || decision.reason || "";
    const elapsed = Number(intent?.elapsedMs ?? decision.latency_ms);
    const parts = [`Qwen route: ${route}`];
    if (reason) parts.push(`${reason}`);
    if (Number.isFinite(elapsed) && elapsed > 0) parts.push(`${Math.round(elapsed)}ms`);
    return parts.join(" | ");
  };

  const selectedCaptureMonitor = () => {
    const value = captureDisplaySelect?.value || localStorage.getItem("capture-monitor") || "auto";
    return value === "auto" ? null : value;
  };

  const loadCaptureMonitors = async () => {
    if (!captureDisplaySelect) return;
    const saved = localStorage.getItem("capture-monitor") || "auto";
    captureDisplaySelect.innerHTML = '<option value="auto">Auto</option>';
    try {
      const response = await fetch(`${API_BASE}/monitors`);
      if (!response.ok) throw new Error(`Monitor API returned ${response.status}`);
      const data = await response.json();
      const monitors = Array.isArray(data.monitors) ? data.monitors : [];
      for (const monitor of monitors.filter((item) => !item.aggregate)) {
        const option = document.createElement("option");
        option.value = String(monitor.index);
        option.textContent = `S${monitor.index}`;
        option.title = `${monitor.label} @ ${monitor.left},${monitor.top}`;
        captureDisplaySelect.appendChild(option);
      }
      if ([...captureDisplaySelect.options].some((option) => option.value === saved)) {
        captureDisplaySelect.value = saved;
      }
    } catch (err) {
      console.warn("Monitor list unavailable:", err);
    }
  };

  const captureScreen = async (profile = "turbo") => {
    const monitor = selectedCaptureMonitor();
    const exclusionApplied = await setAppCaptureExclusion(true);
    const restoreAfterCapture = exclusionApplied && !captureProtectionEnabled && !liveStateEnabled;
    const params = new URLSearchParams({
      mode: monitor ? "screen" : "foreground",
      // App-initiated captures use Tauri capture exclusion first. The backend
      // keeps this flag for metadata/fallback policy but should not move windows.
      redact: "1",
      profile
    });
    try {
      if (monitor) params.set("monitor", monitor);
      const res = await fetch(`${API_BASE}/screenshot?${params.toString()}`);
      if (!res.ok) {
        const detail = await res.text().catch(() => "");
        throw new Error(`Screenshot failed: ${res.status}${detail ? ` ${detail}` : ""}`);
      }
      const data = await res.json();
      if (!data.image_base64) throw new Error("Screenshot API returned no image.");
      return data;
    } finally {
      if (restoreAfterCapture) await setAppCaptureExclusion(false);
    }
  };

  const liveStateCapturePayload = () => {
    const monitor = selectedCaptureMonitor();
    return {
      mode: monitor ? "screen" : "foreground",
      monitor: monitor ? Number(monitor) : null
    };
  };

  const setLiveStateUi = (status = {}) => {
    latestLiveStateStatus = status || {};
    const enabled = Boolean(status.enabled);
    const state = String(status.status || (enabled ? "Watching" : "Off"));
    liveStateEnabled = enabled;
    localStorage.setItem("live-state", enabled ? "on" : "off");
    liveStateBtn?.classList.toggle("active", enabled && !["Off", "Error"].includes(state));
    liveStateBtn?.classList.toggle("thinking", state === "Thinking");
    liveStateBtn?.classList.toggle("paused", state === "Paused" || state === "uncertain");
    liveStateBtn?.classList.toggle("error", state === "Error" || state === "capture_failed");
    if (liveStateBtn) {
      const label = state === "Thinking"
        ? "Think"
        : state === "Paused"
          ? "Pause"
          : state === "Error" || state === "capture_failed"
            ? "Err"
            : enabled
              ? "Watch"
              : "Live";
      setButtonContent(liveStateBtn, enabled ? "radio" : "square", label);
      const scene = status.scene ? ` Scene: ${status.scene}` : "";
      const error = status.last_error ? ` Error: ${status.last_error}` : "";
      liveStateBtn.title = `Live State: ${state}.${scene}${error}`;
    }
    if (liveStateAnalyzeBtn) {
      liveStateAnalyzeBtn.disabled = state === "Thinking";
      liveStateAnalyzeBtn.title = state === "Thinking"
        ? "Live State is analyzing now"
        : "Scan current screen with Hermes agent";
    }
    syncToolPanelState();
  };

  const refreshLiveStateStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/live-state/status`);
      if (!res.ok) throw new Error(`Live State status ${res.status}`);
      const status = await res.json();
      setLiveStateUi(status);
      return status;
    } catch (err) {
      if (liveStateBtn) {
        liveStateBtn.classList.add("error");
        setButtonContent(liveStateBtn, "square", "Err");
        liveStateBtn.title = `Live State status unavailable: ${err.message || err}`;
      }
      return null;
    }
  };

  const setLiveStatePolling = (enabled) => {
    if (liveStatePollTimer) {
      window.clearInterval(liveStatePollTimer);
      liveStatePollTimer = null;
    }
    if (enabled) {
      liveStatePollTimer = window.setInterval(() => {
        refreshLiveStateStatus().catch(() => {});
      }, 5000);
    }
  };

  const startLiveState = async () => {
    const payload = liveStateCapturePayload();
    await setAppCaptureExclusion(true);
    try {
      const res = await fetch(`${API_BASE}/live-state/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(`Live State start failed: ${res.status}`);
      const status = await res.json();
      setLiveStateUi(status);
      setLiveStatePolling(true);
      return status;
    } catch (err) {
      if (!captureProtectionEnabled) await setAppCaptureExclusion(false);
      throw err;
    }
  };

  const stopLiveState = async () => {
    const res = await fetch(`${API_BASE}/live-state/stop`, { method: "POST" });
    if (!res.ok) throw new Error(`Live State stop failed: ${res.status}`);
    const status = await res.json();
    setLiveStateUi(status);
    setLiveStatePolling(false);
    if (!captureProtectionEnabled) await setAppCaptureExclusion(false);
    return status;
  };

  const analyzeLiveStateNow = async () => {
    const payload = { ...liveStateCapturePayload(), force: true };
    const exclusionApplied = await setAppCaptureExclusion(true);
    const restoreAfterAnalyze = exclusionApplied && !captureProtectionEnabled && !liveStateEnabled;
    if (liveStateAnalyzeBtn) liveStateAnalyzeBtn.disabled = true;
    setButtonContent(liveStateAnalyzeBtn, "loader", "Scan");
    try {
      const res = await fetch(`${API_BASE}/live-state/analyze-now`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(`Live State analyze failed: ${res.status}`);
      const data = await res.json();
      setLiveStateUi(data.status || {});
      return data;
    } finally {
      setButtonContent(liveStateAnalyzeBtn, "target", "Scan");
      if (liveStateAnalyzeBtn) liveStateAnalyzeBtn.disabled = false;
      if (restoreAfterAnalyze) await setAppCaptureExclusion(false);
    }
  };

  const formatLiveStateResult = (data) => {
    const status = data?.status || {};
    const state = data?.state || {};
    if (state.scene) {
      const reason = status.last_skip_reason || state.last_skip_reason || "";
      const error = status.last_error || state.last_error || "";
      const retainedPrevious = reason === "parse_failed_retained_previous";
      const headline = retainedPrevious
        ? "Live State 這次輸出格式不完整，已保留上一筆有效狀態。"
        : data?.fresh
        ? "Live State 已更新。"
        : "Live State 目前使用上一筆可用狀態。";
      const objects = Array.isArray(state.visible_objects) && state.visible_objects.length
        ? `\n可見物件：${state.visible_objects.slice(0, 5).join("、")}`
        : "";
      const ui = Array.isArray(state.visible_ui) && state.visible_ui.length
        ? `\n畫面 UI：${state.visible_ui.slice(0, 4).join("、")}`
        : "";
      return [
        headline,
        `場景：${state.scene}`,
        `狀態：${state.player_status || "未知"} / 意圖：${state.possible_intent || "未知"}`,
        `信心：${Math.round(Number(state.confidence || 0) * 100)}%${objects}${ui}`,
        retainedPrevious && error ? `本次未更新原因：${error}` : ""
      ].filter(Boolean).join("\n");
    }
    const reason = status.last_skip_reason || state.last_skip_reason || "";
    const error = status.last_error || state.last_error || "";
    if (reason === "image_input_unsupported" || /image input is not supported|mmproj/i.test(error)) {
      return "Live State 截圖成功，但目前地端 Qwen 是文字 GGUF，不能讀圖片。需要改成支援 vision/mmproj 的地端模型，或把 Live State 改走雲端 vision。";
    }
    return `Live State 沒有產生可用狀態。${reason ? `\n原因：${reason}` : ""}${error ? `\n錯誤：${error}` : ""}`.trim();
  };

  const appendUserMessage = (text, imageBase64, mimeType = "image/jpeg") => {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", "user-message");
    msgDiv.textContent = text || "[screenshot]";
    if (imageBase64) {
      const img = document.createElement("img");
      img.src = `data:${mimeType || "image/jpeg"};base64,${imageBase64}`;
      img.addEventListener("load", scrollChatToBottom, { once: true });
      msgDiv.appendChild(document.createElement("br"));
      msgDiv.appendChild(img);
    }
    chatWindow.appendChild(msgDiv);
    scrollChatToBottom();
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
    await setVirtualCursorActiveWindow("tasks");
  };

  const closeTasksWindow = async () => {
    if (invoke) {
      await invoke("hide_tasks_window").catch((err) => {
        appendMessage(`Task close failed: ${err?.message || err}`, "bot");
      });
    }
    await setVirtualCursorActiveWindow("main");
  };

  const openSearchWindow = async () => {
    const game = selectedGameId || localStorage.getItem("currentGameId") || "";
    localStorage.setItem("currentGameId", game);
    localStorage.removeItem("igpu-game-search-pending-keyword");
    localStorage.removeItem("igpu-game-search-auto-run");
    if (invoke) {
      await invoke("show_search_window").catch((err) => {
        appendMessage(`Search window failed: ${err?.message || err}`, "bot");
      });
    }
    await events.emit?.("search:context", { game }).catch(() => {});
    await setVirtualCursorActiveWindow("search");
  };

  const closeSearchWindow = async () => {
    localStorage.removeItem("igpu-game-search-pending-keyword");
    localStorage.removeItem("igpu-game-search-auto-run");
    if (invoke) {
      await invoke("hide_search_window").catch((err) => {
        appendMessage(`Search close failed: ${err?.message || err}`, "bot");
      });
    }
    await setVirtualCursorActiveWindow("main");
  };

  const openGamePathWindow = async () => {
    const game = selectedGameId || localStorage.getItem("currentGameId") || "";
    localStorage.setItem("currentGameId", game);
    if (invoke) {
      await invoke("show_gamepath_window").catch((err) => {
        appendMessage(`GamePath window failed: ${err?.message || err}`, "bot");
      });
    }
    await events.emit?.("gamepath:context", { game }).catch(() => {});
    await setVirtualCursorActiveWindow("gamepath");
  };

  const closeGamePathWindow = async () => {
    if (invoke) {
      await invoke("hide_gamepath_window").catch((err) => {
        appendMessage(`GamePath close failed: ${err?.message || err}`, "bot");
      });
    }
    await setVirtualCursorActiveWindow("main");
  };

  const restoreMainFromStandby = async () => {
    await invoke?.("restore_main_from_standby").catch(async (err) => {
      console.warn("Could not restore main from standby:", err);
      await appWindow?.show?.().catch(() => {});
      await appWindow?.setFocus?.().catch(() => {});
    });
    await setVirtualCursorActiveWindow("main");
  };

  const openToolsWindow = async (options = {}) => {
    const { quiet = false, retry = false } = options;
    if (invoke) {
      await invoke("show_tools_window").catch((err) => {
        if (retry) {
          window.setTimeout(() => openToolsWindow({ quiet: true }), 450);
        } else if (!quiet) {
          appendMessage(`Tools panel failed: ${err?.message || err}`, "bot");
        }
      });
    }
    syncToolPanelState(true);
  };

  const notifyGamePathChanged = async (details = {}) => {
    const game = selectedGameId || localStorage.getItem("currentGameId") || "";
    const payload = {
      ...details,
      game,
      changed_at: Date.now()
    };
    localStorage.setItem("igpu-gamepath-last-change", JSON.stringify(payload));
    await events.emit?.("gamepath:changed", payload).catch(() => {});
  };

  toolsBtn?.addEventListener("click", (event) => {
    event.preventDefault();
    openToolsWindow();
  });
  const collapseToStandby = async (event) => {
    event.preventDefault();
    event.stopPropagation();
    await invoke?.("collapse_main_to_standby").catch((err) => {
      appendMessage(`Standby failed: ${err?.message || err}`, "bot");
    });
  };

  standbyBtn?.addEventListener("pointerdown", collapseToStandby);
  standbyBtn?.addEventListener("click", collapseToStandby);
  document.addEventListener("pointerdown", (event) => {
    if (!event.target?.closest?.("#standbyBtn")) return;
    collapseToStandby(event);
  }, true);
  tasksBtn?.addEventListener("click", (event) => {
    event.preventDefault();
    openTasksWindow();
  });
  searchBtn?.addEventListener("click", (event) => {
    event.preventDefault();
    openSearchWindow();
  });
  gamepathBtn?.addEventListener("click", (event) => {
    event.preventDefault();
    openGamePathWindow();
  });

  const normalizeCommandText = (text) => (text || "")
    .toLowerCase()
    .replace(/[，。！？、,.!?]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const compactCommandText = (text) => normalizeCommandText(text).replace(/\s+/g, "");

  const includesAny = (text, patterns) => patterns.some((pattern) => (
    pattern instanceof RegExp ? pattern.test(text) : text.includes(pattern)
  ));

  const hasUiActionPrefix = (text) => includesAny(text, [
    "\u5e6b\u6211",
    "\u8acb",
    "\u53ef\u4ee5",
    "\u958b",
    "\u958b\u555f",
    "\u6253\u958b",
    "\u95dc",
    "\u95dc\u9589",
    "\u5207\u63db",
    "\u8abf",
    "\u8a2d",
    "open",
    "show",
    "toggle",
    "set",
    "start",
    "stop",
    "turn"
  ]);

  const extractPercent = (text) => {
    const match = text.match(/(\d{1,3})\s*%?/);
    if (!match) return null;
    const numeric = Number(match[1]);
    if (!Number.isFinite(numeric)) return null;
    return Math.max(10, Math.min(100, numeric));
  };

  const extractSearchKeyword = (rawText) => {
    let keyword = normalizeCommandText(rawText);
    const cleanup = [
      /\b(game search|search|google|wiki|youtube)\b/gi,
      /\b(open|show|find|look up|搜尋|查詢|查|找|攻略|資料|關鍵字)\b/gi,
      /幫我|請|可以|一下|現在|直接|開啟|打開|用|在|裡面|裡|遊戲/gi
    ];
    cleanup.forEach((pattern) => {
      keyword = keyword.replace(pattern, " ");
    });
    return keyword.replace(/\s+/g, " ").trim();
  };

  const requestedWindowAction = (compact, wantsOn, wantsOff) => {
    if (wantsOff) return "close";
    if (includesAny(compact, ["\u5207\u63db", "\u958b\u95dc", "toggle"])) return "toggle";
    if (wantsOn || includesAny(compact, ["\u986f\u793a", "\u53eb\u51fa", "\u62c9\u51fa", "show"])) return "open";
    return "open";
  };

  const setPerfMode = (enabled) => {
    document.body.classList.toggle("perf-mode", enabled);
    localStorage.setItem("perf-mode", enabled ? "true" : "false");
  };

  const handleUiCommand = async (rawText, options = {}) => {
    const text = normalizeCommandText(rawText);
    const compact = compactCommandText(rawText);
    const appendCommandUserMessage = () => {
      if (!options.userAlreadyAppended) appendUserMessage(rawText);
    };
    const directUiTarget = includesAny(compact, [
      "gamesearch",
      "\u904a\u6232\u641c\u5c0b",
      "\u641c\u5c0b\u8996\u7a97",
      "\u95dc\u9589\u641c\u5c0b",
      "\u95dc\u6389\u641c\u5c0b",
      "\u4efb\u52d9\u8996\u7a97",
      "\u95dc\u9589\u4efb\u52d9",
      "\u95dc\u6389\u4efb\u52d9",
      "\u5167\u5bb9\u4fdd\u8b77",
      "\u4fdd\u8b77\u5167\u5bb9",
      "\u622a\u5716\u4fdd\u8b77",
      "\u8a9e\u97f3\u6a21\u5f0f",
      "\u622a\u5716",
      "taskwindow",
      "voice mode",
      "screenshot"
    ]);
    if (!text || (!hasUiActionPrefix(text) && !directUiTarget)) return false;

    const wantsOff = includesAny(compact, ["\u95dc\u9589", "\u95dc\u6389", "\u95dc\u8d77", "\u96b1\u85cf", "\u6536\u8d77", "\u53d6\u6d88", "\u505c\u6b62", "close", "hide", "off", "stop", "disable"]);
    const wantsOn = includesAny(compact, ["\u958b\u555f", "\u6253\u958b", "\u958b\u8d77", "\u958b", "\u555f\u52d5", "open", "on", "start", "enable"]);
    const wantsStore = includesAny(compact, ["\u5b58\u4e0b\u4f86", "\u5132\u5b58", "\u4fdd\u5b58", "\u5b58\u8d77\u4f86", "\u5b58", "\u8a18\u9304", "\u8a18\u4e0b", "save", "store", "record"]);

    if (includesAny(compact, ["task", "\u4efb\u52d9", "\u76ee\u6a19\u6e05\u55ae", "\u5f85\u8fa6"]) && includesAny(compact, ["\u8996\u7a97", "\u7a97\u53e3", "\u9762\u677f", "\u958b", "\u95dc", "\u96b1\u85cf", "\u6536\u8d77", "\u986f\u793a", "open", "show", "close", "hide", "toggle"])) {
      const action = requestedWindowAction(compact, wantsOn, wantsOff);
      appendCommandUserMessage();
      if (action === "close") {
        await closeTasksWindow();
        appendMessage("UI command: Task window closed.", "bot");
      } else if (action === "toggle") {
        if (invoke) {
          await invoke("toggle_tasks_window").catch(async () => openTasksWindow());
        } else {
          await openTasksWindow();
        }
        appendMessage("UI command: Task window toggled.", "bot");
      } else {
        await openTasksWindow();
        appendMessage("UI command: Task window opened.", "bot");
      }
      if (messageInput) messageInput.value = "";
      return true;
    }

    if (!wantsStore && includesAny(compact, ["gamesearch", "\u904a\u6232\u641c\u5c0b", "\u641c\u5c0b", "\u67e5\u8a62", "\u641c\u5c0b\u8996\u7a97", "\u653b\u7565\u8996\u7a97"])) {
      appendCommandUserMessage();
      if (wantsOff) {
        await closeSearchWindow();
        appendMessage("UI command: Game Search closed.", "bot");
      } else if (includesAny(compact, ["\u5207\u63db", "\u958b\u95dc", "toggle"])) {
        if (invoke) {
          await invoke("toggle_search_window").catch(async () => openSearchWindow());
        } else {
          await openSearchWindow();
        }
        appendMessage("UI command: Game Search toggled.", "bot");
      } else {
        await openSearchWindow();
        appendMessage("UI command: Game Search opened.", "bot");
      }
      if (messageInput) messageInput.value = "";
      return true;
    }

    if (includesAny(compact, ["\u900f\u660e", "opacity"])) {
      const percent = extractPercent(text);
      if (percent) {
        appendCommandUserMessage();
        applyOpacity(percent);
        appendMessage(`UI command: Opacity set to ${percent}%.`, "bot");
        if (messageInput) messageInput.value = "";
        return true;
      }
    }

    if (includesAny(compact, ["\u5167\u5bb9\u4fdd\u8b77", "\u4fdd\u8b77\u5167\u5bb9", "\u622a\u5716\u4fdd\u8b77", "protection", "protect"])) {
      appendCommandUserMessage();
      if ((wantsOn && !captureProtectionEnabled) || (wantsOff && captureProtectionEnabled) || (!wantsOn && !wantsOff)) {
        await toggleCaptureProtection();
      } else {
        appendMessage(captureProtectionEnabled ? "UI command: Content protection is already on." : "UI command: Content protection is already off.", "bot");
      }
      if (messageInput) messageInput.value = "";
      return true;
    }

    if (includesAny(compact, ["\u622a\u5716", "\u64f7\u53d6\u756b\u9762", "screenshot", "capture"]) && !includesAny(compact, ["\u4efb\u52d9", "task"])) {
      appendCommandUserMessage();
      await takeScreenshot();
      if (messageInput) messageInput.value = "";
      return true;
    }

    if (includesAny(compact, ["hud", "\u6a19\u8a18"]) && includesAny(compact, ["\u6e05\u9664", "\u95dc\u6389", "\u6d88\u6389", "clear", "hide"])) {
      appendCommandUserMessage();
      await clearHudOverlay();
      appendMessage("UI command: HUD cleared.", "bot");
      if (messageInput) messageInput.value = "";
      return true;
    }

    if (includesAny(compact, ["hud"]) && includesAny(compact, ["test", "\u6e2c\u8a66"])) {
      appendCommandUserMessage();
      const target = hudTargetFromSource(null);
      target.imageWidth = Number(target.width);
      target.imageHeight = Number(target.height);
      const ok = await showHudOverlay(makeTestOverlay(target), null);
      appendMessage(ok ? "UI command: HUD test sent." : `UI command: HUD test failed. ${lastHudError}`, "bot");
      if (messageInput) messageInput.value = "";
      return true;
    }

    if (includesAny(compact, ["perf", "\u6027\u80fd", "\u6548\u80fd"])) {
      appendCommandUserMessage();
      const enabled = wantsOff ? false : wantsOn ? true : !document.body.classList.contains("perf-mode");
      setPerfMode(enabled);
      appendMessage(enabled ? "UI command: Perf mode on." : "UI command: Perf mode off.", "bot");
      if (messageInput) messageInput.value = "";
      return true;
    }

    if (includesAny(compact, ["\u8a9e\u97f3", "voice", "mic", "\u9ea5\u514b\u98a8"])) {
      appendCommandUserMessage();
      if (wantsOff) {
        stopVoiceMode();
        appendMessage("UI command: Voice mode off.", "bot");
      } else if (wantsOn) {
        await startVoiceMode();
        appendMessage("UI command: Voice mode on.", "bot");
      } else {
        toggleVoiceMode();
        appendMessage("UI command: Voice mode toggled.", "bot");
      }
      if (messageInput) messageInput.value = "";
      return true;
    }

    return false;
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
      gameId: analysis?.game_id || selectedGameId || "global",
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
    const parentBusy = context.parentBusy === true;
    if (isSending && !parentBusy) return;
    const originalNote = (note || "").trim();
    let imageBase64 = context.imageBase64 ?? pendingImageBase64;
    let imageMimeType = context.imageMimeType ?? pendingImageMimeType;
    let captureSource = context.captureSource ?? pendingCaptureSource;
    const routeStatus = context.intent ? `${formatIntentRouteStatus(context.intent, "task_memory")}\n` : "";
    const status = appendMessage(`${routeStatus}Building task from screen...`, "bot");
    if (!parentBusy) setBusy(true);
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

      if (!context.userAlreadyAppended) {
        appendUserMessage(originalNote || "Create a task from the current screen.", imageBase64, imageMimeType);
      }
      const response = await fetch(`${API_BASE}/tasks/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: originalNote,
          image_base64: imageBase64,
          game_id: selectedGameId || null,
          source_title: captureSource?.window_title || ""
        }),
        signal: context.signal || abortController?.signal
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
      status.textContent = err?.name === "AbortError"
        ? "Task logging stopped."
        : `Task logging failed: ${err.message || err}`;
    } finally {
      if (!parentBusy) setBusy(false);
      if (taskCaptureBtn) {
        taskCaptureBtn.disabled = false;
        setButtonContent(taskCaptureBtn, "flag", "Task");
      }
      setButtonContent(screenshotBtn, "camera", "Shot");
      screenshotBtn.disabled = false;
      scrollChatToBottom();
    }
  };

  const GAME_AUTO_CONFIDENCE = 0.55;
  const isGameAutoMode = () => gameDetectMode !== "manual";

  const gameOptionLabel = (gameId, name = "") => {
    const label = String(name || "").trim();
    return label && label.toLowerCase() !== "game" ? label : String(gameId || "").trim();
  };

  const currentGameLabel = () => {
    const selected = gameSelect?.selectedOptions?.[0]?.textContent?.trim() || "";
    if (selected && selected.toLowerCase() !== "game") return selected;
    if (selectedGameName) return selectedGameName;
    const known = gameCatalog.find((item) => item.id === selectedGameId);
    return known?.name || selectedGameId;
  };

  const ensureGameOption = (gameId, name = "") => {
    const value = String(gameId || "").trim();
    if (!value) return null;
    const label = gameOptionLabel(value, name);
    const existing = gameCatalog.find((item) => item.id === value);
    if (existing) {
      if (label) existing.name = label;
    } else {
      gameCatalog.push({ id: value, name: label || value });
    }
    gameCatalog.sort((a, b) => (a.id ? 1 : -1) - (b.id ? 1 : -1) || a.name.localeCompare(b.name));
    if (!gameSelect) {
      syncToolPanelState();
      return { value, textContent: label || value };
    }
    let option = Array.from(gameSelect.options).find((item) => item.value === value);
    if (!option) {
      option = document.createElement("option");
      option.value = value;
      gameSelect.appendChild(option);
    }
    if (label) option.textContent = label;
    syncToolPanelState();
    return option;
  };

  const updateGameAutoButton = (detection = null) => {
    if (!gameAutoBtn) return;
    const auto = isGameAutoMode();
    setButtonContent(gameAutoBtn, auto ? "radio" : "square", auto ? "Auto" : "Manual");
    gameAutoBtn.classList.toggle("active", auto);
    if (auto && detection?.game_id) {
      const confidence = Math.round(Number(detection.confidence || 0) * 100);
      gameAutoBtn.title = `Auto detect game: ${detection.name || detection.game_id} (${confidence}%)`;
    } else {
      gameAutoBtn.title = auto
        ? "Auto detect game from foreground window"
        : "Manual game selection; selecting a game teaches auto detection";
    }
  };

  const setGameDetectMode = (mode) => {
    gameDetectMode = mode === "manual" ? "manual" : "auto";
    localStorage.setItem("game-detect-mode", gameDetectMode);
    updateGameAutoButton();
    syncToolPanelState();
  };

  const applyGameSelection = (gameId, name = "", options = {}) => {
    const value = String(gameId || "").trim();
    const previous = selectedGameId || "";
    if (value) ensureGameOption(value, name);
    selectedGameId = value;
    selectedGameName = value ? gameOptionLabel(value, name) : "";
    if (gameSelect) gameSelect.value = value;
    localStorage.setItem("currentGameId", value);
    if (options.emit !== false && (options.forceEmit || previous !== value)) {
      events.emit?.("search:context", { game: value, source: options.source || "game-select" }).catch(() => {});
      events.emit?.("gamepath:context", { game: value, source: options.source || "game-select" }).catch(() => {});
    }
    syncToolPanelState();
    return previous !== value;
  };

  const learnCurrentGameMapping = async (gameId) => {
    const value = String(gameId || "").trim();
    if (!value) return null;
    const resp = await fetch(`${API_BASE}/game-profiles/learn`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ game_id: value, name: currentGameLabel() || value })
    });
    if (!resp.ok) {
      const detail = await resp.text().catch(() => "");
      throw new Error(`Game learn failed: ${resp.status}${detail ? ` ${detail}` : ""}`);
    }
    const data = await resp.json();
    if (data?.profile?.name) {
      ensureGameOption(value, data.profile.name);
      if (value === selectedGameId) selectedGameName = gameOptionLabel(value, data.profile.name);
    }
    updateGameAutoButton(data?.detection || null);
    syncToolPanelState();
    return data;
  };

  const pollActiveGame = async (options = {}) => {
    if (!isGameAutoMode()) return null;
    try {
      const resp = await fetch(`${API_BASE}/active-game`, { cache: "no-store" });
      if (!resp.ok) return null;
      const detection = await resp.json();
      updateGameAutoButton(detection);
      const gameId = String(detection?.game_id || "").trim();
      const confidence = Number(detection?.confidence || 0);
      if (!gameId || confidence < GAME_AUTO_CONFIDENCE) return detection;
      const changed = applyGameSelection(gameId, detection.name || gameId, { source: "auto-game" });
      const key = `${gameId}:${detection.process_name || ""}:${detection.source || ""}`;
      if (options.announce && (changed || key !== lastDetectedGameKey)) {
        appendMessage(`Auto game: ${detection.name || gameId}`, "bot");
      }
      lastDetectedGameKey = key;
      return detection;
    } catch (err) {
      console.warn("Active game detection unavailable:", err);
      return null;
    }
  };

  const loadGuideGames = async () => {
    try {
      const resp = await fetch(`${API_BASE}/game-profiles`);
      const data = await resp.json();
      const savedGameId = localStorage.getItem("currentGameId") || "";
      const profiles = data.profiles || {};
      gameCatalog = [{ id: "", name: "Game" }];
      if (gameSelect) gameSelect.innerHTML = '<option value="">Game</option>';
      for (const game of data.games || []) {
        ensureGameOption(game, profiles[game]?.name || game);
      }
      if (savedGameId) ensureGameOption(savedGameId, profiles[savedGameId]?.name || savedGameId);
      selectedGameId = savedGameId;
      const savedProfileName = profiles[savedGameId]?.name || savedGameId;
      selectedGameName = savedGameId ? gameOptionLabel(savedGameId, savedProfileName) : "";
      if (gameSelect) gameSelect.value = savedGameId;
      updateGameAutoButton();
      syncToolPanelState();
    } catch (err) {
      console.warn("Guide list unavailable:", err);
      updateGameAutoButton();
      syncToolPanelState();
    }
  };

  gameSelect?.addEventListener("change", () => {
    const gameId = gameSelect.value || "";
    setGameDetectMode("manual");
    applyGameSelection(gameId, currentGameLabel(), { source: "manual-game", forceEmit: true });
    learnCurrentGameMapping(gameId).catch((err) => console.warn("Game profile learn unavailable:", err));
  });

  const handleGameAutoToggle = async () => {
    const nextMode = isGameAutoMode() ? "manual" : "auto";
    setGameDetectMode(nextMode);
    if (nextMode === "auto") {
      const detection = await pollActiveGame({ announce: true });
      if (!detection?.game_id) appendMessage("Auto game: no foreground game detected yet.", "bot");
    } else if (selectedGameId) {
      learnCurrentGameMapping(selectedGameId).catch((err) => console.warn("Game profile learn unavailable:", err));
    }
    syncToolPanelState();
  };

  gameAutoBtn?.addEventListener("click", handleGameAutoToggle);

  await loadGuideGames();
  await pollActiveGame();
  window.setInterval(() => pollActiveGame(), 3000);
  captureDisplaySelect?.addEventListener("change", () => {
    localStorage.setItem("capture-monitor", captureDisplaySelect.value || "auto");
  });
  await loadCaptureMonitors();
  const initialLiveStateStatus = await refreshLiveStateStatus();
  setLiveStatePolling(Boolean(initialLiveStateStatus?.enabled));
  const handleLiveStateToggle = async () => {
    try {
      if (liveStateEnabled) {
        await stopLiveState();
        appendMessage("Live State 已關閉。", "bot");
      } else {
        const status = await startLiveState();
        const scene = status?.scene ? `\n目前上一筆狀態：${status.scene}` : "";
        appendMessage(`Live State 已開啟。\n背景觀察不會主動回覆；按 Scan 會擷取目前畫面並交給雲端 Hermes agent 分析。${scene}`, "bot");
      }
    } catch (err) {
      appendMessage(`Live State failed: ${err.message || err}`, "bot");
      await refreshLiveStateStatus();
    }
  };

  const handleLiveStateAnalyze = async () => {
    const pendingMsg = appendMessage("Scan 正在擷取目前畫面...\n接著會交給雲端 Hermes agent 分析。", "bot");
    try {
      if (liveStateAnalyzeBtn) liveStateAnalyzeBtn.disabled = true;
      setButtonContent(liveStateAnalyzeBtn, "loader", "Scan");
      const data = await captureScreen("turbo");
      const imageBase64 = data.image_base64;
      const imageMimeType = data.mime_type || "image/jpeg";
      const captureSource = data.source || null;
      const sourceWidth = data.source?.capture_width || data.original_width || data.width;
      const sourceHeight = data.source?.capture_height || data.original_height || data.height;
      updateBotMessage(
        pendingMsg,
        `Scan 已擷取 ${data.width}x${data.height}（來源 ${sourceWidth}x${sourceHeight}），正在送給 Hermes agent...`
      );
      appendUserMessage("Scan current game screen with Hermes.", imageBase64, imageMimeType);
      await sendToAI(
        [
          "請透過這張目前遊戲畫面，用繁體中文幫玩家做短分析。",
          "請優先回答：",
          "1. 你判斷玩家目前可能在什麼狀態或場景。",
          "2. 下一步可以做什麼，先給一句短教學，再補充必要細節；不要使用分級提示標籤。",
          "3. 如果畫面不足以判斷，請明確說還需要玩家補什麼資訊。",
          "不要主動暴雷；除非玩家問題需要，否則不要列來源。"
        ].join("\n"),
        imageBase64,
        captureSource
      );
      await refreshLiveStateStatus();
    } catch (err) {
      updateBotMessage(pendingMsg, `Hermes Scan failed: ${err.message || err}`);
      await refreshLiveStateStatus();
    } finally {
      setButtonContent(liveStateAnalyzeBtn, "target", "Scan");
      if (liveStateAnalyzeBtn) liveStateAnalyzeBtn.disabled = false;
    }
  };

  liveStateBtn?.addEventListener("click", handleLiveStateToggle);
  liveStateAnalyzeBtn?.addEventListener("click", handleLiveStateAnalyze);

  const handleToolPanelCommand = async (payload = {}) => {
    const command = String(payload.command || "");
    try {
      switch (command) {
        case "game_select": {
          const gameId = String(payload.game_id || "");
          setGameDetectMode("manual");
          applyGameSelection(gameId, String(payload.name || gameId), {
            source: "tools-panel",
            forceEmit: true
          });
          if (gameId) {
            learnCurrentGameMapping(gameId).catch((err) => console.warn("Game profile learn unavailable:", err));
          }
          break;
        }
        case "game_auto_toggle":
          await handleGameAutoToggle();
          break;
        case "live_toggle":
          await handleLiveStateToggle();
          break;
        case "live_scan":
          await handleLiveStateAnalyze();
          break;
        case "open_search":
          await openSearchWindow();
          break;
        case "open_tasks":
          await openTasksWindow();
          break;
        case "open_gamepath":
          await openGamePathWindow();
          break;
        case "hud_clear":
          await handleHudClear();
          break;
        case "hud_test":
          await handleHudTest();
          break;
        case "protect_toggle":
          await toggleCaptureProtection();
          break;
        case "cursor_toggle":
          virtualCursor.toggle();
          break;
        case "perf_toggle": {
          const enabled = document.body.classList.toggle("perf-mode");
          localStorage.setItem("perf-mode", enabled ? "true" : "false");
          syncToolPanelState();
          break;
        }
        case "opacity_set":
          applyOpacity(payload.opacity);
          break;
        case "state_request":
          syncToolPanelState(true);
          break;
        default:
          if (command) console.warn("Unknown tool panel command:", command);
      }
    } finally {
      syncToolPanelState();
    }
  };

  await events.listen?.("tool-panel-command", (event) => {
    handleToolPanelCommand(event.payload || {}).catch((err) => {
      appendMessage(`Tool panel command failed: ${err?.message || err}`, "bot");
      syncToolPanelState();
    });
  }).catch(() => {});
  await events.listen?.("tool-panel-ready", () => {
    syncToolPanelState(true);
  }).catch(() => {});
  await openToolsWindow({ quiet: true, retry: true });

  const setBusy = (busy) => {
    const nextMode = busy
      ? (standbyWindowMode === "collapsed" ? "collapsed" : "thinking")
      : (standbyDetailedConversation ? "detail" : (standbyWindowMode === "response" || standbyWindowMode === "detail" ? standbyWindowMode : (standbyWindowMode === "typein" || standbyWindowMode === "thinking" ? "typein" : "collapsed")));

    isSending = busy;
    sendBtn.style.display = busy ? "none" : "block";
    stopBtn.style.display = busy ? "block" : "none";

    events.emit?.("standby:set-mode", {
      expanded: nextMode !== "collapsed",
      mode: nextMode
    }).catch(() => {});
  };

  const setVoiceRecording = (recording) => {
    isVoiceRecording = recording;
    voiceBtn?.classList.toggle("recording", recording);
    voiceBtn?.classList.toggle("voice-mode", isVoiceModeEnabled);
    if (voiceBtn) {
      const active = recording || isVoiceModeEnabled;
      setButtonContent(voiceBtn, active ? "radio" : "mic", active ? "On" : "Mic");
      voiceBtn.title = isVoiceModeEnabled
        ? "Voice mode on: speak anytime, pauses auto-send"
        : "Voice mode";
      voiceBtn.disabled = isVoiceBusy && !recording;
    }
  };

  const setVoiceBusy = (busy) => {
    isVoiceBusy = busy;
    if (voiceBtn && !isVoiceRecording && !isVoiceModeEnabled) {
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

  const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

  const drainVoiceSendQueue = async () => {
    if (isVoiceQueueRunning) return;
    isVoiceQueueRunning = true;
    try {
      while (voiceSendQueue.length) {
        const nextText = voiceSendQueue.shift();
        while (isSending) {
          await sleep(250);
        }
        await sendMessage(nextText);
      }
    } finally {
      isVoiceQueueRunning = false;
    }
  };

  const queueVoiceMessage = (text) => {
    const normalized = normalizeSpeechText(text);
    if (!normalized) return;
    voiceSendQueue.push(normalized);
    drainVoiceSendQueue().catch((err) => {
      appendMessage(`Voice queue failed: ${err.message || err}`, "bot");
    });
  };

  const resetVoiceModePhrase = () => {
    liveVoiceBaseText = "";
    liveVoiceFinalText = "";
    liveVoiceInterimText = "";
    liveVoiceSent = false;
    skipVoiceBlobTranscription = false;
    if (messageInput) messageInput.value = "";
    if (voiceStatusMessage && isVoiceModeEnabled) {
      voiceStatusMessage.textContent = "Voice mode on. Speak anytime; I will send after you pause.";
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
    const now = Date.now();
    if (spoken === lastSentVoiceText && now - lastSentVoiceAt < 3500) {
      clearLiveSpeechSilence();
      clearLiveVoiceDraft({ clearInput: true });
      return false;
    }

    clearLiveSpeechSilence();
    liveVoiceSent = true;
    lastSentVoiceText = spoken;
    lastSentVoiceAt = now;
    skipVoiceBlobTranscription = true;
    clearLiveVoiceDraft({ clearInput: true });
    if (voiceStatusMessage) voiceStatusMessage.textContent = `Heard: ${spoken}`;
    queueVoiceMessage(messageText);
    return true;
  };

  const scheduleLiveSpeechAutoSend = () => {
    clearLiveSpeechSilence();
    if (!liveVoiceSpokenText(true) || liveVoiceSent) return;

    liveSpeechSilenceTimer = window.setTimeout(() => {
      if (!isVoiceRecording || liveVoiceSent || !liveVoiceSpokenText(true)) return;
      if (sendLiveVoiceTranscript({ includeInterim: true })) {
        if (isVoiceModeEnabled) {
          window.setTimeout(resetVoiceModePhrase, 250);
        } else {
          stopLiveSpeechRecognition();
          stopRecorderAfterLiveSpeech();
          voiceStatusMessage = null;
        }
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
    if (speechRecognition) return true;

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
          voiceStatusMessage.textContent = isVoiceModeEnabled
            ? "Voice mode on. Speak anytime; I will send after you pause."
            : "Listening live... speech will appear in the input box.";
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
        if (error === "no-speech") {
          if (voiceStatusMessage && isVoiceModeEnabled) {
            voiceStatusMessage.textContent = "Voice mode on. Waiting for speech...";
          }
          return;
        }
        if (["not-allowed", "service-not-allowed", "audio-capture"].includes(error)) {
          liveSpeechBlocked = true;
        }
        if (error !== "aborted" && voiceStatusMessage) {
          voiceStatusMessage.textContent = isVoiceModeEnabled
            ? "Voice mode paused; trying to reconnect the microphone..."
            : "Live speech unavailable; recording fallback is still running.";
        }
      });

      recognition.addEventListener("end", () => {
        if (speechRecognition !== recognition) return;
        speechRecognition = null;
        if (liveVoiceSent && !isVoiceModeEnabled) return;
        liveVoiceInterimText = "";
        updateLiveVoiceInput();

        if (sendLiveVoiceTranscript({ includeInterim: false })) {
          if (isVoiceModeEnabled) {
            window.setTimeout(resetVoiceModePhrase, 250);
          } else {
            stopRecorderAfterLiveSpeech();
            voiceStatusMessage = null;
            return;
          }
        }

        if (!liveVoiceStopRequested && !liveSpeechBlocked && (isVoiceRecording || isVoiceModeEnabled)) {
          liveSpeechRestartTimer = window.setTimeout(() => {
            if (
              liveVoiceStopRequested ||
              liveSpeechBlocked ||
              (!isVoiceRecording && !isVoiceModeEnabled) ||
              (liveVoiceSent && !isVoiceModeEnabled)
            ) {
              return;
            }
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

  const startVoiceMode = async () => {
    if (isVoiceModeEnabled || isVoiceStarting) return;
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      appendMessage("Voice mode is not available in this WebView. Use the Mic button as one-shot recording after updating WebView2.", "bot");
      return;
    }

    isVoiceModeEnabled = true;
    voiceStopRequested = false;
    liveVoiceStopRequested = false;
    liveSpeechBlocked = false;
    liveVoiceSent = false;
    skipVoiceBlobTranscription = true;
    liveVoiceBaseText = messageInput.value.trim();
    liveVoiceFinalText = "";
    liveVoiceInterimText = "";
    setVoiceRecording(true);
    voiceStatusMessage = appendMessage("Voice mode on. Speak anytime; I will send after you pause.", "bot");

    const started = startLiveSpeechRecognition();
    if (!started) {
      isVoiceModeEnabled = false;
      setVoiceRecording(false);
      if (voiceStatusMessage) {
        voiceStatusMessage.textContent = "Voice mode could not start.";
        voiceStatusMessage = null;
      }
    }
  };

  const stopVoiceMode = () => {
    if (!isVoiceModeEnabled && !isVoiceRecording) return;
    isVoiceModeEnabled = false;
    voiceStopRequested = true;
    liveVoiceStopRequested = true;
    clearLiveSpeechRestart();
    clearLiveSpeechSilence();
    stopLiveSpeechRecognition(true);
    clearLiveVoiceDraft({ clearInput: true });
    setVoiceRecording(false);
    if (voiceStatusMessage) {
      voiceStatusMessage.textContent = "Voice mode off.";
      voiceStatusMessage = null;
    }
  };

  const toggleVoiceMode = (options = {}) => {
    const source = options?.source || "button";
    const now = Date.now();
    const debounceMs = source === "virtual-cursor" ? 900 : 350;
    if (now - lastVoiceToggleAt < debounceMs) return;
    lastVoiceToggleAt = now;
    if (isVoiceModeEnabled || isVoiceRecording) {
      stopVoiceMode();
    } else {
      startVoiceMode();
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
    if (isVoiceModeEnabled && !mediaRecorder) {
      stopVoiceMode();
      return;
    }
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

  const sendToAI = async (text, imageBase64 = null, captureSource = null, options = {}) => {
    const manageBusy = options.manageBusy !== false;
    if (manageBusy && isSending) return;
    if (manageBusy) {
      setBusy(true);
      abortController = new AbortController();
    } else if (!abortController) {
      abortController = new AbortController();
    }

    const { statusDiv, routeLogDiv, routeTrace, contentDiv } = createBotResponseMessage(
      imageBase64 ? "Reading compressed screenshot..." : "Preparing response..."
    );
    let collected = "";
    let showedOverlay = false;
    let lastStandbyResponseText = "";
    let lastStandbyDetailText = "";
    let latestResponseFormat = null;

    const formatStandbyResponseText = (value) => {
      const cleaned = String(value || "")
        .replace(/```[\s\S]*?```/g, " ")
        .replace(/[`*_>#-]/g, " ")
        .replace(/\s+/g, " ")
        .trim();
      if (cleaned.length <= 118) return cleaned;
      return `${cleaned.slice(0, 115).trim()}...`;
    };

    const normalizeResponseFormat = (payload) => {
      const source = payload?.response_format || payload || {};
      const shortText = String(source.short ?? source.Short ?? "").trim();
      const longText = String(source.long ?? source.Long ?? source.content ?? "").trim();
      if (!shortText && !longText) return null;
      return {
        short: shortText || formatStandbyResponseText(longText),
        long: longText || shortText
      };
    };

    const emitStandbyResponse = (shortText, longText) => {
      if (standbyWindowMode === "collapsed") return;
      const responseText = String(shortText || "").trim();
      const detailText = String(longText || shortText || "").trim();
      if (!responseText || (responseText === lastStandbyResponseText && detailText === lastStandbyDetailText)) return;
      lastStandbyResponseText = responseText;
      lastStandbyDetailText = detailText;
      const responseMode = standbyDetailedConversation || standbyWindowMode === "detail" ? "detail" : "response";
      standbyWindowMode = responseMode;
      events.emit?.("standby:set-response", {
        text: responseText,
        detailText,
        question: text,
        title: selectedGameName || selectedGameId || "Game Companion",
        mode: responseMode
      }).catch(() => {});
    };

    const syncStandbyResponse = () => {
      if (latestResponseFormat) {
        emitStandbyResponse(latestResponseFormat.short, latestResponseFormat.long);
        return;
      }
      const detailText = collected.trim();
      emitStandbyResponse(formatStandbyResponseText(detailText), detailText);
    };

    try {
      const body = {
        message: text,
        game_id: selectedGameId || null,
        use_memory: true,
        use_live_state: liveStateEnabled
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
      const handleSseLine = async (line) => {
        if (!line.startsWith("data: ")) return;
        const dataStr = line.slice(6).trim();
        if (!dataStr || dataStr === "[DONE]") return;

        try {
          const dataObj = JSON.parse(dataStr);
          if (dataObj.lookup_status) {
            recordLookupStatus(dataObj.lookup_status, statusDiv, routeLogDiv, routeTrace);
            if (["gamepath_stored", "gamepath_disputed"].includes(dataObj.lookup_status.stage)) {
              await notifyGamePathChanged(dataObj.lookup_status);
            }
            scrollChatToBottom();
            return;
          }
          if (dataObj.overlay) {
            const hudShown = await showHudOverlay(dataObj.overlay, captureSource);
            showedOverlay = showedOverlay || hudShown;
            if (!hudShown) {
              collected += `\nHUD 顯示失敗。${lastHudError}`;
              renderFormattedMessage(contentDiv, collected.trim());
              scrollChatToBottom();
            }
          }
          if (dataObj.response_format) {
            const responseFormat = normalizeResponseFormat(dataObj.response_format);
            if (responseFormat) {
              latestResponseFormat = responseFormat;
              collected = responseFormat.long || collected;
              renderFormattedMessage(contentDiv, collected.trim());
              syncStandbyResponse();
              scrollChatToBottom();
            }
            return;
          }
          const content = dataObj.content || "";
          if (!content) return;
          collected += content;
          renderFormattedMessage(contentDiv, collected.trimStart());
          syncStandbyResponse();
          scrollChatToBottom();
        } catch (err) {
          console.warn("Could not parse SSE line:", line, err);
        }
      };

      if (response.body?.getReader) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split(/\r?\n/);
          buffer = lines.pop() || "";
          for (const line of lines) await handleSseLine(line.trimEnd());
        }
        buffer += decoder.decode();
        if (buffer) {
          const lines = buffer.split(/\r?\n/);
          for (const line of lines) await handleSseLine(line.trimEnd());
        }
      } else {
        const rawResponse = await response.text();
        const lines = rawResponse.split(/\r?\n/);
        for (const line of lines) await handleSseLine(line.trimEnd());
      }

      renderFormattedMessage(contentDiv, collected.trim() || (
        showedOverlay
          ? "HUD 已標記。"
          : "這次沒有產生可用回覆；請換個問法，或指定要看的畫面位置。"
      ));
      scrollChatToBottom();
    } catch (error) {
      if (error.name === "AbortError") {
        renderFormattedMessage(contentDiv, `${collected.trim()}\n\nStopped.`.trim());
      } else {
        console.error(error);
        statusDiv.textContent = "Connection failed";
        statusDiv.dataset.stage = "error";
        renderFormattedMessage(contentDiv, `${error.message || error}`);
      }
    } finally {
      if (manageBusy) {
        abortController = null;
        setBusy(false);
      }
    }
  };

  const sendMessage = async (overrideText = null) => {
    const text = (overrideText ?? messageInput.value).trim();
    let imageBase64 = pendingImageBase64;
    let imageMimeType = pendingImageMimeType;
    let captureSource = pendingCaptureSource;
    if (!text && !imageBase64) return;
    if (isSendInFlight || isSending) return;
    isSendInFlight = true;
    let busyManagedHere = false;
    let userAlreadyAppended = false;

    try {
      await exitVirtualCursorTextEntry("send");

      setBusy(true);
      busyManagedHere = true;
      abortController = new AbortController();

      messageInput.value = "";
      clearImagePreview();
      appendUserMessage(text || "Analyze this screenshot.", imageBase64, imageMimeType);
      userAlreadyAppended = true;

      const intent = text ? await routeUserIntent(text, abortController.signal) : null;
      if (abortController?.signal?.aborted) return;
      const intentRoute = String(intent?.route || "").trim();

      if (text && !imageBase64 && intentRoute === "ui_command") {
        abortController = null;
        setBusy(false);
        busyManagedHere = false;
        if (await handleUiCommand(text, { userAlreadyAppended })) {
          return;
        }
        setBusy(true);
        busyManagedHere = true;
        abortController = new AbortController();
      }

      if (text && intentRoute === "task_memory") {
        await createTaskFromContext(text, {
          imageBase64,
          imageMimeType,
          captureSource,
          capture: true,
          intent,
          userAlreadyAppended,
          parentBusy: true,
          signal: abortController?.signal
        });
        return;
      }

      if (!imageBase64 && isScreenshotIntentRoute(intentRoute)) {
        const status = appendMessage(`${formatIntentRouteStatus(intent, intentRoute || "screenshot_visual")}\nQwen requested screen context. Auto-capturing for Hermes vision...`, "bot");
        try {
          screenshotBtn.disabled = true;
          setButtonContent(screenshotBtn, "camera", "Shot...");
          const data = await captureScreen("turbo");
          imageBase64 = data.image_base64;
          imageMimeType = data.mime_type || "image/jpeg";
          captureSource = data.source || null;
          const sourceWidth = data.source?.capture_width || data.original_width || data.width;
          const sourceHeight = data.source?.capture_height || data.original_height || data.height;
          status.textContent = `Auto-captured ${data.width}x${data.height} from source ${sourceWidth}x${sourceHeight}. Sending to cloud Hermes agent...`;
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

      const fixedSourceHint = captureSource?.window_title
        ? `\n\nScreenshot source window title: ${captureSource.window_title}`
        : "";
      await sendToAI(
        (text || "Analyze this screenshot and give one useful next step.") + fixedSourceHint,
        imageBase64,
        captureSource,
        { manageBusy: false }
      );
    } catch (err) {
      if (err?.name === "AbortError") {
        appendMessage("Stopped.", "bot");
      } else {
        console.error(err);
        appendMessage(`Send failed: ${err.message || err}`, "bot");
      }
    } finally {
      if (busyManagedHere) {
        abortController = null;
        setBusy(false);
      }
      isSendInFlight = false;
    }
  };

  const takeScreenshot = async () => {
    if (isScreenshotInFlight || screenshotBtn.disabled || isSending) return;
    isScreenshotInFlight = true;
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
      isScreenshotInFlight = false;
    }
  };

  sendBtn?.addEventListener("click", () => sendMessage());
  messageInput?.addEventListener("keydown", (event) => {
    if (event.isComposing || event.keyCode === 229) return;
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
  voiceBtn?.addEventListener("virtual-cursor-activate", (event) => {
    event.preventDefault();
    event.stopPropagation();
    toggleVoiceMode({ source: "virtual-cursor" });
  });
  voiceBtn?.addEventListener("click", (event) => {
    if (!event.isTrusted && (isVoiceModeEnabled || isVoiceRecording || isVoiceStarting)) return;
    toggleVoiceMode({ source: event.isTrusted ? "button" : "programmatic" });
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
    await invoke?.("collapse_main_to_standby").catch(async (err) => {
      console.warn("Could not collapse to standby:", err);
      await appWindow?.minimize?.().catch(() => {});
    });
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
  await events.listen?.("protect-state-changed", (event) => {
    const enabled = Boolean(event.payload?.enabled);
    if (captureProtectionEnabled === enabled) return;
    captureProtectionEnabled = enabled;
    updateProtectButton();
    if (!enabled && liveStateEnabled) {
      setAppCaptureExclusion(true).catch(() => {});
    }
    appendMessage(enabled ? "Content protection on. (F4)" : "Screenshot protection off. (F4)", "bot");
  }).catch(() => {});
  await events.listen?.("task-hotkey", () => {
    createTaskFromContext(messageInput?.value || "", { capture: true });
  }).catch(() => {});
  await events.listen?.("voice-hotkey-start", () => {
    const now = Date.now();
    if (now - lastVoiceHotkeyToggleAt < 500) return;
    lastVoiceHotkeyToggleAt = now;
    toggleVoiceMode();
  }).catch(() => {});
  await events.listen?.("voice-hotkey-stop", () => {}).catch(() => {});
  await events.listen?.("clear-hud-hotkey", async () => {
    await clearHudOverlay();
    appendMessage("HUD cleared.", "bot");
  }).catch(() => {});

  await events.listen?.("standby:submit", async (event) => {
    const text = String(event.payload?.text || "").trim();
    if (!text) return;
    standbyDetailedConversation = event.payload?.source === "detail";
    messageInput.value = text;
    messageInput.dispatchEvent(new Event("input", { bubbles: true }));
    await sendMessage(text);
  }).catch(() => {});

  await events.listen?.("standby:voice-toggle", async () => {
    await restoreMainFromStandby();
    toggleVoiceMode({ source: "standby" });
  }).catch(() => {});

  await events.listen?.("standby:mode-change", (event) => {
    const mode = normalizeStandbyMode(event.payload?.mode)
      || (event.payload?.expanded ? "typein" : "collapsed");
    standbyWindowMode = mode;
    if (mode === "detail") {
      standbyDetailedConversation = true;
    } else if (mode === "collapsed" || mode === "typein" || mode === "response") {
      standbyDetailedConversation = false;
    }
  }).catch(() => {});

  const applyGamePathAsk = async (entry = {}) => {
    const title = entry.title || entry.question || "GamePath";
    messageInput.value = entry.message || `根據 GamePath「${title}」，請幫我整理下一步`;
    messageInput.dispatchEvent(new Event("input", { bubbles: true }));
    localStorage.removeItem("igpu-gamepath-ask-pending");
    await appWindow?.show?.().catch(() => {});
    await appWindow?.setFocus?.().catch(() => {});
    await setVirtualCursorActiveWindow("main");
    messageInput?.focus();
  };

  await events.listen?.("tasks:ask", async (event) => {
    const task = event.payload || {};
    const title = task.title || "目前任務";
    messageInput.value = `根據任務「${title}」，請告訴我下一步怎麼做`;
    await appWindow?.show?.().catch(() => {});
    await appWindow?.setFocus?.().catch(() => {});
    messageInput?.focus();
  }).catch(() => {});

  await events.listen?.("gamepath:ask", async (event) => {
    await applyGamePathAsk(event.payload || {});
  }).catch(() => {});

  window.addEventListener("storage", (event) => {
    if (event.key !== "igpu-gamepath-ask-pending" || !event.newValue) return;
    try {
      applyGamePathAsk(JSON.parse(event.newValue));
    } catch (err) {
      console.warn("Pending GamePath ask failed:", err);
    }
  });

  await globalShortcut.register?.("F9", takeScreenshot).catch((err) => {
    console.warn("F9 registration failed:", err);
  });

  await globalShortcut.register?.("F10", () => {
    clearHudOverlay();
    appendMessage("HUD cleared.", "bot");
  }).catch(() => {});
});
