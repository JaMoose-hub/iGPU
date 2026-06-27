import lottie from "lottie-web";
import loadingAnimation from "./assets/standby-thinking/loading.json";
import thinkingLightAnimation from "./assets/standby-thinking/thinking_light.json";
import thinkingGlowAUrl from "./assets/standby-thinking/e96f88b672c0f9e7785f4a2a288091eea871c166.webp";
import thinkingGlowBUrl from "./assets/standby-thinking/ec50226adaafbe9d3f9c2384e2ad25a7b9c88aee.webp";

const createAnimationData = (animation, imageUrls = {}) => {
  const data = JSON.parse(JSON.stringify(animation));
  for (const asset of data.assets || []) {
    if (!imageUrls[asset.id]) continue;
    asset.u = "";
    asset.p = imageUrls[asset.id];
  }
  return data;
};

const createThinkingLightData = () => {
  const imageUrls = {
    e96f88b672c0f9e7785f4a2a288091eea871c166: thinkingGlowAUrl,
    ec50226adaafbe9d3f9c2384e2ad25a7b9c88aee: thinkingGlowBUrl,
  };
  return createAnimationData(thinkingLightAnimation, imageUrls);
};

const standbyModeFromInput = (value) => {
  if (typeof value !== "string") return null;
  const normalized = value.trim().toLowerCase();
  if (["collapsed", "typein", "thinking", "response", "detail"].includes(normalized)) return normalized;
  return null;
};

const isStandbyModeCollapsed = (mode) => mode === "collapsed";

const resolveStandbyMode = (nextMode, fallback = "typein") => {
  if (typeof nextMode === "string") {
    return standbyModeFromInput(nextMode) || fallback;
  }
  return nextMode ? fallback : "collapsed";
};

const runWhenDomReady = (callback) => {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", callback, { once: true });
  } else {
    callback();
  }
};

const COLLAPSED_IDLE_DELAY_MS = 3000;
const COLLAPSED_ARM_DELAY_MS = 800;
const COLLAPSED_LONG_PRESS_MS = 1000;
const STANDBY_WAKE_CLICK_WINDOW_MS = 5000;
const STANDBY_OPENING_FADE_MS = 90;
const STANDBY_OPENING_SETTLE_MS = 36;
const STANDBY_COLLAPSE_ANIMATION_MS = 180;

const wait = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));
const nextFrame = () => new Promise((resolve) => window.requestAnimationFrame(() => resolve()));
const nextPaint = async () => {
  await nextFrame();
  await nextFrame();
};

runWhenDomReady(async () => {
  const tauri = window.__TAURI__ || {};
  const events = tauri.event || {};
  const invoke = tauri.core?.invoke;
  const appWindow = tauri.window?.getCurrentWindow?.();

  const root = document.getElementById("standbyRoot");
  const collapsedButton = document.getElementById("standbyCollapsedButton");
  const form = document.getElementById("standbyForm");
  const input = document.getElementById("standbyInput");
  const sendBtn = document.getElementById("standbySendBtn");
  const thinkingPanel = document.getElementById("standbyThinkingPanel");
  const thinkingLight = document.getElementById("standbyThinkingLight");
  const thinkingLoading = document.getElementById("standbyThinkingLoading");
  const responsePanel = document.getElementById("standbyResponsePanel");
  const responseText = document.getElementById("standbyResponseText");
  const detailPanel = document.getElementById("standbyDetailPanel");
  const detailTitle = document.getElementById("standbyDetailTitle");
  const detailQuestion = document.getElementById("standbyDetailQuestion");
  const detailAnswer = document.getElementById("standbyDetailAnswer");
  const detailForm = document.getElementById("standbyDetailForm");
  const detailInput = document.getElementById("standbyDetailInput");
  const detailSendBtn = document.getElementById("standbyDetailSendBtn");
  const detailCloseBtn = document.getElementById("standbyDetailCloseBtn");

  if (thinkingLight) {
    try {
      lottie.loadAnimation({
        container: thinkingLight,
        renderer: "svg",
        loop: true,
        autoplay: true,
        animationData: createThinkingLightData(),
        rendererSettings: {
          preserveAspectRatio: "xMidYMid slice",
        },
      });
    } catch (err) {
      console.warn("Could not start standby thinking light:", err);
    }
  }

  if (thinkingLoading) {
    try {
      lottie.loadAnimation({
        container: thinkingLoading,
        renderer: "svg",
        loop: true,
        autoplay: true,
        animationData: createAnimationData(loadingAnimation),
        rendererSettings: {
          preserveAspectRatio: "xMidYMid meet",
        },
      });
    } catch (err) {
      console.warn("Could not start standby loading lottie:", err);
    }
  }

  let standbyMode = "collapsed";
  let lastResponseSummary = "";
  let lastDetailTitle = "Game Companion";
  let lastDetailQuestion = "Your last question";
  let lastDetailAnswer = "I have a response ready.";
  let collapsedIdleTimer = null;
  let collapsedArmTimer = null;
  let collapsedLongPressTimer = null;
  let collapseAnimationTimer = null;
  let standbyWakeTimer = null;
  let standbyWakeActive = false;
  let nativeModeEchoToIgnore = null;
  let ignoreNextCollapsedClick = false;
  let pointerInsideStandby = false;

  const hasDraftText = () => Boolean((input?.value || "").trim());
  const hasDetailDraftText = () => Boolean((detailInput?.value || "").trim());

  const updateActionButtonMode = () => {
    if (!sendBtn) return;
    const typing = hasDraftText();
    sendBtn.classList.toggle("typing", typing);
    sendBtn.classList.toggle("default", !typing);
    sendBtn.title = typing ? "Send" : "Voice mode";
    sendBtn.setAttribute("aria-label", typing ? "Send" : "Voice mode");
  };

  const updateDetailActionButtonMode = () => {
    if (!detailSendBtn) return;
    const typing = hasDetailDraftText();
    detailSendBtn.classList.toggle("typing", typing);
    detailSendBtn.classList.toggle("default", !typing);
    detailSendBtn.title = typing ? "Reply" : "Voice mode";
    detailSendBtn.setAttribute("aria-label", typing ? "Reply" : "Voice mode");
  };

  const renderDetailContext = () => {
    if (detailTitle) detailTitle.textContent = lastDetailTitle || "Game Companion";
    if (detailQuestion) detailQuestion.textContent = lastDetailQuestion || "Your last question";
    if (detailAnswer) detailAnswer.textContent = lastDetailAnswer || lastResponseSummary || "I have a response ready.";
  };

  const announceMode = (mode) => {
    events.emit?.("standby:mode-change", {
      mode,
      expanded: !isStandbyModeCollapsed(mode),
    }).catch(() => {});
  };

  const clearCollapsedIdleTimer = () => {
    if (!collapsedIdleTimer) return;
    window.clearTimeout(collapsedIdleTimer);
    collapsedIdleTimer = null;
  };

  const clearCollapsedArmTimer = () => {
    if (!collapsedArmTimer) return;
    window.clearTimeout(collapsedArmTimer);
    collapsedArmTimer = null;
  };

  const clearCollapsedLongPressTimer = () => {
    if (!collapsedLongPressTimer) return;
    window.clearTimeout(collapsedLongPressTimer);
    collapsedLongPressTimer = null;
  };

  const clearCollapseAnimationTimer = () => {
    if (!collapseAnimationTimer) return;
    window.clearTimeout(collapseAnimationTimer);
    collapseAnimationTimer = null;
  };

  const clearStandbyWakeTimer = () => {
    if (!standbyWakeTimer) return;
    window.clearTimeout(standbyWakeTimer);
    standbyWakeTimer = null;
  };

  const setStandbyPointerPassthrough = async (passthrough) => {
    await invoke?.("set_standby_pointer_passthrough", { passthrough }).catch((err) => {
      console.warn("Could not update standby pointer passthrough:", err);
    });
  };

  const setCollapsedWakeWindow = async (awake, timeoutMs = STANDBY_WAKE_CLICK_WINDOW_MS) => {
    clearStandbyWakeTimer();
    if (!isStandbyModeCollapsed(standbyMode)) {
      standbyWakeActive = false;
      root?.classList.remove("awake");
      await setStandbyPointerPassthrough(false);
      return;
    }
    standbyWakeActive = Boolean(awake);
    root?.classList.toggle("awake", Boolean(awake));
    setCollapsedArmed(Boolean(awake));
    await setStandbyPointerPassthrough(!awake);
    if (!awake) {
      scheduleCollapsedIdle();
      return;
    }
    setCollapsedIdle(false);
    standbyWakeTimer = window.setTimeout(() => {
      standbyWakeTimer = null;
      if (!isStandbyModeCollapsed(standbyMode)) return;
      setCollapsedWakeWindow(false).catch(() => {});
    }, Math.max(1200, Number(timeoutMs) || STANDBY_WAKE_CLICK_WINDOW_MS));
  };

  const setCollapsedIdle = (idle) => {
    root?.classList.toggle("idle", Boolean(idle && isStandbyModeCollapsed(standbyMode)));
  };

  const setCollapsedArmed = (armed) => {
    root?.classList.toggle("armed", Boolean(armed && isStandbyModeCollapsed(standbyMode)));
  };

  const scheduleCollapsedIdle = () => {
    clearCollapsedIdleTimer();
    if (!isStandbyModeCollapsed(standbyMode) || pointerInsideStandby) {
      setCollapsedIdle(false);
      return;
    }
    collapsedIdleTimer = window.setTimeout(() => {
      setCollapsedIdle(true);
      collapsedIdleTimer = null;
    }, COLLAPSED_IDLE_DELAY_MS);
  };

  const scheduleCollapsedArmed = () => {
    clearCollapsedArmTimer();
    setCollapsedArmed(false);
    if (!isStandbyModeCollapsed(standbyMode) || !pointerInsideStandby) return;
    collapsedArmTimer = window.setTimeout(() => {
      setCollapsedArmed(true);
      collapsedArmTimer = null;
    }, COLLAPSED_ARM_DELAY_MS);
  };

  const wakeCollapsedStandby = () => {
    if (!isStandbyModeCollapsed(standbyMode)) return;
    setCollapsedIdle(false);
    scheduleCollapsedIdle();
  };

  const beginCollapsedLongPress = () => {
    clearCollapsedLongPressTimer();
    if (!isStandbyModeCollapsed(standbyMode)) return;
    collapsedLongPressTimer = window.setTimeout(async () => {
      collapsedLongPressTimer = null;
      if (!isStandbyModeCollapsed(standbyMode) || !pointerInsideStandby) return;
      ignoreNextCollapsedClick = true;
      clearCollapsedArmTimer();
      setCollapsedArmed(false);
      await setMode("typein", { syncWindow: true, emitMode: true, focusInput: false });
      root?.classList.add("listening");
      await events.emit?.("standby:voice-toggle", { source: "standby-long-press" }).catch(() => {});
    }, COLLAPSED_LONG_PRESS_MS);
  };

  const applyVisualMode = (mode, options = {}) => {
    const { opening = false } = options;
    root?.classList.remove("expanded", "collapsed", "typein", "thinking", "response", "detail", "idle", "armed", "awake", "closing", "opening");
    if (isStandbyModeCollapsed(mode)) {
      root?.classList.add("collapsed");
      if (standbyWakeActive) root?.classList.add("awake", "armed");
    } else {
      root?.classList.add("expanded", mode);
      if (opening) root?.classList.add("opening");
    }
  };

  const syncStandbyWindowMode = async (mode) => {
    if (!invoke) return;
    nativeModeEchoToIgnore = mode;
    await invoke("set_standby_window_mode", { mode }).catch(async () => {
      await invoke("set_standby_window_expanded", { expanded: !isStandbyModeCollapsed(mode) });
    }).catch((err) => {
      console.warn("Could not resize standby window:", err);
    }).finally(() => {
      window.setTimeout(() => {
        if (nativeModeEchoToIgnore === mode) nativeModeEchoToIgnore = null;
      }, 0);
    });
  };

  const applyStandbyMode = async (nextMode, options = {}) => {
    const { syncWindow = true, emitMode = true, focusInput = false } = options;
    clearCollapseAnimationTimer();
    const mode = resolveStandbyMode(nextMode, "typein");
    const wasCollapsed = isStandbyModeCollapsed(standbyMode);
    const willExpand = wasCollapsed && !isStandbyModeCollapsed(mode);
    standbyMode = mode;
    if (isStandbyModeCollapsed(mode)) {
      await setStandbyPointerPassthrough(!standbyWakeActive);
    } else {
      standbyWakeActive = false;
      clearStandbyWakeTimer();
      await setStandbyPointerPassthrough(false);
    }

    if (willExpand && syncWindow) {
      root?.classList.remove("idle", "armed");
      root?.classList.add("opening");
      if (collapsedButton) collapsedButton.disabled = true;
      await wait(STANDBY_OPENING_FADE_MS);
      await syncStandbyWindowMode(mode);
      await wait(STANDBY_OPENING_SETTLE_MS);
      applyVisualMode(mode, { opening: true });
      await nextPaint();
      root?.classList.remove("opening");
      if (collapsedButton) collapsedButton.disabled = false;
    } else {
      applyVisualMode(mode);
    }

    if (collapsedButton) collapsedButton.hidden = !isStandbyModeCollapsed(mode);
    if (form) {
      const isTypeIn = mode === "typein";
      form.setAttribute("aria-hidden", isTypeIn ? "false" : "true");
    }
    if (thinkingPanel) {
      const isThinking = mode === "thinking";
      thinkingPanel.setAttribute("aria-hidden", isThinking ? "false" : "true");
    }
    if (responsePanel) {
      const isResponse = mode === "response";
      responsePanel.setAttribute("aria-hidden", isResponse ? "false" : "true");
    }
    if (detailPanel) {
      const isDetail = mode === "detail";
      detailPanel.setAttribute("aria-hidden", isDetail ? "false" : "true");
    }
    if (input) input.disabled = mode !== "typein";
    if (detailInput) detailInput.disabled = mode !== "detail";
    if (sendBtn) sendBtn.hidden = mode !== "typein";
    if (detailSendBtn) detailSendBtn.hidden = mode !== "detail";
    if (mode === "typein" && focusInput) {
      window.setTimeout(() => input?.focus(), willExpand ? 180 : 80);
    }
    if (mode === "detail") {
      renderDetailContext();
      window.setTimeout(() => detailInput?.focus(), 80);
    }
    updateActionButtonMode();
    updateDetailActionButtonMode();
    clearCollapsedArmTimer();
    clearCollapsedLongPressTimer();
    scheduleCollapsedIdle();

    if (syncWindow && !willExpand) {
      await syncStandbyWindowMode(mode);
    }
    if (emitMode) announceMode(mode);
  };

  const setMode = (nextMode, options = {}) => {
    return applyStandbyMode(nextMode, options);
  };

  const submit = async () => {
    const text = (input?.value || "").trim();
    if (!text) {
      await invoke?.("restore_main_from_standby").catch(() => {});
      await setMode("collapsed", { syncWindow: false, emitMode: false });
      return;
    }
    await setMode("thinking", { syncWindow: true });
    input.value = "";
    updateActionButtonMode();
    await events.emit?.("standby:submit", { text }).catch(() => {});
  };

  const submitDetail = async () => {
    const text = (detailInput?.value || "").trim();
    if (!text) {
      root?.classList.toggle("listening");
      await events.emit?.("standby:voice-toggle", {}).catch(() => {});
      return;
    }
    await setMode("thinking", { syncWindow: true });
    detailInput.value = "";
    updateDetailActionButtonMode();
    await events.emit?.("standby:submit", { text, source: "detail" }).catch(() => {});
  };

  const goTypeIn = async () => {
    standbyWakeActive = false;
    clearStandbyWakeTimer();
    await setMode("typein", { syncWindow: true, emitMode: true, focusInput: true });
  };

  const goDetail = async () => {
    renderDetailContext();
    await setMode("detail", { syncWindow: true, emitMode: true });
  };

  const collapseStandby = async (event) => {
    if (event?.isComposing) return;
    if (standbyMode === "collapsed" || root?.classList.contains("closing")) return;
    event?.preventDefault?.();
    event?.stopPropagation?.();
    root?.classList.remove("listening");
    standbyWakeActive = false;
    clearStandbyWakeTimer();
    clearCollapsedArmTimer();
    clearCollapsedLongPressTimer();
    setCollapsedArmed(false);
    setCollapsedIdle(false);
    input?.blur();
    detailInput?.blur();
    root?.classList.add("closing");
    collapseAnimationTimer = window.setTimeout(() => {
      collapseAnimationTimer = null;
      setMode("collapsed", { syncWindow: true });
    }, STANDBY_COLLAPSE_ANIMATION_MS);
  };

  window.__igpuSetStandbyExpanded = (nextExpanded) => {
    setMode(nextExpanded, { syncWindow: false, emitMode: false });
  };

  collapsedButton?.addEventListener("click", (event) => {
    if (ignoreNextCollapsedClick) {
      ignoreNextCollapsedClick = false;
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    goTypeIn();
  });
  sendBtn?.addEventListener("click", async () => {
    if (standbyMode !== "typein") return;
    if (hasDraftText()) {
      await submit();
      return;
    }
    root?.classList.toggle("listening");
    await events.emit?.("standby:voice-toggle", {}).catch(() => {});
  });

  form?.addEventListener("submit", (event) => {
    event.preventDefault();
    submit();
  });

  input?.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      collapseStandby(event);
    }
  });

  input?.addEventListener("input", updateActionButtonMode);

  detailSendBtn?.addEventListener("click", async () => {
    if (standbyMode !== "detail") return;
    await submitDetail();
  });

  detailForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    submitDetail();
  });

  detailInput?.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      collapseStandby(event);
      return;
    }
    if (event.key.toLowerCase() === "m" && event.shiftKey) {
      event.preventDefault();
      root?.classList.toggle("listening");
      events.emit?.("standby:voice-toggle", {}).catch(() => {});
    }
  });

  detailInput?.addEventListener("input", updateDetailActionButtonMode);

  detailCloseBtn?.addEventListener("click", () => {
    setMode("response", { syncWindow: true });
  });

  collapsedButton?.addEventListener("pointerenter", () => {
    pointerInsideStandby = true;
    wakeCollapsedStandby();
    scheduleCollapsedArmed();
  });

  collapsedButton?.addEventListener("pointerleave", () => {
    pointerInsideStandby = false;
    clearCollapsedArmTimer();
    clearCollapsedLongPressTimer();
    setCollapsedArmed(false);
    scheduleCollapsedIdle();
  });

  collapsedButton?.addEventListener("pointermove", () => {
    if (root.classList.contains("idle")) wakeCollapsedStandby();
  });

  collapsedButton?.addEventListener("pointerdown", () => {
    pointerInsideStandby = true;
    wakeCollapsedStandby();
    scheduleCollapsedArmed();
    beginCollapsedLongPress();
  });

  collapsedButton?.addEventListener("pointerup", clearCollapsedLongPressTimer);
  collapsedButton?.addEventListener("pointercancel", clearCollapsedLongPressTimer);

  collapsedButton?.addEventListener("focus", wakeCollapsedStandby);

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      collapseStandby(event);
    } else if (event.key === "Enter" && standbyMode === "response") {
      event.preventDefault();
      goTypeIn();
    } else if (event.key.toLowerCase() === "d" && event.shiftKey && standbyMode === "response") {
      event.preventDefault();
      goDetail();
    }
  });

  await events.listen?.("standby:set-mode", (event) => {
    const mode = standbyModeFromInput(event.payload?.mode) || (event.payload?.expanded ? "typein" : "collapsed");
    if (nativeModeEchoToIgnore === mode) {
      nativeModeEchoToIgnore = null;
      return;
    }
    setMode(mode, { syncWindow: false, emitMode: false });
  }).catch(() => {});

  await events.listen?.("standby:wake", (event) => {
    const timeoutMs = Number(event.payload?.timeoutMs) || STANDBY_WAKE_CLICK_WINDOW_MS;
    (async () => {
      if (!isStandbyModeCollapsed(standbyMode)) {
        await setMode("collapsed", { syncWindow: false, emitMode: false });
      }
      await setCollapsedWakeWindow(true, timeoutMs);
    })().catch(() => {});
  }).catch(() => {});

  await events.listen?.("standby:set-response", (event) => {
    const text = String(event.payload?.text || "").trim();
    const detailText = String(event.payload?.detailText || "").trim();
    const question = String(event.payload?.question || "").trim();
    const title = String(event.payload?.title || "").trim();
    lastResponseSummary = text || lastResponseSummary;
    lastDetailAnswer = detailText || text || lastDetailAnswer;
    lastDetailQuestion = question || lastDetailQuestion;
    lastDetailTitle = title || lastDetailTitle;
    if (responseText) {
      responseText.textContent = text || "I have a response ready.";
    }
    renderDetailContext();
    const responseMode = standbyModeFromInput(event.payload?.mode) || (standbyMode === "detail" ? "detail" : "response");
    const modeChanged = standbyMode !== responseMode;
    setMode(responseMode, { syncWindow: modeChanged, emitMode: modeChanged });
  }).catch(() => {});

  await events.listen?.("standby:set-tone", (event) => {
    const tone = event.payload?.tone || "pink";
    root?.setAttribute("data-tone", tone);
  }).catch(() => {});

  window.addEventListener("blur", () => {
    root?.classList.remove("listening");
  });

  scheduleCollapsedIdle();
  await appWindow?.setAlwaysOnTop?.(true).catch(() => {});
});
