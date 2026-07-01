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
const STANDBY_VOICE_SILENCE_MS = 1400;

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
  const responseEnterBtn = document.getElementById("standbyResponseEnterBtn");
  const detailPanel = document.getElementById("standbyDetailPanel");
  const detailTitle = document.getElementById("standbyDetailTitle");
  const detailQuestion = document.getElementById("standbyDetailQuestion");
  const detailContext = document.getElementById("standbyDetailContext");
  const detailContextList = document.getElementById("standbyDetailContextList");
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
  let lastDetailContext = [];
  let lastDetailRouteTrace = [];
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
  let standbySpeechRecognition = null;
  let standbySpeechSilenceTimer = null;
  let standbyVoiceActive = false;
  let standbyVoiceStopRequested = false;
  let standbyVoiceSent = false;
  let standbyVoiceBaseText = "";
  let standbyVoiceFinalText = "";
  let standbyVoiceInterimText = "";
  const TYPEWRITER_DELAY_MS = 16;
  let detailAutoScrollFrame = 0;

  const scrollDetailContextToBottom = () => {
    if (!detailContextList) return;
    if (detailAutoScrollFrame) return;
    detailAutoScrollFrame = window.requestAnimationFrame(() => {
      detailAutoScrollFrame = 0;
      detailContextList.scrollTop = detailContextList.scrollHeight;
    });
  };

  const prefersReducedMotion = () => {
    try {
      return window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
    } catch {
      return false;
    }
  };

  const commonPrefixByChar = (left, right) => {
    const leftChars = Array.from(String(left || ""));
    const rightChars = Array.from(String(right || ""));
    let index = 0;
    while (index < leftChars.length && index < rightChars.length && leftChars[index] === rightChars[index]) {
      index += 1;
    }
    return rightChars.slice(0, index).join("");
  };

  const createTypewriter = (fallbackText = "", onCommit = null) => {
    let element = null;
    let target = "";
    let visible = "";
    let timer = null;

    const stop = () => {
      if (timer) {
        window.clearTimeout(timer);
        timer = null;
      }
    };

    const commit = () => {
      if (!element) return;
      element.textContent = visible || fallbackText;
      onCommit?.();
    };

    const tick = () => {
      timer = null;
      if (visible === target) {
        commit();
        return;
      }
      const targetChars = Array.from(target);
      const visibleLength = Array.from(visible).length;
      visible = targetChars.slice(0, Math.min(targetChars.length, visibleLength + 1)).join("");
      commit();
      if (visible !== target) {
        timer = window.setTimeout(tick, TYPEWRITER_DELAY_MS);
      }
    };

    return {
      reset(nextText = "") {
        stop();
        target = String(nextText || "");
        visible = "";
        commit();
      },
      setElement(nextElement) {
        element = nextElement || null;
        commit();
      },
      setTarget(nextText, options = {}) {
        const nextTarget = String(nextText || fallbackText || "");
        const instant = Boolean(options.instant) || prefersReducedMotion();
        if (nextTarget === target && visible === target) {
          commit();
          return;
        }
        target = nextTarget;
        if (instant) {
          stop();
          visible = target;
          commit();
          return;
        }
        if (!target.startsWith(visible)) {
          visible = commonPrefixByChar(visible, target);
        }
        commit();
        if (!timer && visible !== target) {
          timer = window.setTimeout(tick, TYPEWRITER_DELAY_MS);
        }
      },
      value() {
        return visible;
      },
    };
  };

  const responseTypewriter = createTypewriter("I have a response ready.");
  const detailTypewriter = createTypewriter("I have a response ready.", scrollDetailContextToBottom);

  const hasDraftText = () => Boolean((input?.value || "").trim());
  const hasDetailDraftText = () => Boolean((detailInput?.value || "").trim());
  const normalizeSpeechText = (text) => (text || "").replace(/\s+/g, " ").trim();

  const updateActionButtonMode = () => {
    if (!sendBtn) return;
    const typing = hasDraftText();
    form?.classList.toggle("has-draft", typing);
    sendBtn.classList.toggle("typing", typing);
    sendBtn.classList.toggle("default", !typing);
    const voiceLabel = standbyVoiceActive ? "Listening..." : "Voice mode";
    sendBtn.title = typing ? "Send" : voiceLabel;
    sendBtn.setAttribute("aria-label", typing ? "Send" : voiceLabel);
  };

  const updateDetailActionButtonMode = () => {
    if (!detailSendBtn) return;
    const typing = hasDetailDraftText();
    detailSendBtn.classList.toggle("typing", typing);
    detailSendBtn.classList.toggle("default", !typing);
    detailSendBtn.title = typing ? "Reply" : "Voice mode";
    detailSendBtn.setAttribute("aria-label", typing ? "Reply" : "Voice mode");
  };

  const focusStandbyInput = ({ select = false } = {}) => {
    if (!input || standbyMode !== "typein") return;
    const focusOnce = () => {
      if (standbyMode !== "typein" || input.disabled) return;
      window.focus();
      input.focus({ preventScroll: true });
      const cursor = input.value.length;
      if (select && input.value) {
        input.select?.();
      } else {
        input.setSelectionRange?.(cursor, cursor);
      }
    };
    focusOnce();
    window.requestAnimationFrame(focusOnce);
    [50, 140, 260].forEach((delay) => window.setTimeout(focusOnce, delay));
  };

  const renderDetailContext = () => {
    if (detailTitle) detailTitle.textContent = lastDetailTitle || "Game Companion";
    if (detailQuestion) detailQuestion.textContent = lastDetailQuestion || "Your last question";
    if (detailContext && detailContextList) {
      detailContextList.textContent = "";
      const contextItems = Array.isArray(lastDetailContext) ? lastDetailContext.slice(-10) : [];
      const conversationItems = contextItems.filter((item) => item.role !== "status");
      const statusItems = lookupTraceToDetailItems(lastDetailRouteTrace);
      const currentQuestion = normalizeSpeechText(lastDetailQuestion || "");
      const currentAnswer = String(lastDetailAnswer || lastResponseSummary || detailTypewriter.value() || "").trim();
      const chatItems = [...conversationItems];
      const lastContextItem = chatItems[chatItems.length - 1];
      if (currentQuestion && !(lastContextItem?.role === "user" && lastContextItem?.text === currentQuestion)) {
        chatItems.push({ role: "user", text: currentQuestion });
      }
      chatItems.push(...statusItems);
      if (currentAnswer) {
        chatItems.push({ role: "assistant", text: currentAnswer, live: true });
      }
      detailContext.hidden = chatItems.length === 0;
      let statusLabelShown = false;
      for (const item of chatItems) {
        const row = document.createElement("div");
        const isAssistant = item.role === "assistant";
        const isStatus = item.role === "status";
        row.className = `standby-detail-chat-message ${isStatus ? "status" : isAssistant ? "assistant" : "user"}`;
        if (isStatus && item.stage) row.dataset.stage = item.stage;

        const role = document.createElement("span");
        role.className = "standby-detail-chat-role";
        if (isStatus && statusLabelShown) {
          role.classList.add("standby-detail-chat-role-repeat");
          role.setAttribute("aria-hidden", "true");
          role.textContent = "Status";
        } else {
          role.textContent = isStatus ? "Status" : isAssistant ? "AI" : "You";
        }
        if (isStatus) statusLabelShown = true;

        const text = document.createElement("span");
        text.className = "standby-detail-chat-bubble";
        if (item.live) {
          detailTypewriter.setElement(text);
          detailTypewriter.setTarget(item.text || "I have a response ready.");
        } else {
          text.textContent = item.text || "";
        }

        row.append(role, text);
        detailContextList.appendChild(row);
      }
      scrollDetailContextToBottom();
    }
    if (detailAnswer) detailAnswer.textContent = detailTypewriter.value() || lastDetailAnswer || lastResponseSummary || "";
  };

  const normalizeDetailContext = (value) => {
    if (!Array.isArray(value)) return [];
    return value
      .map((item) => {
        const roleValue = String(item?.role || "").trim().toLowerCase();
        const role = roleValue === "assistant" ? "assistant" : roleValue === "status" ? "status" : "user";
        return {
          role,
          text: String(item?.text || "").replace(/\s+/g, " ").trim(),
        };
      })
      .filter((item) => item.text)
      .slice(-10);
  };

  const normalizeLookupTrace = (value) => {
    const items = Array.isArray(value) ? value : [];
    return items
      .map((item) => ({
        stage: String(item?.stage || "").trim(),
        summary: String(item?.summary || "").replace(/\s+/g, " ").trim(),
        detail: String(item?.detail || "").replace(/\s+/g, " ").trim(),
        time: String(item?.time || "").trim(),
      }))
      .filter((item) => item.summary || item.detail)
      .slice(-8);
  };

  const lookupTraceToDetailItems = (trace) => normalizeLookupTrace(trace)
    .map((item) => {
      const lines = [];
      if (item.summary) lines.push(item.summary);
      if (item.detail && item.detail !== item.summary) lines.push(item.detail);
      return {
        role: "status",
        text: lines.join("\n"),
        stage: item.stage,
      };
    })
    .filter((item) => item.text);

  const appendDetailContextMessage = (role, text) => {
    const normalizedRole = role === "assistant" ? "assistant" : "user";
    const normalizedText = String(text || "").replace(/\s+/g, " ").trim();
    if (!normalizedText || normalizedText === "Your last question" || normalizedText === "I have a response ready.") return;
    const items = normalizeDetailContext(lastDetailContext);
    const last = items[items.length - 1];
    if (last?.role !== normalizedRole || last?.text !== normalizedText) {
      items.push({ role: normalizedRole, text: normalizedText });
    }
    lastDetailContext = items.slice(-10);
  };

  const promoteCurrentDetailExchange = () => {
    appendDetailContextMessage("user", lastDetailQuestion);
    appendDetailContextMessage("assistant", lastDetailAnswer || lastResponseSummary || detailTypewriter.value());
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
    await setStandbyPointerPassthrough(false);
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
      startStandbyVoice().catch(() => {});
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
      await setStandbyPointerPassthrough(false);
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
      window.setTimeout(() => focusStandbyInput(), willExpand ? 180 : 80);
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

  const standbyVoiceSpokenText = (includeInterim = true) => normalizeSpeechText(
    `${standbyVoiceFinalText} ${includeInterim ? standbyVoiceInterimText : ""}`
  );

  const composeStandbyVoiceText = (includeInterim = true) => {
    const spoken = standbyVoiceSpokenText(includeInterim);
    return normalizeSpeechText([standbyVoiceBaseText, spoken].filter(Boolean).join(" "));
  };

  const clearStandbySpeechSilence = () => {
    if (!standbySpeechSilenceTimer) return;
    window.clearTimeout(standbySpeechSilenceTimer);
    standbySpeechSilenceTimer = null;
  };

  const setStandbyVoiceActive = (active) => {
    standbyVoiceActive = Boolean(active);
    root?.classList.toggle("listening", standbyVoiceActive);
    updateActionButtonMode();
  };

  const updateStandbyVoiceInput = () => {
    if (!input || standbyVoiceSent) return;
    input.value = composeStandbyVoiceText(true);
    input.focus();
    const cursor = input.value.length;
    input.setSelectionRange?.(cursor, cursor);
    updateActionButtonMode();
  };

  const stopStandbySpeechRecognition = (abort = false) => {
    standbyVoiceStopRequested = true;
    clearStandbySpeechSilence();
    if (!standbySpeechRecognition) return;
    try {
      if (abort) {
        standbySpeechRecognition.abort?.();
      } else {
        standbySpeechRecognition.stop?.();
      }
    } catch (err) {
      console.warn("Standby speech stop failed:", err);
    }
  };

  const resetStandbyVoiceDraft = ({ restoreBase = false } = {}) => {
    const baseText = standbyVoiceBaseText;
    standbyVoiceBaseText = "";
    standbyVoiceFinalText = "";
    standbyVoiceInterimText = "";
    standbyVoiceSent = false;
    if (restoreBase && input) {
      input.value = baseText;
      updateActionButtonMode();
    }
  };

  const sendStandbyVoiceTranscript = async ({ includeInterim = false } = {}) => {
    if (standbyVoiceSent) return false;
    const spoken = standbyVoiceSpokenText(includeInterim);
    if (!spoken) return false;

    const messageText = composeStandbyVoiceText(includeInterim);
    standbyVoiceSent = true;
    stopStandbySpeechRecognition(true);
    setStandbyVoiceActive(false);
    resetStandbyVoiceDraft();
    if (input) {
      input.value = messageText;
      updateActionButtonMode();
    }
    await submit();
    return true;
  };

  const scheduleStandbyVoiceAutoSend = () => {
    clearStandbySpeechSilence();
    if (!standbyVoiceSpokenText(true) || standbyVoiceSent) return;
    standbySpeechSilenceTimer = window.setTimeout(() => {
      if (!standbyVoiceActive || standbyVoiceSent || !standbyVoiceSpokenText(true)) return;
      sendStandbyVoiceTranscript({ includeInterim: true }).catch((err) => {
        console.warn("Standby voice auto-send failed:", err);
        setStandbyVoiceActive(false);
      });
    }, STANDBY_VOICE_SILENCE_MS);
  };

  const startStandbyVoice = async () => {
    if (standbyMode !== "typein") {
      await setMode("typein", { syncWindow: true, emitMode: true, focusInput: true });
    }
    if (standbyVoiceActive) return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn("Standby voice is unavailable in this WebView.");
      return;
    }

    standbyVoiceBaseText = (input?.value || "").trim();
    standbyVoiceFinalText = "";
    standbyVoiceInterimText = "";
    standbyVoiceSent = false;
    standbyVoiceStopRequested = false;
    clearStandbySpeechSilence();

    try {
      const recognition = new SpeechRecognition();
      recognition.lang = localStorage.getItem("speech-lang") || "zh-TW";
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;

      recognition.addEventListener("start", () => {
        setStandbyVoiceActive(true);
        input?.focus();
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
          standbyVoiceFinalText = normalizeSpeechText(`${standbyVoiceFinalText} ${finalText}`);
        }
        standbyVoiceInterimText = normalizeSpeechText(interimText);
        updateStandbyVoiceInput();
        scheduleStandbyVoiceAutoSend();
      });

      recognition.addEventListener("error", (event) => {
        const error = event.error || "speech-recognition";
        if (error === "no-speech") return;
        console.warn("Standby voice error:", error);
        if (["not-allowed", "service-not-allowed", "audio-capture"].includes(error)) {
          standbyVoiceStopRequested = true;
          setStandbyVoiceActive(false);
        }
      });

      recognition.addEventListener("end", () => {
        if (standbySpeechRecognition !== recognition) return;
        standbySpeechRecognition = null;
        standbyVoiceInterimText = "";
        updateStandbyVoiceInput();
        if (standbyVoiceSent) return;
        if (standbyVoiceSpokenText(false)) {
          sendStandbyVoiceTranscript({ includeInterim: false }).catch(() => {});
          return;
        }
        if (!standbyVoiceStopRequested && standbyVoiceActive) {
          window.setTimeout(() => {
            if (standbyVoiceStopRequested || !standbyVoiceActive || standbySpeechRecognition) return;
            startStandbyVoice().catch(() => {});
          }, 150);
          return;
        }
        setStandbyVoiceActive(false);
        resetStandbyVoiceDraft({ restoreBase: true });
      });

      recognition.start();
      standbySpeechRecognition = recognition;
      setStandbyVoiceActive(true);
    } catch (err) {
      console.warn("Standby voice start failed:", err);
      setStandbyVoiceActive(false);
      resetStandbyVoiceDraft({ restoreBase: true });
    }
  };

  const stopStandbyVoice = async ({ submitTranscript = true } = {}) => {
    if (!standbyVoiceActive && !standbySpeechRecognition) return;
    if (submitTranscript && await sendStandbyVoiceTranscript({ includeInterim: true })) return;
    standbyVoiceStopRequested = true;
    stopStandbySpeechRecognition(true);
    setStandbyVoiceActive(false);
    resetStandbyVoiceDraft({ restoreBase: true });
  };

  const toggleStandbyVoice = async () => {
    if (standbyVoiceActive || standbySpeechRecognition) {
      await stopStandbyVoice({ submitTranscript: true });
      return;
    }
    await startStandbyVoice();
  };

  const submit = async () => {
    const text = (input?.value || "").trim();
    if (standbyVoiceActive || standbySpeechRecognition) {
      standbyVoiceSent = true;
      stopStandbySpeechRecognition(true);
      clearStandbySpeechSilence();
      setStandbyVoiceActive(false);
      standbyVoiceBaseText = "";
      standbyVoiceFinalText = "";
      standbyVoiceInterimText = "";
    }
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
    promoteCurrentDetailExchange();
    lastDetailQuestion = text;
    lastDetailAnswer = "";
    lastResponseSummary = "";
    lastDetailRouteTrace = [];
    detailTypewriter.reset();
    await setMode("detail", { syncWindow: false, emitMode: true });
    detailInput.value = "";
    updateDetailActionButtonMode();
    renderDetailContext();
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
    stopStandbyVoice({ submitTranscript: false }).catch(() => {});
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

  window.__igpuSetStandbyMode = (nextMode, options = {}) => {
    setMode(nextMode, {
      syncWindow: false,
      emitMode: false,
      focusInput: Boolean(options?.focusInput),
    });
  };

  window.__igpuFocusStandbyInput = () => {
    focusStandbyInput();
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
    await toggleStandbyVoice();
  });

  form?.addEventListener("submit", (event) => {
    event.preventDefault();
    submit();
  });

  input?.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      collapseStandby(event);
      return;
    }
    if (!event.isComposing && event.key.toLowerCase() === "m" && event.shiftKey) {
      event.preventDefault();
      event.stopPropagation();
      toggleStandbyVoice().catch(() => {});
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

  responseEnterBtn?.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (standbyMode !== "response") return;
    goTypeIn();
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
    } else if (!event.isComposing && event.key.toLowerCase() === "m" && event.shiftKey && standbyMode === "typein") {
      event.preventDefault();
      toggleStandbyVoice().catch(() => {});
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
    setMode(mode, {
      syncWindow: false,
      emitMode: false,
      focusInput: Boolean(event.payload?.focusInput),
    });
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

  await events.listen?.("standby:focus-input", () => {
    focusStandbyInput();
  }).catch(() => {});

  await events.listen?.("standby:set-response", (event) => {
    const text = String(event.payload?.text || "").trim();
    const detailText = String(event.payload?.detailText || "").trim();
    const question = String(event.payload?.question || "").trim();
    const title = String(event.payload?.title || "").trim();
    lastDetailRouteTrace = normalizeLookupTrace(event.payload?.routeTrace || event.payload?.trace || lastDetailRouteTrace);
    const isNewQuestion = Boolean(question && question !== lastDetailQuestion);
    if (isNewQuestion) {
      responseTypewriter.reset();
      detailTypewriter.reset();
    }
    lastResponseSummary = text || lastResponseSummary;
    lastDetailAnswer = detailText || text || lastDetailAnswer;
    lastDetailQuestion = question || lastDetailQuestion;
    lastDetailContext = normalizeDetailContext(event.payload?.context);
    lastDetailTitle = title || lastDetailTitle;
    if (responseText) {
      responseTypewriter.setElement(responseText);
      responseTypewriter.setTarget(text || "I have a response ready.");
    }
    renderDetailContext();
    const responseMode = standbyModeFromInput(event.payload?.mode) || (standbyMode === "detail" ? "detail" : "response");
    const modeChanged = standbyMode !== responseMode;
    setMode(responseMode, { syncWindow: modeChanged, emitMode: modeChanged });
  }).catch(() => {});

  await events.listen?.("standby:set-lookup-status", (event) => {
    lastDetailRouteTrace = normalizeLookupTrace(event.payload?.trace || []);
    if (standbyMode === "detail") {
      renderDetailContext();
    }
  }).catch(() => {});

  await events.listen?.("standby:clear-response", () => {
    lastResponseSummary = "";
    lastDetailTitle = "Game Companion";
    lastDetailQuestion = "Your last question";
    lastDetailContext = [];
    lastDetailRouteTrace = [];
    lastDetailAnswer = "";
    responseTypewriter.reset();
    detailTypewriter.reset();
    if (responseText) responseText.textContent = "";
    if (detailAnswer) detailAnswer.textContent = "";
    if (detailQuestion) detailQuestion.textContent = "";
    if (detailContextList) detailContextList.textContent = "";
    if (detailContext) detailContext.hidden = true;
    if (standbyMode === "response" || standbyMode === "detail") {
      setMode("typein", { syncWindow: true, emitMode: true, focusInput: true });
    }
  }).catch(() => {});

  await events.listen?.("standby:set-tone", (event) => {
    const tone = event.payload?.tone || "pink";
    root?.setAttribute("data-tone", tone);
  }).catch(() => {});

  window.addEventListener("blur", () => {
    if (!standbyVoiceActive) root?.classList.remove("listening");
  });

  scheduleCollapsedIdle();
  await appWindow?.setAlwaysOnTop?.(true).catch(() => {});
});
