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
  const events = tauri.event || {};
  const invoke = tauri.core?.invoke;
  initVirtualCursor({ events, invoke, windowLabel: "tools" });

  const closeBtn = document.getElementById("closeBtn");
  const gameSelect = document.getElementById("gameSelect");
  const gameAutoBtn = document.getElementById("gameAutoBtn");
  const liveStateBtn = document.getElementById("liveStateBtn");
  const liveStateAnalyzeBtn = document.getElementById("liveStateAnalyzeBtn");
  const searchBtn = document.getElementById("searchBtn");
  const tasksBtn = document.getElementById("tasksBtn");
  const gamepathBtn = document.getElementById("gamepathBtn");
  const hudBtn = document.getElementById("hudBtn");
  const hudTestBtn = document.getElementById("hudTestBtn");
  const protectBtn = document.getElementById("protectBtn");
  const virtualCursorBtn = document.getElementById("virtualCursorBtn");
  const perfBtn = document.getElementById("perfBtn");
  const opacitySlider = document.getElementById("opacitySlider");
  const opacityValue = document.getElementById("opacityValue");

  const iconSvg = {
    blend: '<circle cx="9" cy="9" r="7"/><circle cx="15" cy="15" r="7"/>',
    crosshair: '<circle cx="12" cy="12" r="10"/><path d="M22 12h-4"/><path d="M6 12H2"/><path d="M12 6V2"/><path d="M12 22v-4"/>',
    cursor: '<path d="m4 4 7.07 16.97 2.51-7.39 7.39-2.51Z"/><path d="m13.58 13.58 5.84 5.84"/>',
    database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
    gauge: '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    list: '<path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
    radio: '<path d="M4.9 19.1C1 15.2 1 8.8 4.9 4.9"/><path d="M7.8 16.2a6 6 0 0 1 0-8.5"/><circle cx="12" cy="12" r="2"/><path d="M16.2 7.8a6 6 0 0 1 0 8.5"/><path d="M19.1 4.9c3.9 3.9 3.9 10.3 0 14.1"/>',
    search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
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
    renderIcon(button.querySelector(".icon"), iconName);
    const labelNode = button.querySelector(".button-label");
    if (labelNode) labelNode.textContent = label;
  };

  const applyOpacity = (value) => {
    const numeric = Math.max(10, Math.min(100, Number(value) || 100));
    const scale = numeric / 100;
    document.documentElement.style.setProperty("--ui-opacity", scale.toFixed(2));
    document.documentElement.style.setProperty("--ui-bg-alpha", (scale * 0.97).toFixed(3));
    if (opacitySlider) opacitySlider.value = String(numeric);
    if (opacityValue) opacityValue.textContent = String(numeric);
  };

  const emitCommand = (command, payload = {}) => {
    events.emit?.("tool-panel-command", { command, ...payload }).catch(() => {});
  };

  const setActive = (button, active) => {
    button?.classList.toggle("active", Boolean(active));
  };

  const applyState = (state = {}) => {
    const games = Array.isArray(state.games) ? state.games : [];
    if (gameSelect) {
      const previous = state.selectedGameId || "";
      gameSelect.innerHTML = "";
      for (const game of games.length ? games : [{ id: "", name: "Game" }]) {
        const option = document.createElement("option");
        option.value = String(game.id || "");
        option.textContent = String(game.name || game.id || "Game");
        gameSelect.appendChild(option);
      }
      if (![...gameSelect.options].some((option) => option.value === previous)) {
        const option = document.createElement("option");
        option.value = previous;
        option.textContent = state.selectedGameName || previous || "Game";
        gameSelect.appendChild(option);
      }
      gameSelect.value = previous;
    }

    const auto = Boolean(state.autoMode);
    setActive(gameAutoBtn, auto);
    setButtonContent(gameAutoBtn, auto ? "radio" : "square", auto ? "Auto" : "Manual");

    const live = state.liveStateStatus || {};
    const liveEnabled = Boolean(state.liveStateEnabled);
    const liveStatus = String(live.status || (liveEnabled ? "Watching" : "Off"));
    liveStateBtn?.classList.toggle("thinking", liveStatus === "Thinking");
    liveStateBtn?.classList.toggle("error", liveStatus === "Error" || liveStatus === "capture_failed");
    setActive(liveStateBtn, liveEnabled && !["Off", "Error"].includes(liveStatus));
    setButtonContent(
      liveStateBtn,
      liveEnabled ? "radio" : "square",
      liveStatus === "Thinking" ? "Think" : liveEnabled ? "Watch" : "Live"
    );
    if (liveStateAnalyzeBtn) {
      liveStateAnalyzeBtn.disabled = liveStatus === "Thinking";
    }

    setActive(protectBtn, state.captureProtectionEnabled);
    setButtonContent(protectBtn, state.captureProtectionEnabled ? "shield" : "shield-off", "Protect");
    setActive(virtualCursorBtn, state.virtualCursorEnabled);
    setActive(perfBtn, state.perfEnabled);
    applyOpacity(state.opacity || localStorage.getItem("ui-opacity") || 100);
  };

  hydrateIcons();
  applyOpacity(localStorage.getItem("ui-opacity") || 100);

  closeBtn?.addEventListener("click", () => invoke?.("hide_tools_window").catch(() => {}));
  gameSelect?.addEventListener("change", () => {
    const option = gameSelect.selectedOptions?.[0];
    emitCommand("game_select", { game_id: gameSelect.value || "", name: option?.textContent || "" });
  });
  gameAutoBtn?.addEventListener("click", () => emitCommand("game_auto_toggle"));
  liveStateBtn?.addEventListener("click", () => emitCommand("live_toggle"));
  liveStateAnalyzeBtn?.addEventListener("click", () => emitCommand("live_scan"));
  searchBtn?.addEventListener("click", () => emitCommand("open_search"));
  tasksBtn?.addEventListener("click", () => emitCommand("open_tasks"));
  gamepathBtn?.addEventListener("click", () => emitCommand("open_gamepath"));
  hudBtn?.addEventListener("click", () => emitCommand("hud_clear"));
  hudTestBtn?.addEventListener("click", () => emitCommand("hud_test"));
  protectBtn?.addEventListener("click", () => emitCommand("protect_toggle"));
  virtualCursorBtn?.addEventListener("click", () => emitCommand("cursor_toggle"));
  perfBtn?.addEventListener("click", () => emitCommand("perf_toggle"));
  opacitySlider?.addEventListener("input", (event) => emitCommand("opacity_set", { opacity: event.target.value }));

  await events.listen?.("tool-panel-state", (event) => applyState(event.payload || {})).catch(() => {});
  await events.listen?.("ui-opacity-updated", (event) => applyOpacity(event.payload?.opacity)).catch(() => {});
  events.emit?.("tool-panel-ready", {}).catch(() => {});
  emitCommand("state_request");
});
