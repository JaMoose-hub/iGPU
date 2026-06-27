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
  const appWindow = tauri.window?.getCurrentWindow?.() || tauri.window?.Window?.getCurrent?.();
  const events = tauri.event || {};
  const invoke = tauri.core?.invoke;
  const virtualCursor = initVirtualCursor({ events, invoke, windowLabel: "gamepath" });

  const API_BASE = "http://127.0.0.1:8000";
  const entryList = document.getElementById("entryList");
  const searchInput = document.getElementById("searchInput");
  const refreshBtn = document.getElementById("refreshBtn");
  const closeBtn = document.getElementById("closeBtn");
  const entryCount = document.getElementById("entryCount");
  const currentGame = document.getElementById("currentGame");
  const emptyTemplate = document.getElementById("emptyTemplate");
  const dragHandle = document.querySelector(".gamepath-drag");

  let activeGame = localStorage.getItem("currentGameId") || "";
  let lastEntries = [];
  let searchTimer = 0;
  let lastDragAt = 0;
  let lastDbRevision = "";
  let isLoading = false;
  let pendingLoad = false;

  const iconSvg = {
    ask: '<path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4Z"/><path d="M9 9h6"/><path d="M9 13h4"/>',
    database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
    refresh: '<path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M3 21v-5h5"/><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/>',
    search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    trash: '<path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v5"/><path d="M14 11v5"/>',
    x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>'
  };

  const renderIcon = (target, name) => {
    if (!target || !iconSvg[name]) return;
    target.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${iconSvg[name]}</svg>`;
  };

  const hydrateIcons = (root = document) => {
    root.querySelectorAll(".icon[data-icon]").forEach((icon) => renderIcon(icon, icon.dataset.icon));
  };

  const applyOpacity = (value) => {
    const numeric = Math.max(10, Math.min(100, Number(value) || 100));
    const scale = numeric / 100;
    document.documentElement.style.setProperty("--ui-opacity", scale.toFixed(2));
    document.documentElement.style.setProperty("--ui-bg-alpha", (scale * 0.97).toFixed(3));
  };

  const formatDate = (value) => {
    if (!value) return "";
    const normalized = String(value).replace(/([+-]\d{2})(\d{2})$/, "$1:$2");
    const date = new Date(normalized);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString([], { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
  };

  const previewText = (value, maxLength = 760) => {
    const text = String(value || "").replace(/\n{3,}/g, "\n\n").trim();
    if (text.length <= maxLength) return text;
    return `${text.slice(0, maxLength).trimEnd()}...`;
  };

  const dbRevision = (health = {}) => [
    health.gamepath_entry_count ?? 0,
    health.gamepath_last_updated_at || ""
  ].join(":");

  const fetchHealth = async () => {
    const response = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Health failed: ${response.status}`);
    return response.json();
  };

  const updateStats = async () => {
    try {
      const health = await fetchHealth();
      entryCount.textContent = String(health.gamepath_entry_count ?? "-");
      currentGame.textContent = activeGame || "All";
      return dbRevision(health);
    } catch {
      entryCount.textContent = "-";
      currentGame.textContent = activeGame || "All";
      return "";
    }
  };

  const fetchEntries = async () => {
    const query = (searchInput?.value || "").trim();
    if (query) {
      const response = await fetch(`${API_BASE}/gamepath/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          game_id: activeGame || null,
          spoiler_level: "full",
          limit: 20
        })
      });
      if (!response.ok) throw new Error(`Search failed: ${response.status}`);
      return (await response.json()).results || [];
    }

    const params = new URLSearchParams({ limit: "20" });
    if (activeGame) params.set("game_id", activeGame);
    const response = await fetch(`${API_BASE}/gamepath/recent?${params.toString()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Recent failed: ${response.status}`);
    return (await response.json()).results || [];
  };

  const askEntry = async (entry) => {
    const title = entry.title || entry.question || "GamePath";
    const query = (searchInput?.value || "").trim();
    const scope = query ? `「${query}」相關的內容` : "最相關的內容";
    const payload = {
      ...entry,
      message: `根據 GamePath「${title}」，請幫我整理下一步`
    };
    payload.message = `請從攻略庫「${title}」中提取${scope}，整理成無暴雷的下一步提示，不要貼整篇攻略`;
    localStorage.setItem("igpu-gamepath-ask-pending", JSON.stringify(payload));
    await events.emit?.("gamepath:ask", payload).catch((err) => {
      console.warn("GamePath ask event failed:", err);
    });
    await invoke?.("set_virtual_cursor_active_window", { label: "main" }).catch(() => {});
    await invoke?.("hide_gamepath_window").catch(() => appWindow?.hide?.().catch(() => {}));
  };

  const deleteEntry = async (entry, button) => {
    if (!entry?.id || !button) return;
    const label = button.querySelector(".delete-label");
    if (button.dataset.confirming !== "1") {
      button.dataset.confirming = "1";
      if (label) label.textContent = "Sure?";
      window.setTimeout(() => {
        if (!button.isConnected || button.dataset.confirming !== "1") return;
        button.dataset.confirming = "0";
        if (label) label.textContent = "Delete";
      }, 2600);
      return;
    }

    button.disabled = true;
    try {
      const response = await fetch(`${API_BASE}/gamepath/${entry.id}`, { method: "DELETE" });
      if (!response.ok) throw new Error(`Delete failed: ${response.status}`);
      await load();
    } catch (err) {
      button.disabled = false;
      button.dataset.confirming = "0";
      if (label) label.textContent = "Delete";
      console.warn("GamePath delete failed:", err);
    }
  };

  const renderEntry = (entry) => {
    const article = document.createElement("article");
    article.className = "entry-card";

    const header = document.createElement("div");
    header.className = "entry-header";
    const titleWrap = document.createElement("div");
    titleWrap.className = "entry-title-wrap";
    const title = document.createElement("h2");
    title.textContent = entry.title || entry.question || "Untitled";
    const meta = document.createElement("div");
    meta.className = "entry-meta";
    meta.textContent = [
      entry.game_id,
      entry.source_type,
      entry.trust_state ? `trust:${entry.trust_state}` : "",
      Number(entry.dispute_count || 0) > 0 ? `disputes:${entry.dispute_count}` : "",
      entry.spoiler_level ? `spoiler:${entry.spoiler_level}` : "",
      formatDate(entry.updated_at || entry.created_at)
    ].filter(Boolean).join(" · ");
    titleWrap.append(title, meta);

    const ask = document.createElement("button");
    ask.className = "ask-entry-btn";
    ask.title = "Ask using this GamePath entry";
    ask.dataset.cursorLabel = "Ask GamePath entry";
    ask.innerHTML = '<span class="icon" data-icon="ask"></span><span>Ask</span>';
    ask.addEventListener("click", () => askEntry(entry));

    const remove = document.createElement("button");
    remove.className = "delete-entry-btn";
    remove.title = "Delete this GamePath entry";
    remove.dataset.cursorLabel = "Delete GamePath entry";
    remove.innerHTML = '<span class="icon" data-icon="trash"></span><span class="delete-label">Delete</span>';
    remove.addEventListener("click", () => deleteEntry(entry, remove));

    const actions = document.createElement("div");
    actions.className = "entry-actions";
    actions.append(ask, remove);
    header.append(titleWrap, actions);
    article.appendChild(header);

    const question = document.createElement("p");
    question.className = "entry-question";
    question.textContent = entry.question || "";
    article.appendChild(question);

    const summary = document.createElement("p");
    summary.className = "entry-summary";
    summary.textContent = previewText(entry.relevant_excerpt || entry.snippet || entry.answer_summary || "");
    if (entry.large_entry) {
      summary.dataset.large = "1";
    }
    article.appendChild(summary);

    const tags = String(entry.tags || "").split(",").map((tag) => tag.trim()).filter(Boolean);
    if (tags.length || entry.markdown_path) {
      const chipRow = document.createElement("div");
      chipRow.className = "entry-chips";
      if (entry.trust_state) {
        const chip = document.createElement("span");
        chip.className = `trust-chip trust-${entry.trust_state}`;
        chip.textContent = entry.trust_state;
        chipRow.appendChild(chip);
      }
      for (const tag of tags) {
        const chip = document.createElement("span");
        chip.textContent = tag;
        chipRow.appendChild(chip);
      }
      if (entry.markdown_path) {
        const chip = document.createElement("span");
        chip.textContent = entry.markdown_path;
        chip.title = entry.markdown_path;
        chipRow.appendChild(chip);
      }
      article.appendChild(chipRow);
    }

    hydrateIcons(article);
    return article;
  };

  const render = (entries = lastEntries) => {
    lastEntries = entries;
    entryList.innerHTML = "";
    if (!entries.length) {
      const empty = emptyTemplate.content.cloneNode(true);
      hydrateIcons(empty);
      entryList.appendChild(empty);
      return;
    }
    for (const entry of entries) {
      entryList.appendChild(renderEntry(entry));
    }
  };

  const load = async () => {
    if (isLoading) {
      pendingLoad = true;
      return;
    }
    isLoading = true;
    const revision = await updateStats();
    try {
      render(await fetchEntries());
      if (revision) lastDbRevision = revision;
      virtualCursor.refresh?.();
    } catch (err) {
      entryList.innerHTML = "";
      const error = document.createElement("section");
      error.className = "empty-state";
      error.innerHTML = '<span class="icon" data-icon="database"></span><p>GamePath 讀取失敗</p><small></small>';
      error.querySelector("small").textContent = err?.message || String(err);
      hydrateIcons(error);
      entryList.appendChild(error);
    } finally {
      isLoading = false;
      if (pendingLoad) {
        pendingLoad = false;
        window.setTimeout(load, 0);
      }
    }
  };

  const refreshIfChanged = async () => {
    if (isLoading) return;
    try {
      const health = await fetchHealth();
      const revision = dbRevision(health);
      entryCount.textContent = String(health.gamepath_entry_count ?? "-");
      currentGame.textContent = activeGame || "All";
      if (!lastDbRevision) {
        lastDbRevision = revision;
        return;
      }
      if (revision && revision !== lastDbRevision) {
        await load();
      }
    } catch {
      // Keep the current list visible; the next explicit refresh will show an error if needed.
    }
  };

  hydrateIcons();
  applyOpacity(localStorage.getItem("ui-opacity") || "100");
  await load();

  const startWindowDrag = (event) => {
    if (event.button !== 0 || event.target?.closest?.("button")) return;
    const now = Date.now();
    if (now - lastDragAt < 180) return;
    lastDragAt = now;
    event.preventDefault();
    event.stopPropagation();
    invoke?.("begin_gamepath_window_drag").catch((err) => {
      console.warn("GamePath native drag failed:", err);
      appWindow?.startDragging?.().catch((fallbackErr) => {
        console.warn("GamePath drag failed:", fallbackErr);
      });
    });
  };

  dragHandle?.addEventListener("pointerdown", startWindowDrag);
  dragHandle?.addEventListener("mousedown", startWindowDrag);

  searchInput?.addEventListener("input", () => {
    window.clearTimeout(searchTimer);
    searchTimer = window.setTimeout(load, 180);
  });
  refreshBtn?.addEventListener("click", load);
  closeBtn?.addEventListener("click", async () => {
    localStorage.setItem("igpu-virtual-cursor-active-window", "main");
    await events.emit?.("virtual-cursor-active-window", { window: "main", source: "gamepath" }).catch(() => {});
    if (invoke) {
      await invoke("hide_gamepath_window").catch(() => appWindow?.hide?.().catch(() => {}));
    } else {
      await appWindow?.hide?.().catch(() => {});
    }
  });

  await events.listen?.("gamepath:context", (event) => {
    activeGame = event.payload?.game || localStorage.getItem("currentGameId") || "";
    load();
  }).catch(() => {});

  await events.listen?.("gamepath:changed", (event) => {
    activeGame = event.payload?.game ?? localStorage.getItem("currentGameId") ?? activeGame;
    load();
  }).catch(() => {});

  await events.listen?.("companion-window-shown", () => {
    applyOpacity(localStorage.getItem("ui-opacity") || "100");
    activeGame = localStorage.getItem("currentGameId") || activeGame;
    load();
    virtualCursor.refresh?.();
    events.emit?.("gamepath:ready", { window: "gamepath" }).catch(() => {});
    requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
  }).catch(() => {});

  await events.listen?.("ui-opacity-updated", (event) => {
    applyOpacity(event.payload?.opacity);
  }).catch(() => {});

  window.addEventListener("focus", refreshIfChanged);
  window.setInterval(refreshIfChanged, 3500);

  window.addEventListener("storage", (event) => {
    if (event.key === "ui-opacity") applyOpacity(event.newValue);
    if (event.key === "currentGameId") {
      activeGame = event.newValue || "";
      load();
    }
    if (event.key === "igpu-gamepath-last-change") {
      activeGame = localStorage.getItem("currentGameId") || activeGame;
      load();
    }
  });
});
