document.addEventListener("DOMContentLoaded", async () => {
  const tauri = window.__TAURI__ || {};
  const appWindow = tauri.window?.getCurrentWindow?.() || tauri.window?.Window?.getCurrent?.();
  const events = tauri.event || {};
  const invoke = tauri.core?.invoke;
  const opener = tauri.opener || {};
  const Webview = tauri.webview?.Webview;

  const gameInput = document.getElementById("gameInput");
  const keywordInput = document.getElementById("keywordInput");
  const searchForm = document.getElementById("searchForm");
  const externalBtn = document.getElementById("externalBtn");
  const closeBtn = document.getElementById("closeBtn");
  const quickChips = document.getElementById("quickChips");
  const browserHost = document.getElementById("browserHost");
  const browserEmpty = document.getElementById("browserEmpty");
  const browserBackBtn = document.getElementById("browserBackBtn");
  const browserForwardBtn = document.getElementById("browserForwardBtn");
  const browserReloadBtn = document.getElementById("browserReloadBtn");
  const browserHomeBtn = document.getElementById("browserHomeBtn");
  const browserAddress = document.getElementById("browserAddress");
  const engineBtns = [...document.querySelectorAll(".engine-btn")];

  const QUICK_KEYWORDS = ["\u653b\u7565", "\u4efb\u52d9", "\u7269\u54c1", "\u5730\u5716", "\u914d\u88dd", "boss", "location", "wiki"];
  const EMBEDDED_BROWSER_LABEL = "game-search-results";
  const browserNavButtons = [browserBackBtn, browserForwardBtn, browserReloadBtn, browserHomeBtn].filter(Boolean);
  let embeddedBrowser = null;
  let lastEmbeddedUrl = "";

  const iconSvg = {
    back: '<path d="m15 18-6-6 6-6"/><path d="M21 12H9"/>',
    external: '<path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>',
    forward: '<path d="m9 18 6-6-6-6"/><path d="M3 12h12"/>',
    gamepad: '<path d="M6 12h4"/><path d="M8 10v4"/><path d="M15 13h.01"/><path d="M18 11h.01"/><path d="M5 8h14a4 4 0 0 1 3.8 5.3l-1 3A3 3 0 0 1 17 18l-2-2H9l-2 2a3 3 0 0 1-4.8-1.7l-1-3A4 4 0 0 1 5 8Z"/>',
    home: '<path d="m3 10 9-7 9 7"/><path d="M5 10v10h14V10"/><path d="M9 20v-6h6v6"/>',
    panel: '<rect width="18" height="14" x="3" y="5" rx="2"/><path d="M3 10h18"/><path d="M8 21h8"/><path d="M12 17v4"/>',
    reload: '<path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/>',
    search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
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

  const activeEngine = () => document.querySelector(".engine-btn.active")?.dataset.engine || "google";

  const setEngine = (engine) => {
    engineBtns.forEach((button) => button.classList.toggle("active", button.dataset.engine === engine));
    localStorage.setItem("igpu-game-search-engine", engine);
  };

  const normalize = (value) => (value || "").replace(/\s+/g, " ").trim();

  const buildQuery = () => normalize([gameInput.value, keywordInput.value].filter(Boolean).join(" "));

  const buildUrl = (query, engine) => {
    const encoded = encodeURIComponent(query);
    if (engine === "youtube") return `https://www.youtube.com/results?search_query=${encoded}`;
    if (engine === "wiki") return `https://www.google.com/search?q=${encoded}%20wiki`;
    return `https://www.google.com/search?q=${encoded}`;
  };

  const openExternalSearch = async (url) => {
    if (opener.openUrl) {
      await opener.openUrl(url);
    } else {
      window.open(url, "_blank", "noopener,noreferrer");
    }
  };

  const setBrowserControls = (enabled) => {
    browserNavButtons.forEach((button) => {
      button.disabled = !enabled;
    });
  };

  const setBrowserAddress = (text) => {
    if (browserAddress) browserAddress.textContent = text || "No page loaded";
  };

  const setBrowserMessage = (message) => {
    if (!browserEmpty) return;
    document.body.classList.remove("browser-open");
    setBrowserControls(false);
    browserEmpty.innerHTML = '<span class="icon" data-icon="panel"></span><span></span>';
    browserEmpty.querySelector("span:last-child").textContent = message;
    hydrateIcons(browserEmpty);
  };

  const browserBounds = () => {
    const rect = browserHost.getBoundingClientRect();
    return {
      x: Math.max(0, Math.round(rect.left)),
      y: Math.max(0, Math.round(rect.top)),
      width: Math.max(240, Math.round(rect.width)),
      height: Math.max(160, Math.round(rect.height))
    };
  };

  const positionEmbeddedBrowser = async () => {
    if (!embeddedBrowser || !browserHost) return;
    const bounds = browserBounds();
    await embeddedBrowser.setPosition?.({ type: "Logical", x: bounds.x, y: bounds.y }).catch(() => {});
    await embeddedBrowser.setSize?.({ type: "Logical", width: bounds.width, height: bounds.height }).catch(() => {});
  };

  const closeEmbeddedBrowser = async () => {
    if (!Webview) {
      embeddedBrowser = null;
      return;
    }
    const existing = embeddedBrowser || await Webview.getByLabel?.(EMBEDDED_BROWSER_LABEL).catch(() => null);
    if (existing) {
      await existing.close?.().catch(() => {});
    }
    embeddedBrowser = null;
  };

  const openEmbeddedSearch = async (url) => {
    if (!Webview || !appWindow || !browserHost) {
      setBrowserMessage("Embedded browser is unavailable in this build.");
      return;
    }

    await closeEmbeddedBrowser();
    await new Promise((resolve) => window.setTimeout(resolve, 120));

    const bounds = browserBounds();
    lastEmbeddedUrl = url;
    setBrowserAddress(url);
    document.body.classList.add("browser-open");
    embeddedBrowser = new Webview(appWindow, EMBEDDED_BROWSER_LABEL, {
      url,
      x: bounds.x,
      y: bounds.y,
      width: bounds.width,
      height: bounds.height,
      focus: true
    });
    embeddedBrowser.once?.("tauri://created", async () => {
      await positionEmbeddedBrowser();
      await embeddedBrowser.setFocus?.().catch(() => {});
      setBrowserControls(true);
    });
    embeddedBrowser.once?.("tauri://error", async () => {
      embeddedBrowser = null;
      lastEmbeddedUrl = "";
      setBrowserMessage("Embedded browser failed to open. Use External if this site blocks WebView.");
    });
  };

  const runBrowserCommand = async (command, args = {}) => {
    if (!invoke) {
      setBrowserMessage("Browser controls are unavailable in this build.");
      return;
    }
    await invoke(command, args).catch((err) => {
      console.warn("Browser command failed:", command, err);
    });
  };

  const openSearch = async (query = buildQuery(), engine = activeEngine(), mode = "embedded") => {
    const finalQuery = normalize(query);
    if (!finalQuery) {
      keywordInput?.focus();
      return;
    }
    const url = buildUrl(finalQuery, engine);
    if (mode === "external") {
      await openExternalSearch(url);
    } else {
      await openEmbeddedSearch(url);
    }
  };

  const renderChips = () => {
    quickChips.innerHTML = "";
    for (const keyword of QUICK_KEYWORDS) {
      const button = document.createElement("button");
      button.className = "chip";
      button.type = "button";
      button.textContent = keyword;
      button.addEventListener("click", () => {
        keywordInput.value = keyword;
        openSearch();
      });
      quickChips.appendChild(button);
    }
  };

  const setGame = (game) => {
    gameInput.value = normalize(game || localStorage.getItem("currentGameId") || "");
  };

  hydrateIcons();
  setBrowserControls(false);
  setBrowserAddress("No page loaded");
  applyOpacity(localStorage.getItem("ui-opacity") || "100");
  setEngine(localStorage.getItem("igpu-game-search-engine") || "google");
  setGame(localStorage.getItem("currentGameId") || "");
  keywordInput.value = localStorage.getItem("igpu-game-search-pending-keyword") || "";
  localStorage.removeItem("igpu-game-search-pending-keyword");
  renderChips();

  engineBtns.forEach((button) => {
    button.addEventListener("click", () => setEngine(button.dataset.engine));
  });

  searchForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    await openSearch().catch((err) => {
      console.warn("Search open failed:", err);
      setBrowserMessage("Search failed inside Game Search.");
    });
  });

  externalBtn?.addEventListener("click", async () => {
    await openSearch(buildQuery(), activeEngine(), "external").catch((err) => {
      console.warn("External search open failed:", err);
    });
  });

  browserBackBtn?.addEventListener("click", async () => {
    await runBrowserCommand("game_search_browser_back");
  });

  browserForwardBtn?.addEventListener("click", async () => {
    await runBrowserCommand("game_search_browser_forward");
  });

  browserReloadBtn?.addEventListener("click", async () => {
    await runBrowserCommand("game_search_browser_reload");
  });

  browserHomeBtn?.addEventListener("click", async () => {
    if (!lastEmbeddedUrl) return;
    setBrowserAddress(lastEmbeddedUrl);
    await runBrowserCommand("game_search_browser_navigate", { url: lastEmbeddedUrl });
  });

  closeBtn?.addEventListener("click", async () => {
    await closeEmbeddedBrowser();
    if (invoke) {
      await invoke("hide_search_window").catch(() => appWindow?.hide?.().catch(() => {}));
    } else {
      await appWindow?.hide?.().catch(() => {});
    }
  });

  window.addEventListener("resize", () => {
    window.requestAnimationFrame(() => {
      positionEmbeddedBrowser();
    });
  });

  await events.listen?.("search:context", (event) => {
    setGame(event.payload?.game || "");
    if (event.payload?.keyword) keywordInput.value = event.payload.keyword;
    localStorage.removeItem("igpu-game-search-pending-keyword");
    keywordInput?.focus();
    positionEmbeddedBrowser();
  }).catch(() => {});

  await events.listen?.("companion-window-shown", async () => {
    applyOpacity(localStorage.getItem("ui-opacity") || "100");
    window.dispatchEvent(new Event("resize"));
    await positionEmbeddedBrowser();
    requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
  }).catch(() => {});

  await events.listen?.("ui-opacity-updated", (event) => {
    applyOpacity(event.payload?.opacity);
  }).catch(() => {});

  window.addEventListener("storage", (event) => {
    if (event.key === "ui-opacity") applyOpacity(event.newValue);
    if (event.key === "currentGameId") setGame(event.newValue);
  });

  keywordInput?.focus();
});
