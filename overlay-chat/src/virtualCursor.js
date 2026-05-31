const isTextEntry = (element) => {
  if (!element) return false;
  const tag = element.tagName?.toLowerCase();
  return tag === "input" || tag === "textarea" || element.isContentEditable;
};

const isSelectElement = (element) => {
  return element?.tagName?.toLowerCase() === "select";
};

const isScrollableElement = (element) => {
  if (!element) return false;
  const style = window.getComputedStyle(element);
  const overflowY = style.overflowY || style.overflow;
  return (
    element.dataset?.cursorScroll !== undefined ||
    ((overflowY === "auto" || overflowY === "scroll") && element.scrollHeight > element.clientHeight + 4)
  );
};

const closestScrollable = (element) => {
  const scrollable = element?.closest?.("[data-cursor-scroll]");
  if (scrollable && isScrollableElement(scrollable)) return scrollable;
  let current = element;
  while (current && current !== document.body && current !== document.documentElement) {
    if (isScrollableElement(current)) return current;
    current = current.parentElement;
  }
  return null;
};

const closestTextEntry = (element) => (
  element?.closest?.("input, textarea, [contenteditable]")
);

const isVirtualCursorControlKey = (event) => {
  const code = event.code || "";
  const key = (event.key || "").toLowerCase();
  return [
    "KeyW",
    "KeyA",
    "KeyS",
    "KeyD",
    "ArrowUp",
    "ArrowLeft",
    "ArrowDown",
    "ArrowRight",
    "Numpad8",
    "Numpad4",
    "Numpad2",
    "Numpad6",
    "Numpad5",
    "Tab",
    "Enter",
    "NumpadEnter",
    "Space"
  ].includes(code) || key === " " || key === "spacebar";
};

const isUsableElement = (element) => {
  if (!element || element.disabled || element.ariaDisabled === "true") return false;
  const rect = element.getBoundingClientRect();
  const style = window.getComputedStyle(element);
  return (
    rect.width >= 8 &&
    rect.height >= 8 &&
    rect.bottom > 0 &&
    rect.right > 0 &&
    rect.left < window.innerWidth &&
    rect.top < window.innerHeight &&
    style.visibility !== "hidden" &&
    style.display !== "none" &&
    Number(style.opacity || 1) > 0.05
  );
};

const elementLabel = (element) => {
  if (!element) return "";
  return (
    element.getAttribute("aria-label") ||
    element.title ||
    element.dataset.cursorLabel ||
    element.innerText ||
    element.value ||
    element.placeholder ||
    element.id ||
    element.tagName ||
    ""
  ).replace(/\s+/g, " ").trim();
};

export function initVirtualCursor({
  events,
  invoke,
  windowLabel = "main",
  controller = false,
  onStateChange = null
} = {}) {
  const layer = document.createElement("div");
  layer.className = "virtual-cursor-layer";
  layer.innerHTML = `
    <div class="virtual-cursor-target-ring"></div>
    <div class="virtual-cursor-badge"></div>
    <div class="virtual-cursor-pointer"></div>
  `;
  document.body.appendChild(layer);

  const pointer = layer.querySelector(".virtual-cursor-pointer");
  const ring = layer.querySelector(".virtual-cursor-target-ring");
  const badge = layer.querySelector(".virtual-cursor-badge");

  let enabled = false;
  let activeWindow = localStorage.getItem("igpu-virtual-cursor-active-window") || "main";
  let x = Math.round(window.innerWidth / 2);
  let y = Math.round(window.innerHeight / 2);
  let targetElement = null;
  let lastToggleAt = 0;
  let typingElement = null;
  let typingCleanup = null;
  let textEntrySuppressUntil = 0;
  let textEntrySuppressHardStop = 0;
  let pendingLocalActivate = false;
  let windowMoveActive = false;
  let scrollDragElement = null;
  let lastScrollDragY = y;
  let lastActivationKey = "";
  let lastActivationAt = 0;
  const textEntrySuppressedKeys = new Set();

  const isActiveWindow = () => enabled && activeWindow === windowLabel;
  const isTypingWithVirtualCursor = () => Boolean(typingElement);
  const isScrollDragging = () => Boolean(scrollDragElement);

  const targets = () => [
    ...document.querySelectorAll([
      "button",
      "input",
      "textarea",
      "select",
      "a[href]",
      "[role='button']",
      "[tabindex]:not([tabindex='-1'])",
      "[data-cursor-action]"
    ].join(","))
  ].filter((element) => !layer.contains(element) && isUsableElement(element));

  const clearTarget = () => {
    targetElement?.classList?.remove("virtual-cursor-target");
    targetElement = null;
    if (ring) ring.style.display = "none";
    if (badge) badge.style.display = "none";
  };

  const setTextEntryMode = (active) => {
    invoke?.("set_virtual_cursor_text_entry", { active: Boolean(active) }).catch((err) => {
      console.warn("Virtual cursor text-entry state failed:", err);
    });
  };

  const syncLocalPosition = () => {
    invoke?.("set_virtual_cursor_window_position", {
      label: windowLabel,
      x,
      y
    }).catch((err) => {
      console.warn("Virtual cursor position sync failed:", err);
    });
  };

  const beginTextEntrySuppression = () => {
    textEntrySuppressUntil = Date.now() + 800;
    textEntrySuppressHardStop = Date.now() + 2500;
    textEntrySuppressedKeys.clear();
  };

  const suppressTextEntryKeyIfNeeded = (event) => {
    if (!typingElement || !isVirtualCursorControlKey(event)) return false;
    const id = event.code || event.key || "";
    const now = Date.now();
    const shouldSuppress = now < textEntrySuppressUntil
      || (now < textEntrySuppressHardStop && textEntrySuppressedKeys.size > 0)
      || (now < textEntrySuppressHardStop && event.repeat);
    if (!shouldSuppress) return false;
    if (id) textEntrySuppressedKeys.add(id);
    event.preventDefault();
    event.stopImmediatePropagation();
    return true;
  };

  const leaveTextEntry = ({ blur = false } = {}) => {
    const element = typingElement;
    if (typingCleanup) typingCleanup();
    typingCleanup = null;
    typingElement = null;
    textEntrySuppressUntil = 0;
    textEntrySuppressHardStop = 0;
    textEntrySuppressedKeys.clear();
    document.body.classList.remove("virtual-cursor-text-mode");
    setTextEntryMode(false);
    if (blur && element && (document.activeElement === element || element.contains?.(document.activeElement))) {
      element.blur?.();
    }
  };

  const setScrollDragMode = (element) => {
    scrollDragElement?.classList?.remove("virtual-cursor-scroll-entry");
    scrollDragElement = element && isScrollableElement(element) ? element : null;
    lastScrollDragY = y;
    scrollDragElement?.classList?.add("virtual-cursor-scroll-entry");
    document.body.classList.toggle("virtual-cursor-scroll-drag", Boolean(scrollDragElement));
    updateVisuals();
  };

  const scrollDragBy = (deltaY) => {
    if (!scrollDragElement || Math.abs(deltaY) < 0.05) return;
    const maxScroll = Math.max(0, scrollDragElement.scrollHeight - scrollDragElement.clientHeight);
    if (!maxScroll) return;
    const ratio = Math.max(1.6, scrollDragElement.scrollHeight / Math.max(1, scrollDragElement.clientHeight));
    scrollDragElement.scrollTop = Math.max(
      0,
      Math.min(maxScroll, scrollDragElement.scrollTop + deltaY * ratio)
    );
  };

  const enterTextEntry = (element) => {
    if (!element) return;
    setScrollDragMode(null);
    leaveTextEntry();
    typingElement = element;
    typingElement.classList.add("virtual-cursor-text-entry");
    document.body.classList.add("virtual-cursor-text-mode");
    beginTextEntrySuppression();
    setTextEntryMode(true);

    const focusElement = () => {
      if (!typingElement) return;
      typingElement.scrollIntoView?.({ block: "nearest", inline: "nearest" });
      typingElement.focus?.({ preventScroll: true });
      typingElement.focus?.({ preventScroll: true });
      if (typingElement?.tagName?.toLowerCase() === "input" && typingElement.type !== "range") {
        const end = String(typingElement.value || "").length;
        typingElement.setSelectionRange?.(end, end);
      }
      if (typingElement?.tagName?.toLowerCase() === "textarea") {
        const end = String(typingElement.value || "").length;
        typingElement.setSelectionRange?.(end, end);
      }
    };

    if (invoke) {
      invoke("focus_virtual_cursor_text_entry", { label: windowLabel, x, y }).catch((err) => {
        console.warn("Virtual cursor physical text focus failed:", err);
        return invoke("focus_companion_window", { label: windowLabel }).catch(() => {});
      }).finally(() => {
        focusElement();
        window.requestAnimationFrame(focusElement);
        window.setTimeout(focusElement, 60);
        window.setTimeout(focusElement, 160);
      });
    } else {
      focusElement();
      window.requestAnimationFrame(focusElement);
    }

    typingCleanup = () => {
      element.classList.remove("virtual-cursor-text-entry");
    };
  };

  document.addEventListener("focusin", (event) => {
    if (!enabled || !isActiveWindow()) return;
    const element = closestTextEntry(event.target);
    if (!element || layer.contains(element)) return;
    typingElement = element;
    typingElement.classList.add("virtual-cursor-text-entry");
    document.body.classList.add("virtual-cursor-text-mode");
    setTextEntryMode(true);
  }, true);

  document.addEventListener("virtual-cursor-activate", (event) => {
    if (event.detail?.action !== "window-move" || event.detail?.windowLabel !== windowLabel) return;
    event.preventDefault();
    event.stopPropagation();
    setScrollDragMode(null);
    leaveTextEntry({ blur: true });
    setWindowMoveMode(!windowMoveActive);
  });

  document.addEventListener("virtual-cursor-activate", (event) => {
    if (event.detail?.action !== "scroll-drag" || event.detail?.windowLabel !== windowLabel) return;
    const target = closestScrollable(event.target);
    if (!target) return;
    event.preventDefault();
    event.stopPropagation();
    leaveTextEntry({ blur: true });
    if (scrollDragElement === target) {
      setScrollDragMode(null);
    } else {
      setWindowMoveMode(false);
      setScrollDragMode(target);
    }
  });

  const setTarget = (element) => {
    if (targetElement === element) return;
    clearTarget();
    targetElement = element || null;
    targetElement?.classList?.add("virtual-cursor-target");
  };

  const targetSelector = [
    "button",
    "input",
    "textarea",
    "select",
    "a[href]",
    "[role='button']",
    "[tabindex]:not([tabindex='-1'])",
    "[data-cursor-action]"
  ].join(",");

  const targetAtCursor = () => {
    const elements = document.elementsFromPoint?.(x, y) || [];
    for (const element of elements) {
      if (layer.contains(element)) continue;
      const textEntry = closestTextEntry(element);
      if (textEntry && !layer.contains(textEntry) && isUsableElement(textEntry)) return textEntry;
    }
    for (const element of elements) {
      if (layer.contains(element)) continue;
      const target = element.closest?.(targetSelector);
      if (target && !layer.contains(target) && isUsableElement(target)) return target;
    }
    for (const element of elements) {
      if (layer.contains(element)) continue;
      const scrollable = closestScrollable(element);
      if (scrollable && !layer.contains(scrollable) && isUsableElement(scrollable)) return scrollable;
    }
    return null;
  };

  const nearestTarget = () => {
    const directTarget = targetAtCursor();
    if (directTarget) return directTarget;

    let best = null;
    let bestDistance = Infinity;
    for (const element of targets()) {
      const rect = element.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      const inside = x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
      const distance = inside ? 0 : Math.hypot(centerX - x, centerY - y);
      if (distance < bestDistance) {
        best = element;
        bestDistance = distance;
      }
    }
    return bestDistance <= 90 ? best : null;
  };

  const updateVisuals = () => {
    document.body.classList.toggle("virtual-cursor-mode", enabled);
    document.body.classList.toggle("virtual-cursor-active-window", isActiveWindow());
    document.body.classList.toggle("virtual-cursor-window-move", isActiveWindow() && windowMoveActive);
    document.body.classList.toggle("virtual-cursor-scroll-drag", isActiveWindow() && isScrollDragging());
    if (!isActiveWindow()) {
      clearTarget();
      return;
    }

    pointer.style.left = `${x}px`;
    pointer.style.top = `${y}px`;

    const nextTarget = scrollDragElement && isActiveWindow() ? scrollDragElement : nearestTarget();
    setTarget(nextTarget);
    if (!targetElement) return;

    const rect = targetElement.getBoundingClientRect();
    ring.style.display = "block";
    ring.style.left = `${rect.left}px`;
    ring.style.top = `${rect.top}px`;
    ring.style.width = `${rect.width}px`;
    ring.style.height = `${rect.height}px`;

    const label = elementLabel(targetElement);
    const badgeText = windowMoveActive ? "Move window" : isScrollDragging() ? "Scroll" : label;
    if (badgeText) {
      badge.textContent = badgeText;
      badge.style.display = "block";
      badge.style.left = `${Math.max(8, Math.min(rect.left, window.innerWidth - 230))}px`;
      badge.style.top = `${Math.max(8, Math.min(rect.bottom + 8, window.innerHeight - 30))}px`;
    }
  };

  const applyRender = (payload = {}) => {
    const nextEnabled = Boolean(payload.enabled);
    const previousY = y;
    if (!nextEnabled) {
      leaveTextEntry({ blur: true });
      setScrollDragMode(null);
    }

    enabled = nextEnabled;
    activeWindow = payload.activeWindow || "main";
    localStorage.setItem("igpu-virtual-cursor-enabled", enabled ? "true" : "false");
    localStorage.setItem("igpu-virtual-cursor-active-window", activeWindow);
    windowMoveActive = payload.movingWindow === windowLabel;
    if (Number.isFinite(Number(payload.x))) x = Number(payload.x);
    if (Number.isFinite(Number(payload.y))) y = Number(payload.y);

    if (!isActiveWindow()) {
      leaveTextEntry({ blur: true });
      setScrollDragMode(null);
    } else if (scrollDragElement) {
      scrollDragBy(y - previousY);
    }
    onStateChange?.(enabled);
    updateVisuals();
  };

  const setEnabled = (nextEnabled) => {
    invoke?.("set_virtual_cursor_global_controls", { enabled: Boolean(nextEnabled) }).catch((err) => {
      console.warn("Virtual cursor enable failed:", err);
    });
  };

  const setWindowMoveMode = (active) => {
    invoke?.("set_virtual_cursor_window_move", {
      label: windowLabel,
      active: Boolean(active)
    }).catch((err) => {
      console.warn("Virtual cursor window move failed:", err);
    });
  };

  const toggleEnabled = () => {
    const now = Date.now();
    if (now - lastToggleAt < 250) return;
    lastToggleAt = now;
    setEnabled(!enabled);
  };

  const focusElement = (element) => {
    if (!element) return;
    element.scrollIntoView?.({ block: "nearest", inline: "nearest" });
    const rect = element.getBoundingClientRect();
    x = Math.round(rect.left + rect.width / 2);
    y = Math.round(rect.top + rect.height / 2);
    syncLocalPosition();
    updateVisuals();
  };

  const focusRelative = (direction) => {
    const list = targets().sort((a, b) => {
      const ar = a.getBoundingClientRect();
      const br = b.getBoundingClientRect();
      return ar.top - br.top || ar.left - br.left;
    });
    if (!list.length) return;
    const current = Math.max(0, list.indexOf(targetElement));
    const nextIndex = targetElement
      ? (current + direction + list.length) % list.length
      : 0;
    focusElement(list[nextIndex]);
  };

  const activateSelect = (element) => {
    const options = [...(element.options || [])].filter((option) => !option.disabled);
    if (!options.length) return;
    const currentIndex = Math.max(0, options.findIndex((option) => option.value === element.value));
    const nextOption = options[(currentIndex + 1) % options.length];
    if (!nextOption) return;
    focusElement(element);
    element.value = nextOption.value;
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  };

  const activationKeyFor = (element) => {
    if (!element) return "";
    return [
      windowLabel,
      element.id || "",
      element.dataset?.cursorAction || "",
      element.getAttribute?.("aria-label") || "",
      element.title || "",
      element.tagName || ""
    ].join("|");
  };

  const activateTarget = () => {
    if (scrollDragElement) {
      setScrollDragMode(null);
      return;
    }
    const element = targetAtCursor() || targetElement || nearestTarget();
    if (!element) return;
    const now = Date.now();
    const activationKey = activationKeyFor(element);
    if (activationKey && activationKey === lastActivationKey && now - lastActivationAt < 450) {
      return;
    }
    lastActivationKey = activationKey;
    lastActivationAt = now;
    if (element.dataset?.cursorAction) {
      const activationEvent = new CustomEvent("virtual-cursor-activate", {
        detail: { x, y, windowLabel, action: element.dataset.cursorAction },
        bubbles: true,
        cancelable: true
      });
      if (!element.dispatchEvent(activationEvent)) return;
    }
    focusElement(element);
    if (isSelectElement(element)) {
      activateSelect(element);
      return;
    }
    if (isTextEntry(element)) {
      enterTextEntry(element);
      return;
    }
    element.click?.();
  };

  const handleKeyDown = (event) => {
    if (event.key === "F11") {
      event.preventDefault();
      return;
    }
    if (!enabled || !isActiveWindow()) return;

    if (isTypingWithVirtualCursor()) {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopImmediatePropagation();
        leaveTextEntry({ blur: true });
      }
      return;
    }

    if (isTextEntry(document.activeElement)) {
      document.activeElement.blur?.();
    }

    if (event.key === "Escape") {
      event.preventDefault();
      if (scrollDragElement) {
        setScrollDragMode(null);
        return;
      }
      if (windowMoveActive) {
        setWindowMoveMode(false);
        return;
      }
      setEnabled(false);
      return;
    }

    const key = event.key.toLowerCase();
    if (key === "arrowleft" || key === "a") {
      event.preventDefault();
      return;
    } else if (key === "arrowright" || key === "d") {
      event.preventDefault();
      return;
    } else if (key === "arrowup" || key === "w") {
      event.preventDefault();
      return;
    } else if (key === "arrowdown" || key === "s") {
      event.preventDefault();
      return;
    } else if (key === "tab") {
      event.preventDefault();
      focusRelative(event.shiftKey ? -1 : 1);
    } else if (key === "enter" || key === " " || key === "spacebar" || event.code === "Space") {
      event.preventDefault();
      pendingLocalActivate = true;
    }
  };

  const handleKeyUp = (event) => {
    if (typingElement && isVirtualCursorControlKey(event)) {
      const id = event.code || event.key || "";
      if (id) textEntrySuppressedKeys.delete(id);
      if (textEntrySuppressedKeys.size === 0) {
        textEntrySuppressUntil = Math.min(textEntrySuppressUntil, Date.now());
      }
      return;
    }

    if (!enabled || !isActiveWindow() || !pendingLocalActivate) return;
    if (event.key === "Enter" || event.key === " " || event.key === "Spacebar" || event.code === "Space") {
      event.preventDefault();
      pendingLocalActivate = false;
      if (!invoke) activateTarget();
    }
  };

  window.addEventListener("keydown", handleKeyDown, true);
  window.addEventListener("keyup", handleKeyUp, true);
  window.addEventListener("resize", () => window.requestAnimationFrame(updateVisuals));
  window.addEventListener("scroll", updateVisuals, true);

  events?.listen?.("virtual-cursor-toggle-request", () => {
    if (controller) toggleEnabled();
  }).catch(() => {});

  events?.listen?.("virtual-cursor-render", (event) => {
    applyRender(event.payload || {});
  }).catch(() => {});

  events?.listen?.("virtual-cursor-active-window", (event) => {
    const label = event.payload?.window || event.payload?.activeWindow;
    if (!label) return;
    activeWindow = label;
    localStorage.setItem("igpu-virtual-cursor-active-window", activeWindow);
    if (!enabled && localStorage.getItem("igpu-virtual-cursor-enabled") === "true") {
      enabled = true;
    }
    if (Number.isFinite(Number(event.payload?.x))) x = Number(event.payload.x);
    if (Number.isFinite(Number(event.payload?.y))) y = Number(event.payload.y);
    if (!isActiveWindow()) {
      leaveTextEntry({ blur: true });
      setScrollDragMode(null);
    }
    updateVisuals();
  }).catch(() => {});

  events?.listen?.("virtual-cursor-transfer", (event) => {
    if (event.payload?.window !== windowLabel) return;
    activeWindow = windowLabel;
    localStorage.setItem("igpu-virtual-cursor-active-window", activeWindow);
    if (!enabled && localStorage.getItem("igpu-virtual-cursor-enabled") === "true") {
      enabled = true;
    }
    if (Number.isFinite(Number(event.payload?.x))) x = Number(event.payload.x);
    if (Number.isFinite(Number(event.payload?.y))) y = Number(event.payload.y);
    leaveTextEntry({ blur: true });
    setScrollDragMode(null);
    updateVisuals();
  }).catch(() => {});

  events?.listen?.("virtual-cursor-frames-changed", () => {
    window.requestAnimationFrame(updateVisuals);
  }).catch(() => {});

  events?.listen?.("virtual-cursor-exit-text-entry", () => {
    leaveTextEntry({ blur: true });
  }).catch(() => {});

  events?.listen?.("virtual-cursor-action", (event) => {
    if (!enabled || event.payload?.activeWindow !== windowLabel) return;
    if (event.payload?.type === "target_next") {
      if (scrollDragElement) {
        setScrollDragMode(null);
        return;
      }
      focusRelative(1);
    } else if (event.payload?.type === "activate") {
      activateTarget();
    }
  }).catch(() => {});

  updateVisuals();

  return {
    isEnabled: () => enabled,
    getPosition: () => ({ x, y, enabled, activeWindow }),
    setEnabled,
    toggle: toggleEnabled,
    syncPosition: syncLocalPosition,
    setActiveWindow: (label) => (
      localStorage.setItem("igpu-virtual-cursor-active-window", label),
      invoke?.("set_virtual_cursor_active_window", { label }).catch((err) => {
        console.warn("Virtual cursor active-window sync failed:", err);
      })
    ),
    adoptActiveWindow: (label) => {
      activeWindow = label;
      localStorage.setItem("igpu-virtual-cursor-active-window", label);
      if (localStorage.getItem("igpu-virtual-cursor-enabled") === "true") {
        enabled = true;
      }
      updateVisuals();
    },
    refresh: updateVisuals
  };
}
