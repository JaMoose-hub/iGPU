document.addEventListener("DOMContentLoaded", async () => {
  const tauri = window.__TAURI__ || {};
  const appWindow = tauri.window?.getCurrentWindow?.();
  const events = tauri.event || {};
  const invoke = tauri.core?.invoke;

  const TASK_STORAGE_KEY = "igpu-task-log-v1";
  const taskList = document.getElementById("taskList");
  const closeBtn = document.getElementById("closeBtn");
  const refreshBtn = document.getElementById("refreshBtn");
  const clearDoneBtn = document.getElementById("clearDoneBtn");
  const searchInput = document.getElementById("searchInput");
  const emptyTemplate = document.getElementById("emptyTemplate");
  const filterBtns = [...document.querySelectorAll(".filter-btn")];

  const applyOpacity = (value) => {
    const numeric = Math.max(10, Math.min(100, Number(value) || 100));
    const scale = numeric / 100;
    document.documentElement.style.setProperty("--ui-opacity", scale.toFixed(2));
    document.documentElement.style.setProperty("--ui-bg-alpha", (scale * 0.97).toFixed(3));
  };

  const iconSvg = {
    check: '<path d="M20 6 9 17l-5-5"/>',
    flag: '<path d="M4 22V4"/><path d="M4 4h12l-1 4 1 4H4"/>',
    list: '<path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
    refresh: '<path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M3 21v-5h5"/><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/>',
    search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    trash: '<path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/>',
    undo: '<path d="M9 14 4 9l5-5"/><path d="M4 9h10a6 6 0 0 1 0 12h-2"/>',
    x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>'
  };

  const renderIcon = (target, name) => {
    if (!target || !iconSvg[name]) return;
    target.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${iconSvg[name]}</svg>`;
  };

  const hydrateIcons = (root = document) => {
    root.querySelectorAll(".icon[data-icon]").forEach((icon) => renderIcon(icon, icon.dataset.icon));
  };

  const loadTasks = () => {
    try {
      const parsed = JSON.parse(localStorage.getItem(TASK_STORAGE_KEY) || "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  };

  const saveTasks = (tasks) => {
    localStorage.setItem(TASK_STORAGE_KEY, JSON.stringify(tasks.slice(0, 80)));
    events.emit?.("tasks:updated", { tasks }).catch(() => {});
  };

  const activeFilter = () => document.querySelector(".filter-btn.active")?.dataset.filter || "active";

  const filteredTasks = () => {
    const filter = activeFilter();
    const query = (searchInput?.value || "").trim().toLowerCase();
    return loadTasks().filter((task) => {
      const statusOk = filter === "all" || (filter === "done" ? task.status === "done" : task.status !== "done");
      if (!statusOk) return false;
      if (!query) return true;
      const haystack = [
        task.title,
        task.itemName,
        task.objective,
        task.why,
        task.summary,
        task.note,
        ...(task.tags || []),
        ...(task.nextSteps || [])
      ].join(" ").toLowerCase();
      return haystack.includes(query);
    });
  };

  const formatDate = (iso) => {
    if (!iso) return "";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleString([], { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
  };

  const updateTask = (id, patch) => {
    const tasks = loadTasks().map((task) => (
      task.id === id ? { ...task, ...patch, updatedAt: new Date().toISOString() } : task
    ));
    saveTasks(tasks);
    render();
  };

  const deleteTask = (id) => {
    saveTasks(loadTasks().filter((task) => task.id !== id));
    render();
  };

  const askTask = (task) => {
    events.emit?.("tasks:ask", {
      id: task.id,
      title: task.title,
      itemName: task.itemName,
      objective: task.objective
    }).catch(() => {});
  };

  const renderTask = (task) => {
    const article = document.createElement("article");
    article.className = `task-card ${task.status === "done" ? "done" : ""}`;

    const header = document.createElement("div");
    header.className = "task-card-header";

    const toggle = document.createElement("button");
    toggle.className = "icon-btn status-btn";
    toggle.title = task.status === "done" ? "Mark active" : "Mark done";
    toggle.innerHTML = `<span class="icon" data-icon="${task.status === "done" ? "undo" : "check"}"></span>`;
    toggle.addEventListener("click", () => updateTask(task.id, { status: task.status === "done" ? "active" : "done" }));

    const titleWrap = document.createElement("div");
    titleWrap.className = "task-title-wrap";
    const title = document.createElement("h2");
    title.textContent = task.title || "Untitled task";
    const meta = document.createElement("div");
    meta.className = "task-meta";
    meta.textContent = [task.category, task.gameId, formatDate(task.createdAt)].filter(Boolean).join(" · ");
    titleWrap.append(title, meta);

    const remove = document.createElement("button");
    remove.className = "icon-btn";
    remove.title = "Delete task";
    remove.innerHTML = '<span class="icon" data-icon="trash"></span>';
    remove.addEventListener("click", () => deleteTask(task.id));

    header.append(toggle, titleWrap, remove);
    article.appendChild(header);

    const details = document.createElement("div");
    details.className = "task-details";
    const lines = [
      task.itemName ? ["物品/指標", task.itemName] : null,
      task.objective ? ["目標", task.objective] : null,
      task.why ? ["線索", task.why] : null,
      task.summary ? ["摘要", task.summary] : null
    ].filter(Boolean);
    for (const [label, value] of lines) {
      const row = document.createElement("p");
      row.innerHTML = `<strong>${label}</strong><span></span>`;
      row.querySelector("span").textContent = value;
      details.appendChild(row);
    }
    article.appendChild(details);

    if (task.nextSteps?.length) {
      const steps = document.createElement("ol");
      steps.className = "step-list";
      for (const step of task.nextSteps) {
        const li = document.createElement("li");
        li.textContent = step;
        steps.appendChild(li);
      }
      article.appendChild(steps);
    }

    if (task.tags?.length || task.sourceTitle || task.sourceSize) {
      const chips = document.createElement("div");
      chips.className = "task-chips";
      for (const tag of task.tags || []) {
        const chip = document.createElement("span");
        chip.textContent = tag;
        chips.appendChild(chip);
      }
      if (task.sourceTitle) {
        const chip = document.createElement("span");
        chip.textContent = task.sourceTitle;
        chips.appendChild(chip);
      }
      if (task.sourceSize) {
        const chip = document.createElement("span");
        chip.textContent = task.sourceSize;
        chips.appendChild(chip);
      }
      article.appendChild(chips);
    }

    const footer = document.createElement("div");
    footer.className = "task-footer";
    const confidence = document.createElement("span");
    confidence.textContent = `信心 ${Math.round((Number(task.confidence) || 0) * 100)}%`;
    const ask = document.createElement("button");
    ask.className = "ask-btn";
    ask.textContent = "問下一步";
    ask.addEventListener("click", () => askTask(task));
    footer.append(confidence, ask);
    article.appendChild(footer);

    hydrateIcons(article);
    return article;
  };

  const render = () => {
    taskList.innerHTML = "";
    const tasks = filteredTasks();
    if (!tasks.length) {
      const empty = emptyTemplate.content.cloneNode(true);
      hydrateIcons(empty);
      taskList.appendChild(empty);
      return;
    }
    for (const task of tasks) {
      taskList.appendChild(renderTask(task));
    }
  };

  hydrateIcons();
  applyOpacity(localStorage.getItem("ui-opacity") || "100");
  render();

  filterBtns.forEach((button) => {
    button.addEventListener("click", () => {
      filterBtns.forEach((candidate) => candidate.classList.toggle("active", candidate === button));
      render();
    });
  });

  searchInput?.addEventListener("input", render);
  refreshBtn?.addEventListener("click", render);
  clearDoneBtn?.addEventListener("click", () => {
    saveTasks(loadTasks().filter((task) => task.status !== "done"));
    render();
  });
  closeBtn?.addEventListener("click", async () => {
    if (invoke) {
      await invoke("hide_tasks_window").catch(() => appWindow?.hide?.().catch(() => {}));
    } else {
      await appWindow?.hide?.().catch(() => {});
    }
  });

  await events.listen?.("tasks:updated", (event) => {
    if (Array.isArray(event.payload?.tasks)) {
      localStorage.setItem(TASK_STORAGE_KEY, JSON.stringify(event.payload.tasks.slice(0, 80)));
    }
    render();
  }).catch(() => {});

  await events.listen?.("companion-window-shown", () => {
    applyOpacity(localStorage.getItem("ui-opacity") || "100");
    render();
    requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
  }).catch(() => {});

  await events.listen?.("ui-opacity-updated", (event) => {
    applyOpacity(event.payload?.opacity);
  }).catch(() => {});

  window.addEventListener("storage", (event) => {
    if (event.key === "ui-opacity") {
      applyOpacity(event.newValue);
    }
  });
});
