document.addEventListener("DOMContentLoaded", async () => {
  const { getCurrentWindow } = window.__TAURI__.window;
  const { invoke } = window.__TAURI__.core;
  const { listen, emit } = window.__TAURI__.event;

  const hudWindow = getCurrentWindow();
  const canvas = document.getElementById("hudCanvas");
  const ctx = canvas.getContext("2d");
  let hideTimer = null;
  let currentOverlay = null;
  let currentTarget = null;
  let lastStoredOverlayId = null;

  const clamp = (value, fallback = 0) => {
    const number = Number(value);
    if (!Number.isFinite(number)) return fallback;
    return Math.max(0, Math.min(1, number));
  };

  const resize = () => {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(window.innerWidth * dpr));
    canvas.height = Math.max(1, Math.floor(window.innerHeight * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  };

  const imageSpace = () => {
    const space = currentOverlay?.coordinate_space || {};
    const imageWidth = Number(space.image_width || currentTarget?.imageWidth || currentTarget?.width);
    const imageHeight = Number(space.image_height || currentTarget?.imageHeight || currentTarget?.height);
    if (!Number.isFinite(imageWidth) || !Number.isFinite(imageHeight) || imageWidth <= 0 || imageHeight <= 0) {
      return null;
    }
    return { width: imageWidth, height: imageHeight };
  };

  const xy = (point) => {
    const space = imageSpace();
    const pixelX = Number(point?.pixel_x);
    const pixelY = Number(point?.pixel_y);
    if (space && Number.isFinite(pixelX) && Number.isFinite(pixelY)) {
      return {
        x: (pixelX / space.width) * window.innerWidth,
        y: (pixelY / space.height) * window.innerHeight
      };
    }
    return {
      x: clamp(point?.x, 0.5) * window.innerWidth,
      y: clamp(point?.y, 0.5) * window.innerHeight
    };
  };

  const radiusPx = (item) => {
    const space = imageSpace();
    const pixelRadius = Number(item?.radius_px);
    if (space && Number.isFinite(pixelRadius) && pixelRadius > 0) {
      return (pixelRadius / Math.min(space.width, space.height)) * Math.min(window.innerWidth, window.innerHeight);
    }
    return clamp(item.radius, 0.06) * Math.min(window.innerWidth, window.innerHeight);
  };

  const drawLabel = (label, x, y, color = "#00E5FF") => {
    if (!label) return;
    ctx.save();
    ctx.font = "600 16px Segoe UI, sans-serif";
    ctx.textBaseline = "middle";
    const paddingX = 10;
    const paddingY = 6;
    const metrics = ctx.measureText(label);
    const width = metrics.width + paddingX * 2;
    const height = 30;
    const left = Math.min(Math.max(12, x + 14), window.innerWidth - width - 12);
    const top = Math.min(Math.max(12, y - height / 2), window.innerHeight - height - 12);
    ctx.fillStyle = "rgba(0, 0, 0, 0.68)";
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    roundRect(left, top, width, height, 6);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, left + paddingX, top + height / 2);
    ctx.restore();
  };

  const roundRect = (x, y, w, h, r) => {
    const radius = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + w, y, x + w, y + h, radius);
    ctx.arcTo(x + w, y + h, x, y + h, radius);
    ctx.arcTo(x, y + h, x, y, radius);
    ctx.arcTo(x, y, x + w, y, radius);
    ctx.closePath();
  };

  const drawArrowHead = (from, to, color) => {
    const angle = Math.atan2(to.y - from.y, to.x - from.x);
    const size = 18;
    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(to.x, to.y);
    ctx.lineTo(to.x - size * Math.cos(angle - Math.PI / 6), to.y - size * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(to.x - size * Math.cos(angle + Math.PI / 6), to.y - size * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  };

  const draw = () => {
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    if (!currentOverlay?.items?.length) return;

    for (const item of currentOverlay.items) {
      const itemType = {
        target: "pin",
        objective: "pin",
        destination: "pin",
        marker: "pin",
        route: "path",
        line: "path",
        guidance: "arrow",
        guide: "arrow",
        direction: "arrow"
      }[item.type] || item.type;
      const color = itemType === "circle" ? "#ff2d2d" : (item.color || "#ff2d2d");
      ctx.save();
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = itemType === "circle" ? 6 : 4;
      ctx.shadowColor = "rgba(0, 0, 0, 0.75)";
      ctx.shadowBlur = 10;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";

      if (itemType === "circle") {
        const point = xy(item);
        const radius = Math.max(18, radiusPx(item));
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 0.14;
        ctx.fill();
        ctx.globalAlpha = 1;
        drawLabel(item.label, point.x + radius, point.y, color);
      } else if (itemType === "arrow") {
        const from = xy(item.from);
        const to = xy(item.to);
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
        drawArrowHead(from, to, color);
        drawLabel(item.label, to.x, to.y, color);
      } else if (itemType === "path") {
        const points = (item.points || []).map(xy);
        if (points.length >= 2) {
          ctx.beginPath();
          ctx.moveTo(points[0].x, points[0].y);
          for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
          ctx.stroke();
          drawArrowHead(points[points.length - 2], points[points.length - 1], color);
          const last = points[points.length - 1];
          drawLabel(item.label, last.x, last.y, color);
        }
      } else if (itemType === "pin" || itemType === "label") {
        const point = xy(item);
        if (itemType === "pin") {
          ctx.beginPath();
          ctx.arc(point.x, point.y, 8, 0, Math.PI * 2);
          ctx.fill();
          ctx.beginPath();
          ctx.arc(point.x, point.y, 18, 0, Math.PI * 2);
          ctx.stroke();
        }
        drawLabel(item.label, point.x, point.y, color);
      }
      ctx.restore();
    }
  };

  const clearHud = async () => {
    if (hideTimer) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
    currentOverlay = null;
    currentTarget = null;
    localStorage.removeItem("hud-overlay");
    document.body.classList.remove("visible");
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    setTimeout(async () => {
      await invoke("hide_hud_window").catch(() => hudWindow.hide());
    }, 180);
  };

  const hudTargetArgs = (target = null) => {
    const args = {
      width: Number(target?.width) || window.screen?.width || window.innerWidth || 1920,
      height: Number(target?.height) || window.screen?.height || window.innerHeight || 1080
    };
    const x = Number(target?.x);
    const y = Number(target?.y);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      args.x = x;
      args.y = y;
    }
    return args;
  };

  const showHud = async (overlay, target = null) => {
    if (!overlay?.items?.length) return;
    if (hideTimer) clearTimeout(hideTimer);
    currentTarget = target || overlay.render_target || null;
    await invoke("show_hud_window", hudTargetArgs(currentTarget)).catch((err) => console.warn("HUD show failed:", err));
    currentOverlay = overlay;
    document.body.classList.add("visible");
    resize();
    hideTimer = setTimeout(clearHud, Math.max(3000, Math.min(Number(overlay.duration_ms) || 6000, 8000)));
  };

  const readStoredOverlay = () => {
    const raw = localStorage.getItem("hud-overlay");
    if (!raw) return;
    try {
      const payload = JSON.parse(raw);
      if (!payload?.id || payload.id === lastStoredOverlayId) return;
      if (!payload.overlay?.items?.length) return;
      lastStoredOverlayId = payload.id;
      showHud(payload.overlay, payload.target);
    } catch (err) {
      console.warn("Could not read stored HUD overlay:", err);
    }
  };

  await listen("hud:show", (event) => showHud(event.payload));
  await listen("hud:clear", clearHud);
  await invoke("configure_hud_window").catch((err) => console.warn("HUD configure failed:", err));
  window.addEventListener("resize", resize);
  resize();
  setInterval(readStoredOverlay, 250);
  readStoredOverlay();
  await emit("hud:ready", { label: "hud" });
});
