(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const camCanvas = $("camCanvas");
  const overlayCanvas = $("overlayCanvas");
  const bevCanvas = $("bevCanvas");
  const ctxCam = camCanvas.getContext("2d");
  const ctxOvl = overlayCanvas.getContext("2d");
  const ctxBev = bevCanvas.getContext("2d");

  const els = {
    connDot: $("connDot"), connText: $("connText"),
    srcBadge: $("srcBadge"), detBadge: $("detBadge"), modeTag: $("modeTag"),
    fpsBadge: $("fpsBadge"), odoText: $("odoText"), bevMeta: $("bevMeta"),
    vSpeed: $("vSpeed"), vSteer: $("vSteer"), vLatency: $("vLatency"), vTtc: $("vTtc"),
    cardTtc: $("cardTtc"), cardSteer: $("cardSteer"), cardLatency: $("cardLatency"),
    statusPill: $("statusPill"), pauseTag: $("pauseTag"),
    btnPause: $("btnPause"), btnReset: $("btnReset"), btnUpload: $("btnUpload"),
    fileInput: $("fileInput"), speedSlider: $("speedSlider"), speedVal: $("speedVal"),
    sparkSpeed: $("sparkSpeed"), sparkSteer: $("sparkSteer"),
    sparkLatency: $("sparkLatency"), sparkTtc: $("sparkTtc"),
    btnInfo: $("btnInfo"), infoModal: $("infoModal"), modalClose: $("modalClose"),
    btnExport: $("btnExport"), btnRefreshLogs: $("btnRefreshLogs"),
    logBody: $("logBody"), logTotal: $("logTotal"), zoneCount: $("zoneCount"),
  };

  const ENTITY_COLORS = {
    pothole: "#f59e0b", animal: "#ef4444", person: "#f472b6", vehicle: "#3b82f6",
  };

  let state = null;
  let pausedLocal = false;
  let ws = null;
  let wsAttempt = 0;
  let frameImg = null;
  let latestFrameUrl = null;
  let imgLoading = false;
  let occCache = { seq: -1, canvas: null };
  const history = { speed: [], steer: [], latency: [], ttc: [] };
  const msgTimes = [];

  const CAM_W = 1280;
  const CAM_H = 720;

  function initCam() {
    if (camCanvas.width !== CAM_W || camCanvas.height !== CAM_H) {
      camCanvas.width = CAM_W;
      camCanvas.height = CAM_H;
    }
    if (overlayCanvas.width !== CAM_W || overlayCanvas.height !== CAM_H) {
      overlayCanvas.width = CAM_W;
      overlayCanvas.height = CAM_H;
    }
  }

  function fit(canvas) {
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return false;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(rect.width * dpr);
    const h = Math.round(rect.height * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    return true;
  }

  function send(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(obj));
    }
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    els.connText.textContent = "CONNECTING…";
    els.connDot.className = "dot wait";
    ws = new WebSocket(proto + "://" + location.host + "/ws");

    ws.onopen = () => {
      wsAttempt = 0;
      els.connDot.className = "dot live";
      els.connText.textContent = "LIVE";
      send({ type: "hello" });
    };

    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.type !== "update") return;
      state = msg;
      msgTimes.push(performance.now());
      latestFrameUrl = msg.frame;
      pumpFrame();
      updateTelemetry(msg.telemetry);
      updateBadges(msg.telemetry);
      if (Array.isArray(msg.events)) msg.events.forEach(addLogRow);
    };

    ws.onclose = () => {
      els.connDot.className = "dot";
      els.connText.textContent = "RECONNECTING…";
      const delay = Math.min(5000, 700 * Math.pow(1.6, wsAttempt++));
      setTimeout(connect, delay);
    };

    ws.onerror = () => { try { ws.close(); } catch {} };
  }

  function pumpFrame() {
    if (imgLoading || !latestFrameUrl) return;
    imgLoading = true;
    const img = new Image();
    img.onload = () => { frameImg = img; imgLoading = false; };
    img.onerror = () => { imgLoading = false; };
    img.src = latestFrameUrl;
  }

  function updateBadges(tel) {
    els.srcBadge.textContent = tel.source.toUpperCase();
    els.detBadge.textContent = String(tel.detector).toUpperCase().replace("SIMULATED-GT", "SIM-GT");
    els.modeTag.textContent = tel.source === "simulation" ? "SIMULATED FEED" : "UPLOADED VIDEO";
    els.bevMeta.textContent = "96 × 144 @ 0.4 m · " + tel.nodes_expanded + " nodes";
    els.odoText.textContent = "odometer " + tel.odometer_m + " m";
    els.zoneCount.textContent = tel.in_zone != null ? tel.in_zone : 0;
  }

  function updateTelemetry(tel) {
    els.vSpeed.textContent = tel.speed_kmh.toFixed(1);
    els.vSteer.textContent = (tel.steering_deg > 0 ? "+" : "") + tel.steering_deg.toFixed(1);
    els.vLatency.textContent = tel.latency_ms.toFixed(1);

    if (tel.ttc_s == null) {
      els.vTtc.innerHTML = "&#8734;";
      els.cardTtc.className = "stat good";
    } else {
      els.vTtc.textContent = tel.ttc_s.toFixed(2);
      els.cardTtc.className = "stat " + (tel.ttc_s < 2 ? "danger" : tel.ttc_s < 4 ? "warn" : "good");
    }

    const st = pausedLocal ? "PAUSED" : tel.status;
    els.statusPill.textContent = st;
    els.statusPill.dataset.s = st;

    push(history.speed, tel.speed_kmh);
    push(history.steer, tel.steering_deg);
    push(history.latency, Math.min(tel.latency_ms, 120));
    push(history.ttc, tel.ttc_s == null ? 10 : Math.min(tel.ttc_s, 10));

    drawSpark(els.sparkSpeed, history.speed, "#34d399");
    drawSpark(els.sparkSteer, history.steer, "#22d3ee", -30, 30);
    drawSpark(els.sparkLatency, history.latency, "#f59e0b", 0, 60);
    drawSpark(els.sparkTtc, history.ttc, "#ef4444", 0, 10);
  }

  function push(arr, v) {
    arr.push(v);
    if (arr.length > 90) arr.shift();
  }

  function drawSpark(canvas, data, color, fixedMin, fixedMax) {
    fit(canvas);
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (!data || data.length < 2) return;
    let min = fixedMin != null ? fixedMin : Math.min(...data);
    let max = fixedMax != null ? fixedMax : Math.max(...data);
    if (max - min < 1e-6) { max += 1; min -= 1; }
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = h - 2 - ((v - min) / (max - min)) * (h - 4);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1.2, (window.devicePixelRatio || 1) * 1.3);
    ctx.lineJoin = "round";
    ctx.globalAlpha = 0.9;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function drawCamera() {
    initCam();
    const W = camCanvas.width, H = camCanvas.height;
    if (frameImg && frameImg.complete && frameImg.naturalWidth > 0) {
      ctxCam.drawImage(frameImg, 0, 0, W, H);
    } else {
      ctxCam.fillStyle = "#04070d";
      ctxCam.fillRect(0, 0, W, H);
    }
    ctxOvl.clearRect(0, 0, W, H);
    if (!state) return;

    const s = W / 640;

    if (state.path_img && state.path_img.length > 1) {
      ctxOvl.beginPath();
      state.path_img.forEach(([x, y], i) => {
        i === 0 ? ctxOvl.moveTo(x * s, y * s) : ctxOvl.lineTo(x * s, y * s);
      });
      ctxOvl.strokeStyle = "#22d3ee";
      ctxOvl.lineWidth = 2.5 * s * (window.devicePixelRatio > 1 ? 1 : 1.4);
      ctxOvl.shadowColor = "rgba(34,211,238,.8)";
      ctxOvl.shadowBlur = 8;
      ctxOvl.stroke();
      ctxOvl.shadowBlur = 0;
    }

    for (const d of state.detections || []) {
      const [x1, y1, x2, y2] = d.box.map((v) => v * s);
      const col = d.color || "#94a3b8";
      ctxOvl.strokeStyle = col;
      ctxOvl.lineWidth = 1.6 * s;
      ctxOvl.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const c = Math.min(12 * s, (x2 - x1) / 2, (y2 - y1) / 2);
      ctxOvl.lineWidth = 3.2 * s;
      ctxOvl.beginPath();
      ctxOvl.moveTo(x1, y1 + c); ctxOvl.lineTo(x1, y1); ctxOvl.lineTo(x1 + c, y1);
      ctxOvl.moveTo(x2 - c, y1); ctxOvl.lineTo(x2, y1); ctxOvl.lineTo(x2, y1 + c);
      ctxOvl.moveTo(x2, y2 - c); ctxOvl.lineTo(x2, y2); ctxOvl.lineTo(x2 - c, y2);
      ctxOvl.moveTo(x1 + c, y2); ctxOvl.lineTo(x1, y2); ctxOvl.lineTo(x1, y2 - c);
      ctxOvl.stroke();

      const label = (d.name || d.label || "?") + " " + Math.round((d.conf || 0) * 100) + "%";
      ctxOvl.font = `${Math.max(11, 11 * s)}px ui-monospace, monospace`;
      const tw = ctxOvl.measureText(label).width;
      const ly = y1 - 18 * s < 2 ? y1 + 3 * s : y1 - 18 * s;
      ctxOvl.fillStyle = "rgba(4,8,15,.78)";
      ctxOvl.fillRect(x1, ly, tw + 10 * s, 15 * s);
      ctxOvl.fillStyle = col;
      ctxOvl.fillText(label, x1 + 5 * s, ly + 11 * s);
    }
  }

  function unpackHard(b64, nCells) {
    const bin = atob(b64);
    const out = new Uint8Array(nCells);
    for (let i = 0; i < nCells; i++) {
      out[i] = (bin.charCodeAt(i >> 3) >> (7 - (i & 7))) & 1;
    }
    return out;
  }

  function unpackSoft(b64, nCells) {
    const bin = atob(b64);
    const out = new Uint8Array(nCells);
    for (let i = 0; i < nCells; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function rebuildOccupancy(g) {
    const n = g.w * g.h;
    const hard = unpackHard(g.hard, n);
    const soft = unpackSoft(g.soft, n);
    const off = document.createElement("canvas");
    off.width = g.w;
    off.height = g.h;
    const octx = off.getContext("2d");
    const img = octx.createImageData(g.w, g.h);
    for (let r = 0; r < g.h; r++) {
      for (let c = 0; c < g.w; c++) {
        const i = r * g.w + c;
        const p = i * 4;
        if (hard[i]) {
          img.data[p] = 239; img.data[p + 1] = 68; img.data[p + 2] = 68; img.data[p + 3] = 235;
        } else if (soft[i] > 8) {
          img.data[p] = 245; img.data[p + 1] = 158; img.data[p + 2] = 11;
          img.data[p + 3] = Math.min(200, soft[i] + 40);
        } else {
          img.data[p] = 16; img.data[p + 1] = 25; img.data[p + 2] = 42; img.data[p + 3] = 110;
        }
      }
    }
    octx.putImageData(img, 0, 0);
    occCache = { seq: state.seq, canvas: off };
  }

  function drawBev() {
    if (!fit(bevCanvas) || !state || !state.grid) return;
    const g = state.grid;
    const W = bevCanvas.width, H = bevCanvas.height;
    ctxBev.clearRect(0, 0, W, H);

    if (occCache.seq !== state.seq) rebuildOccupancy(g);

    const pad = Math.max(14, Math.min(W, H) * 0.06);
    const sc = Math.min((W - pad * 2) / g.w, (H - pad * 2) / g.h);
    const ox = (W - g.w * sc) / 2;
    const oy = (H - g.h * sc) / 2;
    const cellX = (c) => ox + (c + 0.5) * sc;
    const cellY = (r) => oy + (r + 0.5) * sc;

    ctxBev.fillStyle = "#050a13";
    ctxBev.fillRect(ox, oy, g.w * sc, g.h * sc);

    ctxBev.imageSmoothingEnabled = false;
    ctxBev.drawImage(occCache.canvas, ox, oy, g.w * sc, g.h * sc);
    ctxBev.imageSmoothingEnabled = true;

    ctxBev.font = `${Math.round(sc * 2.6)}px ui-monospace, monospace`;
    for (let zm = 10; zm <= g.z_max; zm += 10) {
      const y = oy + ((g.z_max - zm) / g.res) * sc;
      ctxBev.strokeStyle = "rgba(148,163,184,.09)";
      ctxBev.lineWidth = 1;
      ctxBev.beginPath(); ctxBev.moveTo(ox, y); ctxBev.lineTo(ox + g.w * sc, y); ctxBev.stroke();
      ctxBev.fillStyle = "rgba(124,138,165,.55)";
      ctxBev.fillText(zm + "m", ox + g.w * sc + 4, y + 3);
    }

    const edgeL = ox + (g.w / 2 - g.road_half_m / g.res) * sc;
    const edgeR = ox + (g.w / 2 + g.road_half_m / g.res) * sc;
    ctxBev.setLineDash([4, 5]);
    ctxBev.strokeStyle = "rgba(203,213,225,.28)";
    ctxBev.lineWidth = 1.2;
    ctxBev.beginPath();
    ctxBev.moveTo(edgeL, oy); ctxBev.lineTo(edgeL, oy + g.h * sc);
    ctxBev.moveTo(edgeR, oy); ctxBev.lineTo(edgeR, oy + g.h * sc);
    ctxBev.stroke();
    ctxBev.setLineDash([]);

    ctxBev.fillStyle = "rgba(100,116,139,.45)";
    for (const [c, r] of state.path_raw_cells || []) {
      ctxBev.fillRect(cellX(c) - 1.2, cellY(r) - 1.2, 2.4, 2.4);
    }

    if ((state.path_smooth_cells || []).length > 1) {
      ctxBev.beginPath();
      state.path_smooth_cells.forEach(([c, r], i) => {
        i === 0 ? ctxBev.moveTo(cellX(c), cellY(r)) : ctxBev.lineTo(cellX(c), cellY(r));
      });
      ctxBev.strokeStyle = "#22d3ee";
      ctxBev.lineWidth = Math.max(2, sc * 0.32);
      ctxBev.lineJoin = "round";
      ctxBev.shadowColor = "rgba(34,211,238,.75)";
      ctxBev.shadowBlur = 10;
      ctxBev.stroke();
      ctxBev.shadowBlur = 0;
    }

    const goalX = cellX(g.w / 2 - 0.5);
    const goalY = oy + 1.5 * sc;
    ctxBev.strokeStyle = "#34d399";
    ctxBev.lineWidth = 1.6;
    ctxBev.beginPath();
    ctxBev.moveTo(goalX, goalY); ctxBev.lineTo(goalX, goalY + sc * 5); ctxBev.stroke();
    ctxBev.fillStyle = "#34d399";
    ctxBev.beginPath();
    ctxBev.moveTo(goalX, goalY);
    ctxBev.lineTo(goalX + sc * 3.4, goalY + sc * 1.4);
    ctxBev.lineTo(goalX, goalY + sc * 2.8);
    ctxBev.closePath();
    ctxBev.fill();

    const carCol = g.w / 2 - 0.5 + (state.telemetry.ego_u || 0) / g.res;
    const carRow = (g.z_max - 1.4) / g.res;
    const carX = cellX(carCol);
    const carY = cellY(carRow);
    ctxBev.save();
    ctxBev.translate(carX, carY);
    ctxBev.rotate(((state.telemetry.steering_deg || 0) * Math.PI) / 180);

    ctxBev.fillStyle = "rgba(34,211,238,.07)";
    ctxBev.beginPath();
    ctxBev.moveTo(0, 0);
    const coneLen = (12 / g.res) * sc;
    const coneHalf = Math.tan((28 * Math.PI) / 180) * coneLen;
    ctxBev.lineTo(-coneHalf, -coneLen);
    ctxBev.lineTo(coneHalf, -coneLen);
    ctxBev.closePath();
    ctxBev.fill();

    const cw = (1.9 / g.res) * sc;
    const chl = (4.3 / g.res) * sc;
    ctxBev.fillStyle = "#e2e8f0";
    ctxBev.strokeStyle = "#22d3ee";
    ctxBev.lineWidth = 1.4;
    roundRect(ctxBev, -cw / 2, -chl * 0.62, cw, chl, cw * 0.24);
    ctxBev.fill();
    ctxBev.stroke();
    ctxBev.fillStyle = "rgba(14,116,144,.85)";
    roundRect(ctxBev, -cw * 0.32, -chl * 0.48, cw * 0.64, chl * 0.26, cw * 0.1);
    ctxBev.fill();
    ctxBev.restore();

    ctxBev.fillStyle = "rgba(124,138,165,.7)";
    ctxBev.font = `${Math.max(10, sc * 2.2)}px ui-monospace, monospace`;
    ctxBev.fillText(
      `A* ${state.telemetry.planner_status.toUpperCase()} · ${state.telemetry.nodes_expanded} nodes · ${state.telemetry.latency_ms.toFixed(1)} ms`,
      ox, oy + g.h * sc + (H - (oy + g.h * sc)) * 0.75
    );
  }

  function roundRect(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  const capWord = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : "Obstacle");

  function fmtTime(ts) {
    return new Date(ts * 1000).toLocaleTimeString("en-GB", { hour12: false });
  }

  function addLogRow(rec) {
    const empty = els.logBody.querySelector(".empty-row");
    if (empty) empty.remove();
    if (els.logBody.querySelector(`[data-tid="${rec.track_id}"]`)) return;

    const tr = document.createElement("tr");
    tr.dataset.tid = rec.track_id;
    const dot = ENTITY_COLORS[rec.label] || "#94a3b8";
    const statusLabel = rec.status === "AVOIDED" ? "Avoided" : "Cleared";
    tr.innerHTML = `
      <td class="mono">${rec.track_id}</td>
      <td class="mono">${fmtTime(rec.entered_at)}</td>
      <td><span class="entity"><span class="swatch" style="background:${dot}"></span>${capWord(rec.name || rec.label)}</span></td>
      <td class="mono">${Number(rec.min_distance_m).toFixed(1)}</td>
      <td class="mono">${String(rec.action).replace(/</g, "&lt;")}</td>
      <td><span class="st ${rec.status}">${statusLabel}</span></td>`;
    els.logBody.prepend(tr);
    while (els.logBody.children.length > 250) els.logBody.lastElementChild.remove();
    els.logTotal.textContent = Number(els.logTotal.textContent || 0) + 1;
  }

  async function fetchLogs() {
    try {
      const res = await fetch("/api/logs?limit=200");
      const j = await res.json();
      const events = j.events || [];
      els.logBody.innerHTML = "";
      els.logTotal.textContent = "0";
      events.forEach(addLogRow);
      els.logTotal.textContent = j.total != null ? j.total : events.length;
      if (!events.length) {
        els.logBody.innerHTML =
          `<tr class="empty-row"><td colspan="6">No obstacle encounters logged yet — monitoring avoidance zone…</td></tr>`;
      }
    } catch {}
  }

  async function exportCsv() {
    try {
      const res = await fetch("/api/logs?limit=1000");
      const j = await res.json();
      const hdr = ["ID", "Timestamp", "Detected Entity", "Distance_m", "Steering_deg",
        "Speed_kmh", "Action", "Status", "Duration_s"];
      const esc = (v) => `"${String(v).replace(/"/g, '""')}"`;
      const rows = (j.events || []).map((e) => [
        e.track_id,
        new Date(e.entered_at * 1000).toISOString(),
        e.name || e.label,
        e.min_distance_m, e.steering_deg, e.speed_kmh,
        esc(e.action), e.status, e.duration_s,
      ].join(","));
      const csv = [hdr.join(","), ...rows].join("\r\n");
      const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = `sih26037_obstacle_log_${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch {}
  }

  function loop() {
    drawCamera();
    drawBev();
    const now = performance.now();
    while (msgTimes.length && now - msgTimes[0] > 1000) msgTimes.shift();
    els.fpsBadge.textContent = msgTimes.length + " msg/s";
    requestAnimationFrame(loop);
  }

  els.btnPause.addEventListener("click", () => {
    pausedLocal = !pausedLocal;
    els.btnPause.textContent = pausedLocal ? "Resume" : "Pause";
    els.pauseTag.classList.toggle("show", pausedLocal);
    send({ type: pausedLocal ? "pause" : "resume" });
    if (state) {
      els.statusPill.textContent = pausedLocal ? "PAUSED" : state.telemetry.status;
      els.statusPill.dataset.s = pausedLocal ? "PAUSED" : state.telemetry.status;
    }
  });

  els.btnReset.addEventListener("click", async () => {
    pausedLocal = false;
    els.btnPause.textContent = "Pause";
    els.pauseTag.classList.remove("show");
    await fetch("/api/source/simulate", { method: "POST" }).catch(() => {});
    send({ type: "reset" });
  });

  let sliderTimer = null;
  els.speedSlider.addEventListener("input", () => {
    els.speedVal.textContent = els.speedSlider.value + " km/h";
    clearTimeout(sliderTimer);
    sliderTimer = setTimeout(() => {
      send({ type: "set_speed", value: Number(els.speedSlider.value) });
    }, 180);
  });

  els.btnUpload.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", async () => {
    const file = els.fileInput.files && els.fileInput.files[0];
    els.fileInput.value = "";
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    els.statusPill.textContent = "UPLOADING…";
    try {
      const res = await fetch("/api/upload", { method: "POST", body: fd });
      const j = await res.json();
      if (!res.ok || !j.ok) throw new Error(j.detail || "upload failed");
    } catch (err) {
      els.statusPill.textContent = "UPLOAD FAILED";
      els.statusPill.dataset.s = "OBSTRUCTED";
    }
  });

  els.btnInfo.addEventListener("click", () => els.infoModal.classList.add("show"));
  els.modalClose.addEventListener("click", () => els.infoModal.classList.remove("show"));
  els.infoModal.addEventListener("click", (ev) => {
    if (ev.target === els.infoModal) els.infoModal.classList.remove("show");
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") els.infoModal.classList.remove("show");
  });

  els.btnExport.addEventListener("click", exportCsv);
  els.btnRefreshLogs.addEventListener("click", fetchLogs);

  connect();
  fetchLogs();
  requestAnimationFrame(loop);
})();
