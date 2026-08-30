import asyncio
import heapq
import math
import random
import sqlite3
import time
from datetime import datetime

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI(title="SIH26037 Adaptive Path Planner")

# ============================================================
# CONFIGURATION
# ============================================================

GRID_WIDTH = 96<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SIH26037 · Adaptive Path Planner &amp; Collision Avoidance</title>
<style>
  :root {
    --bg: #070b14;
    --panel: #0c1424;
    --panel-2: #0a1120;
    --line: rgba(148, 163, 184, 0.14);
    --text: #e2e8f0;
    --muted: #7c8aa5;
    --accent: #22d3ee;
    --accent-dim: rgba(34, 211, 238, 0.14);
    --green: #34d399;
    --amber: #f59e0b;
    --red: #ef4444;
    --blue: #3b82f6;
    --pink: #f472b6;
    --radius: 14px;
    font-size: 15px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    background:
      radial-gradient(1200px 500px at 80% -10%, rgba(34, 211, 238, 0.06), transparent 60%),
      radial-gradient(900px 420px at -10% 110%, rgba(59, 130, 246, 0.05), transparent 60%),
      var(--bg);
    color: var(--text);
    font-family: "Segoe UI", ui-sans-serif, system-ui, Roboto, Helvetica, Arial, sans-serif;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  header {
    display: flex; align-items: center; gap: 18px;
    padding: 12px 20px;
    border-bottom: 1px solid var(--line);
    backdrop-filter: blur(8px);
    flex-wrap: wrap;
    flex-shrink: 0;
  }
  .brand { display: flex; align-items: center; gap: 12px; margin-right: auto; }
  .brand-mark {
    width: 38px; height: 38px; border-radius: 11px;
    background: linear-gradient(135deg, rgba(34,211,238,.25), rgba(59,130,246,.18));
    border: 1px solid rgba(34,211,238,.35);
    display: grid; place-items: center;
    box-shadow: 0 0 18px rgba(34,211,238,.25);
  }
  .brand h1 { font-size: 1.02rem; font-weight: 650; letter-spacing: .2px; }
  .brand p { font-size: .72rem; color: var(--muted); letter-spacing: .4px; }

  .conn { display: flex; align-items: center; gap: 7px; font-size: .74rem; color: var(--muted); }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--red); transition: background .3s; }
  .dot.live { background: var(--green); box-shadow: 0 0 8px rgba(52,211,153,.8); animation: pulse 2s infinite; }
  .dot.wait { background: var(--amber); }
  @keyframes pulse { 50% { opacity: .55; } }

  .controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .btn {
    background: var(--panel-2); color: var(--text);
    border: 1px solid var(--line); border-radius: 9px;
    padding: 7px 13px; font-size: .78rem; cursor: pointer;
    transition: all .18s ease; font-family: inherit;
  }
  .btn:hover { border-color: rgba(34,211,238,.45); color: var(--accent);
    transform: translateY(-1px); box-shadow: 0 4px 14px rgba(34,211,238,.12); }
  .btn.primary { background: linear-gradient(135deg, rgba(34,211,238,.2), rgba(59,130,246,.15));
    border-color: rgba(34,211,238,.4); }
  .speed-ctl { display: flex; align-items: center; gap: 8px; font-size: .74rem; color: var(--muted);
    border: 1px solid var(--line); border-radius: 9px; padding: 5px 11px; }
  input[type=range] { width: 110px; accent-color: var(--accent); cursor: pointer; }
  #speedVal { color: var(--text); font-family: ui-monospace, monospace; min-width: 58px; text-align: right; }

  .badge {
    font-size: .68rem; letter-spacing: .8px; padding: 5px 10px; border-radius: 999px;
    border: 1px solid var(--line); color: var(--muted); white-space: nowrap;
  }
  .badge b { color: var(--accent); font-weight: 600; }

  main {
    flex: 0 0 auto;
    height: 400px;
    min-height: 0;
    display: grid; grid-template-columns: 1.35fr 1fr; gap: 14px;
    padding: 14px 20px 0;
  }
  .panel {
    background: linear-gradient(180deg, var(--panel), var(--panel-2));
    border: 1px solid var(--line); border-radius: var(--radius);
    display: flex; flex-direction: column; min-height: 0; min-width: 0;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(2, 6, 17, .5);
  }
  .panel-head {
    display: flex; align-items: center; justify-content: space-between; gap: 10px;
    padding: 11px 16px; border-bottom: 1px solid var(--line);
  }
  .panel-head h2 { font-size: .82rem; font-weight: 600; letter-spacing: .3px; display:flex; align-items:center; gap:8px;}
  .panel-head h2::before { content: ""; width: 8px; height: 8px; border-radius: 2px;
    background: linear-gradient(135deg, var(--accent), var(--blue)); }
  .panel-head span { font-size: .68rem; color: var(--muted); }
  .stage-wrap { flex: 1; min-height: 0; padding: 12px 16px 14px; display: flex; }
  .stage { position: relative; flex: 1; border-radius: 10px; overflow: hidden;
    border: 1px solid var(--line); background: #04070d; min-height: 0; min-width: 0; }
  canvas.fill { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
  #camCanvas, #overlayCanvas { object-fit: contain; }

  .legend { display: flex; gap: 14px; padding: 0 16px 12px; flex-wrap: wrap; }
  .legend div { display: flex; align-items: center; gap: 6px; font-size: .68rem; color: var(--muted); }
  .swatch { width: 10px; height: 10px; border-radius: 3px; }

  footer#telemetry {
    display: grid; grid-template-columns: repeat(4, 1fr) auto; gap: 12px;
    padding: 0 20px 16px;
    flex-shrink: 0;
    margin-top: auto;
  }
  .stat {
    background: linear-gradient(180deg, var(--panel), var(--panel-2));
    border: 1px solid var(--line); border-radius: var(--radius);
    padding: 10px 14px 8px;
    display: flex; flex-direction: column; gap: 2px; min-width: 0;
    transition: border-color .3s;
  }
  .stat-top { display: flex; justify-content: space-between; align-items: baseline; }
  .stat-label { font-size: .68rem; letter-spacing: 1px; text-transform: uppercase; color: var(--muted); }
  .stat-unit { font-size: .64rem; color: var(--muted); }
  .stat-value {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 1.65rem; font-weight: 600; line-height: 1.15;
  }
  .spark { width: 100%; height: 30px; opacity: .9; }
  .stat.warn { border-color: rgba(245, 158, 11, .55); }
  .stat.warn .stat-value { color: var(--amber); }
  .stat.danger { border-color: rgba(239, 68, 68, .6); animation: pulse 1.1s infinite; }
  .stat.danger .stat-value { color: var(--red); }
  .stat.good .stat-value { color: var(--green); }

  .status-pill {
    align-self: stretch; min-width: 150px;
    border-radius: var(--radius); border: 1px solid var(--line);
    background: linear-gradient(180deg, var(--panel), var(--panel-2));
    display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 3px;
    padding: 8px 16px;
  }
  #statusPill {
    font-size: .95rem; font-weight: 700; letter-spacing: 1.5px; color: var(--green);
    font-family: ui-monospace, monospace;
  }
  .status-pill small { font-size: .62rem; color: var(--muted); letter-spacing: .6px; }
  #statusPill[data-s="BRAKE"], #statusPill[data-s="OBSTRUCTED"] { color: var(--red); }
  #statusPill[data-s="REROUTED"] { color: var(--amber); }
  #statusPill[data-s="PAUSED"] { color: var(--muted); }

  .mw { color: var(--accent); font-weight: 600; white-space: nowrap; }
  .btn.small { padding: 5px 10px; font-size: .7rem; }

  .log-wrap { flex: 0 1 auto; min-height: 0; padding: 0 20px 12px; display: flex; }
  .log-wrap .panel { flex: 1; min-width: 0; min-height: 0; }
  .log-actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .table-wrap {
    flex: 0 1 auto; min-height: 0; max-height: 220px;
    overflow-y: auto; overscroll-behavior: contain;
  }
  #logTable { width: 100%; border-collapse: collapse; font-size: .74rem; }
  #logTable thead th {
    position: sticky; top: 0; z-index: 1; background: #0d1626;
    text-align: left; padding: 9px 14px; font-size: .64rem; font-weight: 600;
    letter-spacing: 1px; text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--line);
  }
  #logTable tbody td { padding: 8px 14px; border-top: 1px solid rgba(148,163,184,.07); white-space: nowrap; }
  #logTable tbody tr { animation: rowIn .5s ease; }
  #logTable tbody tr:hover { background: rgba(34,211,238,.045); }
  @keyframes rowIn { from { background: rgba(34,211,238,.16); } }
  td.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .76rem; }
  .empty-row td { text-align: center; color: var(--muted); padding: 24px !important; }
  .entity { display: inline-flex; align-items: center; gap: 7px; }
  .st { font-size: .62rem; font-weight: 700; letter-spacing: .8px; padding: 3px 9px; border-radius: 999px; }
  .st.AVOIDED { color: #fbbf24; background: rgba(245,158,11,.12); border: 1px solid rgba(245,158,11,.35); }
  .st.CLEARED { color: #34d399; background: rgba(52,211,153,.1); border: 1px solid rgba(52,211,153,.3); }

  .modal-backdrop {
    position: fixed; inset: 0; z-index: 60; padding: 22px;
    background: rgba(3, 6, 12, .7); backdrop-filter: blur(7px);
    display: none; place-items: center;
  }
  .modal-backdrop.show { display: grid; }
  .modal {
    position: relative; width: 100%; max-width: 780px; max-height: 86vh; overflow: auto;
    background: linear-gradient(180deg, #101a2e, #0a1220);
    border: 1px solid rgba(34, 211, 238, .28); border-radius: 18px;
    padding: 28px 30px 24px; box-shadow: 0 30px 80px rgba(0, 0, 0, .6);
    animation: pop .24s cubic-bezier(.2, .9, .3, 1.15);
  }
  @keyframes pop { from { opacity: 0; transform: translateY(14px) scale(.97); } }
  .modal-x {
    position: absolute; top: 12px; right: 16px; background: none; border: none;
    color: var(--muted); font-size: 1.6rem; cursor: pointer; line-height: 1;
  }
  .modal-x:hover { color: var(--text); }
  .modal-title {
    font-size: 1.18rem; font-weight: 700; margin-bottom: 6px;
    background: linear-gradient(90deg, #22d3ee, #818cf8);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .modal-sub { font-size: .8rem; color: var(--muted); margin-bottom: 18px; line-height: 1.55; }
  .modal-sub b { color: var(--accent); }
  .m-sec { margin-bottom: 16px; }
  .m-sec h3 {
    font-size: .74rem; letter-spacing: 1.2px; text-transform: uppercase; color: var(--accent);
    margin-bottom: 7px; display: flex; align-items: center; gap: 8px;
  }
  .m-sec h3::before {
    content: ""; width: 7px; height: 7px; border-radius: 2px;
    background: linear-gradient(135deg, #22d3ee, #3b82f6);
  }
  .m-sec p, .m-sec li { font-size: .8rem; color: #b8c4d9; line-height: 1.65; }
  .m-sec p { margin-bottom: 4px; }
  .m-sec b { color: var(--text); }
  .m-sec ul { list-style: none; }
  .m-sec li { padding-left: 16px; position: relative; margin-bottom: 4px; }
  .m-sec li::before { content: "▸"; position: absolute; left: 0; color: var(--accent); font-size: .68rem; }
  .chips { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 4px; }
  .chips span {
    font-size: .66rem; letter-spacing: .6px; padding: 4px 10px; border-radius: 999px;
    border: 1px solid var(--line); color: var(--muted); background: rgba(148,163,184,.05);
  }

  .overlay-tag {
    position: absolute; top: 10px; left: 10px; z-index: 3;
    font-size: .66rem; letter-spacing: 1px; padding: 4px 9px; border-radius: 6px;
    background: rgba(4, 8, 15, .72); border: 1px solid var(--line); color: var(--muted);
  }
  .overlay-tag b { color: var(--accent); }
  #pauseTag { position: absolute; inset: 0; z-index: 4; display: none;
    place-items: center; background: rgba(4, 8, 15, .55);
    font-weight: 700; letter-spacing: 4px; color: var(--muted); font-size: 1rem; }
  #pauseTag.show { display: grid; }

  @media (max-width: 1080px) {
    html, body { height: auto; }
    body { overflow: auto; }
    main { height: auto; grid-template-columns: 1fr; }
    .stage-wrap { min-height: 300px; }
    .log-wrap { flex: 0 0 auto; }
    footer#telemetry { grid-template-columns: repeat(2, 1fr); }
    .status-pill { grid-column: span 2; min-height: 60px; }
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="brand-mark">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="9"/><path d="M12 3v4M12 17v4M3 12h4M17 12h4"/>
        <path d="M7.5 15.5 L12 8 L16.5 15.5"/>
      </svg>
    </div>
    <div>
      <h1>SIH26037: Adaptive Path Planning &amp; Collision Avoidance <span class="mw">(MathWorks)</span></h1>
      <p>LIVE PERCEPTION · A* REPLANNING @ 10 HZ · TTC SAFETY · PROTOTYPE</p>
    </div>
  </div>

  <div class="controls">
    <button class="btn" id="btnInfo">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" style="vertical-align:-2px;margin-right:4px">
        <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
      </svg>Info / Problem Statement</button>
    <button class="btn primary" id="btnPause">Pause</button>
    <button class="btn" id="btnReset">Reset</button>
    <label class="speed-ctl">Target
      <input type="range" id="speedSlider" min="15" max="90" step="5" value="45">
      <span id="speedVal">45 km/h</span>
    </label>
    <button class="btn" id="btnUpload">Upload Video</button>
    <input type="file" id="fileInput" accept="video/*" hidden>
    <span class="badge">SRC <b id="srcBadge">SIMULATION</b></span>
    <span class="badge">DET <b id="detBadge">SIM-GT</b></span>
    <span class="badge" id="fpsBadge">— msg/s</span>
  </div>
  <div class="conn"><span class="dot wait" id="connDot"></span><span id="connText">CONNECTING…</span></div>
</header>

<main>
  <section class="panel">
    <div class="panel-head">
      <h2>Perception · Camera Feed</h2>
      <span id="odoText">odometer 0 m</span>
    </div>
    <div class="stage-wrap">
      <div class="stage">
        <span class="overlay-tag"><b>SIH26037</b> · <span id="modeTag">SIMULATED FEED</span></span>
        <canvas id="camCanvas" class="fill"></canvas>
        <canvas id="overlayCanvas" class="fill"></canvas>
        <div id="pauseTag">PAUSED</div>
      </div>
    </div>
    <div class="legend">
      <div><span class="swatch" style="background:#f59e0b"></span>Pothole</div>
      <div><span class="swatch" style="background:#ef4444"></span>Animal</div>
      <div><span class="swatch" style="background:#f472b6"></span>Person</div>
      <div><span class="swatch" style="background:#3b82f6"></span>Vehicle</div>
      <div><span class="swatch" style="background:#22d3ee"></span>Planned trajectory</div>
    </div>
  </section>

  <section class="panel">
    <div class="panel-head">
      <h2>Bird's-Eye Planner · A*</h2>
      <span id="bevMeta">96 × 144 @ 0.4 m</span>
    </div>
    <div class="stage-wrap">
      <div class="stage">
        <canvas id="bevCanvas" class="fill"></canvas>
      </div>
    </div>
  </section>
</main>

<section class="log-wrap">
  <div class="panel">
    <div class="panel-head">
      <h2>Live Obstacle Log · Avoidance Zone (&le;28 m)</h2>
      <div class="log-actions">
        <span class="badge">IN ZONE <b id="zoneCount">0</b></span>
        <span class="badge">TOTAL <b id="logTotal">0</b></span>
        <button class="btn small" id="btnRefreshLogs">Refresh</button>
        <button class="btn small primary" id="btnExport">Export Log to CSV</button>
      </div>
    </div>
    <div class="table-wrap">
      <table id="logTable">
        <thead>
          <tr>
            <th>ID</th>
            <th>Timestamp</th>
            <th>Detected Entity</th>
            <th>Distance (m)</th>
            <th>Action Taken (Steering / Speed)</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id="logBody">
          <tr class="empty-row"><td colspan="6">No obstacle encounters logged yet — monitoring avoidance zone…</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</section>

<footer id="telemetry">
  <div class="stat" id="cardSpeed">
    <div class="stat-top"><span class="stat-label">Speed</span><span class="stat-unit">km/h</span></div>
    <div class="stat-value" id="vSpeed">0.0</div>
    <canvas class="spark" id="sparkSpeed"></canvas>
  </div>
  <div class="stat" id="cardSteer">
    <div class="stat-top"><span class="stat-label">Steering Angle</span><span class="stat-unit">deg</span></div>
    <div class="stat-value" id="vSteer">0.0</div>
    <canvas class="spark" id="sparkSteer"></canvas>
  </div>
  <div class="stat" id="cardLatency">
    <div class="stat-top"><span class="stat-label">Re-plan Latency</span><span class="stat-unit">ms</span></div>
    <div class="stat-value" id="vLatency">0.0</div>
    <canvas class="spark" id="sparkLatency"></canvas>
  </div>
  <div class="stat" id="cardTtc">
    <div class="stat-top"><span class="stat-label">Time To Collision</span><span class="stat-unit">s</span></div>
    <div class="stat-value" id="vTtc">&#8734;</div>
    <canvas class="spark" id="sparkTtc"></canvas>
  </div>
  <div class="status-pill">
    <span id="statusPill" data-s="CRUISE">CRUISE</span>
    <small>PLANNER STATE</small>
  </div>
</footer>

<div class="modal-backdrop" id="infoModal">
  <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modalTitle">
    <button class="modal-x" id="modalClose" aria-label="Close">&times;</button>
    <h2 class="modal-title" id="modalTitle">SIH26037 · Problem Statement</h2>
    <p class="modal-sub">Adaptive Path Planning &amp; Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads — sponsored by <b>MathWorks</b>.</p>

    <section class="m-sec">
      <h3>The Challenge</h3>
      <p>Autonomous navigation on Indian roads must cope with <b>lane-less traffic, stray livestock,
      potholes and irregular obstacles</b> that appear with little warning. This prototype demonstrates a
      complete onboard stack: camera perception, bird's-eye obstacle mapping, continuous A* replanning and
      time-to-collision based emergency braking — running end-to-end at 10 Hz.</p>
    </section>

    <section class="m-sec">
      <h3>Perception Layer</h3>
      <ul>
        <li><b>Simulated feed</b> — synthetic unstructured-road scene with ground-truth potholes, crossing cattle and oncoming traffic (640 × 360 @ 10 Hz).</li>
        <li><b>YOLOv8n</b> (Ultralytics) — optional live inference on uploaded video; COCO classes mapped to animal / vehicle / person.</li>
        <li><b>Classical fallback</b> — dark-blob contour detector for potholes when no model is available.</li>
        <li><b>Ground projection</b> — pinhole inverse-perspective maps every bounding box to metres in the ego frame.</li>
      </ul>
    </section>

    <section class="m-sec">
      <h3>Path Planner</h3>
      <ul>
        <li><b>Occupancy grid</b> — 96 × 144 cells @ 0.4 m (38.4 m × 57.6 m horizon); hard-inflated obstacles for animals/vehicles, soft costs for potholes, road-corridor walls.</li>
        <li><b>A* search</b> — 8-connected, octile heuristic, partial-path fallback when the goal is unreachable.</li>
        <li><b>Smoothing</b> — line-of-sight shortcutting + Catmull-Rom spline resampled at 0.35 m.</li>
        <li><b>Control</b> — pure-pursuit steering (2.6 m wheelbase) with curvature- and TTC-based speed governor.</li>
      </ul>
    </section>

    <section class="m-sec">
      <h3>Safety &amp; Telemetry</h3>
      <ul>
        <li><b>TTC monitor</b> — time-to-collision from BEV obstacle range; &lt; 1.6 s triggers full brake.</li>
        <li><b>Event logger</b> — every zone encounter (≤ 28 m) persisted to SQLite with closest approach, steering and speed; exportable to CSV.</li>
        <li><b>Live stream</b> — WebSocket push of frames, grid, trajectory and telemetry at 10 Hz.</li>
      </ul>
    </section>

    <div class="chips">
      <span>FastAPI</span><span>WebSocket</span><span>OpenCV</span><span>NumPy</span>
      <span>SQLite</span><span>YOLOv8</span><span>A* Search</span><span>Catmull-Rom</span>
      <span>Pure Pursuit</span><span>Canvas 2D</span>
    </div>
  </div>
</div>

<script src="/static/app.js"></script>
</body>
</html>

GRID_HEIGHT = 144
CELL_SIZE = 0.4

MAX_RANGE = 57.6
AVOIDANCE_ZONE = 28.0

WHEELBASE = 2.6
BRAKE_TTC = 1.6

TARGET_SPEED = 45.0
UPDATE_HZ = 10


# ============================================================
# DATABASE
# ============================================================

db = sqlite3.connect("obstacles.db", check_same_thread=False)

db.execute("""
CREATE TABLE IF NOT EXISTS obstacle_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    entity TEXT,
    distance REAL,
    steering REAL,
    speed REAL,
    status TEXT
)
""")

db.commit()


def log_obstacle(entity, distance, steering, speed, status):
    db.execute("""
        INSERT INTO obstacle_log
        (timestamp, entity, distance, steering, speed, status)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        entity,
        distance,
        steering,
        speed,
        status
    ))

    db.commit()


# ============================================================
# OBSTACLE MODEL
# ============================================================

class Obstacle:

    def __init__(self, entity, x, y, width=1.5, height=1.5):
        self.entity = entity
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def distance(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)


# ============================================================
# SIMULATION
# ============================================================

class Simulation:

    def __init__(self):
        self.reset()

    def reset(self):

        self.vehicle_x = 0.0
        self.vehicle_y = 0.0

        self.speed = 45.0
        self.steering = 0.0

        self.obstacles = []

        self.status = "CRUISE"

        self.odometer = 0.0

        self.create_obstacles()

    def create_obstacles(self):

        self.obstacles = [

            Obstacle(
                "vehicle",
                18,
                random.uniform(-4, 4),
                2.0,
                4.0
            ),

            Obstacle(
                "animal",
                30,
                random.uniform(-5, 5),
                1.5,
                2.0
            ),

            Obstacle(
                "pothole",
                22,
                random.uniform(-3, 3),
                1.0,
                1.0
            ),

            Obstacle(
                "person",
                40,
                random.uniform(-5, 5),
                0.7,
                0.7
            )
        ]

    def update(self, dt):

        speed_ms = self.speed / 3.6

        self.odometer += speed_ms * dt

        # Move obstacles toward vehicle
        for obstacle in self.obstacles:
            obstacle.x -= speed_ms * dt

        # Remove objects behind vehicle
        self.obstacles = [
            o for o in self.obstacles
            if o.x > -5
        ]

        # Add new obstacles
        if random.random() < 0.02:

            entity = random.choice([
                "vehicle",
                "animal",
                "pothole",
                "person"
            ])

            self.obstacles.append(
                Obstacle(
                    entity,
                    random.uniform(35, 55),
                    random.uniform(-6, 6)
                )
            )


simulation = Simulation()


# ============================================================
# OCCUPANCY GRID
# ============================================================

def create_grid(obstacles):

    grid = np.zeros(
        (GRID_HEIGHT, GRID_WIDTH),
        dtype=np.float32
    )

    center_x = GRID_WIDTH // 2

    for obstacle in obstacles:

        gx = int(
            center_x +
            obstacle.y / CELL_SIZE
        )

        gy = int(
            obstacle.x / CELL_SIZE
        )

        radius = 2

        if obstacle.entity in ["vehicle", "animal", "person"]:
            cost = 1000
        else:
            cost = 20

        for dy in range(-radius, radius + 1):

            for dx in range(-radius, radius + 1):

                x = gx + dx
                y = gy + dy

                if (
                    0 <= x < GRID_WIDTH
                    and
                    0 <= y < GRID_HEIGHT
                ):
                    grid[y, x] = max(
                        grid[y, x],
                        cost
                    )

    # Road boundaries

    road_left = int(center_x - 12)
    road_right = int(center_x + 12)

    grid[:, :road_left] = 1000
    grid[:, road_right:] = 1000

    return grid


# ============================================================
# A* PATH PLANNER
# ============================================================

MOVES = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1)
]


def heuristic(a, b):

    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])

    return max(dx, dy) + (
        math.sqrt(2) - 1
    ) * min(dx, dy)


def astar(grid, start, goal):

    queue = []

    heapq.heappush(
        queue,
        (0, start)
    )

    came_from = {}
    cost_so_far = {
        start: 0
    }

    while queue:

        _, current = heapq.heappop(queue)

        if current == goal:

            path = []

            while current in came_from:

                path.append(current)
                current = came_from[current]

            path.append(start)

            return path[::-1]

        for dx, dy in MOVES:

            nx = current[0] + dx
            ny = current[1] + dy

            if not (
                0 <= nx < GRID_WIDTH
                and
                0 <= ny < GRID_HEIGHT
            ):
                continue

            if grid[ny, nx] >= 1000:
                continue

            movement_cost = (
                math.sqrt(2)
                if dx != 0 and dy != 0
                else 1
            )

            new_cost = (
                cost_so_far[current]
                + movement_cost
                + grid[ny, nx] * 0.05
            )

            neighbor = (nx, ny)

            if (
                neighbor not in cost_so_far
                or
                new_cost < cost_so_far[neighbor]
            ):

                cost_so_far[neighbor] = new_cost

                priority = (
                    new_cost
                    + heuristic(
                        neighbor,
                        goal
                    )
                )

                heapq.heappush(
                    queue,
                    (priority, neighbor)
                )

                came_from[neighbor] = current

    return []


# ============================================================
# PATH SMOOTHING
# ============================================================

def smooth_path(path):

    if len(path) < 3:
        return path

    result = [path[0]]

    for i in range(1, len(path) - 1):

        p0 = np.array(path[i - 1])
        p1 = np.array(path[i])
        p2 = np.array(path[i + 1])

        midpoint = (
            0.25 * p0
            +
            0.5 * p1
            +
            0.25 * p2
        )

        result.append(
            tuple(midpoint)
        )

    result.append(path[-1])

    return result


# ============================================================
# PURE PURSUIT
# ============================================================

def pure_pursuit(path):

    if not path:
        return 0.0

    center = GRID_WIDTH // 2

    # Choose look-ahead point

    look_index = min(
        10,
        len(path) - 1
    )

    gx, gy = path[look_index]

    lateral_error = (
        gx - center
    ) * CELL_SIZE

    lookahead = max(
        3.0,
        simulation.speed / 10
    )

    angle = math.atan2(
        lateral_error,
        lookahead
    )

    steering = math.degrees(
        math.atan(
            2 *
            WHEELBASE *
            math.sin(angle)
            /
            lookahead
        )
    )

    return max(
        -35,
        min(35, steering)
    )


# ============================================================
# TTC SAFETY
# ============================================================

def calculate_ttc(obstacles):

    dangerous = [
        o for o in obstacles
        if o.x > 0
        and abs(o.y) < 2.0
    ]

    if not dangerous:
        return float("inf")

    closest = min(
        dangerous,
        key=lambda o: o.x
    )

    speed_ms = max(
        simulation.speed / 3.6,
        0.1
    )

    return closest.x / speed_ms


def safety_controller(ttc):

    if ttc < BRAKE_TTC:

        simulation.status = "BRAKE"
        simulation.speed = 0.0

        return

    if ttc < 4.0:

        simulation.status = "REROUTED"

        simulation.speed = min(
            simulation.speed,
            25.0
        )

    else:

        simulation.status = "CRUISE"

        simulation.speed = min(
            TARGET_SPEED,
            simulation.speed + 1
        )


# ============================================================
# PLANNER
# ============================================================

def run_planner():

    grid = create_grid(
        simulation.obstacles
    )

    start = (
        GRID_WIDTH // 2,
        2
    )

    goal = (
        GRID_WIDTH // 2,
        GRID_HEIGHT - 10
    )

    start_time = time.perf_counter()

    path = astar(
        grid,
        start,
        goal
    )

    latency = (
        time.perf_counter()
        - start_time
    ) * 1000

    path = smooth_path(path)

    steering = pure_pursuit(path)

    ttc = calculate_ttc(
        simulation.obstacles
    )

    safety_controller(ttc)

    simulation.steering = steering

    return grid, path, latency, ttc


# ============================================================
# JSON DATA
# ============================================================

def get_state():

    grid, path, latency, ttc = run_planner()

    obstacles = []

    for obstacle in simulation.obstacles:

        distance = obstacle.distance()

        if distance <= AVOIDANCE_ZONE:

            obstacles.append({
                "entity": obstacle.entity,
                "x": round(obstacle.x, 2),
                "y": round(obstacle.y, 2),
                "distance": round(distance, 2)
            })

            log_obstacle(
                obstacle.entity,
                distance,
                simulation.steering,
                simulation.speed,
                simulation.status
            )

    return {

        "status": simulation.status,

        "speed": round(
            simulation.speed,
            2
        ),

        "steering": round(
            simulation.steering,
            2
        ),

        "latency_ms": round(
            latency,
            2
        ),

        "ttc": (
            None
            if math.isinf(ttc)
            else round(ttc, 2)
        ),

        "odometer": round(
            simulation.odometer,
            2
        ),

        "obstacles": obstacles,

        "path": [
            {
                "x": p[0],
                "y": p[1]
            }
            for p in path
        ],

        "grid_size": [
            GRID_WIDTH,
            GRID_HEIGHT
        ]
    }


# ============================================================
# WEB SOCKET
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()

    previous = time.time()

    try:

        while True:

            now = time.time()

            dt = now - previous
            previous = now

            simulation.update(dt)

            state = get_state()

            await websocket.send_json(
                state
            )

            await asyncio.sleep(
                1 / UPDATE_HZ
            )

    except Exception:
        pass


# ============================================================
# SIMPLE PYTHON DASHBOARD
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<title>SIH26037 Python Dashboard</title>

<style>

body {
    background: #10151f;
    color: white;
    font-family: Arial;
    padding: 30px;
}

.card {
    background: #192232;
    padding: 20px;
    margin: 10px;
    border-radius: 10px;
    display: inline-block;
}

.value {
    font-size: 30px;
    font-weight: bold;
}

#log {
    margin-top: 20px;
    background: #080b10;
    padding: 15px;
    height: 250px;
    overflow: auto;
}

</style>

</head>

<body>

<h1>
SIH26037: Adaptive Path Planning &
Collision Avoidance
</h1>

<div class="card">
Speed
<div class="value" id="speed">0</div>
km/h
</div>

<div class="card">
Steering
<div class="value" id="steering">0</div>
degrees
</div>

<div class="card">
TTC
<div class="value" id="ttc">∞</div>
seconds
</div>

<div class="card">
Planner State
<div class="value" id="status">CRUISE</div>
</div>

<div class="card">
A* Latency
<div class="value" id="latency">0</div>
ms
</div>

<h2>Detected Obstacles</h2>

<div id="log"></div>

<script>

const socket = new WebSocket(
    "ws://" + location.host + "/ws"
);

socket.onmessage = function(event) {

    const data = JSON.parse(
        event.data
    );

    document.getElementById(
        "speed"
    ).innerText = data.speed;

    document.getElementById(
        "steering"
    ).innerText = data.steering;

    document.getElementById(
        "latency"
    ).innerText = data.latency_ms;

    document.getElementById(
        "status"
    ).innerText = data.status;

    document.getElementById(
        "ttc"
    ).innerText =
        data.ttc === null
        ? "∞"
        : data.ttc;

    let html = "";

    data.obstacles.forEach(
        function(obstacle) {

            html +=
                "<p>" +
                "<b>" +
                obstacle.entity +
                "</b> — " +
                obstacle.distance +
                " m" +
                "</p>";

        }
    );

    document.getElementById(
        "log"
    ).innerHTML = html;

};

</script>

</body>

</html>
"""


@app.get("/")
async def home():

    return HTMLResponse(
        HTML
    )


# ============================================================
# RESET
# ============================================================

@app.post("/reset")
async def reset():

    simulation.reset()

    return {
        "message": "Simulation reset"
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
