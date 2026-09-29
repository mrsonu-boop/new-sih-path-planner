"""SIH26037 · Adaptive Path Planning & Collision Avoidance — Streamlit dashboard.

Single-file app reusing the backend pipeline (scene simulator / YOLOv8 perception,
BEV occupancy grid, A* planner, obstacle tracker, SQLite event log).
Run with:  streamlit run app.py
"""
from __future__ import annotations

import math
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_webrtc as webrtc

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.logger import EventLogger
from backend.main import Engine, LABEL_COLORS, ObstacleTracker
from backend.occupancy_grid import (
    CELL_RES,
    GRID_H,
    GRID_W,
    ROAD_HALF_WIDTH_M,
    Z_MAX,
    build_grid,
    constant_road_profile,
    project_image_to_world,
    variable_road_profile,
)
from backend.path_planner import plan_path, pure_pursuit_steer
from backend.perception import IMG_H, IMG_W, PerceptionEngine, SceneSimulator

TICK_S = 0.4
WHEELBASE_M = 2.6
BEV_SCALE = 4
BEV_ROW_START = 64
BEV_PX_W = GRID_W * BEV_SCALE
BEV_PX_H = (GRID_H - BEV_ROW_START) * BEV_SCALE
COL_PATH = (238, 211, 34)
COL_CAR = (235, 232, 226)
COL_GRID = (52, 60, 74)
COL_TEXT = (150, 160, 178)


def hex_bgr(hexcol: str) -> tuple:
    return tuple(int(c) for c in bytes.fromhex(hexcol.lstrip("#")))[::-1]


class WebcamFeed:
    """Latest-frame holder for the WebRTC webcam stream.

    ``streamlit-webrtc`` invokes ``video_frame_callback`` on the aiortc
    worker thread, not the Streamlit script thread, so every frame crosses a
    thread boundary. Session state is not safe to touch from that thread,
    hence this plain lock-guarded holder.
    """

    STALE_S = 1.5

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._ts = 0.0
        self._fps = 0.0

    def put(self, frame) -> None:
        """Store the newest frame. Called on the aiortc worker thread."""
        rgb = frame.to_ndarray(format="rgb24")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with self._lock:
            now = time.time()
            if self._ts:
                dt = now - self._ts
                if 0.0 < dt < 1.0:
                    inst = 1.0 / dt
                    self._fps = inst if self._fps == 0.0 else 0.7 * self._fps + 0.3 * inst
            self._frame = bgr
            self._ts = now

    def latest(self, max_age: float = STALE_S) -> np.ndarray | None:
        """Most recent BGR frame, or None if the stream is idle or has stalled."""
        with self._lock:
            if self._frame is None or (time.time() - self._ts) > max_age:
                return None
            return self._frame

    def reset(self) -> None:
        with self._lock:
            self._frame = None
            self._ts = 0.0
            self._fps = 0.0

    @property
    def live(self) -> bool:
        return self.latest() is not None

    @property
    def fps(self) -> float:
        return self._fps


st.set_page_config(
    page_title="SIH26037 · Adaptive Path Planner",
    page_icon=":material/route:",
    layout="wide",
)


@st.cache_resource
def get_perception() -> PerceptionEngine:
    return PerceptionEngine()


@st.cache_resource
def get_event_logger() -> EventLogger:
    return EventLogger()


def init_state() -> None:
    defaults = {
        "sim": SceneSimulator(seed=int(time.time())),
        "tracker": ObstacleTracker(),
        "speed_kmh": 38.0,
        "steering_deg": 0.0,
        "ego_u": 0.0,
        "odometer_m": 0.0,
        "paused": False,
        "seq": 0,
        "video_acc": 0.0,
        "fps": 25.0,
        "source_key": None,
        "cap": None,
        "webcam": None,
        "webcam_on": True,
        "webcam_playing": False,
        "hist_speed": [],
        "hist_steer": [],
        "hist_lat": [],
        "hist_ttc": [],
        "last_cam": None,
        "last_bev": None,
        "tel": {
            "status": "CRUISE", "latency_ms": 0.0, "ttc_s": None,
            "planner_status": "optimal", "in_zone": 0, "detector": "simulated-gt",
            "source": "simulation",
        },
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_world() -> None:
    ss = st.session_state
    ss.sim = SceneSimulator(seed=int(time.time()))
    ss.tracker.reset()
    ss.speed_kmh = 38.0
    ss.steering_deg = 0.0
    ss.ego_u = 0.0
    ss.odometer_m = 0.0
    ss.video_acc = 0.0
    ss.hist_speed, ss.hist_steer, ss.hist_lat, ss.hist_ttc = [], [], [], []
    ss.last_cam, ss.last_bev = None, None


def resolve_source(source: str, upload) -> str:
    ss = st.session_state
    file_id = getattr(upload, "id", None) if source == "Video file" else None
    key = (source, file_id, bool(ss.get("webcam_on")))
    if ss.source_key == key:
        return source if (source != "Video file" or upload) else "Simulation"

    if ss.cap is not None:
        ss.cap.release()
        ss.cap = None
    if ss.get("webcam") is not None:
        try:
            ss.webcam.reset()
        except Exception:
            pass
    reset_world()

    if source == "Webcam":
        if not ss.get("webcam_on", True):
            st.sidebar.caption(":red[Webcam access disabled — falling back to simulation.]")
            ss.source_key = ("Simulation", None, False)
            return "Simulation"
        if ss.get("webcam") is None:
            ss.webcam = WebcamFeed()
        ss.source_key = key
        return "Webcam"

    if source == "Video file" and upload is not None:
        path = Path(tempfile.gettempdir()) / f"sih26037_{file_id}_{upload.name}"
        if not path.exists():
            path.write_bytes(upload.getvalue())
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            ss.cap = cap
            ss.fps = max(cap.get(cv2.CAP_PROP_FPS), 1.0) or 25.0
            ss.source_key = key
            return "Video file"
        st.sidebar.caption(":red[Could not decode that video — falling back to simulation.]")

    ss.source_key = ("Simulation", None, bool(ss.get("webcam_on")))
    return "Simulation"


def grab_frame(active: str, yolo_on: bool):
    ss = st.session_state
    ss.sim.unstructured = bool(ss.get("unstructured_mode"))
    if active == "Simulation":
        return (*ss.sim.step(TICK_S, ss.speed_kmh / 3.6), "simulation", "simulated-gt")

    frame = None
    if active == "Webcam":
        feed = ss.get("webcam")
        if feed is not None:
            frame = feed.latest()
            if frame is not None:
                ss.fps = feed.fps or 30.0
    elif ss.cap is not None:
        ss.video_acc += TICK_S * ss.fps
        while ss.video_acc >= 1.0:
            ss.video_acc -= 1.0
            ok, f = ss.cap.read()
            if not ok:
                ss.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, f = ss.cap.read()
            if ok:
                frame = f
    if frame is None and active == "Webcam":
        return (*ss.sim.step(TICK_S, ss.speed_kmh / 3.6), "simulation", "simulated-gt")
    if frame is None:
        blank = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        cv2.putText(blank, "NO SIGNAL", (205, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 90, 90), 2)
        return blank, [], "video", "off"
    if frame.shape[:2] != (IMG_H, IMG_W):
        frame = cv2.resize(frame, (IMG_W, IMG_H))

    engine = get_perception()
    if yolo_on:
        dets = engine.detect(frame)
        detector = "yolov8" if engine.model is not None else "classical"
    else:
        dets, detector = [], "off"
    return frame, dets, "video", detector


def compute_ttc(dets, speed_kmh):
    v = speed_kmh / 3.6
    best = None
    for det in dets:
        if det.get("label") == "pothole":
            continue
        box = det["box"]
        p = project_image_to_world((box[0] + box[2]) / 2.0, min(box[3], IMG_H - 2))
        if p is None:
            continue
        u, z = p
        if not (0.5 < z < 55.0) or abs(u) > 2.3:
            continue
        ttc = z / max(v + {"vehicle": 6.0}.get(det.get("label"), 0.4), 0.6)
        best = ttc if best is None else min(best, ttc)
    return best


def corridor_obstruction(grid):
    c0, c1 = GRID_W // 2 - 3, GRID_W // 2 + 4
    for r in range(GRID_H - 6, 2, -1):
        if grid.hard[r, c0:c1].any():
            return ((GRID_H - 1) - r) * CELL_RES
    return None


def step_world(active: str, yolo_on: bool) -> None:
    ss = st.session_state
    frame, dets, source, detector = grab_frame(active, yolo_on)

    profile = variable_road_profile if ss.get("unstructured_mode") else constant_road_profile
    grid = build_grid(dets, road_profile=profile)
    t0 = time.perf_counter()
    plan = plan_path(grid, ego_u=ss.ego_u, target_u=0.0)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    raw_steer = pure_pursuit_steer(plan.world, ss.ego_u, ss.speed_kmh / 3.6, WHEELBASE_M)
    ss.steering_deg = 0.6 * raw_steer + 0.4 * ss.steering_deg

    ttc = compute_ttc(dets, ss.speed_kmh)
    obstruction = corridor_obstruction(grid)

    target = st.session_state.get("target_kmh", 45.0)
    status = "CRUISE"
    if plan.blocked:
        target_speed, status = 0.0, "OBSTRUCTED"
    elif ttc is not None and ttc < 1.6:
        target_speed, status = 0.0, "BRAKE"
    elif obstruction is not None and obstruction < 10.0:
        target_speed, status = min(target, 20.0), "OBSTRUCTED"
    elif plan.status != "optimal":
        status = "REROUTED"
    else:
        target_speed = target

    if target_speed == 0.0:
        accel = -45.0
    elif target_speed >= ss.speed_kmh:
        accel = 9.0
    else:
        accel = -22.0
    new_speed = ss.speed_kmh + accel * TICK_S
    new_speed = min(new_speed, target_speed) if accel >= 0 else max(new_speed, target_speed)
    ss.speed_kmh = max(0.0, new_speed)

    v = ss.speed_kmh / 3.6
    ss.ego_u = max(-3.3, min(3.3,
        ss.ego_u + math.sin(math.radians(ss.steering_deg)) * v * TICK_S * 0.35))
    ss.odometer_m += v * TICK_S
    ss.seq += 1

    now = time.time()
    _, closed = ss.tracker.update(dets, ss.steering_deg, ss.speed_kmh, now)
    for tr in closed:
        get_event_logger().log_event(Engine._close_record(tr))

    ss.tel = {
        "status": status, "latency_ms": latency_ms, "ttc_s": ttc,
        "planner_status": plan.status, "in_zone": len(ss.tracker.tracks),
        "detector": detector, "source": source, "grid": grid, "plan": plan,
        "dets": dets, "frame": frame,
    }
    for hist, val in ((ss.hist_speed, ss.speed_kmh), (ss.hist_steer, ss.steering_deg),
                      (ss.hist_lat, min(latency_ms, 120.0)),
                      (ss.hist_ttc, 10.0 if ttc is None else min(ttc, 10.0))):
        hist.append(float(val))
        if len(hist) > 48:
            hist.pop(0)


def draw_camera() -> np.ndarray:
    ss = st.session_state
    tel = ss.tel
    frame = tel["frame"].copy()
    plan = tel["plan"]

    if len(plan.image) > 1:
        pts = np.array([[int(x), int(y)] for x, y in plan.image], np.int32)
        cv2.polylines(frame, [pts], False, COL_PATH, 2, cv2.LINE_AA)

    for det in tel["dets"]:
        x1, y1, x2, y2 = det["box"]
        color = hex_bgr(LABEL_COLORS.get(det.get("label", ""), "#94a3b8"))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        c = max(4, min(10, (x2 - x1) // 2, (y2 - y1) // 2))
        for (px, py), (dx, dy) in (((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
                                   ((x2, y2), (-1, -1)), ((x1, y2), (1, -1))):
            cv2.line(frame, (px, py), (px + c * dx, py), color, 2)
            cv2.line(frame, (px, py), (px, py + c * dy), color, 2)
        label = f"{det.get('name', det['label'])} {det.get('conf', 0):.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ly = y1 - th - 8 if y1 - th - 8 > 0 else y1 + 4
        cv2.rectangle(frame, (x1, ly), (x1 + tw + 8, ly + th + 6), (18, 14, 10), -1)
        cv2.putText(frame, label, (x1 + 4, ly + th + 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"SIH26037 · {tel['source'].upper()} · {tel['detector'].upper()}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 220, 235), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{ss.speed_kmh:.0f} km/h", (IMG_W - 110, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_PATH, 1, cv2.LINE_AA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_bev() -> np.ndarray:
    ss = st.session_state
    tel = ss.tel
    grid, plan = tel["grid"], tel["plan"]

    cell = np.full((GRID_H, GRID_W, 3), (30, 24, 18), np.uint8)
    cell[grid.hard] = (68, 68, 239)
    soft_mask = grid.soft > 0.02
    if soft_mask.any():
        a = np.clip(grid.soft[soft_mask], 0.0, 1.0)[:, None]
        amber = np.array([11, 158, 245], dtype=float)
        cell[soft_mask] = (a * amber + (1.0 - a) * cell[soft_mask]).astype(np.uint8)

    bev = cv2.resize(cell, (BEV_PX_W, GRID_H * BEV_SCALE), interpolation=cv2.INTER_NEAREST)
    ox = lambda c: int((c + 0.5) * BEV_SCALE)
    oy = lambda r: int((r + 0.5) * BEV_SCALE)

    for zm in range(10, int(Z_MAX), 10):
        r = int((Z_MAX - zm) / CELL_RES)
        if r < BEV_ROW_START:
            continue
        y = oy(r)
        cv2.line(bev, (0, y), (BEV_PX_W, y), COL_GRID, 1)
        cv2.putText(bev, f"{zm}m", (BEV_PX_W - 36, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, COL_TEXT, 1, cv2.LINE_AA)
    for u_edge in (-ROAD_HALF_WIDTH_M, ROAD_HALF_WIDTH_M):
        c_edge = int(round(GRID_W / 2 + u_edge / CELL_RES))
        x = min(max(ox(c_edge - 0.5), 1), BEV_PX_W - 2)
        cv2.line(bev, (x, oy(BEV_ROW_START)), (x, BEV_PX_H - 1), (170, 178, 190), 1)

    for c, r in plan.cells_raw:
        cv2.circle(bev, (ox(c), oy(r)), 2, (110, 120, 138), -1)
    if len(plan.cells_smooth) > 1:
        pts = np.array([[ox(c), oy(r)] for c, r in plan.cells_smooth], np.int32)
        cv2.polylines(bev, [pts], False, COL_PATH, 2, cv2.LINE_AA)

    goal_row = max(int(7.5), BEV_ROW_START + 2)
    goal = (ox(GRID_W // 2 - 1), oy(goal_row))
    cv2.line(bev, goal, (goal[0], goal[1] + 18), (99, 211, 52), 2)
    cv2.fillPoly(bev, [np.array([[goal[0], goal[1]], [goal[0] + 14, goal[1] + 5],
                                 [goal[0], goal[1] + 10]], np.int32)], (99, 211, 52))

    car = (ox(GRID_W / 2 - 0.5 + ss.ego_u / CELL_RES), oy((Z_MAX - 1.4) / CELL_RES))
    cone_len = int(12 / CELL_RES * BEV_SCALE)
    overlay = bev.copy()
    cv2.fillPoly(overlay, [np.array([[car[0], car[1]],
                                     [car[0] - 90, car[1] - cone_len],
                                     [car[0] + 90, car[1] - cone_len]], np.int32)],
                 (120, 110, 34))
    cv2.addWeighted(overlay, 0.16, bev, 0.84, 0, bev)
    box = cv2.boxPoints(((car[0], car[1] - 2),
                         (int(1.9 / CELL_RES * BEV_SCALE), int(4.3 / CELL_RES * BEV_SCALE)),
                         -ss.steering_deg)).astype(np.int32)
    cv2.fillPoly(bev, [box], COL_CAR)
    cv2.polylines(bev, [box], True, COL_PATH, 1)

    bev = bev[BEV_ROW_START * BEV_SCALE:, :].copy()
    cv2.putText(bev, f"A* {tel['planner_status'].upper()} · {plan.nodes} nodes · "
                     f"{tel['latency_ms']:.1f} ms",
                (8, BEV_PX_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_TEXT, 1, cv2.LINE_AA)
    return cv2.cvtColor(bev, cv2.COLOR_BGR2RGB)


init_state()

with st.sidebar:
    st.markdown("### :material/route: SIH26037 planner")
    st.caption("Adaptive path planning & collision avoidance for unstructured roads")

    source = st.segmented_control(
        "Input source", ["Simulation", "Webcam", "Video file"], default="Simulation")
    upload = None
    if source == "Video file":
        upload = st.file_uploader("Upload a road clip", type=["mp4", "avi", "mov", "mkv"])
    if source == "Webcam":
        ss = st.session_state
        ss.webcam_on = st.toggle(
            "Use browser webcam", value=bool(ss.get("webcam_on", True)),
            help="Streams through WebRTC, so it works on Streamlit Cloud where "
                 "no local camera hardware exists. Turn off to use the simulated "
                 "road instead.")
        if ss.get("webcam") is None:
            ss.webcam = WebcamFeed()
        if ss.webcam_on:
            try:
                ctx = webrtc.webrtc_streamer(
                    key="webcam_rtc",
                    mode=webrtc.WebRtcMode.SENDONLY,
                    media_stream_constraints={"video": True, "audio": False},
                    video_frame_callback=ss.webcam.put,
                    sendback_video=False,
                    sendback_audio=False,
                )
                ss.webcam_playing = bool(getattr(ctx.state, "playing", False))
            except Exception as exc:
                st.caption(f":red[WebRTC unavailable ({type(exc).__name__}) — "
                           f"running the simulated road.]")
                ctx = None
            if ctx is not None:
                if ss.webcam.live:
                    st.caption(f":green[Webcam live · {ss.webcam.fps:.0f} fps to planner]")
                elif ss.webcam_playing:
                    st.caption(":orange[Stream connected, waiting for frames…]")
                else:
                    st.caption(":orange[Press START and allow camera access. Until "
                               "frames arrive the planner runs the simulated road.]")
        else:
            st.caption(":orange[Webcam off — running the simulated road.]")
    yolo_on = st.toggle(
        "YOLOv8 detection", value=True, disabled=(source == "Simulation"),
        help="Simulation uses ground-truth boxes. Webcam and uploaded video run "
             "YOLOv8n (Ultralytics), with a classical dark-blob fallback if the "
             "model is unavailable.")

    st.session_state.target_kmh = st.slider("Target speed (km/h)", 15, 90, 45, 5)

    st.session_state.unstructured_mode = st.toggle(
        "Unstructured road mode", value=False,
        help="Replaces the fixed-width marked corridor with a variable-width, "
             "unmarked road that narrows to a single-vehicle pinch point, with "
             "soft (crossable) shoulders instead of hard walls — closer to a "
             "real unmarked village road than a structured lane.")

    with st.container(horizontal=True):
        if st.button("Pause" if not st.session_state.paused else "Resume",
                     icon=":material/pause:" if not st.session_state.paused
                     else ":material/play_arrow:",
                     width="stretch"):
            st.session_state.paused = not st.session_state.paused
            st.rerun()
        if st.button("Reset", icon=":material/refresh:", width="stretch"):
            reset_world()
            st.rerun()

    engine = get_perception()
    if source == "Simulation":
        st.caption(":green[Detector: simulated ground truth]")
    elif engine.ensure_model():
        st.caption(":green[Detector: YOLOv8n ready]")
    else:
        st.caption(f":orange[YOLO unavailable ({str(engine.error)[:60]}…) — classical fallback]")

    with st.expander("About SIH26037", icon=":material/info:"):
        st.markdown(
            "Adaptive path planning and collision avoidance for autonomous vehicles "
            "on unstructured Indian roads — potholes, stray livestock and irregular "
            "traffic, replanned in real time.\n\n"
            "- **Perception** — simulated GT feed, YOLOv8n or classical fallback\n"
            "- **Planner** — 96×144 BEV grid @ 0.4 m, A* + Catmull-Rom smoothing\n"
            "- **Safety** — pure-pursuit steering, TTC-based emergency braking")

active = resolve_source(source, upload)


@st.fragment(run_every=TICK_S)
def live_dashboard():
    ss = st.session_state
    if not ss.paused or ss.last_cam is None:
        step_world(active, yolo_on)
        ss.last_cam = draw_camera()
        ss.last_bev = draw_bev()

    tel = ss.tel
    status_color = {"CRUISE": "green", "REROUTED": "orange",
                    "OBSTRUCTED": "orange", "BRAKE": "red"}.get(tel["status"], "grey")

    cam_col, bev_col = st.columns([1.5, 1], gap="small")
    with cam_col:
        with st.container(border=True):
            st.markdown(f"**Perception · camera feed**  :{status_color}-badge[{tel['status']}]")
            st.image(ss.last_cam, width="stretch")
    with bev_col:
        with st.container(border=True):
            st.markdown("**Bird's-eye planner · A***")
            st.image(ss.last_bev, width="stretch")

    ttc_text = "clear" if tel["ttc_s"] is None else f"{tel['ttc_s']:.2f} s"
    spark = lambda h: h if len(h) > 1 else None
    with st.container(horizontal=True):
        st.metric("Speed", f"{ss.speed_kmh:.1f} km/h",
                  chart_data=spark(ss.hist_speed), chart_type="line", border=True)
        st.metric("Steering angle", f"{ss.steering_deg:+.1f}°",
                  chart_data=spark(ss.hist_steer), chart_type="line", border=True)
        st.metric("Re-plan latency", f"{tel['latency_ms']:.1f} ms",
                  chart_data=spark(ss.hist_lat), chart_type="line", border=True)
        st.metric("Time to collision", ttc_text,
                  chart_data=spark(ss.hist_ttc), chart_type="line", border=True)
        st.metric("Odometer", f"{ss.odometer_m:.0f} m", border=True)

    with st.container(border=True):
        head_l, head_r = st.columns([3, 1], vertical_alignment="center")
        with head_l:
            st.markdown(f"**Live obstacle log · avoidance zone (≤ 28 m)** · "
                        f"{tel['in_zone']} tracked")
        with head_r:
            events = get_event_logger().recent(200)
            if events:
                csv = ("ID,Timestamp,Entity,Distance_m,Steering_deg,Speed_kmh,Action,Status\n"
                       + "".join(
                           f"{e['track_id']},"
                           f"{datetime.fromtimestamp(e['entered_at']).isoformat()},"
                           f"{e['name']},{e['min_distance_m']},{e['steering_deg']},"
                           f"{e['speed_kmh']},\"{e['action']}\",{e['status']}\n"
                           for e in reversed(events)))
                st.download_button(":material/download: Export CSV", csv,
                                   mime="text/csv",
                                   file_name="sih26037_obstacle_log.csv",
                                   icon=":material/download:")
        if events:
            df = pd.DataFrame([{
                "ID": e["track_id"],
                "Time": datetime.fromtimestamp(e["entered_at"]).strftime("%H:%M:%S"),
                "Entity": (e["name"] or e["label"]).capitalize(),
                "Distance (m)": e["min_distance_m"],
                "Action": e["action"],
                "Status": "Avoided" if e["status"] == "AVOIDED" else "Cleared",
            } for e in events])
            st.dataframe(
                df, hide_index=True, height=190,
                column_config={
                    "Distance (m)": st.column_config.NumberColumn(format="%.1f"),
                })
        else:
            st.caption("No obstacle encounters logged yet — monitoring avoidance zone…")


live_dashboard()

