"""SIH26037 prototype server: streams simulated camera frames, detections,
occupancy grid and A* trajectory over a WebSocket at ~10 Hz."""
from __future__ import annotations

import asyncio
import base64
import json
import math
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from .logger import EventLogger
    from .occupancy_grid import (
        CELL_RES, GRID_H, GRID_W, ROAD_HALF_WIDTH_M, Z_MAX, build_grid, project_image_to_world,
    )
    from .path_planner import plan_path, pure_pursuit_steer
    from .perception import IMG_H, IMG_W, PerceptionEngine, SceneSimulator
except ImportError:
    from logger import EventLogger
    from occupancy_grid import (
        CELL_RES, GRID_H, GRID_W, ROAD_HALF_WIDTH_M, Z_MAX, build_grid, project_image_to_world,
    )
    from path_planner import plan_path, pure_pursuit_steer
    from perception import IMG_H, IMG_W, PerceptionEngine, SceneSimulator

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

TICK_S = 0.1
WHEELBASE_M = 2.6
CRUISE_KMH_DEFAULT = 45.0

LABEL_COLORS = {
    "pothole": "#f59e0b",
    "animal": "#ef4444",
    "person": "#f472b6",
    "vehicle": "#3b82f6",
}

ZONE_Z_MAX_M = 28.0
ZONE_U_MAX_M = 5.2
MATCH_GATE_M = 5.0
TRACK_LOST_S = 1.0
ZONE_EXIT_TICKS = 3
OPEN_MARGIN_M = 1.5
POS_EMA = 0.65
LOGGED_LABELS = {"animal", "person", "vehicle"}


class ObstacleTracker:
    """Nearest-neighbour multi-object tracker for the collision-avoidance zone."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tracks: Dict[str, Dict] = {}
        self.seq = 0

    @staticmethod
    def _in_zone(u: float, z: float) -> bool:
        return -2.0 < z <= ZONE_Z_MAX_M and abs(u) <= ZONE_U_MAX_M

    def update(self, detections: List[Dict], steer_deg: float, speed_kmh: float,
               now: float):
        opened, closed = [], []
        obs = []
        for det in detections:
            if det.get("label") not in LOGGED_LABELS:
                continue
            box = det["box"]
            p = project_image_to_world((box[0] + box[2]) / 2.0, min(box[3], IMG_H - 2))
            if p is not None:
                obs.append((det, p[0], p[1]))

        free = set(range(len(obs)))
        matched = set()
        candidates = []
        for tid, tr in self.tracks.items():
            for i, (_, u, z) in enumerate(obs):
                dist = math.hypot(u - tr["u"], z - tr["z"])
                if dist <= MATCH_GATE_M:
                    candidates.append((dist, tid, i))
        candidates.sort(key=lambda c: c[0])
        for _, tid, i in candidates:
            if tid in matched or i not in free:
                continue
            matched.add(tid)
            free.discard(i)
            det, u, z = obs[i]
            tr = self.tracks[tid]
            tr["label"] = det.get("label", tr["label"])
            tr["name"] = det.get("name") or tr["name"]
            self._observe(tr, u, z, steer_deg, speed_kmh, now)

        for i in sorted(free):
            det, u, z = obs[i]
            if not self._in_zone(u, z) or z > ZONE_Z_MAX_M - OPEN_MARGIN_M:
                continue
            self.seq += 1
            tid = f"OBS-{self.seq:03d}"
            tr = {
                "tid": tid, "label": det.get("label", "obstacle"),
                "name": det.get("name") or det.get("label", "obstacle"),
                "u": u, "z": z, "entered": now, "last": now, "out": 0,
                "min_dist": math.hypot(u, z),
                "steer_at_min": steer_deg, "speed_at_min": speed_kmh,
            }
            self.tracks[tid] = tr
            opened.append(tr)

        for tid, tr in list(self.tracks.items()):
            if tid in matched:
                tr["out"] = 0
                continue
            inside = self._in_zone(tr["u"], tr["z"])
            tr["out"] = 0 if inside else tr.get("out", 0) + 1
            status = None
            if tr["out"] >= ZONE_EXIT_TICKS:
                status = "CLEARED"
            elif now - tr["last"] > TRACK_LOST_S:
                status = "AVOIDED" if tr["z"] < 8.0 else "CLEARED"
            if status:
                tr["status"] = status
                tr["cleared"] = now
                closed.append(tr)
                del self.tracks[tid]
        return opened, closed

    @staticmethod
    def _observe(tr: Dict, u: float, z: float, steer_deg: float,
                 speed_kmh: float, now: float) -> None:
        tr["u"] += POS_EMA * (u - tr["u"])
        tr["z"] += POS_EMA * (z - tr["z"])
        tr["last"] = now
        dist = math.hypot(tr["u"], tr["z"])
        if dist < tr["min_dist"]:
            tr["min_dist"] = dist
            tr["steer_at_min"] = steer_deg
            tr["speed_at_min"] = speed_kmh


class Engine:
    def __init__(self) -> None:
        self.perception = PerceptionEngine()
        self.tracker = ObstacleTracker()
        self.paused = False
        self._video_acc = 0.0
        self._last_frame: Optional[np.ndarray] = None
        self.reset_sim()

    def reset_sim(self) -> None:
        self.mode = "sim"
        self.sim = SceneSimulator()
        if getattr(self, "cap", None) is not None:
            self.cap.release()
            self.cap = None
        self.video_name = ""
        self.speed_kmh = 38.0
        self.target_kmh = CRUISE_KMH_DEFAULT
        self.ego_u = 0.0
        self.steering_deg = 0.0
        self.odometer_m = 0.0
        self.seq = 0
        self.tracker.reset()

    def load_video(self, path: str, name: str = "") -> bool:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
        if getattr(self, "cap", None) is not None and self.cap is not None:
            self.cap.release()
        self.cap = cap
        self.fps = max(cap.get(cv2.CAP_PROP_FPS), 1.0) or 25.0
        self.mode = "video"
        self.video_name = name
        self.speed_kmh = 30.0
        self.target_kmh = CRUISE_KMH_DEFAULT
        self.ego_u = 0.0
        self.odometer_m = 0.0
        self._video_acc = 0.0
        return True

    def _grab(self, dt: float):
        if self.mode == "sim":
            frame, dets = self.sim.step(dt, self.speed_kmh / 3.6)
            source = "simulation"
            return frame, dets, source

        self._video_acc += dt * self.fps
        n = max(int(self._video_acc), 1)
        self._video_acc -= n
        frame = None
        for _ in range(n):
            ok, f = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, f = self.cap.read()
                if not ok:
                    break
            frame = f
        if frame is None:
            frame = self._last_frame
        else:
            if frame.shape[:2] != (IMG_H, IMG_W):
                frame = cv2.resize(frame, (IMG_W, IMG_H))
            self._last_frame = frame
        dets = self.perception.detect(frame)
        return frame, dets, "video"

    def _corridor_obstruction_m(self, grid) -> Optional[float]:
        c0, c1 = GRID_W // 2 - 3, GRID_W // 2 + 4
        for r in range(GRID_H - 6, 2, -1):
            if grid.hard[r, c0:c1].any():
                return ((GRID_H - 1) - r) * CELL_RES
        return None

    def _compute_ttc(self, detections: List[Dict]) -> Optional[float]:
        v = self.speed_kmh / 3.6
        best = None
        for det in detections:
            label = det.get("label")
            if label == "pothole":
                continue
            box = det["box"]
            p = project_image_to_world((box[0] + box[2]) / 2.0, min(box[3], IMG_H - 2))
            if p is None:
                continue
            u, z = p
            if not (0.5 < z < 55.0) or abs(u) > 2.3:
                continue
            extra = {"vehicle": 6.0}.get(label, 0.4)
            ttc = z / max(v + extra, 0.6)
            best = ttc if best is None else min(best, ttc)
        return best

    def tick(self) -> Dict:
        dt = TICK_S
        frame, dets, source = self._grab(dt)

        grid = build_grid(dets)
        plan = plan_path(grid, ego_u=self.ego_u, target_u=0.0)

        speed_mps = self.speed_kmh / 3.6
        raw_steer = pure_pursuit_steer(plan.world, self.ego_u, speed_mps, WHEELBASE_M)
        self.steering_deg = 0.6 * raw_steer + 0.4 * self.steering_deg

        ttc = self._compute_ttc(dets)
        obstruction = self._corridor_obstruction_m(grid)

        desired = self.target_kmh
        status = "CRUISE"
        if plan.blocked:
            desired, status = 0.0, "OBSTRUCTED"
        elif ttc is not None and ttc < 1.6:
            desired, status = 0.0, "BRAKE"
        elif obstruction is not None and obstruction < 10.0:
            desired, status = min(desired, 20.0), "OBSTRUCTED"
        elif plan.status != "optimal":
            status = "REROUTED"

        if plan.blocked or (ttc is not None and ttc < 1.6):
            accel = -45.0
        elif desired >= self.speed_kmh:
            accel = 9.0
        else:
            accel = -22.0
        new_speed = self.speed_kmh + accel * dt
        new_speed = min(new_speed, desired) if accel >= 0 else max(new_speed, desired)
        self.speed_kmh = max(0.0, new_speed)

        v = self.speed_kmh / 3.6
        self.ego_u += math.sin(math.radians(self.steering_deg)) * v * dt * 0.35
        self.ego_u = max(-3.3, min(3.3, self.ego_u))
        self.odometer_m += v * dt
        self.seq += 1

        now = time.time()
        _, closed_tracks = self.tracker.update(
            dets, self.steering_deg, self.speed_kmh, now)
        event_records = [self._close_record(tr) for tr in closed_tracks]
        for rec in event_records:
            event_logger.log_event(rec)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        frame_b64 = base64.b64encode(buf).decode("ascii") if ok else ""

        telemetry = {
            "speed_kmh": round(self.speed_kmh, 1),
            "target_kmh": round(self.target_kmh, 1),
            "steering_deg": round(self.steering_deg, 1),
            "latency_ms": round(plan.time_ms, 1),
            "ttc_s": None if ttc is None else round(min(ttc, 99.0), 2),
            "status": status,
            "obstruction_m": None if obstruction is None else round(obstruction, 1),
            "odometer_m": round(self.odometer_m, 1),
            "ego_u": round(self.ego_u, 2),
            "source": source,
            "detector": "simulated-gt" if self.mode == "sim"
                        else ("yolov8" if self.perception.model is not None else "classical"),
            "planner_status": plan.status,
            "nodes_expanded": plan.nodes,
            "in_zone": len(self.tracker.tracks),
        }

        for det in dets:
            det["color"] = LABEL_COLORS.get(det.get("label", ""), "#94a3b8")

        return {
            "type": "update",
            "seq": self.seq,
            "frame": "data:image/jpeg;base64," + frame_b64,
            "detections": dets,
            "path_img": [[round(x, 1), round(y, 1)] for x, y in plan.image],
            "path_raw_cells": [[c, r] for r, c in plan.cells_raw],
            "path_smooth_cells": [[c, r] for r, c in plan.cells_smooth],
            "grid": {
                "w": GRID_W, "h": GRID_H, "res": CELL_RES,
                "road_half_m": ROAD_HALF_WIDTH_M, "z_max": Z_MAX,
                "hard": base64.b64encode(np.packbits(grid.hard.flatten()).tobytes()).decode(),
                "soft": base64.b64encode(
                    (grid.soft * 255).astype(np.uint8).tobytes()).decode(),
            },
            "telemetry": telemetry,
            "events": event_records,
        }

    @staticmethod
    def _close_record(tr: Dict) -> Dict:
        steer = tr.get("steer_at_min", 0.0)
        speed = tr.get("speed_at_min", 0.0)
        if speed <= 6.5:
            verb = "BRAKE"
        elif abs(steer) >= 5.0:
            verb = "STEER"
        else:
            verb = "YIELD"
        action = f"{verb} {steer:+.1f} deg @ {speed:.0f} km/h"
        return {
            "track_id": tr["tid"],
            "label": tr["label"],
            "name": tr.get("name") or tr["label"],
            "entered_at": round(tr["entered"], 3),
            "cleared_at": round(tr["cleared"], 3),
            "duration_s": round(max(0.0, tr["cleared"] - tr["entered"]), 2),
            "min_distance_m": round(tr["min_dist"], 2),
            "steering_deg": round(steer, 1),
            "speed_kmh": round(speed, 1),
            "action": action,
            "status": tr["status"],
        }


app = FastAPI(title="SIH26037 Adaptive Path Planner", version="0.1.0")
engine = Engine()
event_logger = EventLogger()


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/logs")
async def api_logs(limit: int = 200):
    return {"events": event_logger.recent(limit), "total": event_logger.count()}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "clip.mp4").suffix.lower() or ".mp4"
    if suffix not in (".mp4", ".avi", ".mov", ".mkv"):
        raise HTTPException(400, "Unsupported file type")
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    with path.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > 300 * 1024 * 1024:
                fh.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, "File too large")
            fh.write(chunk)
    if not engine.load_video(str(path), file.filename or path.name):
        path.unlink(missing_ok=True)
        raise HTTPException(400, "Could not decode this video with OpenCV")
    return JSONResponse({"ok": True, "name": engine.video_name, "mode": "video"})


@app.post("/api/source/simulate")
async def use_simulation():
    engine.reset_sim()
    return JSONResponse({"ok": True, "mode": "sim"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    stop = asyncio.Event()

    async def reader():
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                mtype = msg.get("type")
                if mtype == "pause":
                    engine.paused = True
                elif mtype == "resume":
                    engine.paused = False
                elif mtype == "reset":
                    engine.reset_sim()
                elif mtype == "set_speed":
                    engine.target_kmh = float(max(15, min(90, msg.get("value", 45))))
                elif mtype == "hello":
                    pass
        except Exception:
            stop.set()

    task = asyncio.create_task(reader())
    try:
        payload = json.dumps(engine.tick(), separators=(",", ":"))
        await ws.send_text(payload)
        last_sent = time.perf_counter()
        while not stop.is_set():
            if not engine.paused:
                payload = json.dumps(engine.tick(), separators=(",", ":"))
                await ws.send_text(payload)
                elapsed = time.perf_counter() - last_sent
                await asyncio.sleep(max(0.02, TICK_S - elapsed))
                last_sent = time.perf_counter()
            else:
                await asyncio.sleep(0.08)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        task.cancel()


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
