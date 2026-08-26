"""Perception layer: simulated unstructured-road scene with ground-truth boxes,
plus optional YOLOv8 inference and a classical dark-blob fallback for uploads."""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from .occupancy_grid import CAMERA, project_world_to_image
except ImportError:
    from occupancy_grid import CAMERA, project_world_to_image

IMG_W = CAMERA["image_w"]
IMG_H = CAMERA["image_h"]
HORIZON = CAMERA["horizon_y"]
FOCAL = CAMERA["focal_px"]

ROAD_HALF = 3.5
COL_SKY_TOP = (40, 28, 16)
COL_SKY_BOT = (78, 58, 44)
COL_GRASS = (48, 72, 56)
COL_ROAD = (45, 42, 40)
COL_LANE = (205, 205, 195)
COL_EDGE = (150, 148, 140)

COCO_LABEL_MAP = {
    0: "person", 1: "vehicle", 2: "vehicle", 3: "vehicle", 5: "vehicle",
    6: "vehicle", 7: "vehicle", 14: "animal", 15: "animal", 16: "animal",
    17: "animal", 18: "animal", 19: "animal", 20: "animal", 21: "animal",
    22: "animal", 23: "animal",
}


class SceneSimulator:
    """Synthetic two-lane rural road with potholes, crossing animals and traffic."""

    def __init__(self, seed: Optional[int] = 7):
        self.rng = random.Random(seed)
        self.t = 0.0
        self.phase = 0.0
        self.animal_timer = 2.5
        self.objects: List[Dict] = []
        self._obj_id = 0
        self._background = self._render_background()
        self.reset()

    def reset(self) -> None:
        self.objects.clear()
        self.t = 0.0
        self.phase = 0.0
        self.animal_timer = self.rng.uniform(2.0, 4.0)
        for k in range(5):
            self._spawn("pothole", z=12.0 + k * 9.0 + self.rng.uniform(0, 4),
                        u=self.rng.uniform(-2.9, 2.9))

    def _spawn(self, kind: str, z: float, u: float) -> None:
        self._obj_id += 1
        obj = {"kind": kind, "z": z, "u": u, "id": self._obj_id,
               "phase": self.rng.uniform(0, 6.28)}
        if kind == "pothole":
            obj.update(w=self.rng.uniform(0.5, 1.1), l=self.rng.uniform(0.4, 0.9),
                       vu=0.0, extra_close=0.0)
        elif kind == "animal":
            s = self.rng.uniform(0.75, 1.45)
            side = self.rng.choice([-1, 1])
            obj.update(w=0.62 * s, l=1.15 * s,
                       vu=-side * self.rng.uniform(1.3, 2.4), extra_close=0.3)
            obj["species"] = self.rng.choice(["cattle", "dog", "goat"])
        elif kind == "oncoming":
            obj.update(w=1.7, l=4.2, vu=0.0,
                       extra_close=self.rng.uniform(6.0, 10.0))
        else:
            obj.update(w=1.85, l=4.4, vu=0.0, extra_close=0.0)
        self.objects.append(obj)

    def step(self, dt: float, ego_speed_mps: float):
        self.t += dt
        v = max(ego_speed_mps, 0.0)
        self.phase = (self.phase + v * dt) % 8.0

        self.animal_timer -= dt
        if self.animal_timer <= 0.0:
            self.animal_timer = self.rng.uniform(9.0, 18.0)
            self._spawn("animal", z=self.rng.uniform(30.0, 46.0),
                        u=self.rng.choice([-6.5, 6.5]))
        n_potholes = sum(1 for o in self.objects if o["kind"] == "pothole")
        if n_potholes < 5 and self.rng.random() < dt * 0.5:
            self._spawn("pothole", z=54.0 + self.rng.uniform(0, 3),
                        u=self.rng.uniform(-2.9, 2.9))
        if not any(o["kind"] == "oncoming" for o in self.objects) and self.rng.random() < dt * 0.06:
            self._spawn("oncoming", z=52.0 + self.rng.uniform(0, 6), u=2.55)
        if not any(o["kind"] == "parked" for o in self.objects) and self.rng.random() < dt * 0.03:
            self._spawn("parked", z=48.0 + self.rng.uniform(0, 10),
                        u=self.rng.choice([3.05, -3.05]))

        for o in self.objects:
            closing = v + o["extra_close"]
            o["z"] -= closing * dt
            o["u"] += o["vu"] * dt
            if o["kind"] == "animal":
                o["u"] += math.sin(self.t * 1.7 + o["phase"]) * 0.15 * dt
        self.objects = [o for o in self.objects
                        if o["z"] > 1.2 and abs(o["u"]) < 8.5]

        frame = self._draw_scene()
        detections = self._ground_truth_boxes()
        return frame, detections

    def _render_background(self) -> np.ndarray:
        hy = int(HORIZON)
        sky = np.linspace(np.array(COL_SKY_TOP, np.float32),
                          np.array(COL_SKY_BOT, np.float32), hy)[:, None, :]
        sky = np.repeat(sky, IMG_W, axis=1).astype(np.uint8)
        ground_h = IMG_H - hy
        grass = np.linspace(np.array(COL_GRASS, np.float32),
                            np.array(COL_GRASS, np.float32) * 0.55,
                            ground_h)[:, None, :]
        grass = np.repeat(grass, IMG_W, axis=1).astype(np.uint8)
        bg = np.vstack([sky, grass])

        rng = random.Random(42)
        x = 0
        while x < IMG_W:
            bw = rng.randint(18, 60)
            bh = rng.randint(4, 16)
            cv2.rectangle(bg, (x, hy - bh), (x + bw, hy), (30, 24, 22), -1)
            x += bw + rng.randint(2, 14)
        return bg

    def _road_poly(self, frame, u_left: float, u_right: float, z_near: float,
                   z_far: float, color) -> None:
        pts = [project_world_to_image(u_left, z_near),
               project_world_to_image(u_right, z_near),
               project_world_to_image(u_right, z_far),
               project_world_to_image(u_left, z_far)]
        poly = np.array([[int(round(x)), min(int(round(y)), IMG_H)] for x, y in pts],
                        dtype=np.int32)
        cv2.fillPoly(frame, [poly], color)

    def _draw_scene(self) -> np.ndarray:
        frame = self._background.copy()
        self._road_poly(frame, -ROAD_HALF - 0.35, ROAD_HALF + 0.35, 2.0, 90.0, COL_ROAD)

        for z0 in np.arange(-self.phase, 58.0, 8.0):
            z1 = min(z0 + 3.0, 58.0)
            if z1 <= 2.0:
                continue
            self._road_poly(frame, -0.09, 0.09, max(z0, 2.0), z1, COL_LANE)
        self._road_poly(frame, ROAD_HALF + 0.15, ROAD_HALF + 0.27, 2.0, 70.0, COL_EDGE)
        self._road_poly(frame, -ROAD_HALF - 0.27, -ROAD_HALF - 0.15, 2.0, 70.0, COL_EDGE)

        for obj in sorted(self.objects, key=lambda o: -o["z"]):
            self._draw_object(frame, obj)
        return frame

    @staticmethod
    def _footprint_corners(obj: Dict):
        hw, hl = obj["w"] / 2.0, obj["l"] / 2.0
        return [(obj["u"] - hw, obj["z"] - hl), (obj["u"] + hw, obj["z"] - hl),
                (obj["u"] + hw, obj["z"] + hl), (obj["u"] - hw, obj["z"] + hl)]

    def _project_footprint(self, obj: Dict):
        pix = [project_world_to_image(u, z) for u, z in self._footprint_corners(obj)]
        xs = [min(max(p[0], -80), IMG_W + 80) for p in pix]
        ys = [min(max(p[1], -80), IMG_H + 80) for p in pix]
        return min(xs), min(ys), max(xs), max(ys)

    def _draw_object(self, frame, obj: Dict) -> None:
        z = max(obj["z"], 1.3)
        cx, cy = project_world_to_image(obj["u"], z)
        scale = FOCAL / z
        kind = obj["kind"]

        if kind == "pothole":
            rx = obj["w"] / 2.0 * scale
            ry = rx * 0.42 + 2
            if rx >= 1.5 and cy < IMG_H + 20:
                cv2.ellipse(frame, (int(cx), int(cy)), (int(rx), int(ry)),
                            0, 0, 360, (28, 26, 25), -1)
                cv2.ellipse(frame, (int(cx), int(cy)), (max(int(rx * 0.55), 1),
                            max(int(ry * 0.55), 1)), 0, 0, 360, (10, 10, 11), -1)
                cv2.ellipse(frame, (int(cx), int(cy)), (int(rx), int(ry)),
                            0, 200, 340, (74, 66, 58), 1)
        elif kind in ("oncoming", "parked"):
            x1, y1, x2, y2 = self._project_footprint(obj)
            body = (188, 186, 182) if kind == "oncoming" else (88, 82, 76)
            poly = np.array([[int(x1), int(y1)], [int(x2), int(y1)],
                             [int(x2), int(y2)], [int(x1), int(y2)]], np.int32)
            if y1 < IMG_H:
                cv2.fillPoly(frame, [poly], body)
                roof_y = int(y1 + (y2 - y1) * 0.22)
                win_y = int(y1 + (y2 - y1) * 0.52)
                inset_x = int((x2 - x1) * 0.12)
                cv2.rectangle(frame, (int(x1) + inset_x, roof_y),
                              (int(x2) - inset_x, win_y), (52, 48, 44), -1)
                lamp_r = max(2, int((x2 - x1) * 0.08))
                ly = int(y2 - (y2 - y1) * 0.12)
                if kind == "oncoming":
                    for lx in (int(x1) + lamp_r, int(x2) - lamp_r):
                        cv2.circle(frame, (lx, ly), lamp_r, (255, 235, 180), -1)
                        cv2.circle(frame, (lx, ly), lamp_r * 2, (120, 110, 80), 1)
                else:
                    for lx in (int(x1) + lamp_r, int(x2) - lamp_r):
                        cv2.circle(frame, (lx, int(y1 + lamp_r)), lamp_r, (40, 40, 210), -1)
        else:
            s = obj["w"] / 0.62
            rx = obj["w"] / 2.0 * scale
            ry = obj["l"] / 2.0 * scale * 0.5
            colors = {"cattle": (90, 96, 150), "dog": (48, 60, 105), "goat": (140, 170, 190)}
            col = colors.get(obj.get("species", "dog"), (60, 70, 110))
            if cy < IMG_H + 20 and rx >= 1:
                cv2.ellipse(frame, (int(cx), int(cy)), (max(int(rx), 2), max(int(ry * 0.7), 2)),
                            0, 0, 360, col, -1)
                head_dx = -math.copysign(rx * 0.95, obj["vu"] or 1.0)
                hr = max(int(rx * 0.42), 2)
                cv2.circle(frame, (int(cx + head_dx), int(cy - ry * 0.55)), hr, col, -1)
                leg_len = max(int(ry * 1.1), 2)
                for lx in (-rx * 0.6, rx * 0.6):
                    p1 = (int(cx + lx), int(cy))
                    p2 = (int(cx + lx * 1.15), int(cy + leg_len))
                    cv2.line(frame, p1, p2, tuple(int(c * 0.6) for c in col), 1)

    def _ground_truth_boxes(self) -> List[Dict]:
        dets = []
        for obj in sorted(self.objects, key=lambda o: -o["z"]):
            if obj["z"] > Z_MAX_VIEW:
                continue
            x1, y1, x2, y2 = self._project_footprint(obj)
            pad = max(2.0, (x2 - x1) * 0.04)
            bx = [max(x1 - pad, 0), max(y1 - pad, 0),
                  min(x2 + pad, IMG_W - 1), min(y2 + pad, IMG_H - 1)]
            box = [int(round(v)) for v in bx]
            if box[2] - box[0] < 4 or box[3] - box[1] < 3 or box[1] > IMG_H - 2:
                continue
            jitter = self.rng.gauss
            box = [box[0] + jitter(0, 1.2), box[1] + jitter(0, 1.2),
                   box[2] + jitter(0, 1.2), box[3] + jitter(0, 1.2)]
            label = obj["kind"] if obj["kind"] != "animal" else "animal"
            dets.append({
                "label": label,
                "name": obj.get("species") or obj["kind"],
                "conf": round(min(0.97, self.rng.uniform(0.68, 0.96)), 2),
                "box": [int(round(v)) for v in box],
            })
        return dets


Z_MAX_VIEW = 56.0


def classical_pothole_detect(frame_bgr: np.ndarray) -> List[Dict]:
    """Dark-blob detector used when no YOLO weights are available."""
    roi_y = int(HORIZON) + 8
    roi = frame_bgr[roi_y:, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    _, th = cv2.threshold(gray, 52, 255, cv2.THRESH_BINARY_INV)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = cv2.contourArea(cnt)
        if area < 140 or area > 12000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 6 or not (0.3 < w / max(h, 1) < 3.5):
            continue
        dets.append({"label": "pothole", "name": "pothole",
                     "conf": round(min(0.92, 0.45 + area / 9000.0), 2),
                     "box": [int(x), int(y + roi_y), int(x + w), int(y + h + roi_y)]})
    return dets


class PerceptionEngine:
    """Lazily loads YOLOv8; falls back to the classical detector."""

    def __init__(self, weights: str = "yolov8n.pt"):
        self.weights = weights
        self.model = None
        self.error: Optional[str] = None

    def ensure_model(self) -> bool:
        if self.model is not None:
            return True
        if self.error is not None:
            return False
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights)
            return True
        except Exception as exc:
            self.error = str(exc)[:180]
            return False

    def detect(self, frame_bgr: np.ndarray) -> List[Dict]:
        if not self.ensure_model():
            return classical_pothole_detect(frame_bgr)
        try:
            results = self.model.predict(frame_bgr, conf=0.35, imgsz=480, verbose=False)
            dets = []
            r = results[0]
            names = r.names
            for b in r.boxes:
                cls_id = int(b.cls.item())
                label = COCO_LABEL_MAP.get(cls_id)
                if label is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                dets.append({"label": label, "name": names.get(cls_id, str(cls_id)),
                             "conf": round(float(b.conf.item()), 2),
                             "box": [int(x1), int(y1), int(x2), int(y2)]})
            return dets
        except Exception:
            return classical_pothole_detect(frame_bgr)
