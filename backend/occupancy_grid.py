"""Bounding-box detections -> bird's-eye-view occupancy grid for path planning."""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

CAMERA = {
    "image_w": 640,
    "image_h": 360,
    "horizon_y": 118.0,
    "focal_px": 380.0,
    "height_m": 1.45,
}

GRID_W = 96
GRID_H = 144
CELL_RES = 0.4
Z_MAX = GRID_H * CELL_RES
ROAD_HALF_WIDTH_M = 4.2

MARGIN_BY_LABEL = {
    "pothole": 0.45,
    "animal": 0.95,
    "person": 1.0,
    "vehicle": 1.15,
}

SOFT_COST_BY_LABEL = {
    "pothole": 0.55,
}


def project_image_to_world(x: float, y: float):
    """Pixel (x, y) on the ground plane -> (u lateral [m], z forward [m]) in ego frame."""
    denom = y - CAMERA["horizon_y"]
    if denom <= 1.0:
        return None
    z = CAMERA["focal_px"] * CAMERA["height_m"] / denom
    u = (x - CAMERA["image_w"] / 2.0) * z / CAMERA["focal_px"]
    return u, z


def project_world_to_image(u: float, z: float):
    """Ground-plane point -> pixel (x, y). Points closer than ~0.35 m clamp forward."""
    z = max(z, 0.35)
    y = CAMERA["horizon_y"] + CAMERA["focal_px"] * CAMERA["height_m"] / z
    x = CAMERA["image_w"] / 2.0 + CAMERA["focal_px"] * u / z
    return float(x), float(y)


def world_to_cell(u: float, z: float):
    col = int(GRID_W / 2 + u / CELL_RES)
    row = int((Z_MAX - z) / CELL_RES)
    return _clamp_cell(row, col)


def cell_to_world(row: int, col: int):
    u = (col - GRID_W / 2 + 0.5) * CELL_RES
    z = (GRID_H - row - 0.5) * CELL_RES
    return u, z


def _clamp_cell(row: int, col: int):
    row = min(max(row, 0), GRID_H - 1)
    col = min(max(col, 0), GRID_W - 1)
    return row, col


@lru_cache(maxsize=64)
def _disk_offsets(radius_cells: int):
    offs = []
    r2 = radius_cells * radius_cells
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if dr * dr + dc * dc <= r2:
                offs.append((dr, dc))
    return tuple(offs)


@dataclass
class OccupancyGrid:
    hard: np.ndarray
    soft: np.ndarray

    @property
    def shape(self):
        return self.hard.shape


EDGE_SOFT_MARGIN_M = 1.2   # width of the soft shoulder ramp outside the nominal road edge
EDGE_SOFT_COST = 0.85      # max soft cost at the edge of the shoulder ramp (below hard block)


def road_half_width_at_z(z: float, unstructured: bool = False,
                          pinch_z_center: float = 24.0, pinch_z_span: float = 12.0,
                          pinch_half_width_m: float = 1.6,
                          base_half_width_m: float = ROAD_HALF_WIDTH_M) -> float:
    """Single source of truth for road half-width at forward distance z (meters).

    Used by both the planner's occupancy grid and the perception renderer so the
    bird's-eye planner and the camera-feed drawing always agree on where the
    road actually narrows, instead of drifting out of sync.
    """
    if not unstructured:
        return base_half_width_m
    half_span = max(pinch_z_span / 2.0, 1e-6)
    dist_from_center = abs(z - pinch_z_center)
    if dist_from_center >= half_span:
        return base_half_width_m
    taper = 1.0 - dist_from_center / half_span  # 0 at edges of pinch window, 1 at center
    return base_half_width_m - taper * (base_half_width_m - pinch_half_width_m)


def constant_road_profile(_row: int) -> float:
    """Default road profile: fixed half-width, matching a structured/marked road."""
    return ROAD_HALF_WIDTH_M


def variable_road_profile(row: int) -> float:
    """Village-road style profile: narrows to a single-vehicle-width pinch point
    ahead of the vehicle, then widens back out. Models an unmarked, irregular
    -width unstructured road. Delegates to road_half_width_at_z so the camera
    view (perception.py) can render the exact same profile."""
    _, z = cell_to_world(row, GRID_W // 2)
    return road_half_width_at_z(z, unstructured=True)


def build_grid(detections, road_profile=None) -> OccupancyGrid:
    """Convert detections [{label, conf, box:[x1,y1,x2,y2]}] into an inflated BEV grid.

    Potholes become soft costs (avoided but traversable if the lane is blocked);
    animals/vehicles/persons and off-road cells are hard obstacles.

    road_profile(row) -> half_width_m lets the road boundary vary per row instead
    of using a single fixed ROAD_HALF_WIDTH_M, and the boundary itself is a soft
    cost ramp (EDGE_SOFT_MARGIN_M wide) rather than an instant hard wall, so the
    planner *can* cross onto the shoulder under pressure (e.g. avoiding an
    obstacle) instead of treating every unmarked edge as impassable. This models
    unstructured, unmarked, variable-width roads instead of a fixed lane corridor.
    """
    profile = road_profile or constant_road_profile
    half_w = np.array([profile(row) for row in range(GRID_H)], dtype=np.float32)  # (GRID_H,)
    c_lo = (GRID_W / 2 - half_w / CELL_RES)[:, None]   # (GRID_H, 1)
    c_hi = (GRID_W / 2 + half_w / CELL_RES)[:, None]
    cols = np.arange(GRID_W, dtype=np.float32)[None, :]  # (1, GRID_W)

    dist_out = np.where(cols < c_lo, c_lo - cols, np.where(cols > c_hi, cols - c_hi, 0.0))
    dist_out_m = dist_out * CELL_RES
    margin_cells = max(EDGE_SOFT_MARGIN_M / CELL_RES, 1e-6)

    hard = dist_out_m > EDGE_SOFT_MARGIN_M
    ramp = np.clip(dist_out / margin_cells, 0.0, 1.0)
    soft = np.where(hard, 0.0, EDGE_SOFT_COST * ramp ** 1.5).astype(np.float32)

    hard[:2, :] = True
    hard[-4:, :] = True
    hard[:, :2] = True
    hard[:, -2:] = True

    for det in detections:
        box = det.get("box")
        label = str(det.get("label", "obstacle"))
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        cx = (x1 + x2) / 2.0
        ground_y = min(y2, CAMERA["image_h"] - 2)
        p = project_image_to_world(cx, ground_y)
        if p is None:
            continue
        u, z = p
        if z > Z_MAX:
            z = Z_MAX * 0.97
        width_m = (x2 - x1) * z / CAMERA["focal_px"]
        radius_m = max(width_m * 0.5, 0.3) + MARGIN_BY_LABEL.get(label, 0.7)
        radius_cells = max(1, int(math.ceil(radius_m / CELL_RES)))
        soft_val = SOFT_COST_BY_LABEL.get(label, 0.0)

        row_c = int(round((Z_MAX - z) / CELL_RES))
        col_c = int(round(GRID_W / 2 + u / CELL_RES))
        for dr, dc in _disk_offsets(radius_cells):
            rr, cc = row_c + dr, col_c + dc
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W:
                if soft_val > 0.0:
                    soft[rr, cc] = max(soft[rr, cc], soft_val)
                else:
                    hard[rr, cc] = True

    return OccupancyGrid(hard=hard, soft=soft)
