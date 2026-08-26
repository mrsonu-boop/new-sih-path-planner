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


def build_grid(detections) -> OccupancyGrid:
    """Convert detections [{label, conf, box:[x1,y1,x2,y2]}] into an inflated BEV grid.

    Potholes become soft costs (avoided but traversable if the lane is blocked);
    animals/vehicles/persons and off-road cells are hard obstacles.
    """
    hard = np.zeros((GRID_H, GRID_W), dtype=bool)
    soft = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    c_lo = int(round(GRID_W / 2 - ROAD_HALF_WIDTH_M / CELL_RES))
    c_hi = int(round(GRID_W / 2 + ROAD_HALF_WIDTH_M / CELL_RES))
    hard[:, :max(c_lo, 0)] = True
    hard[:, min(c_hi, GRID_W):] = True
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
