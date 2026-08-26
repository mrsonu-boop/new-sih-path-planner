"""A* grid search + Catmull-Rom smoothed trajectory generation on the BEV occupancy grid."""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    from .occupancy_grid import (
        CELL_RES,
        GRID_H,
        GRID_W,
        OccupancyGrid,
        cell_to_world,
        project_world_to_image,
        world_to_cell,
        Z_MAX,
    )
except ImportError:
    from occupancy_grid import (
        CELL_RES,
        GRID_H,
        GRID_W,
        OccupancyGrid,
        cell_to_world,
        project_world_to_image,
        world_to_cell,
        Z_MAX,
    )

SQRT2 = math.sqrt(2.0)
NEIGHBORS = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
)
MAX_EXPANSIONS = 60000
SOFT_WEIGHT = 3.0


@dataclass
class PlanResult:
    cells_raw: List[Tuple[int, int]] = field(default_factory=list)
    cells_smooth: List[Tuple[int, int]] = field(default_factory=list)
    world: List[Tuple[float, float]] = field(default_factory=list)
    image: List[Tuple[float, float]] = field(default_factory=list)
    status: str = "blocked"
    nodes: int = 0
    time_ms: float = 0.0

    @property
    def blocked(self) -> bool:
        return not self.world or sum(
            math.dist(self.world[i], self.world[i + 1]) for i in range(len(self.world) - 1)
        ) < 6.0


def _heuristic(row: int, col: int, goal_r: int, goal_c: int) -> float:
    dr, dc = abs(row - goal_r), abs(col - goal_c)
    return (max(dr, dc) + (SQRT2 - 1.0) * min(dr, dc)) * 1.001


def _nearest_free(hard: np.ndarray, row: int, col: int) -> Optional[Tuple[int, int]]:
    if not hard[row, col]:
        return row, col
    for radius in range(1, 9):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                rr, cc = row + dr, col + dc
                if 0 <= rr < GRID_H and 0 <= cc < GRID_W and not hard[rr, cc]:
                    return rr, cc
    return None


def a_star(grid: OccupancyGrid, start, goal, soft_weight: float = SOFT_WEIGHT):
    """Returns (cells, truncated, nodes_expanded). Falls back to best-effort partial path."""
    hard, soft = grid.hard, grid.soft
    s = _nearest_free(hard, *start)
    g = _nearest_free(hard, *goal)
    if s is None or g is None:
        return [], True, 0
    if s == g:
        return [s], False, 0

    g_cost = np.full((GRID_H, GRID_W), np.inf, dtype=np.float32)
    came = np.full((GRID_H, GRID_W), -1, dtype=np.int64)
    closed = np.zeros((GRID_H, GRID_W), dtype=bool)

    def idx(r, c):
        return r * GRID_W + c

    sr, sc = s
    gr, gc = g
    g_cost[s] = 0.0
    counter = 0
    heap = [(_heuristic(sr, sc, gr, gc), 0, s)]
    best_node = s
    best_h = _heuristic(sr, sc, gr, gc)
    expansions = 0

    while heap and expansions < MAX_EXPANSIONS:
        _, _, cur = heapq.heappop(heap)
        cr, cc = cur
        if closed[cr, cc]:
            continue
        closed[cr, cc] = True
        expansions += 1

        h = _heuristic(cr, cc, gr, gc)
        if h < best_h:
            best_h = h
            best_node = cur
        if cur == g:
            return _reconstruct(came, cur, idx), False, expansions

        for dr, dc, step in NEIGHBORS:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < GRID_H and 0 <= nc < GRID_W) or hard[nr, nc]:
                continue
            ng = g_cost[cr, cc] + step + soft_weight * float(soft[nr, nc])
            if ng < g_cost[nr, nc]:
                g_cost[nr, nc] = ng
                came[nr, nc] = idx(cr, cc)
                counter += 1
                heapq.heappush(heap, (ng + _heuristic(nr, nc, gr, gc), counter, (nr, nc)))

    return _reconstruct(came, best_node, idx), True, expansions


def _reconstruct(came: np.ndarray, node, idx_fn) -> List[Tuple[int, int]]:
    cells = []
    r, c = node
    while r >= 0 and c >= 0:
        cells.append((r, c))
        prev = came[r, c]
        if prev < 0:
            break
        r, c = int(prev // GRID_W), int(prev % GRID_W)
    cells.reverse()
    return cells


def _line_of_sight(grid: OccupancyGrid, a, b) -> bool:
    (ar, ac), (br, bc) = a, b
    steps = max(2, int(math.hypot(br - ar, bc - ac) * 2))
    for k in range(1, steps):
        t = k / steps
        rr = int(round(ar + (br - ar) * t))
        cc = int(round(ac + (bc - ac) * t))
        if grid.hard[rr, cc]:
            return False
    return True


def _shortcut(grid: OccupancyGrid, cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if len(cells) < 3:
        return cells[:]
    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not _line_of_sight(grid, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def _catmull_rom(points: List[Tuple[float, float]], spacing: float = 0.35):
    """Dense smooth trajectory through waypoints in world coordinates."""
    pts = [(float(u), float(z)) for u, z in points]
    if len(pts) < 3:
        dense = []
        for k in range(len(pts) - 1):
            (u0, z0), (u1, z1) = pts[k], pts[k + 1]
            seg = max(1, int(math.hypot(u1 - u0, z1 - z0) / spacing))
            for t in range(seg):
                dense.append((u0 + (u1 - u0) * t / seg, z0 + (z1 - z0) * t / seg))
        dense.append(pts[-1])
        return dense

    P = [pts[0]] + pts + [pts[-1]]
    dense = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        samples = max(4, int(seg_len / 0.08))
        for k in range(samples):
            t = k / samples
            t2, t3 = t * t, t * t * t
            u = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
                       (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            z = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
                       (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            dense.append((u, z))
    dense.append(pts[-1])

    resampled = [dense[0]]
    for p in dense[1:]:
        if math.dist(p, resampled[-1]) >= spacing:
            resampled.append(p)
    return resampled[:800]


def plan_path(grid: OccupancyGrid, ego_u: float = 0.0, target_u: float = 0.0,
              goal_z: float = Z_MAX - 3.0) -> PlanResult:
    """Full pipeline: A* -> shortcut -> spline -> world/image/cell trajectories."""
    t0 = time.perf_counter()
    result = PlanResult()

    start = world_to_cell(ego_u, 1.4)
    goal = world_to_cell(target_u, goal_z)

    cells, truncated, nodes = a_star(grid, start, goal, soft_weight=SOFT_WEIGHT)
    status = "optimal"
    if truncated or not cells:
        status = "rerouted"
        cells, truncated, extra_nodes = a_star(grid, start, goal, soft_weight=0.0)
        nodes += extra_nodes
    result.nodes = nodes

    if cells:
        cells = _shortcut(grid, cells)
        world = [cell_to_world(r, c) for r, c in cells]
        world[0] = (ego_u, min(world[0][1], 1.4))
        smooth = _catmull_rom(world)
        result.cells_raw = cells
        result.cells_smooth = [world_to_cell(u, z) for u, z in smooth]
        result.world = smooth
        result.image = [project_world_to_image(u, z) for u, z in smooth]
        result.status = "optimal" if (status == "optimal" and not truncated) else "rerouted"
    else:
        result.status = "blocked"

    result.time_ms = (time.perf_counter() - t0) * 1000.0
    return result


def pure_pursuit_steer(world_path: List[Tuple[float, float]], ego_u: float,
                       speed_mps: float, wheelbase: float = 2.6) -> float:
    """Return commanded front-wheel angle in degrees (+right / -left)."""
    if len(world_path) < 2:
        return 0.0
    lookahead = min(max(0.9 * max(speed_mps, 1.0), 4.0), 14.0)
    car = (ego_u, 1.0)
    target = world_path[-1]
    acc = 0.0
    prev = car
    for pt in world_path:
        seg = math.dist(pt, prev)
        if acc + seg >= lookahead:
            t = (lookahead - acc) / max(seg, 1e-6)
            target = (prev[0] + (pt[0] - prev[0]) * t, prev[1] + (pt[1] - prev[1]) * t)
            break
        acc += seg
        prev = pt
    alpha = math.atan2(target[0] - car[0], max(target[1] - car[1], 0.1))
    delta = math.atan2(2.0 * wheelbase * math.sin(alpha), lookahead)
    return max(-30.0, min(30.0, math.degrees(delta)))
