"""LiDAR obstacle detection without machine learning: geometry only.

Input: the roof LiDAR's point cloud for one sweep, in the sensor frame (x forward, y right,
z up, metres; CARLA convention) plus where the sensor is mounted on the car.

1. Ground removal  A point is above the road if it's more than GROUND_MARGIN_M above the road
                   plane under the car. On hills that alone would flag rising road, so a grid cell
                   only counts if its points also span at least MIN_EXTENT_M vertically (a car,
                   a person, a barrier does; a patch of sloping road doesn't).
2. Clustering      DBSCAN-style on a 2D grid: CELL_M cells holding >= MIN_POINTS points are the
                   "core" cells, and touching core cells (8-neighbourhood) form one cluster. This
                   is DBSCAN with eps = one cell, but in O(points) with NumPy, fast on the Jetson.
3. Obstacles       each cluster -> position, nearest point, size, height (vehicle frame metres).
4. Rear clearance  nearest above-ground point directly behind the car.
"""

import base64
from collections import deque

import numpy as np

GROUND_MARGIN_M = 0.3
MAX_HEIGHT_M = 2.6           # ignore overhanging trees, bridges, signs
MIN_EXTENT_M = 0.25
CELL_M = 0.5
MIN_POINTS = 3               # per core cell
MIN_CLUSTER_POINTS = 8
ROI = {"ahead": (-20.0, 45.0), "side": 20.0}
EGO_HALF_LENGTH_M, EGO_HALF_WIDTH_M = 2.5, 1.15  # the car itself (Tesla Model 3 + margin)


def decode(payload: bytes) -> np.ndarray:
    """Raw float32 xyz bytes from the host -> (N, 3) array."""
    return np.frombuffer(payload, dtype=np.float32).reshape(-1, 3)


def process(points: np.ndarray, mount: dict | None = None) -> dict:
    """Point cloud -> {"clusters": [...], "rear_clearance_m": float | None, "view": base64 points for the dashboard}."""
    mount = {"x": 0.0, "z": 2.0, **(mount or {})}
    if points is None or len(points) == 0:
        return {"clusters": [], "rear_clearance_m": None, "view": None}
    ahead = points[:, 0] + mount["x"]      # vehicle frame: from the car's centre
    right = points[:, 1]
    height = points[:, 2] + mount["z"]     # above the road under the car

    keep = ((ahead > ROI["ahead"][0]) & (ahead < ROI["ahead"][1]) & (np.abs(right) < ROI["side"])
            & (height > GROUND_MARGIN_M) & (height < MAX_HEIGHT_M)
            & ~((np.abs(ahead) < EGO_HALF_LENGTH_M) & (np.abs(right) < EGO_HALF_WIDTH_M)))
    a, r, h = ahead[keep], right[keep], height[keep]

    # Grid cells; keep cells with enough points that also span some height (not a sloping road patch).
    ci = np.floor(a / CELL_M).astype(np.int32)
    cj = np.floor(r / CELL_M).astype(np.int32)
    cell_id = ci.astype(np.int64) * 100000 + cj
    uniq, first, inverse, counts = np.unique(cell_id, return_index=True, return_inverse=True, return_counts=True)
    hmin = np.full(len(uniq), np.inf)
    hmax = np.full(len(uniq), -np.inf)
    np.minimum.at(hmin, inverse, h)
    np.maximum.at(hmax, inverse, h)
    core = (counts >= MIN_POINTS) & (hmax - hmin >= MIN_EXTENT_M)
    core_cells = {(int(ci[first[k]]), int(cj[first[k]])): int(k) for k in np.flatnonzero(core)}

    # Connected components over touching core cells (BFS).
    label_of_cell, clusters = {}, []
    for start in core_cells:
        if start in label_of_cell:
            continue
        lab = len(clusters)
        queue, members = deque([start]), []
        label_of_cell[start] = lab
        while queue:
            c = queue.popleft()
            members.append(core_cells[c])
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    n = (c[0] + di, c[1] + dj)
                    if n in core_cells and n not in label_of_cell:
                        label_of_cell[n] = lab
                        queue.append(n)
        clusters.append(members)

    point_label = np.full(len(a), -1)
    for lab, members in enumerate(clusters):
        point_label[np.isin(inverse, members)] = lab
    out = []
    for lab in range(len(clusters)):
        sel = point_label == lab
        if sel.sum() < MIN_CLUSTER_POINTS:
            continue
        ca, cr, ch = a[sel], r[sel], h[sel]
        front = ca[ca > 0]
        out.append({
            "ahead_m": round(float(ca.mean()), 2), "right_m": round(float(cr.mean()), 2),
            "nearest_ahead_m": round(float(front.min()), 2) if len(front) else None,
            "extent": [round(float(ca.min()), 2), round(float(ca.max()), 2),   # ahead min/max
                       round(float(cr.min()), 2), round(float(cr.max()), 2)],  # right min/max
            "height_m": round(float(ch.max()), 2), "points": int(sel.sum()),
            # A sample of the cluster's points (ahead, right): a long object such as a wall beside the road
            # is judged by where its points are, not by its bounding box (which can overlap the path).
            "pts": [[round(float(x), 2), round(float(y), 2)] for x, y in
                    zip(ca[::max(1, len(ca) // 60)], cr[::max(1, len(cr) // 60)])],
        })

    behind = (a < -EGO_HALF_LENGTH_M + 0.1) & (np.abs(r) < EGO_HALF_WIDTH_M)
    rear = round(float(-a[behind].max() - EGO_HALF_LENGTH_M + 0.1), 2) if behind.any() else None

    # Bird's-eye view for the dashboard: up to 1500 above-ground points as int16 decimetres.
    step = max(1, len(a) // 1500)
    view_pts = np.stack([a[::step] * 10, r[::step] * 10], axis=1).astype(np.int16)
    return {"clusters": out, "rear_clearance_m": rear,
            "view": base64.b64encode(view_pts.tobytes()).decode()}


def bearing_deg(ahead_m: float, right_m: float, cam_x_m: float) -> float:
    """Horizontal angle of a point as seen from the camera (+ = right)."""
    return float(np.degrees(np.arctan2(right_m, ahead_m - cam_x_m)))
