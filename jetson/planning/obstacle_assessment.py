"""Finds objects in the vehicle's path and how far away they are.

Distance comes from the depth model when it ran this frame (median depth over the
lower half of the box: the object's own pixels, not the background around it), and
otherwise from flat-road geometry (where the bottom of the box touches the road).
Both are reported so they can be compared against CARLA's ground truth. When the LiDAR sees the
object, its measured distance wins (see in_path_obstacles).
"""

import math

import numpy as np

from perception.camera_geometry import Camera

PEOPLE = {0, 1}            # person, bicycle: stop and wait
VEHICLES = {2, 3, 5, 7}    # car, motorcycle, bus, truck: slow down / stop behind


def locate(det: dict, cam: Camera, depth: np.ndarray | None) -> dict | None:
    """Position of one detection in the vehicle frame, or None if it can't be on the road near us."""
    x1, y1, x2, y2 = det["bbox"]
    u = (x1 + x2) / 2
    geom = cam.pixel_to_ground(u, y2)
    pos = {"distance_geom_m": round(geom[0], 2) if geom else None}
    if depth is not None:
        h, w = depth.shape
        r0, r1 = int(max(0, (y1 + y2) / 2)), int(min(h, y2))
        c0, c1 = int(max(0, x1)), int(min(w, x2))
        if r1 > r0 and c1 > c0:
            ahead, right = cam.from_depth(u, float(np.median(depth[r0:r1, c0:c1])))
            return {**pos, "distance_m": round(ahead, 2), "lateral_m": round(right, 2), "source": "depth"}
    if geom is None:
        return None
    return {**pos, "distance_m": round(geom[0], 2), "lateral_m": round(geom[1], 2), "source": "geometry"}


# Half-width of the corridor around the path centre. People get extra margin (they move sideways);
# vehicles in the next lane (~3.5 m over) must stay outside it.
CORRIDOR_HALF_WIDTH_M = {"person": 2.0, "vehicle": 1.4}
LIDAR_CORRIDOR_HALF_WIDTH_M = 1.1   # our car's half width + a little, for LiDAR-only objects
FRONT_BUMPER_M = 2.4                # from the car's centre


def match_cluster(det: dict, cam: Camera, clusters: list, geom_dist: float | None) -> dict | None:
    """The LiDAR cluster behind a camera detection: inside the box's horizontal angle range, nearest first,
    and (if the camera has a distance estimate) not wildly different from it."""
    from perception.lidar import bearing_deg
    x1, _, x2, _ = det["bbox"]
    lo = math.degrees(math.atan((x1 - cam.cx) / cam.f)) - 1.5
    hi = math.degrees(math.atan((x2 - cam.cx) / cam.f)) + 1.5
    best = None
    for c in clusters:
        near = c["nearest_ahead_m"]
        if near is None or near <= cam.x_m or not lo <= bearing_deg(c["ahead_m"], c["right_m"], cam.x_m) <= hi:
            continue
        if geom_dist is not None and abs(near - geom_dist) > max(4.0, 0.5 * geom_dist):
            continue
        if best is None or near < best["nearest_ahead_m"]:
            best = c
    return best


def in_path_obstacles(perception: dict, path_right_at, look_ahead_m: float = 30.0) -> list[dict]:
    """Obstacles inside the corridor around the planned path, nearest first.

    path_right_at(ahead_m) -> lateral position of the path centre at that distance (0 = straight ahead),
    so the corridor bends with the lane instead of always pointing straight ahead.

    Camera detections get their distance from the matching LiDAR cluster when there is one (measured,
    not estimated); LiDAR clusters no camera detection explains (barriers, debris, a car YOLO missed)
    are obstacles too ("object").
    """
    image = perception.get("image")
    clusters = (perception.get("lidar") or {}).get("clusters") or []
    found, used = [], set()
    cam = Camera(image.shape[1], image.shape[0], perception.get("camera")) if image is not None else None
    for det in perception.get("detections", []) if cam else []:
        if det["class_id"] not in PEOPLE | VEHICLES:
            continue
        pos = locate(det, cam, perception.get("depth"))
        cluster = match_cluster(det, cam, clusters, pos["distance_geom_m"] if pos else None)
        if cluster is not None:
            used.add(id(cluster))
            pos = {**(pos or {"distance_geom_m": None}), "distance_m": cluster["nearest_ahead_m"],
                   "lateral_m": cluster["right_m"], "extent": cluster["extent"], "source": "lidar"}
        if pos is None or pos["distance_m"] > look_ahead_m:
            continue
        kind = "person" if det["class_id"] in PEOPLE else "vehicle"
        offset = pos["lateral_m"] - path_right_at(pos["distance_m"])
        if abs(offset) <= CORRIDOR_HALF_WIDTH_M[kind]:
            found.append({"class": det["class"], "kind": kind,
                          "confidence": det["confidence"], "bbox": det["bbox"], **pos})
    for c in clusters:
        if id(c) in used or c["height_m"] < 0.5:
            continue
        # Points of the cluster that are in front of our bumper and within our car's width of the path
        # at their own distance. (A wall or guardrail alongside the road isn't, even on a tight bend.)
        hits = [a for a, r in c.get("pts", []) if FRONT_BUMPER_M <= a <= look_ahead_m
                and abs(r - path_right_at(a)) <= LIDAR_CORRIDOR_HALF_WIDTH_M]
        if len(hits) >= 3:
            found.append({"class": "object", "kind": "object", "confidence": 1.0, "bbox": None,
                          "distance_m": round(min(hits), 2), "distance_geom_m": None, "lateral_m": c["right_m"],
                          "extent": c["extent"], "source": "lidar"})
    return sorted(found, key=lambda o: o["distance_m"])
