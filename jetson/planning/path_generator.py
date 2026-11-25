"""Builds the lane centreline from the lane detector, in the vehicle frame.

All positions are in the vehicle frame (ahead_m forward, right_m to the right), so the
controller needs no map and no GPS: only what the camera sees.

Pair the ego lane's left and right boundary points on the same image row, take the
midpoint and convert it to metres. With only one boundary, offset it by half a lane.

Road segmentation does not steer: in a closed-loop test, steering to the middle of the
segmented road (wrong lane on a multi-lane road) and rejecting lane targets with a mask
that was a few frames old made the car weave. It is kept for the dashboard and later checks.
"""

import numpy as np

from perception.camera_geometry import Camera

HALF_LANE_M = 1.75          # CARLA lanes are ~3.5 m wide
LANE_WIDTH_RANGE_M = (2.4, 5.0)
SPLIT_MARGIN_M = 0.4        # lane this much wider ahead than near the car = the lane is splitting
SPLIT_MAX_WIDTH_M = 9.0     # keep diverging line pairs up to this far apart so a split can be seen


def lane_centreline(lanes: dict, cam: Camera, keep: str = "right") -> tuple[list[tuple[float, float]], float | None]:
    """(centre of the ego lane as (ahead_m, right_m) points near -> far, measured lane width or None).

    keep: which branch to follow where the lane splits ("right" or "left"), see below.
    """
    left = {v: u for u, v in lanes.get("ego_left", [])}
    right = {v: u for u, v in lanes.get("ego_right", [])}
    pairs = []  # (ahead_m, left_right_m, right_right_m), near -> far
    for v in sorted(set(left) & set(right), reverse=True):
        a, b = cam.pixel_to_ground(left[v], v), cam.pixel_to_ground(right[v], v)
        if a and b and LANE_WIDTH_RANGE_M[0] <= b[1] - a[1] <= SPLIT_MAX_WIDTH_M:
            pairs.append((a[0], a[1], b[1]))
    near = [r - lft for _, lft, r in pairs[:max(3, len(pairs) // 3)]]
    near_width = float(np.median(near)) if near else None
    if len(pairs) >= 3 and LANE_WIDTH_RANGE_M[0] <= near_width <= LANE_WIDTH_RANGE_M[1]:
        # Lane split: further ahead the two lines diverge, e.g. an exit-only lane curving onto a ramp
        # while the other line runs on into the gore. The midpoint then heads for the gore barrier
        # (a closed-loop test on Town04 hit it). Which branch to take is a route decision the camera
        # can't make, so it's a setting: follow that side's line at the normal half-width.
        points = []
        for ahead, lft, r in pairs:
            if r - lft > near_width + SPLIT_MARGIN_M:
                points.append((ahead, r - near_width / 2 if keep == "right" else lft + near_width / 2))
            else:
                points.append((ahead, (lft + r) / 2))
        return points, near_width
    # Only one boundary is usable: follow it, half a lane in.
    for name, sign in (("ego_left", 1), ("ego_right", -1)):
        pts = [cam.pixel_to_ground(u, v) for u, v in sorted(lanes.get(name, []), key=lambda p: -p[1])]
        pts = [(g[0], g[1] + sign * HALF_LANE_M) for g in pts if g]
        if len(pts) >= 3:
            return pts, None
    return [], None


def point_at(points: list[tuple[float, float]], ahead_m: float) -> tuple[float, float] | None:
    """Path point at `ahead_m`, interpolated; the nearest end point if the path doesn't reach that far."""
    if not points:
        return None
    pts = sorted(points)
    aheads = [p[0] for p in pts]
    if ahead_m >= aheads[-1]:
        return pts[-1]
    if ahead_m <= aheads[0]:
        return pts[0]
    return ahead_m, float(np.interp(ahead_m, aheads, [p[1] for p in pts]))


def curvature(points: list[tuple[float, float]]) -> float:
    """Approximate path curvature (1/m) from a quadratic fit right = c0 + c1*a + c2*a^2 over 3..20 m."""
    pts = [p for p in points if 3.0 <= p[0] <= 20.0]
    if len(pts) < 5:
        return 0.0
    a, r = np.array(pts).T
    c2, c1, _ = np.polyfit(a, r, 2)
    return float(abs(2 * c2) / (1 + c1 ** 2) ** 1.5)


def find_lane_path(perception: dict, keep: str = "right") -> dict:
    """{"centreline": [(ahead_m, right_m), ...] (may be empty), "lane_width_m": float | None}"""
    image = perception.get("image")
    if image is None:
        return {"centreline": [], "lane_width_m": None}
    cam = Camera(image.shape[1], image.shape[0], perception.get("camera"))
    centre, width = lane_centreline(perception.get("lanes") or {}, cam, keep)
    centre = [p for p in centre if p[0] >= 3.0]  # ignore points under the bonnet
    return {"centreline": centre, "lane_width_m": None if width is None else round(width, 2)}
