"""Decides where to steer and how fast to go.

Path source (chosen on the dashboard, sent by the host in state["path_source"]):
  "perception"  lane detection (kept consistent between frames by PathTracker), with the route as guide:
                the route from the host (Dijkstra to the chosen destination, or "follow the road") is
                followed instead of the lanes where lanes can't do the job:
                  - the lanes are lost (unmarked bends, inside junctions),
                  - a junction is close (the route says which way to go),
                  - the lanes disagree with the route (e.g. lane fragments running straight on at a bend).
  "map"         route waypoints only (comparison).
At the destination (route_remaining_m from the host) the car slows down and stops.

Speed:
  - person/bicycle in the path  -> stop, and wait until the path has been clear for `clear_for_s`
  - vehicle in the path         -> follow: slow down as the gap closes, stop if it gets too small
  - curve                       -> at most sqrt(MAX_LATERAL_ACCEL / curvature)
  - dead-reckoning (no lanes)   -> at most PREDICT_SPEED_FACTOR x target
  - no path at all              -> stop until a path has been back for `path_back_s`

Getting unstuck (planning/maneuvers.py): first decide whether the obstacle is traffic or a real blockage
(planning/obstacle_tracker.py). Traffic: wait (re-check after a long time, never reverse). A stationary
obstacle for a few seconds: swerve around it if there's a gap, else reverse a little and retry.
"""

import math
import time

from planning.maneuvers import Maneuvers
from planning.obstacle_assessment import in_path_obstacles
from planning.obstacle_tracker import ObstacleTracker
from planning.path_generator import curvature, find_lane_path, point_at
from planning.path_tracker import PathTracker, move_path

STOP_GAP_M = 8.0        # stop if a vehicle ahead is this close
FOLLOW_GAP_M = 25.0     # start slowing down for a vehicle ahead from here
MAX_LATERAL_ACCEL = 2.0  # m/s^2, sets the speed limit in curves
PREDICT_SPEED_FACTOR = 0.6


def extend_path(points: list, to_ahead_m: float, step_m: float = 2.0) -> list:
    """Continue a path straight on along its last direction until it reaches `to_ahead_m`."""
    pts = sorted(points)
    if len(pts) < 2 or pts[-1][0] >= to_ahead_m:
        return pts
    (a0, r0), (a1, r1) = pts[-2], pts[-1]
    slope = (r1 - r0) / (a1 - a0) if a1 - a0 > 1e-3 else 0.0
    a = a1
    while a < to_ahead_m:
        a += step_m
        pts.append((a, r1 + slope * (a - a1)))
    return pts


def lookahead_for(speed_mps: float) -> float:
    """Pure-pursuit look-ahead: longer at speed, which damps the steering (less over-correction)."""
    return min(15.0, max(7.0, 1.2 * speed_mps))


class PathPlanner:
    def __init__(self, clear_for_s: float = 1.0, path_back_s: float = 0.5):
        self.clear_for_s, self.path_back_s = clear_for_s, path_back_s
        self.tracker = PathTracker()
        self.maneuvers = Maneuvers()
        self.obstacle_tracker = ObstacleTracker()
        self.reset()

    def reset(self) -> None:
        """Back to the start state (called when the host disconnects, i.e. between experiments)."""
        self.tracker.reset()
        self.maneuvers.reset()
        self.obstacle_tracker.reset()
        self.wants_to_move = False
        self.disagree = 0          # consecutive frames the lanes disagree with the route
        self.swerve_base = None   # the original lane path while swerving (vehicle frame, moved by odometry)
        self.base_time = self.base_yaw = None
        self.person_stop = False
        self.clear_since = None
        self.no_path = False
        self.path_back_since = None

    def plan(self, state: dict, perception: dict, now: float | None = None) -> dict:
        # Simulation time if the host sends it (exact), else this computer's clock.
        if now is None:
            now = state.get("sim_time_s")
        if now is None:
            now = time.monotonic()
        speed = state.get("speed_mps", 0.0)
        fwd_speed = state.get("forward_speed_mps", speed)  # signed: negative while reversing
        lookahead = lookahead_for(speed)

        # 1. Where to go.
        lane_width = None
        route = self._map_path(state) if state.get("waypoints") else []
        route_why = None
        if state.get("path_source") == "map":
            centre, note, use_route = route, "map", True
        else:
            found = find_lane_path(perception, state.get("at_split", "right"))
            lane_width = found["lane_width_m"]
            centre, note = self.tracker.update(found["centreline"], fwd_speed, state.get("yaw_deg"), now, lookahead)
            junction = state.get("junction_ahead_m")
            lane_t, route_t = point_at(centre, lookahead), point_at(route, lookahead)
            self.disagree = self.disagree + 1 if lane_t and route_t and abs(lane_t[1] - route_t[1]) > 1.5 else 0
            if route and note in ("none", "predicted"):
                route_why = "lanes lost"
            elif route and junction is not None and junction < 12.0:
                route_why = "junction ahead"
            elif route and self.disagree >= 2:
                route_why = "lanes disagree with the route"
            use_route = route_why is not None
            if use_route:
                centre = route

        # 2. What is in the way of the normal lane path, and do we need to get round it?
        def right_at(points):
            def f(ahead_m: float) -> float:
                p = point_at(points, ahead_m)
                return p[1] if p else 0.0
            return f
        lane_obstacles = in_path_obstacles(perception, right_at(centre))
        blocker = next((o for o in lane_obstacles if (o["kind"] == "person" and o["distance_m"] <= 20.0)
                        or (o["kind"] in ("vehicle", "object") and o["distance_m"] <= STOP_GAP_M + 1.0)), None)
        # Same vehicle over time: has it moved? is it traffic or a real blockage?
        self.obstacle_tracker.update(lane_obstacles, now, fwd_speed, perception.get("detections", []))
        blockage = (self.obstacle_tracker.classify(blocker, lane_obstacles, state.get("junction_ahead_m"), now)
                    if blocker is not None else None)
        # Space behind: the rear parking sensor and the LiDAR; the closer reading wins.
        rears = [x for x in (state.get("rear_clearance_m"), (perception.get("lidar") or {}).get("rear_clearance_m"))
                 if x is not None]
        man = self.maneuvers.update(now, fwd_speed, blocker, perception, min(rears) if rears else None,
                                    right_at(centre), self.wants_to_move, blockage)
        if man["offset_m"]:
            # Swerving: steer to the ORIGINAL lane shifted sideways. The original lane is frozen when the swerve
            # starts and moved by our own motion; live detections would report the next lane once we're in it,
            # and shifting that again pushed the car off the road in a test.
            yaw = state.get("yaw_deg")
            if self.swerve_base is None:
                self.swerve_base = list(centre)
            elif self.base_time is not None:
                dyaw = math.radians((yaw - self.base_yaw + 180) % 360 - 180) if yaw is not None and self.base_yaw is not None else 0.0
                self.swerve_base = [p for p in move_path(self.swerve_base, fwd_speed * max(0.0, now - self.base_time), dyaw)
                                    if p[0] > -2.0]
            self.base_time, self.base_yaw = now, yaw
            self.swerve_base = extend_path(self.swerve_base, 30.0)  # it was detected ~20 m out; keep it long enough
            centre = [(a, r + man["offset_m"]) for a, r in self.swerve_base if a > 0.5]
            obstacles = in_path_obstacles(perception, right_at(centre))
        else:
            if self.swerve_base is not None:  # swerve finished: hand the original lane back to the tracker
                self.tracker.set_reference([p for p in self.swerve_base if p[0] > 0.5], now)
                self.swerve_base = self.base_time = self.base_yaw = None
            obstacles = lane_obstacles
        target = point_at([p for p in centre if p[0] > 0], lookahead) if centre else None
        self._update_no_path(target is not None or man["reverse"], now)

        person = next((o for o in obstacles if o["kind"] == "person" and o["distance_m"] <= 20.0), None)
        vehicle = next((o for o in obstacles if o["kind"] in ("vehicle", "object")), None)  # object: LiDAR only
        if person is not None:
            self.person_stop, self.clear_since = True, None
        elif self.person_stop:
            self.clear_since = self.clear_since or now
            if now - self.clear_since >= self.clear_for_s:
                self.person_stop, self.clear_since = False, None

        # 3. How fast.
        speed_target = state.get("target_speed_mps", 8.0)
        slow_reason = None
        kappa = curvature(centre) if centre else 0.0
        if kappa > 1e-3 and math.sqrt(MAX_LATERAL_ACCEL / kappa) < speed_target:
            speed_target = math.sqrt(MAX_LATERAL_ACCEL / kappa)
            slow_reason = f"curve (radius {1 / kappa:.0f} m)"
        if note == "predicted" and not use_route:
            speed_target = min(speed_target, PREDICT_SPEED_FACTOR * state.get("target_speed_mps", 8.0))
            slow_reason = "lanes lost: dead-reckoning"
        remaining = state.get("route_remaining_m")
        arrived = remaining is not None and remaining < 4.0
        if remaining is not None and remaining < 25.0:
            speed_target = min(speed_target, max(0.0, (remaining - 3.0) * 0.6))
            slow_reason = f"destination in {remaining:.0f} m"
        if man["speed_cap"] is not None and not man["reverse"]:
            speed_target = min(speed_target, man["speed_cap"])
            slow_reason = man["reason"]
        if vehicle is not None:
            gap = vehicle["distance_m"]
            speed_target = min(speed_target, speed_target * max(0.0, (gap - STOP_GAP_M) / (FOLLOW_GAP_M - STOP_GAP_M)))
            slow_reason = f"following {vehicle['class']} {gap:.1f} m ahead"
        stop = self.person_stop or self.no_path or arrived or (vehicle is not None and vehicle["distance_m"] <= STOP_GAP_M)
        if arrived:
            reason = "destination reached"
        elif self.person_stop:
            reason = "person in path"
        elif self.no_path:
            reason = "no path"
        elif stop:
            reason = f"{vehicle['class']} {vehicle['distance_m']:.1f} m ahead"
        else:
            reason = slow_reason
        if man["mode"] in ("wait", "traffic_wait", "reverse", "recovery_failed") and man["reason"]:
            reason = man["reason"]
        reverse = man["reverse"]  # backing away is fine even with a person in front; only forward is blocked
        if reverse:
            stop = False
        self.wants_to_move = not stop and not reverse and speed_target > 0.5

        return {
            "reverse": reverse, "maneuver": man["mode"], "blockage": blockage,
            "ahead_m": target[0] if target else None, "right_m": target[1] if target else None,
            "speed_mps": 0.0 if stop else (man["speed_cap"] if reverse else speed_target), "stop": stop,
            "decision": "reverse" if reverse else ("stop" if stop else ("slow" if slow_reason else "go")),
            "reason": reason,
            "obstacle": {k: v for k, v in (person or vehicle).items() if k != "track"} if (person or vehicle) else None,
            "path": {"source": "route" if use_route else ("lanes" if centre else "none"), "note": route_why or note,
                     "target": [round(target[0], 2), round(target[1], 2)] if target else None,
                     "lane_width_m": lane_width, "curvature": round(kappa, 4),
                     "centreline": [[round(a, 2), round(r, 2)] for a, r in centre]},
        }

    def _update_no_path(self, have_path: bool, now: float) -> None:
        """Stop when there is no path; resume only once a path has been back for `path_back_s`."""
        if not have_path:
            self.no_path, self.path_back_since = True, None
        elif self.no_path:
            self.path_back_since = self.path_back_since or now
            if now - self.path_back_since >= self.path_back_s:
                self.no_path, self.path_back_since = False, None

    @staticmethod
    def _map_path(state: dict) -> list[tuple[float, float]]:
        """CARLA's waypoints (world frame) -> vehicle frame."""
        yaw = math.radians(state["yaw_deg"])
        centre = []
        for wx, wy in state.get("waypoints") or []:
            dx, dy = wx - state["x"], wy - state["y"]
            centre.append((math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy))
        return [p for p in centre if p[0] > 0]
