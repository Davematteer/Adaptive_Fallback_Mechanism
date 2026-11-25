"""Follows each obstacle in our path over time and decides whether it's traffic or a real blockage.

Every frame the obstacles in our corridor are matched to the ones seen before (same kind, where the
old one should now be given our own movement), so each physical vehicle keeps one id.

Movement: an obstacle's position ALONG our path is   p = distance_ahead + our own travelled distance.
A parked car keeps p constant while we drive up to it; a car in traffic changes p. The obstacle's
own speed is the slope of p over the last ~1.5 s (a least-squares fit, so single noisy distance
readings don't count as movement).

Classification
  moving               it's moving now
  traffic_stopped      stopped, but there's evidence it's traffic: it moved earlier, or there's a
                       queue (more vehicles beyond it in our lane), a junction close ahead, or a
                       traffic light in view
  stationary_obstacle  stopped, never seen moving although we watched it for long enough while
                       approaching, and no traffic evidence: a parked / broken-down car, debris...
  unknown              not enough evidence yet (treated like traffic: wait, don't manoeuvre)
"""

from collections import deque

import numpy as np

MATCH_DIST_M = 3.0
MATCH_LATERAL_M = 1.5
FORGET_S = 2.0           # drop a track not seen for this long
WINDOW_S = 1.5           # speed fit window
MOVING_MPS = 1.2         # obstacle's own speed above this = moving
MIN_OBSERVED_S = 1.5     # watched at least this long (while we were approaching) to call it stationary
APPROACH_M = 4.0         # ... and we closed in by at least this much meanwhile
JUNCTION_NEAR_M = 50.0   # a junction this close ahead counts as traffic evidence
LIGHT_MEMORY_S = 3.0     # a traffic light seen in the last few seconds counts as traffic evidence
TRAFFIC_LIGHT_CLASS_ID = 9  # COCO


class ObstacleTracker:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.tracks = []
        self.next_id = 1
        self.odometry = 0.0       # our signed travelled distance (m)
        self.last_time = None
        self.light_seen_at = None

    def update(self, obstacles: list, now: float, forward_speed: float, detections: list) -> None:
        """Match this frame's in-path obstacles to existing tracks; annotate each obstacle with obs["track"]."""
        dt = 0.0 if self.last_time is None else max(0.0, now - self.last_time)
        self.last_time = now
        moved = forward_speed * dt
        self.odometry += moved
        if any(d.get("class_id") == TRAFFIC_LIGHT_CLASS_ID for d in detections):
            self.light_seen_at = now

        unmatched = list(self.tracks)
        for obs in obstacles:
            best = None
            for tr in unmatched:
                predicted = tr["distance"] - (self.odometry - tr["odometry"])  # if it stood still
                err = abs(obs["distance_m"] - predicted)
                if (tr["kind"] == obs["kind"] and err < MATCH_DIST_M + abs(tr["speed"]) * dt
                        and abs(obs["lateral_m"] - tr["lateral"]) < MATCH_LATERAL_M and (best is None or err < best[0])):
                    best = (err, tr)
            if best is None:
                tr = {"id": self.next_id, "kind": obs["kind"], "first_seen": now, "first_odometry": self.odometry,
                      "history": deque(), "ever_moved": False, "speed": 0.0}
                self.next_id += 1
                self.tracks.append(tr)
            else:
                tr = best[1]
                unmatched.remove(tr)
            tr.update(distance=obs["distance_m"], lateral=obs["lateral_m"], odometry=self.odometry, last_seen=now)
            tr["history"].append((now, obs["distance_m"] + self.odometry))
            while tr["history"] and now - tr["history"][0][0] > WINDOW_S:
                tr["history"].popleft()
            tr["speed"] = self._speed(tr["history"])
            if abs(tr["speed"]) > MOVING_MPS:
                tr["ever_moved"] = True
            obs["track"] = tr
        self.tracks = [t for t in self.tracks if now - t["last_seen"] <= FORGET_S]

    @staticmethod
    def _speed(history) -> float:
        """Least-squares slope of the along-path position: the obstacle's own speed (m/s)."""
        if len(history) < 5 or history[-1][0] - history[0][0] < 0.8:
            return 0.0
        t, p = np.array(history).T
        return float(np.polyfit(t - t[0], p, 1)[0])

    def classify(self, blocker: dict, lane_obstacles: list, junction_ahead_m: float | None, now: float) -> dict:
        """Traffic or a real blockage? Returns the evidence too (for the log and the dashboard)."""
        tr = blocker.get("track")
        if tr is None:
            return {"state": "unknown", "id": None, "ever_moved": None, "queue": False,
                    "junction": False, "light": False, "evidence": []}
        queue = any(o is not blocker and o["kind"] in ("vehicle", "object") and vehicle_like(o)
                    and o["distance_m"] > blocker["distance_m"] + 3.0 for o in lane_obstacles)
        junction = junction_ahead_m is not None and junction_ahead_m <= JUNCTION_NEAR_M
        light = self.light_seen_at is not None and now - self.light_seen_at <= LIGHT_MEMORY_S
        evidence = [name for name, on in (("moved earlier", tr["ever_moved"]), ("queue ahead", queue),
                                          ("junction ahead", junction), ("traffic light", light)) if on]
        watched_s = now - tr["first_seen"]
        approached_m = self.odometry - tr["first_odometry"]
        if abs(tr["speed"]) > MOVING_MPS:
            state = "moving"
        elif evidence:
            state = "traffic_stopped"
        elif watched_s >= MIN_OBSERVED_S and approached_m >= APPROACH_M:
            state = "stationary_obstacle"
        else:
            state = "unknown"
        return {"state": state, "id": tr["id"], "ever_moved": tr["ever_moved"], "queue": queue,
                "junction": junction, "light": light, "evidence": evidence}


def vehicle_like(obs: dict) -> bool:
    """A camera-detected vehicle, or a LiDAR cluster with a car-like size (DBSCAN finds blobs, not cars)."""
    if obs["kind"] == "vehicle":
        return True
    ext = obs.get("extent")
    return bool(ext) and ((ext[3] - ext[2]) >= 1.0 or (ext[1] - ext[0]) >= 1.5)
