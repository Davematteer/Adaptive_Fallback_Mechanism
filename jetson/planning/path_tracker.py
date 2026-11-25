"""Keeps the lane path consistent between frames.

The centreline is in the vehicle frame, so it goes stale as soon as the car moves or
turns. Each frame the previous path is moved by the car's own motion since then
(distance = speed * dt, heading change from the yaw: the odometry / IMU a real car has;
no position or map is used). That predicted path is used to:

  - dead-reckon:  if the lanes drop out, keep steering along the predicted path for up to
                  `max_predict_s`, instead of holding a target that no longer means anything;
  - gate flips:   if a new detection is more than `gate_m` away from the prediction at the
                  look-ahead distance, the model has probably switched to a neighbouring
                  lane line; ignore it for up to `max_rejects` frames, accept it if it persists.
"""

import math

from planning.path_generator import point_at


def move_path(points, distance_m: float, dyaw_rad: float):
    """Re-express vehicle-frame points after the car drove `distance_m` and turned `dyaw_rad` (+ = right)."""
    ta, tr = distance_m * math.cos(dyaw_rad / 2), distance_m * math.sin(dyaw_rad / 2)  # chord of the arc
    c, s = math.cos(dyaw_rad), math.sin(dyaw_rad)
    out = []
    for a, r in points:
        a, r = a - ta, r - tr
        out.append((c * a + s * r, -s * a + c * r))
    return out


class PathTracker:
    def __init__(self, gate_m: float = 0.9, max_rejects: int = 3, max_predict_s: float = 1.5,
                 max_lateral_m: float = 3.0):
        self.gate_m, self.max_rejects, self.max_predict_s = gate_m, max_rejects, max_predict_s
        self.max_lateral_m = max_lateral_m
        self.reset()

    def reset(self) -> None:
        self.path = []            # current best centreline (vehicle frame)
        self.last_time = self.last_yaw = None
        self.predicted_for = 0.0  # seconds since the last accepted detection
        self.rejects = 0
        self.candidate = None     # lateral position (at the look-ahead) of the last unconfirmed detection
        self.protect_until = None  # until then only detections that agree with the path are accepted

    def set_reference(self, path: list, now: float, protect_s: float = 2.0) -> None:
        """Replace the path (e.g. the original lane after a swerve) and, for `protect_s`, only accept detections
        that agree with it: while the car is still straddling the next lane the detector reports that lane."""
        self.path, self.predicted_for, self.rejects, self.candidate = list(path), 0.0, 0, None
        self.protect_until = now + protect_s

    def _accept(self, detected: list) -> tuple[list, str]:
        self.path, self.predicted_for, self.rejects, self.candidate = list(detected), 0.0, 0, None
        return self.path, "lanes"

    def update(self, detected: list, speed_mps: float, yaw_deg: float | None, now: float,
               lookahead_m: float) -> tuple[list, str]:
        """New detection (may be empty) -> (centreline to follow, note: lanes | gate | predicted | none)."""
        # 1. Move the previous path by our own motion since the last frame.
        if self.path and self.last_time is not None:
            dt = max(0.0, now - self.last_time)
            dyaw = 0.0
            if yaw_deg is not None and self.last_yaw is not None:
                dyaw = math.radians((yaw_deg - self.last_yaw + 180) % 360 - 180)
            self.path = [p for p in move_path(self.path, speed_mps * dt, dyaw) if p[0] > 0.5]
            self.predicted_for += dt
        self.last_time, self.last_yaw = now, yaw_deg

        # 2. Accept, gate or fall back to the prediction.
        fresh = point_at(detected, lookahead_m) if detected else None
        if fresh is not None and abs(fresh[1]) > self.max_lateral_m:
            # Our own lane can't be this far sideways at the look-ahead (even a 20 m-radius curve gives
            # ~1.2 m at 7 m): the detector has picked up some other line. Treat as no detection.
            detected, fresh = [], None
        if detected:
            agrees_with_candidate = (self.candidate is not None and fresh is not None
                                     and abs(fresh[1] - self.candidate) <= self.gate_m)
            if not self.path:
                # Nothing to compare with (start-up, or after losing the path): only trust a detection
                # once two in a row agree. A single wrong first frame otherwise becomes the reference
                # the gate then defends against the correct detections.
                if agrees_with_candidate:
                    return self._accept(detected)
                self.candidate = fresh[1] if fresh else None
                return [], "none"
            predicted = point_at(self.path, lookahead_m)
            jump = abs(fresh[1] - predicted[1]) if predicted and fresh else 0.0
            # Accept if it matches the prediction, or if it was rejected before but keeps agreeing with
            # itself (then the prediction is what's wrong), or after too many rejections.
            protected = self.protect_until is not None and now < self.protect_until
            if jump <= self.gate_m or (not protected and (agrees_with_candidate or self.rejects >= self.max_rejects)):
                return self._accept(detected)
            self.rejects += 1
            self.candidate = fresh[1] if fresh else None
            return self.path, "gate"
        self.candidate = None
        if self.path and self.predicted_for <= self.max_predict_s:
            return self.path, "predicted"
        self.path = []
        return [], "none"
