"""Getting unstuck: swerve around a stationary blockage, or reverse and try again.

First: is it traffic or a real blockage? (planning/obstacle_tracker.py)
    traffic (the car ahead moved earlier, a queue, a junction or traffic light ahead, or not sure yet)
        -> TRAFFIC_WAIT. After TRAFFIC_REASSESS_S still stopped: RE-CHECK for a safe way round (swerve
           only, with all the gap checks); never reverse into traffic. No safe way: keep waiting.
    stationary obstacle (never moved while we watched it on the way in, no traffic evidence)
        -> WAIT for WAIT_S, then:

    DRIVE --blocked by a stationary obstacle for WAIT_S--> choose:
        SWERVE  a side gap exists: shift the path sideways around it, pass, shift back -> DRIVE
        REVERSE no gap, rear clear: back up REVERSE_M slowly, then decide again
        WAIT    no gap, rear blocked: keep waiting, decide again every RETRY_S
        RECOVERY_FAILED  reversed MAX_REVERSES times and still no gap: stay stopped, keep re-checking
    DRIVE --stuck (wants to move, speed ~0 for STUCK_S, e.g. nosed into a kerb)--> REVERSE -> DRIVE

"Blocked" only counts an obstacle that stays put while we are stopped: a pedestrian who is
crossing leaves the corridor within seconds and never triggers a maneuver.
The rear clearance comes from the host's rear parking sensor (None = nothing behind).
"""

from perception.camera_geometry import Camera

WAIT_S = 5.0            # how long to wait behind a stationary blockage before doing something
TRAFFIC_REASSESS_S = 60.0  # in traffic: how long before re-checking whether we can get round at all
STUCK_S = 2.0           # wanting to move but not moving for this long = stuck
RETRY_S = 3.0           # how often to re-check for a way round while waiting
REVERSE_M = 4.0         # how far to back up each time
REAR_MIN_M = 1.5        # never reverse with something closer than this behind
MAX_REVERSES = 2        # reverse attempts per blockage before just waiting
STATIONARY_M = 1.0      # obstacle moved less than this (distance and sideways) = stationary
EGO_HALF_WIDTH_M = 1.0
MARGIN_M = {"person": 1.2, "vehicle": 0.8}
MAX_OFFSET_M = 3.8      # never swerve further than about one lane
RETURN_AFTER_M = 6.0    # go back to the lane centre this far past the obstacle
CAR_LENGTH_M = 4.7
SWERVE_SPEED_MPS = 3.0
REVERSE_SPEED_MPS = 1.5


class Maneuvers:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.mode = "drive"
        self.blocked_since = self.block_ref = None
        self.stuck_since = None
        self.travelled = 0.0          # signed odometry (m) since the current maneuver started
        self.offset = 0.0             # sideways path shift while swerving (+ = right)
        self.swerve_end_m = 0.0
        self.reverses = 0
        self.next_retry = None
        self.last_time = None
        self.reason = None
        self.decide_now = False       # after a reverse: decide at the next blockage without waiting again

    # ------------------------------------------------------------------
    def update(self, now: float, forward_speed: float, blocker: dict | None, perception: dict,
               rear_clearance: float | None, lane_right_at, wants_to_move: bool,
               blockage: dict | None = None) -> dict:
        """Called once per frame.

        blocker:       nearest obstacle the planner is stopping for (on the normal lane path), or None
        lane_right_at: ahead_m -> lateral of the lane centreline there (to size the swerve)
        wants_to_move: last frame the planner asked for a speed > 0 (for stuck detection)
        blockage:      ObstacleTracker.classify() for the blocker: traffic or a stationary obstacle
        Returns {"mode", "offset_m", "reverse", "speed_cap", "reason"}.
        """
        dt = 0.0 if self.last_time is None else max(0.0, now - self.last_time)
        self.last_time = now
        self.travelled += forward_speed * dt

        if self.mode == "reverse":
            behind_close = rear_clearance is not None and rear_clearance < REAR_MIN_M
            if -self.travelled >= REVERSE_M or behind_close:
                self.reverses += 1
                self._start("drive", now, "reversed: trying again" if not behind_close
                            else "stopped reversing: something behind")
                self.decide_now = True
            else:
                return self._out(reverse=True, speed_cap=REVERSE_SPEED_MPS)

        if self.mode == "swerve":
            if self.travelled >= self.swerve_end_m:
                self._start("drive", now, "passed the obstacle: back to the lane")
                self.offset = 0.0
            else:
                return self._out(speed_cap=SWERVE_SPEED_MPS)

        # --- DRIVE / WAIT: watch for a blockage or for being stuck --------------
        stationary = abs(forward_speed) < 0.3
        if blocker is not None and stationary:
            info = blockage or {"state": "unknown", "evidence": []}
            # People: wait WAIT_S whatever (a crossing one leaves the corridor long before). Vehicles/objects:
            # only a confirmed stationary obstacle gets the short wait; anything else is treated as traffic.
            traffic = blocker.get("kind") != "person" and info["state"] != "stationary_obstacle"
            wait_s = TRAFFIC_REASSESS_S if traffic else WAIT_S
            if self.block_ref is None or not self._same_obstacle(blocker):
                self.blocked_since, self.block_ref = (now - wait_s if self.decide_now else now), dict(blocker)
                self.decide_now = False
            elif now - self.blocked_since >= wait_s and (self.next_retry is None or now >= self.next_retry):
                return self._decide(now, blocker, perception, rear_clearance, lane_right_at, traffic)
            if self.mode in ("drive", "wait", "traffic_wait"):
                self.mode = "traffic_wait" if traffic else "wait"
            why = f" ({', '.join(info['evidence'])})" if traffic and info.get("evidence") else ""
            self.reason = (f"{blocker['class']} {blocker['distance_m']:.1f} m ahead, "
                           f"{'traffic' + why if traffic else 'stationary obstacle'}: waiting {now - self.blocked_since:.0f} s")
            self.stuck_since = None
            return self._out()

        if blocker is None:
            self.blocked_since = self.block_ref = self.next_retry = None
            if self.mode in ("wait", "traffic_wait", "recovery_failed"):
                self._start("drive", now, "path clear")
            if self.mode == "drive" and forward_speed > 2.0:  # driving normally again: fresh start
                self.reverses, self.decide_now = 0, False

        # Stuck: asked to move, not moving, nothing detected in the way (e.g. nose against a kerb/wall).
        if wants_to_move and stationary and blocker is None:
            self.stuck_since = self.stuck_since or now
            if now - self.stuck_since >= STUCK_S:
                self.stuck_since = None
                if (rear_clearance is None or rear_clearance >= REAR_MIN_M + 0.5) and self.reverses < MAX_REVERSES:
                    self._start("reverse", now, "stuck: reversing")
                    return self._out(reverse=True, speed_cap=REVERSE_SPEED_MPS)
                self.reason = "stuck and can't reverse"
        else:
            self.stuck_since = None
        return self._out()

    # ------------------------------------------------------------------
    def _decide(self, now, blocker, perception, rear_clearance, lane_right_at, traffic: bool = False) -> dict:
        """Blocked long enough: re-check. Swerve if there's a safe gap; else reverse if the rear is clear
        (never in traffic); else keep waiting. A timeout never forces a move by itself."""
        for side in ("left", "right"):  # left first: the overtaking side in right-hand traffic
            offset = swerve_offset(blocker, side, perception, lane_right_at)
            if offset is not None:
                self._start("swerve", now, f"{blocker['class']} {blocker['distance_m']:.1f} m ahead "
                                           f"for {now - self.blocked_since:.0f} s: swerving {side}")
                self.offset = offset
                self.swerve_end_m = blocker["distance_m"] + CAR_LENGTH_M + RETURN_AFTER_M
                self.blocked_since = self.block_ref = None
                return self._out(speed_cap=SWERVE_SPEED_MPS)
        if traffic:  # in traffic: never reverse (there's often a queue behind us too)
            self.mode, self.next_retry = "traffic_wait", now + RETRY_S
            self.reason = f"{blocker['class']} ahead, still in traffic and no safe way round: waiting"
            return self._out()
        if (rear_clearance is None or rear_clearance >= REAR_MIN_M + 0.5) and self.reverses < MAX_REVERSES:
            self._start("reverse", now, f"{blocker['class']} ahead, no gap to pass: reversing to get room")
            self.blocked_since = self.block_ref = None
            return self._out(reverse=True, speed_cap=REVERSE_SPEED_MPS)
        failed = self.reverses >= MAX_REVERSES
        self.mode, self.next_retry = ("recovery_failed" if failed else "wait"), now + RETRY_S
        self.reason = (f"{blocker['class']} ahead: reversed {self.reverses}x, still no gap - staying stopped" if failed
                       else f"{blocker['class']} ahead, no gap and can't reverse: waiting")
        return self._out()

    def _same_obstacle(self, blocker: dict) -> bool:
        ref = self.block_ref
        return (abs(blocker["distance_m"] - ref["distance_m"]) < STATIONARY_M
                and abs(blocker["lateral_m"] - ref["lateral_m"]) < STATIONARY_M)

    def _start(self, mode: str, now: float, reason: str) -> None:
        self.mode, self.reason, self.travelled = mode, reason, 0.0

    def _out(self, reverse: bool = False, speed_cap: float | None = None) -> dict:
        return {"mode": f"swerve_{'left' if self.offset < 0 else 'right'}" if self.mode == "swerve" else self.mode,
                "offset_m": self.offset if self.mode == "swerve" else 0.0,
                "reverse": reverse, "speed_cap": speed_cap, "reason": self.reason}


def swerve_offset(blocker: dict, side: str, perception: dict, lane_right_at) -> float | None:
    """Sideways path shift that passes `blocker` on `side` with a safety margin, or None if there's no gap."""
    image = perception.get("image")
    if image is None:
        return None
    cam = Camera(image.shape[1], image.shape[0], perception.get("camera"))
    sign = -1.0 if side == "left" else 1.0
    # Sideways extent of the blocker: the union of what the camera box and the LiDAR cluster cover.
    # (The LiDAR often sees only one corner of a car; the camera box alone can be off in distance.)
    spans = []
    if blocker.get("bbox"):
        x1, _, x2, _ = blocker["bbox"]
        scale = (blocker["distance_m"] - cam.x_m) / cam.f
        spans.append(((x1 - cam.cx) * scale, (x2 - cam.cx) * scale))
    if blocker.get("extent"):
        spans.append((blocker["extent"][2], blocker["extent"][3]))
    if not spans:
        return None
    left_edge, right_edge = min(sp[0] for sp in spans), max(sp[1] for sp in spans)
    half_width, lateral = (right_edge - left_edge) / 2, (right_edge + left_edge) / 2
    kind = blocker.get("kind", "vehicle")
    pass_at = lateral + sign * (half_width + MARGIN_M.get(kind, 1.0) + EGO_HALF_WIDTH_M)
    offset = pass_at - lane_right_at(blocker["distance_m"])
    if abs(offset) > MAX_OFFSET_M or offset * sign <= 0:
        return None

    # Is there drivable space there? Road mask if segmentation is running, else a neighbouring lane line.
    road = perception.get("road_mask")
    if road is not None:
        for ahead in (max(4.0, blocker["distance_m"] - 4), blocker["distance_m"], blocker["distance_m"] + 4):
            px = cam.ground_to_pixel(ahead, pass_at)
            if px is None or not (0 <= px[0] < cam.width and 0 <= px[1] < cam.height) or not road[px[1], px[0]]:
                return None
    else:
        # No segmentation: the outer lane line on that side must be visible, and the car must fit inside it.
        outer = [cam.pixel_to_ground(u, v) for u, v in (perception.get("lanes") or {}).get(f"outer_{side}", [])]
        near = [g[1] for g in outer if g and abs(g[0] - blocker["distance_m"]) <= 6.0]
        if not near:
            return None
        line = min(near) if side == "right" else max(near)  # the tightest point of that line near the obstacle
        car_edge = pass_at + sign * (EGO_HALF_WIDTH_M + 0.3)
        if (side == "left" and car_edge < line) or (side == "right" and car_edge > line):
            return None

    # LiDAR: nothing solid in the strip the car would drive through (other than the blocker itself).
    lo, hi = pass_at - EGO_HALF_WIDTH_M - 0.3, pass_at + EGO_HALF_WIDTH_M + 0.3
    for c in (perception.get("lidar") or {}).get("clusters") or []:
        a0, a1, r0, r1 = c["extent"]
        if c.get("extent") == blocker.get("extent") or a1 < max(1.0, blocker["distance_m"] - 4) or a0 > blocker["distance_m"] + 15:
            continue
        if r0 < hi and r1 > lo and c["height_m"] >= 0.5:
            return None

    # LiDAR behind and beside us: nothing coming up in the lane we'd move into (it could be overtaking us).
    for c in (perception.get("lidar") or {}).get("clusters") or []:
        a0, a1, r0, r1 = c["extent"]
        if a0 < 2.0 and a1 > -20.0 and r0 < hi + 0.3 and r1 > lo - 0.3 and c["height_m"] >= 0.5 and (r1 - r0) >= 0.5:
            return None

    # Nothing else in the way on that side (another car, a person) up to well past the blocker.
    from planning.obstacle_assessment import locate  # local import: avoids a cycle at module load
    for det in perception.get("detections", []):
        if det["bbox"] == blocker.get("bbox"):
            continue
        pos = locate(det, cam, perception.get("depth"))
        if pos and pos["distance_m"] < blocker["distance_m"] + 15 and abs(pos["lateral_m"] - pass_at) < 2.0:
            return None
    return offset
