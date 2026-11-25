"""Turns the planner's target into steering, throttle and brake.

Steering: pure pursuit, steer along the arc that passes through the target point.
The target is in the vehicle frame (ahead_m forward, right_m to the right), so this
works the same whether the path came from perception or from CARLA's map.
Speed: proportional controller (later: full PID).
"""

import math

WHEELBASE_M = 2.9      # Tesla Model 3, the host's default vehicle
MAX_STEER_RAD = 1.22   # ~70 degrees: CARLA's steer = 1.0 maps to the wheel's maximum angle


def compute_command(state: dict, target: dict | None, speed_gain: float = 0.3) -> dict:
    if target is not None and target.get("reverse"):
        # Back up slowly and straight (maneuvers.py decides when and how far; the host's rear sensor guards it).
        too_fast = abs(state.get("forward_speed_mps", -state.get("speed_mps", 0.0))) > target.get("speed_mps", 1.5)
        return {"steer": 0.0, "throttle": 0.0 if too_fast else 0.3, "brake": 0.0, "reverse": True}
    if target is None or target.get("ahead_m") is None:
        return {"steer": 0.0, "throttle": 0.0, "brake": 1.0}  # nowhere to go: stop

    # Pure pursuit: wheel angle = atan(2 * L * lateral offset / distance^2).
    ahead, right = target["ahead_m"], target["right_m"]
    dist_sq = max(ahead ** 2 + right ** 2, 1e-6)
    wheel_angle = math.atan2(2 * WHEELBASE_M * right, dist_sq)
    steer = max(-1.0, min(1.0, wheel_angle / MAX_STEER_RAD))  # positive = right in CARLA

    if target.get("stop"):  # planner says stop: brake fully, keep steering along the path
        return {"steer": round(steer, 4), "throttle": 0.0, "brake": 1.0}

    # Speed: push throttle when too slow, brake when clearly too fast.
    error = target["speed_mps"] - state["speed_mps"]
    throttle = max(0.0, min(0.75, speed_gain * error))
    brake = max(0.0, min(1.0, -speed_gain * error)) if error < -1.0 else 0.0
    return {"steer": round(steer, 4), "throttle": round(throttle, 4), "brake": round(brake, 4)}
