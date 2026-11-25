"""Pedestrians that step out and cross the road in front of the ego vehicle.

Each walker waits at the roadside some distance down the ego's lane. When the ego
gets within `trigger_m`, it walks straight across the lane and stops on the far
side. Walkers are steered directly with carla.WalkerControl rather than CARLA's
AI walker controller, because the AI only uses sidewalks and crossings and would
never cross a highway lane.
"""

import math
import random

import carla


class PedestrianManager:
    def __init__(self, world: carla.World):
        self.world = world
        self.map = world.get_map()  # get_map() rebuilds the whole road map: fetch it once (calling it every tick crashed the client)
        self.walkers = []  # dicts: actor, start, direction (unit carla.Vector3D), distance_m, state

    def spawn_crossers(self, vehicle: carla.Vehicle, count: int, first_m: float = 50.0, spacing_m: float = 60.0,
                       side_offset_m: float = 4.0, speed_mps: float = 1.4, seed: int = 0,
                       behaviour: str = "cross", stand_s: float = 15.0) -> list[str]:
        """Place `count` walkers along the ego's lane. Returns messages for the dashboard event log.

        behaviour: "cross" walks straight across; "stop_in_lane" stops in the middle of the ego lane for
        `stand_s` seconds, then carries on (to test waiting / swerving / reversing).
        """
        messages = []
        rng = random.Random(seed)  # same pedestrians every run, for repeatable experiments
        blueprints = list(self.world.get_blueprint_library().filter("walker.pedestrian.*"))
        here = self.map.get_waypoint(vehicle.get_location())
        for i in range(count):
            ahead_m = first_m + i * spacing_m
            nxt = here.next(ahead_m)
            if not nxt:
                messages.append(f"pedestrian {i + 1}: road ends before {ahead_m:.0f} m, skipped")
                continue
            wp = nxt[0]
            bp = rng.choice(blueprints)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            right = wp.transform.get_right_vector()
            # Try the right roadside first; if something is in the way (barrier, wall), closer in, then the left side.
            for offset in (side_offset_m, side_offset_m - 1.5, -side_offset_m, -(side_offset_m - 1.5)):
                loc = wp.transform.location + carla.Location(x=right.x * offset, y=right.y * offset, z=1.0)
                yaw = wp.transform.rotation.yaw + (-90.0 if offset > 0 else 90.0)  # face across the road
                actor = self.world.try_spawn_actor(bp, carla.Transform(loc, carla.Rotation(yaw=yaw)))
                if actor is not None:
                    sign = -1.0 if offset > 0 else 1.0  # walk towards the other side
                    self.walkers.append({
                        "actor": actor, "start": loc, "speed": speed_mps,
                        "direction": carla.Vector3D(right.x * sign, right.y * sign, 0.0),
                        "distance_m": 2 * abs(offset), "state": "waiting", "ahead_m": ahead_m,
                        "stop_at_m": abs(offset) if behaviour == "stop_in_lane" else None, "stand_s": stand_s,
                    })
                    messages.append(f"pedestrian {i + 1} placed {ahead_m:.0f} m ahead, "
                                    f"{abs(offset):.1f} m {'right' if offset > 0 else 'left'} of the lane")
                    break
            else:
                messages.append(f"pedestrian {i + 1}: no free spot {ahead_m:.0f} m ahead, skipped")
        return messages

    def update(self, ego_location: carla.Location, trigger_m: float, sim_time: float = 0.0) -> tuple[float | None, list[str]]:
        """Call every tick. Starts/stops walkers; returns (closest ego-walker distance in m, event messages)."""
        closest, messages = None, []
        for i, w in enumerate(self.walkers):
            loc = w["actor"].get_location()
            dist = math.hypot(loc.x - ego_location.x, loc.y - ego_location.y)
            closest = dist if closest is None else min(closest, dist)

            if w["state"] == "waiting" and dist <= trigger_m:
                w["state"] = "crossing"
                messages.append(f"pedestrian {i + 1} starts crossing ({dist:.1f} m from the car)")
            if w["state"] == "standing":
                if sim_time >= w["stand_until"]:
                    w["state"] = "crossing"
                    messages.append(f"pedestrian {i + 1} walks on")
                else:
                    w["actor"].apply_control(carla.WalkerControl(direction=w["direction"], speed=0.0))
            if w["state"] == "crossing":
                walked = math.hypot(loc.x - w["start"].x, loc.y - w["start"].y)
                if w["stop_at_m"] is not None and walked >= w["stop_at_m"]:
                    w["stop_at_m"] = None  # only once
                    w["state"], w["stand_until"] = "standing", sim_time + w["stand_s"]
                    w["actor"].apply_control(carla.WalkerControl(direction=w["direction"], speed=0.0))
                    messages.append(f"pedestrian {i + 1} stops in the lane for {w['stand_s']:.0f} s")
                elif walked >= w["distance_m"]:
                    w["state"] = "done"
                    w["actor"].apply_control(carla.WalkerControl(direction=w["direction"], speed=0.0))
                    messages.append(f"pedestrian {i + 1} reached the other side")
                else:
                    w["actor"].apply_control(carla.WalkerControl(direction=w["direction"], speed=w["speed"]))
        return closest, messages

    def destroy(self) -> None:
        for w in self.walkers:
            try:
                w["actor"].destroy()
            except RuntimeError:
                pass  # already gone
        self.walkers = []
