"""Other cars on the road, driven by CARLA's Traffic Manager (autopilot).

Cars are placed at the spawn points nearest the ego vehicle (but not on top of it), so
they share its roads and show up in the camera. In synchronous mode the Traffic Manager
must be synchronous too, and must be switched back afterwards or CARLA waits on it.
"""

import random
import socket

import carla


def free_port() -> int:
    """A free TCP port for this run's Traffic Manager.

    CARLA remembers a Traffic Manager by port. If the host process that created it has exited
    (e.g. restarted), a new client on the same port connects to that dead one and every tick stalls
    (seen as a 20 s timeout mid-run). A fresh port per run avoids that. Never 8000 (the dashboard).
    """
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TrafficManager:
    def __init__(self, client: carla.Client, world: carla.World):
        self.client, self.world = client, world
        self.map = world.get_map()  # get_map() rebuilds the whole road map: fetch it once (calling it every tick crashed the client)
        self.tm = None
        self.port = None
        self.vehicles = []

    def spawn(self, ego: carla.Vehicle, count: int, seed: int = 0, min_gap_m: float = 15.0,
              slower_percent: float = 20.0) -> list[str]:
        """Spawn `count` autopilot cars near the ego. Returns messages for the dashboard event log."""
        self._ensure_tm(seed)

        rng = random.Random(seed)
        blueprints = [bp for bp in self.world.get_blueprint_library().filter("vehicle.*")
                      if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels")) == 4]
        ego_loc = ego.get_location()
        spawn_points = sorted((sp for sp in self.map.get_spawn_points()
                               if sp.location.distance(ego_loc) >= min_gap_m),
                              key=lambda sp: sp.location.distance(ego_loc))
        for sp in spawn_points:
            if len(self.vehicles) >= count:
                break
            bp = rng.choice(blueprints)
            if bp.has_attribute("color"):
                bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
            actor = self.world.try_spawn_actor(bp, sp)
            if actor is None:
                continue
            actor.set_autopilot(True, self.port)
            self.tm.vehicle_percentage_speed_difference(actor, slower_percent)  # a bit under the speed limit
            self.vehicles.append(actor)
        msg = f"{len(self.vehicles)} traffic vehicles spawned"
        return [msg if len(self.vehicles) == count else f"{msg} (asked for {count}: not enough free spawn points)"]

    def _ensure_tm(self, seed: int = 0) -> None:
        if self.tm is None:
            self.port = free_port()
            self.tm = self.client.get_trafficmanager(self.port)
            self.tm.set_synchronous_mode(True)
            self.tm.set_random_device_seed(seed)  # same traffic behaviour every run
            self.tm.set_global_distance_to_leading_vehicle(2.5)

    def spawn_red_light_queue(self, ego: carla.Vehicle, count: int = 3, red_s: float = 30.0) -> list[str]:
        """Cars queued at the next traffic light ahead of the ego, that light held red for `red_s` seconds.
        Tests that the ego waits in a traffic queue instead of trying to swerve round or reverse."""
        ego_wp = self.map.get_waypoint(ego.get_location())
        lights = self.world.get_traffic_lights_from_waypoint(ego_wp, 250.0)
        if not lights:
            return ["red-light queue: no traffic light within 250 m ahead, not placed"]
        light = lights[0]
        stops = light.get_stop_waypoints()
        stop = next((w for w in stops if (w.road_id, w.lane_id) == (ego_wp.road_id, ego_wp.lane_id)), None)
        stop = stop or min(stops, key=lambda w: w.transform.location.distance(ego.get_location()))
        light.set_state(carla.TrafficLightState.Red)
        light.set_red_time(red_s)
        self._ensure_tm()
        placed = 0
        for k in range(count):
            prev = stop.previous(3.0 + 7.0 * k)
            if not prev:
                break
            tf = carla.Transform(prev[0].transform.location + carla.Location(z=0.3), prev[0].transform.rotation)
            bp = self.world.get_blueprint_library().find("vehicle.audi.a2")
            actor = self.world.try_spawn_actor(bp, tf)
            if actor is not None:
                actor.set_autopilot(True, self.port)  # they obey the light: go when it turns green
                self.vehicles.append(actor)
                placed += 1
        dist = stop.transform.location.distance(ego.get_location())
        return [f"red-light queue: {placed} cars waiting at a red light {dist:.0f} m ahead (red for {red_s:.0f} s)"]

    def spawn_parked(self, ego: carla.Vehicle, ahead_m: float = 60.0, all_lanes: bool = False) -> list[str]:
        """A stopped car (handbrake on, no autopilot) in the ego's lane `ahead_m` ahead: a blockage to get round.
        all_lanes: also park cars beside it in the neighbouring lanes, so there is no gap (tests reversing)."""
        nxt = self.map.get_waypoint(ego.get_location()).next(ahead_m)
        if not nxt:
            return [f"blocked-lane car: road ends before {ahead_m:.0f} m, not placed"]
        lanes = [nxt[0]]
        if all_lanes:
            for side in (nxt[0].get_left_lane(), nxt[0].get_right_lane()):
                if side is not None and side.lane_type in (carla.LaneType.Driving, carla.LaneType.Shoulder,
                                                           carla.LaneType.Parking):
                    lanes.append(side)
        placed = 0
        for wp in lanes:
            tf = carla.Transform(wp.transform.location + carla.Location(z=0.3), wp.transform.rotation)
            actor = self.world.try_spawn_actor(self.world.get_blueprint_library().find("vehicle.audi.a2"), tf)
            if actor is not None:
                actor.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
                self.vehicles.append(actor)
                placed += 1
        if not placed:
            return ["blocked-lane car: no free spot, not placed"]
        return [f"{placed} parked car(s) {ahead_m:.0f} m ahead ({'ego lane and the lanes beside it' if all_lanes else 'ego lane'})"]

    def nearest_ahead(self, ego: carla.Vehicle) -> float | None:
        """Ground truth: distance to the nearest traffic car ahead in the ego's lane (for the log).
        Uses CARLA's lane ids rather than a straight line, so it also works around curves."""
        carla_map = self.map
        tf = ego.get_transform()
        fwd = tf.get_forward_vector()
        ego_wp = carla_map.get_waypoint(tf.location)
        best = None
        for v in self.vehicles:
            loc = v.get_location()
            d = loc - tf.location
            if d.x * fwd.x + d.y * fwd.y <= 0:
                continue  # behind us
            wp = carla_map.get_waypoint(loc)
            if (wp.road_id, wp.lane_id) == (ego_wp.road_id, ego_wp.lane_id):
                dist = loc.distance(tf.location)
                best = dist if best is None else min(best, dist)
        return best

    def destroy(self) -> None:
        if self.vehicles:
            self.client.apply_batch([carla.command.DestroyActor(v) for v in self.vehicles])
            self.vehicles = []
        if self.tm is not None:
            self.tm.set_synchronous_mode(False)
            # Stop this Traffic Manager for good. Left running inside the host process, it gets torn down
            # underneath the CARLA client when a different town is loaded, which segfaulted the host.
            self.tm.shut_down()
            self.tm = None
