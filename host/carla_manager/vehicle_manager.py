"""Spawning the ego vehicle, applying driving commands and reading its state."""

import math

import carla


class VehicleManager:
    def __init__(self, world: carla.World):
        self.world = world
        self.map = world.get_map()  # get_map() rebuilds the whole road map: fetch it once (calling it every tick crashed the client)
        self.vehicle = None

    def spawn(self, blueprint: str = "vehicle.tesla.model3", spawn_index: int = 0,
              lateral_offset_m: float = 0.0) -> carla.Vehicle:
        """Spawn at a map spawn point, optionally shifted sideways (+ = right) to test lane-keeping recovery."""
        bp = self.world.get_blueprint_library().find(blueprint)
        spawn_points = self.map.get_spawn_points()
        tf = spawn_points[spawn_index % len(spawn_points)]
        if lateral_offset_m:
            right = tf.get_right_vector()
            tf = carla.Transform(tf.location + carla.Location(x=right.x * lateral_offset_m, y=right.y * lateral_offset_m),
                                 tf.rotation)
        self.vehicle = self.world.spawn_actor(bp, tf)
        return self.vehicle

    def apply(self, command: dict) -> None:
        """Apply a {"steer", "throttle", "brake", "reverse"} command, clamped to CARLA's valid ranges."""
        self.vehicle.apply_control(carla.VehicleControl(
            steer=max(-1.0, min(1.0, float(command.get("steer", 0.0)))),
            throttle=max(0.0, min(1.0, float(command.get("throttle", 0.0)))),
            brake=max(0.0, min(1.0, float(command.get("brake", 0.0)))),
            reverse=bool(command.get("reverse", False)),
        ))

    def lane_offset(self) -> float:
        """Ground truth: signed distance (m) from the lane centre CARLA's map puts the car in (+ = right)."""
        loc = self.vehicle.get_location()
        wp = self.map.get_waypoint(loc, project_to_road=True)
        right = wp.transform.get_right_vector()
        d = loc - wp.transform.location
        return d.x * right.x + d.y * right.y

    def state(self, num_waypoints: int = 25, spacing_m: float = 2.0) -> dict:
        """Vehicle pose and speed, plus the next lane-centre waypoints ahead of it ("follow the road",
        first branch at junctions). With a destination, the scenario manager replaces them with the
        planned route (route_planner.py).
        """
        tf = self.vehicle.get_transform()
        v = self.vehicle.get_velocity()
        fwd = tf.get_forward_vector()

        waypoints, junction_ahead = [], None
        wp = self.map.get_waypoint(tf.location)
        for _ in range(num_waypoints):
            nxt = wp.next(spacing_m)
            if not nxt:
                break
            wp = nxt[0]  # at junctions, simply take the first branch
            waypoints.append([round(wp.transform.location.x, 2), round(wp.transform.location.y, 2)])
            if junction_ahead is None and wp.is_junction:
                junction_ahead = len(waypoints) * spacing_m

        return {
            "x": tf.location.x,
            "y": tf.location.y,
            "yaw_deg": tf.rotation.yaw,
            "speed_mps": math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2),
            "forward_speed_mps": v.x * fwd.x + v.y * fwd.y + v.z * fwd.z,  # signed: negative when reversing
            "waypoints": waypoints,  # "follow the road" (first branch at junctions); replaced by a planned route
            "junction_ahead_m": junction_ahead,
        }

    def follow_with_spectator(self, distance_m: float = 8.0, height_m: float = 3.5) -> None:
        """Move the CARLA window's camera (the "spectator") to a chase view behind the car.

        Viewing only: this does not affect the camera sensor sent to the Jetson.
        """
        tf = self.vehicle.get_transform()
        back = tf.get_forward_vector() * -distance_m  # point behind the car, along its heading
        location = tf.location + carla.Location(x=back.x, y=back.y, z=height_m)
        rotation = carla.Rotation(pitch=-15.0, yaw=tf.rotation.yaw)  # look slightly down at the car
        self.world.get_spectator().set_transform(carla.Transform(location, rotation))

    def destroy(self) -> None:
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None
