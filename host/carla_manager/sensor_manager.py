"""Attaching sensors to the vehicle and keeping the latest reading of each."""

import time

import carla


class SensorManager:
    def __init__(self, world: carla.World):
        self.world = world
        self.camera = None
        self.latest_image = None  # most recent carla.Image; older frames are simply replaced
        self.collision_sensor = None
        self.first_collision = None  # type of the first actor hit, if any
        self.camera_info = {}  # fov and mount position, sent to the Jetson for distance estimates
        self.rear_sensor = None
        self.lidar = None
        self.latest_lidar = None  # float32 (N, 3) x forward, y right, z up (sensor frame), newest sweep
        self.lidar_mount = {"x": 0.0, "z": 2.0}
        self._rear_hit = None  # (distance_m, time.monotonic()) of the latest thing seen behind

    def attach_camera(self, vehicle: carla.Vehicle, width: int = 640, height: int = 360, fov: int = 90) -> None:
        """Front-facing RGB camera. Smaller images = less data to send and faster inference."""
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        mount = carla.Transform(carla.Location(x=1.5, z=1.6))  # roughly at the windscreen
        self.camera_info = {"fov_deg": float(fov), "height_m": 1.6, "x_m": 1.5}
        self.camera = self.world.spawn_actor(bp, mount, attach_to=vehicle)
        # CARLA calls this from its own thread every time a frame is rendered.
        self.camera.listen(self._on_image)

    def _on_image(self, image: carla.Image) -> None:
        self.latest_image = image

    def attach_collision_sensor(self, vehicle: carla.Vehicle) -> None:
        """Records the first thing the vehicle hits (for the 'collision status' metric)."""
        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        self.collision_sensor.listen(self._on_collision)

    def _on_collision(self, event: carla.CollisionEvent) -> None:
        if self.first_collision is None:
            self.first_collision = event.other_actor.type_id  # e.g. "static.pole", "vehicle.audi.a2"

    def attach_rear_sensor(self, vehicle: carla.Vehicle, range_m: float = 8.0) -> None:
        """Rear parking sensor (CARLA's obstacle detector, facing backwards), so the Jetson can reverse safely."""
        bp = self.world.get_blueprint_library().find("sensor.other.obstacle")
        # Mounted at bumper height with a thin sweep: lower / wider and the sweep touches the road itself
        # (it then reports "static.road" at 0.0 m and reversing would never be allowed).
        for key, value in (("distance", str(range_m)), ("hit_radius", "0.4"), ("only_dynamics", "false"),
                           ("sensor_tick", "0.0")):
            bp.set_attribute(key, value)
        half_length = vehicle.bounding_box.extent.x
        mount = carla.Transform(carla.Location(x=-half_length - 0.1, z=0.9), carla.Rotation(yaw=180))
        self.rear_sensor = self.world.spawn_actor(bp, mount, attach_to=vehicle)
        parent_id = vehicle.id
        self.rear_sensor.listen(lambda ev: None if ev.other_actor.id == parent_id
                                or ev.other_actor.type_id.startswith("static.road")
                                else setattr(self, "_rear_hit", (ev.distance, time.monotonic())))

    def rear_clearance_m(self) -> float | None:
        """Distance to the nearest thing behind (m), or None if nothing has been seen in the last 0.3 s."""
        hit = self._rear_hit
        return round(hit[0], 2) if hit and time.monotonic() - hit[1] < 0.3 else None

    def attach_lidar(self, vehicle: carla.Vehicle, step_s: float, channels: int = 32, range_m: float = 40.0,
                     points_per_second: int = 120000) -> None:
        """Roof LiDAR doing one full 360-degree sweep per simulation step (so every frame has a complete scan)."""
        import numpy as np
        bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
        for key, value in (("channels", channels), ("range", range_m), ("points_per_second", points_per_second),
                           ("rotation_frequency", 1.0 / step_s), ("upper_fov", 10.0), ("lower_fov", -25.0),
                           ("dropoff_general_rate", 0.0)):
            bp.set_attribute(key, str(value))
        mount = carla.Transform(carla.Location(x=self.lidar_mount["x"], z=self.lidar_mount["z"]))
        self.lidar = self.world.spawn_actor(bp, mount, attach_to=vehicle)
        self.lidar.listen(lambda m: setattr(self, "latest_lidar", np.frombuffer(
            bytes(m.raw_data), dtype=np.float32).reshape(-1, 4)[:, :3].copy()))

    def destroy(self) -> None:
        for sensor in (self.camera, self.collision_sensor, self.rear_sensor, self.lidar):
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        self.camera = self.collision_sensor = self.rear_sensor = self.lidar = None
        self.latest_image = None
