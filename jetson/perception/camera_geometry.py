"""Converting between image pixels and positions on the road around the vehicle.

Pinhole camera, level (pitch 0), mounted `height_m` above a flat road and `x_m`
ahead of the vehicle's centre. Vehicle frame: ahead = forward (m), right = to the
right (m), the same convention as CARLA and the controller.

    f       = (width / 2) / tan(fov / 2)            focal length in pixels
    ahead   = f * height_m / (v - cy) + x_m          for a road pixel below the horizon (v > cy)
    right   = (u - cx) * (ahead - x_m) / f
"""

import math

DEFAULT_CAMERA = {"fov_deg": 90.0, "height_m": 1.6, "x_m": 1.5}  # matches host/carla_manager/sensor_manager.py


class Camera:
    def __init__(self, width: int, height: int, camera: dict | None = None):
        cam = {**DEFAULT_CAMERA, **(camera or {})}
        self.width, self.height = width, height
        self.f = (width / 2) / math.tan(math.radians(cam["fov_deg"]) / 2)
        self.cx, self.cy = width / 2, height / 2  # cy is also the horizon row (level camera)
        self.height_m, self.x_m = cam["height_m"], cam["x_m"]

    def pixel_to_ground(self, u: float, v: float) -> tuple[float, float] | None:
        """Road pixel -> (ahead_m, right_m), or None at/above the horizon."""
        below = v - self.cy
        if below <= 1:
            return None
        dist = self.f * self.height_m / below
        return dist + self.x_m, (u - self.cx) * dist / self.f

    def ground_to_pixel(self, ahead_m: float, right_m: float) -> tuple[int, int] | None:
        """(ahead_m, right_m) on the road -> pixel, or None if behind the camera."""
        dist = ahead_m - self.x_m
        if dist <= 0.1:
            return None
        return round(self.cx + right_m * self.f / dist), round(self.cy + self.f * self.height_m / dist)

    def row_for_distance(self, ahead_m: float) -> int:
        """Image row where the road is `ahead_m` ahead."""
        return round(self.cy + self.f * self.height_m / max(ahead_m - self.x_m, 0.1))

    def from_depth(self, u: float, depth_m: float) -> tuple[float, float]:
        """Pixel column + measured depth -> (ahead_m, right_m). Works for objects, not only road."""
        return depth_m + self.x_m, (u - self.cx) * depth_m / self.f
