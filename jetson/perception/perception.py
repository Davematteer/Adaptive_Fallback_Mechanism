"""Turns each camera frame into what the planner needs, using up to four models.

  YOLO objects      every frame
  lanes (UFLDv2)    every frame
  road segmentation every SEG_EVERY-th frame (the road shape changes slowly; the last mask is reused)
  depth             when YOLO sees something we may have to stop for, and every DEPTH_EVERY-th
                    frame otherwise (so the dashboard's depth view stays fresh)
  LiDAR             every frame when the host sends a sweep (geometry only, perception/lidar.py)

The models are set by main.py; any of them can be None (switched off).
"""

import base64
import time

import cv2
import numpy as np

detector = None   # ObjectDetector
lanes = None      # LaneDetector
segmenter = None  # RoadSegmenter
depth = None      # DepthEstimator
use_lidar = True  # process the LiDAR sweep if the host sends one

SEG_EVERY = 3
DEPTH_EVERY = 5
OBSTACLE_CLASSES = {0, 1, 2, 3, 5, 7}  # COCO: person, bicycle, car, motorcycle, bus, truck
VIEW_SIZE = (128, 72)  # downscaled maps sent to the dashboard

_frame = 0
_road_mask = _classes = None


def reset() -> None:
    """Forget cached maps (called between experiments)."""
    global _frame, _road_mask, _classes
    _frame, _road_mask, _classes = 0, None, None


def decode_image(header: dict, payload: bytes):
    """Raw BGRA bytes from CARLA -> (height, width, 3) BGR numpy array, or None if no image.
    The payload is [image][lidar]; header["image_size"] says where the image ends."""
    payload = payload[:header.get("image_size", len(payload))]
    if not payload or "width" not in header:
        return None
    img = np.frombuffer(payload, dtype=np.uint8).reshape(header["height"], header["width"], 4)
    return np.ascontiguousarray(img[:, :, :3])  # drop alpha; contiguous for OpenCV


def _timed(timings: dict, name: str, fn, *args):
    started = time.perf_counter()
    out = fn(*args)
    timings[name] = round((time.perf_counter() - started) * 1000, 1)
    return out


def _small(arr: np.ndarray, interpolation) -> str:
    """Downscaled uint8 map as base64, for the dashboard overlay."""
    return base64.b64encode(cv2.resize(arr, VIEW_SIZE, interpolation=interpolation).tobytes()).decode()


def perceive(header: dict, payload: bytes) -> dict:
    global _frame, _road_mask, _classes
    _frame += 1
    image = decode_image(header, payload)
    timings, view = {}, {}
    out = {"image": image, "camera": header.get("camera") or {}, "detections": [], "lanes": {},
           "road_mask": None, "depth": None, "lidar": None, "timings_ms": timings, "view": view}
    lidar_bytes = payload[header.get("image_size", len(payload)):]
    if use_lidar and lidar_bytes:
        from perception import lidar as lidar_mod
        out["lidar"] = _timed(timings, "lidar", lidar_mod.process, lidar_mod.decode(lidar_bytes), header.get("lidar_mount"))
        view["lidar"] = out["lidar"].pop("view")
    if image is None:
        return out

    if detector is not None:
        out["detections"] = _timed(timings, "yolo", detector.detect, image)
    if lanes is not None:
        out["lanes"] = _timed(timings, "lanes", lanes.detect, image)
    if segmenter is not None and (_road_mask is None or _frame % SEG_EVERY == 0):
        _classes = _timed(timings, "seg", segmenter.segment, image)
        _road_mask = _classes == 0
        view["seg"] = _small(_classes, cv2.INTER_NEAREST)
    out["road_mask"] = _road_mask
    need_depth = any(d["class_id"] in OBSTACLE_CLASSES for d in out["detections"])
    if depth is not None and (need_depth or _frame % DEPTH_EVERY == 0):
        out["depth"] = _timed(timings, "depth", depth.estimate, image)
        view["depth"] = _small(np.clip(out["depth"] * 4, 0, 255).astype(np.uint8), cv2.INTER_LINEAR)  # 0.25 m steps
    if view:
        view["size"] = list(VIEW_SIZE)
    out["inference_ms"] = round(sum(timings.values()), 1)
    return out
