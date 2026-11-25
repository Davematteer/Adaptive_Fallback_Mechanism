"""Jetson entry point: perception -> planning -> control for every frame from the host.

Run on the Jetson from inside the jetson/ folder (with the environment that has ultralytics + tensorrt):
    python3 main.py                        # all four models from ~/models
    python3 main.py --no-depth --no-seg    # switch models off (e.g. to compare timings)
    python3 main.py --models-dir ../models # e.g. on a PC: falls back to the .onnx files
The host then connects to port 5555.

Expected layout of --models-dir (the .engine is used if present, else the .onnx):
    object_detection/yolo_fp32.engine        lane_detection/culane_res18_fp32.engine
    semantic_segmentation/yolo26n-sem_fp32.engine   depth_estimation/yolo26n-depth_fp32.engine
"""

import argparse
import time
from pathlib import Path

from communication.communication_server import CommunicationServer
from control.controller import compute_command
from monitoring.pipeline_health import PipelineHealth
from perception import perception
from perception.camera_geometry import Camera
from planning.path_planner import PathPlanner

health = PipelineHealth()
planner = PathPlanner()

MODELS = {  # name -> (sub-folder, engine file, onnx file)
    "detector": ("object_detection", "yolo_fp32.engine", "yolo26n.onnx"),
    "lanes": ("lane_detection", "culane_res18_fp32.engine", "culane_res18.onnx"),
    "segmenter": ("semantic_segmentation", "yolo26n-sem_fp32.engine", "yolo26n-sem.onnx"),
    "depth": ("depth_estimation", "yolo26n-depth_fp32.engine", "yolo26n-depth.onnx"),
}


def model_path(models_dir: Path, name: str) -> Path:
    folder, engine, onnx = MODELS[name]
    engine_path = models_dir / folder / engine
    return engine_path if engine_path.exists() else models_dir / folder / onnx


def run_pipeline(header: dict, payload: bytes) -> dict:
    """One full driving step. The heartbeat advances only if every stage finishes."""
    health.apply_pending_fault()  # simulated failures (hang / crash) trigger here
    started = time.perf_counter()

    state = header["state"]
    seen = perception.perceive(header, payload)
    target = planner.plan(state, seen)
    command = compute_command(state, target)

    command.update({
        "decision": target["decision"], "reason": target["reason"], "obstacle": target["obstacle"],
        "maneuver": target["maneuver"], "reverse": bool(command.get("reverse", False)),
        "blockage": target["blockage"],  # traffic or a real blockage, with the evidence (obstacle_tracker.py)
        "path": dict(target["path"], **_path_pixels(seen, target)),
        "detections": seen["detections"], "n_detections": len(seen["detections"]),
        "lanes": {name: [list(p) for p in pts] for name, pts in seen["lanes"].items()},
        "view": seen["view"],  # downscaled segmentation / depth maps when computed this frame, LiDAR points
        "lidar_clusters": (seen.get("lidar") or {}).get("clusters"),
        "lidar_rear_m": (seen.get("lidar") or {}).get("rear_clearance_m"),
        "timings_ms": seen["timings_ms"], "inference_ms": seen.get("inference_ms"),
    })
    command["heartbeat"] = health.beat()
    command["pipeline_ms"] = round((time.perf_counter() - started) * 1000, 2)  # per-step timing for latency analysis
    return command


def _path_pixels(seen: dict, target: dict) -> dict:
    """Where the look-ahead target and the path centreline are in the image, for the dashboard overlay."""
    image = seen.get("image")
    if image is None:
        return {"target_px": None, "centreline_px": []}
    cam = Camera(image.shape[1], image.shape[0], seen.get("camera"))
    px = cam.ground_to_pixel(target["ahead_m"], target["right_m"]) if target.get("ahead_m") is not None else None
    line = [cam.ground_to_pixel(a, r) for a, r in target["path"]["centreline"] if a > 3.0]
    line = [list(q) for q in line if q and 0 <= q[1] < cam.height]
    return {"target_px": list(px) if px else None, "centreline_px": line}


def handle_fault(fault: str) -> None:
    if fault == "comm_failure":
        server.muted = True  # keep computing, stop replying
    else:
        health.inject(fault)


def on_disconnect() -> None:
    health.clear()
    planner.reset()
    perception.reset()


def load_models(args) -> None:
    models_dir = Path(args.models_dir).expanduser()
    wanted = {"detector": not args.no_detector, "lanes": not args.no_lanes,
              "segmenter": not args.no_seg, "depth": not args.no_depth}
    for name, enabled in wanted.items():
        if not enabled:
            print(f"{name}: off", flush=True)
            continue
        path = model_path(models_dir, name)
        started = time.perf_counter()
        if name == "detector":
            from perception.object_detector import ObjectDetector
            model = ObjectDetector(path)
        elif name == "lanes":
            from perception.lane_detector import LaneDetector
            model = LaneDetector(path)
        elif name == "segmenter":
            from perception.road_segmenter import RoadSegmenter
            model = RoadSegmenter(path)
        else:
            from perception.depth_estimator import DepthEstimator
            model = DepthEstimator(path)
        setattr(perception, name, model)
        print(f"{name}: {path.name} ready ({time.perf_counter() - started:.1f} s incl. warm-up)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jetson driving service")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--models-dir", default="~/models", help="folder with the four model sub-folders")
    parser.add_argument("--no-detector", action="store_true", help="run without object detection")
    parser.add_argument("--no-lanes", action="store_true", help="run without lane detection")
    parser.add_argument("--no-seg", action="store_true", help="run without road segmentation")
    parser.add_argument("--no-depth", action="store_true", help="run without depth estimation")
    parser.add_argument("--no-lidar", action="store_true", help="ignore the LiDAR sweep from the host")
    args = parser.parse_args()

    load_models(args)
    perception.use_lidar = not args.no_lidar
    server = CommunicationServer(on_frame=run_pipeline, on_fault=handle_fault, on_disconnect=on_disconnect, port=args.port)
    server.serve_forever()
