"""Run the Jetson's perception + planning on saved CARLA frames and compare the path with ground truth.

Uses the same code as main.py. With the .onnx files it runs on a PC (ONNX Runtime); on the
Jetson it uses the .engine files. For every frame it reports which path source was used and
how far the look-ahead target is from CARLA's true lane centre, and saves an overlay image:
  dots   = lane boundaries (green ego-left, red ego-right, blue/yellow outer)
  cyan   = perception path, magenta circle = look-ahead target
  white  = CARLA's true lane centre
  tinted = road pixels (segmentation)

Usage (from the jetson/ folder):
    python tools/offline_check.py --models-dir ../models --frames ../models/test_images/carla [--no-roi]
"""

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
from perception import perception
from perception.camera_geometry import Camera
from planning.path_generator import point_at
from planning.path_planner import PathPlanner

COLOURS = {"ego_left": (0, 255, 0), "ego_right": (0, 0, 255), "outer_left": (255, 0, 0), "outer_right": (0, 255, 255)}


def draw(img, seen, plan, cam, gt):
    out = img.copy()
    if seen["road_mask"] is not None:
        out[seen["road_mask"]] = (0.6 * out[seen["road_mask"]] + [0, 60, 0]).astype(np.uint8)
    for name, pts in seen["lanes"].items():
        for u, v in pts:
            cv2.circle(out, (int(u), int(v)), 3, COLOURS[name], -1)
    for points, colour in ((gt, (255, 255, 255)), (plan["path"]["centreline"], (255, 255, 0))):
        px = [cam.ground_to_pixel(a, r) for a, r in points if a > 3]
        px = [p for p in px if p and 0 <= p[1] < cam.height]
        for a, b in pairwise(px):
            cv2.line(out, a, b, colour, 2)
    if plan["ahead_m"] is not None:
        p = cam.ground_to_pixel(plan["ahead_m"], plan["right_m"])
        if p:
            cv2.circle(out, p, 7, (255, 0, 255), 2)
    for det in seen["detections"]:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 1)
    return out


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="~/models")
    ap.add_argument("--frames", required=True, help="folder with frames (+ ground_truth.json from CARLA)")
    ap.add_argument("--out", default=None, help="where to save overlays (default: <frames>/check)")
    ap.add_argument("--no-roi", action="store_true", help="feed the whole frame to the lane model (no CULane crop)")
    ap.add_argument("--speed", type=float, default=8.0)
    args = ap.parse_args()

    main.load_models(argparse.Namespace(models_dir=args.models_dir, no_detector=False, no_lanes=False,
                                        no_seg=False, no_depth=False))
    if args.no_roi:
        perception.lanes.use_roi = False
    frames_dir = Path(args.frames)
    out_dir = Path(args.out) if args.out else frames_dir / ("check_noroi" if args.no_roi else "check")
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = json.loads((frames_dir / "ground_truth.json").read_text()) if (frames_dir / "ground_truth.json").exists() else {}

    errors, sources = [], {}
    for path in sorted(p for p in frames_dir.iterdir() if p.suffix in (".png", ".jpg")):
        img = cv2.imread(str(path))
        if img.shape[:2] != (360, 640):
            img = cv2.resize(img, (640, 360))
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        header = {"width": 640, "height": 360, "camera": {}, "state": {}}
        perception.reset()  # each frame stands alone: run every model on it
        seen = perception.perceive(header, bgra.tobytes())
        # Each saved frame stands alone, but the planner's tracker only trusts a path once two frames
        # agree, so show it the same frame twice (0.1 s apart, car not moving).
        planner, st = PathPlanner(), {"speed_mps": 0.0, "target_speed_mps": args.speed, "path_source": "perception"}
        planner.plan(st, seen, now=0.0)
        plan = planner.plan(st, seen, now=0.1)
        cam = Camera(640, 360)
        gt = truth.get(path.stem, {}).get("gt_centreline", [])
        source = plan["path"]["source"]
        sources[source] = sources.get(source, 0) + 1
        line = f"{path.stem:<18} source {source:<5} lanes {sorted(seen['lanes'])!s:<52}"
        if plan["ahead_m"] is not None:
            line += f" target ahead {plan['ahead_m']:5.1f} m right {plan['right_m']:+5.2f} m"
            g = point_at(gt, plan["ahead_m"]) if gt else None
            if g:
                err = plan["right_m"] - g[1]
                errors.append(abs(err))
                line += f" | true centre {g[1]:+5.2f} m -> error {err:+5.2f} m"
        print(line + f"  ({seen['inference_ms']} ms)")
        cv2.imwrite(str(out_dir / f"{path.stem}.jpg"), draw(img, seen, plan, cam, gt))

    print(f"\npath sources: {sources}")
    if errors:
        print(f"lateral error vs CARLA lane centre: mean {np.mean(errors):.2f} m, max {np.max(errors):.2f} m "
              f"over {len(errors)} frames with a path")
    print(f"overlays saved in {out_dir}")


if __name__ == "__main__":
    main_cli()
