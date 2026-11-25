"""Lane detection with Ultra-Fast-Lane-Detection-v2 (CULane, ResNet18).

Pre-processing and decoding match the repo's demo/test pipeline (what the model was
trained with); verified on the PC: ONNX and PyTorch give identical lane points
(see models/lane_detection/ufld.py, which this mirrors).

CULane frames are 1640x590 (2.78:1) with the horizon about 40 % down. The CARLA
camera is 16:9 with the horizon in the middle, so a 2.78:1 band of the image is cut
out first, placed so the horizon sits at the same relative height as in CULane.
"""

import cv2
import numpy as np

from perception.trt_runner import load_runner

# configs/culane_res18.py
TRAIN_W, TRAIN_H, CROP_RATIO = 1600, 320, 0.6
NUM_ROW, NUM_COL = 72, 81
ROW_ANCHOR = np.linspace(0.42, 1, NUM_ROW)
COL_ANCHOR = np.linspace(0, 1, NUM_COL)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CULANE_ASPECT = 1640 / 590
CULANE_HORIZON = 0.40  # horizon height as a fraction of the CULane frame
LANE_NAMES = {0: "outer_left", 1: "ego_left", 2: "ego_right", 3: "outer_right"}

# An ego-lane boundary is kept if the model marks it present on more than this fraction of the 72
# row anchors. The repo uses 0.5 (tuned on real CULane footage); on CARLA frames the model is less
# sure of clearly visible markings and dropped them in ~15 % of frames in a closed-loop test. 0.25
# recovered most of those with unchanged accuracy against CARLA's true lane centre (mean 3.6 cm).
MIN_ROW_FRACTION = 0.25


class LaneDetector:
    def __init__(self, model_path, use_roi: bool = True):
        self.runner = load_runner(model_path)
        self.use_roi = use_roi
        self.detect(np.zeros((360, 640, 3), dtype=np.uint8))  # warm-up

    def roi(self, height: int, width: int, horizon_y: float) -> tuple[int, int]:
        """Rows (top, bottom) of the CULane-shaped band."""
        if not self.use_roi:
            return 0, height
        band = min(height, round(width / CULANE_ASPECT))
        top = round(horizon_y - CULANE_HORIZON * band)
        top = max(0, min(height - band, top))
        return top, top + band

    def detect(self, bgr: np.ndarray, horizon_y: float | None = None) -> dict[str, list[tuple[int, int]]]:
        """BGR frame -> {"ego_left": [(u, v), ...], ...} in full-frame pixels, ordered far -> near.
        Lanes the model isn't confident about are left out."""
        h, w = bgr.shape[:2]
        top, bottom = self.roi(h, w, h / 2 if horizon_y is None else horizon_y)
        band = bgr[top:bottom]
        pred = self.runner.run(preprocess(band))
        lanes = decode(pred, band.shape[1], band.shape[0])
        return {name: [(u, v + top) for u, v in pts] for name, pts in lanes.items()}


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """Resize to (320 / 0.6, 1600), RGB, ImageNet-normalise, keep the bottom 320 rows -> (1, 3, 320, 1600)."""
    resized = cv2.resize(bgr, (TRAIN_W, int(TRAIN_H / CROP_RATIO)), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = ((rgb - MEAN) / STD)[-TRAIN_H:]
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def decode(pred: dict, width: int, height: int, local_width: int = 1,
           min_row_fraction: float = MIN_ROW_FRACTION) -> dict[str, list[tuple[int, int]]]:
    """NumPy port of the repo's pred2coords, keeping track of which lane is which
    (and with the ego-lane presence threshold as a parameter, see MIN_ROW_FRACTION)."""
    loc_row, loc_col = pred["loc_row"], pred["loc_col"]
    num_grid_row, num_cls_row = loc_row.shape[1:3]
    num_grid_col, num_cls_col = loc_col.shape[1:3]
    max_row, valid_row = loc_row.argmax(1), pred["exist_row"].argmax(1)
    max_col, valid_col = loc_col.argmax(1), pred["exist_col"].argmax(1)

    lanes = {}
    for i in (1, 2):  # ego-lane boundaries: one point per row anchor
        if valid_row[0, :, i].sum() > num_cls_row * min_row_fraction:
            pts = []
            for k in range(num_cls_row):
                if valid_row[0, k, i]:
                    c = max_row[0, k, i]
                    idx = np.arange(max(0, c - local_width), min(num_grid_row - 1, c + local_width) + 1)
                    x = (_softmax(loc_row[0, idx, k, i]) * idx).sum() + 0.5
                    pts.append((int(x / (num_grid_row - 1) * width), int(ROW_ANCHOR[k] * height)))
            lanes[LANE_NAMES[i]] = pts
    for i in (0, 3):  # outer boundaries: one point per column anchor
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            pts = []
            for k in range(num_cls_col):
                if valid_col[0, k, i]:
                    c = max_col[0, k, i]
                    idx = np.arange(max(0, c - local_width), min(num_grid_col - 1, c + local_width) + 1)
                    y = (_softmax(loc_col[0, idx, k, i]) * idx).sum() + 0.5
                    pts.append((int(COL_ANCHOR[k] * width), int(y / (num_grid_col - 1) * height)))
            lanes[LANE_NAMES[i]] = sorted(pts, key=lambda p: p[1])
    return lanes
