"""Metric depth with YOLO26n-depth: distance in metres for every pixel.

Pre-processing matches Ultralytics' depth predictor (plain resize to the engine's
square size, RGB, 0..1); the depth map is resized back to the frame's size.
"""

import cv2
import numpy as np

from perception.trt_runner import load_runner


class DepthEstimator:
    def __init__(self, model_path):
        self.runner = load_runner(model_path)
        self.size = int(self.runner.input_shape[-1])  # 768 for the default export
        self.estimate(np.zeros((360, 640, 3), dtype=np.uint8))  # warm-up

    def estimate(self, bgr: np.ndarray) -> np.ndarray:
        """BGR frame -> (H, W) float32 depth in metres."""
        h, w = bgr.shape[:2]
        x = cv2.cvtColor(cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        x = x.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = next(iter(self.runner.run(x).values()))
        depth = np.asarray(out, dtype=np.float32).reshape(self.size, self.size)
        return cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
