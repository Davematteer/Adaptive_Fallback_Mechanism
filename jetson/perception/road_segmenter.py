"""Road segmentation with YOLO26n-sem (Cityscapes classes): which pixels are drivable road.

Pre-processing matches Ultralytics (letterbox to the engine's square size, grey padding,
RGB, 0..1); the exported model already outputs one class id per pixel.
"""

import cv2
import numpy as np

from perception.trt_runner import load_runner

ROAD = 0  # Cityscapes class ids: 0 road, 1 sidewalk, ... 11 person, 13 car (see models/semantic_segmentation)


class RoadSegmenter:
    def __init__(self, model_path):
        self.runner = load_runner(model_path)
        self.size = int(self.runner.input_shape[-1])  # 1024 for the default export
        self.segment(np.zeros((360, 640, 3), dtype=np.uint8))  # warm-up

    def segment(self, bgr: np.ndarray) -> np.ndarray:
        """BGR frame -> (H, W) uint8 class-id map at the frame's size."""
        h, w = bgr.shape[:2]
        r = min(self.size / h, self.size / w)
        nw, nh = round(w * r), round(h * r)
        left, top = round((self.size - nw) / 2 - 0.1), round((self.size - nh) / 2 - 0.1)
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = next(iter(self.runner.run(x).values()))
        classes = np.asarray(out).reshape(self.size, self.size)[top:top + nh, left:left + nw].astype(np.uint8)
        return cv2.resize(classes, (w, h), interpolation=cv2.INTER_NEAREST)
