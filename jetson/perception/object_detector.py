"""YOLO object detection on the Jetson (TensorRT engine via Ultralytics).

The engine in ~/models was built with trtexec, so it carries no Ultralytics
metadata: class names come back as "class0", "class1", ... We therefore use the
COCO class IDs the model was trained on and name them ourselves.
"""

import time
from pathlib import Path

import numpy as np

# The COCO classes we care about for driving (ID -> name). Other IDs are still returned, as "class<N>".
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
              9: "traffic light", 11: "stop sign"}

DEFAULT_MODEL = Path.home() / "models" / "object_detection" / "yolo_fp32.engine"


class ObjectDetector:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL, imgsz: int = 640, conf: float = 0.4):
        from ultralytics import YOLO  # imported here so the rest of the app runs without ultralytics

        model_path = Path(model_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path} (pass --model, or --no-detector to run without)")
        self.imgsz, self.conf = imgsz, conf
        started = time.perf_counter()
        self.model = YOLO(str(model_path), task="detect")
        import torch
        self.device = 0 if torch.cuda.is_available() else "cpu"  # GPU on the Jetson; CPU when testing on a PC
        # The first TensorRT call is slow (allocations, CUDA context). Do it now, not on the first
        # frame, or that frame would take long enough to trip the host's 0.5 s watchdog.
        self.detect(np.zeros((360, 640, 3), dtype=np.uint8))
        print(f"detector ready: {model_path.name} ({(time.perf_counter() - started):.1f} s incl. warm-up)", flush=True)

    def detect(self, bgr: np.ndarray) -> list[dict]:
        """BGR image -> [{"class_id", "class", "confidence", "bbox": [x1, y1, x2, y2]}] in image pixels."""
        result = self.model.predict(np.ascontiguousarray(bgr), imgsz=self.imgsz, conf=self.conf,
                                    device=self.device, verbose=False)[0]
        detections = []
        for (x1, y1, x2, y2), cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                               result.boxes.conf.tolist()):
            cls = int(cls)
            detections.append({"class_id": cls, "class": COCO_NAMES.get(cls, f"class{cls}"),
                               "confidence": round(conf, 3),
                               "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})
        return detections
