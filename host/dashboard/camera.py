"""Live camera view: streams the camera frames with the Jetson's YOLO boxes to the dashboard.

The host already has every frame (it sends them to the Jetson), so only the box
coordinates come back from the Jetson. Per update the browser receives a JSON text
message (boxes, decision, timings) followed by the frame as a JPEG binary message.
Nothing is encoded while no browser is watching.
"""

import asyncio
import io

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image

router = APIRouter()

FPS = 8


def encode_jpeg(image, quality: int = 70) -> bytes:
    """carla.Image (BGRA) -> JPEG bytes."""
    rgb = Image.frombuffer("RGBA", (image.width, image.height), bytes(image.raw_data), "raw", "BGRA", 0, 1)
    out = io.BytesIO()
    rgb.convert("RGB").save(out, format="JPEG", quality=quality)
    return out.getvalue()


@router.websocket("/ws/camera")
async def camera_stream(websocket: WebSocket):
    await websocket.accept()
    manager = websocket.app.state.manager
    last_sent, maps_sent = None, {}
    try:
        while True:
            view = manager.view()
            if view is not None:
                frame_id, image, command = view
                key = (frame_id, command.get("frame_id"))
                if key != last_sent:
                    jpeg = await asyncio.to_thread(encode_jpeg, image)  # keep the event loop free
                    # Segmentation / depth maps only arrive every few frames: send each new one once.
                    maps = {}
                    for name, m in manager.view_maps().items():
                        if maps_sent.get(name) != m["frame_id"]:
                            maps[name], maps_sent[name] = m, m["frame_id"]
                    await websocket.send_json({
                        "frame_id": frame_id, "width": image.width, "height": image.height,
                        "detections": command.get("detections") or [], "obstacle": command.get("obstacle"),
                        "decision": command.get("decision"), "reason": command.get("reason"),
                        "maneuver": command.get("maneuver"), "lidar_clusters": command.get("lidar_clusters") or [],
                        "path": command.get("path"), "lanes": command.get("lanes") or {}, "maps": maps,
                        "timings_ms": command.get("timings_ms") or {}, "inference_ms": command.get("inference_ms"),
                        "pipeline_ms": command.get("pipeline_ms"),
                    })
                    await websocket.send_bytes(jpeg)
                    last_sent = key
            await asyncio.sleep(1 / FPS)
    except WebSocketDisconnect:
        pass  # browser tab closed
