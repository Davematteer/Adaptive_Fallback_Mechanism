"""Live status stream: pushes the experiment status to the browser a few times a second."""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws")
async def status_stream(websocket: WebSocket):
    await websocket.accept()
    manager = websocket.app.state.manager
    try:
        while True:
            await websocket.send_json(manager.status())
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        pass  # browser tab closed
