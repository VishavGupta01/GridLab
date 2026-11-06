from __future__ import annotations

import asyncio
import json

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .sessions import Session


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        return super().default(obj)


async def ws_session_handler(websocket: WebSocket, session: Session) -> None:
    await websocket.accept()
    session.last_activity = __import__("time").time()

    try:
        status = {
            "type": "status",
            "state": session.state,
            "speed": session._speed,
            "episode": session.trainer.episode,
            "budget": session.budget,
        }
        await websocket.send_text(json.dumps(status, cls=NumpyEncoder))

        while True:
            try:
                msg = await asyncio.wait_for(
                    session.broadcast_queue.get(), timeout=0.5
                )
                text = json.dumps(msg, cls=NumpyEncoder)
                await websocket.send_text(text)
            except asyncio.TimeoutError:
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                except asyncio.TimeoutError:
                    pass

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
