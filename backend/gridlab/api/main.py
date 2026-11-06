from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ..core.env import GridWorldEnv
from ..core.grid import is_reachable, list_presets, load_preset, parse_grid, validate_grid
from ..core.solver import bfs_shortest_path, solve_grid, value_iteration
from .schemas import (
    AgentConfig,
    ControlCommand,
    EditCommand,
    ExplainRequest,
    GridJSON,
    GridValidation,
    RolloutRequest,
    SessionCreate,
)
from .sessions import SessionManager
from .ws import ws_session_handler

def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return obj


app = FastAPI(title="GridLab", version="0.1.0")
session_manager = SessionManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/presets")
async def get_presets():
    return {"presets": list_presets()}


@app.get("/api/presets/{name}")
async def get_preset(name: str):
    try:
        grid = load_preset(name)
        return grid.to_json()
    except FileNotFoundError:
        raise HTTPException(404, f"Preset '{name}' not found")


@app.post("/api/grids/validate")
async def validate(grid_json: GridJSON):
    try:
        grid = parse_grid(grid_json.model_dump())
    except Exception as e:
        return GridValidation(valid=False, issues=[str(e)])

    issues = validate_grid(grid)
    if issues:
        return GridValidation(valid=False, issues=issues)

    reachable, visited = is_reachable(grid)
    if not reachable:
        return GridValidation(valid=True, solvable=False, issues=["Goal not reachable from start"])

    env = GridWorldEnv(grid)
    model = env.model()
    sol = solve_grid(model, grid.version)

    return GridValidation(
        valid=True,
        solvable=True,
        optimal_path_length=sol["bfs_shortest"],
        v_star_s0=sol["v_star_s0"],
    )


@app.post("/api/sessions")
async def create_session(req: SessionCreate):
    loop = asyncio.get_event_loop()
    agent_dicts = [a.model_dump() for a in req.agents]
    session = session_manager.create(
        grid_data=req.grid.model_dump(),
        agent_configs=agent_dicts,
        master_seed=req.master_seed,
        budget=req.budget,
        eval_every=req.eval_every,
        loop=loop,
    )
    return {"session_id": session.session_id}


@app.post("/api/sessions/{session_id}/control")
async def control_session(session_id: str, cmd: ControlCommand):
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    if cmd.action == "start":
        session.start()
    elif cmd.action == "set_speed":
        session.send_command("set_speed", {"speed": cmd.speed or 1})
    else:
        session.send_command(cmd.action)

    return {"ok": True, "state": session.state}


@app.post("/api/sessions/{session_id}/edit")
async def edit_session(session_id: str, cmd: EditCommand):
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    edits = [e.model_dump() for e in cmd.edits]
    if cmd.at_episode is not None:
        session.trainer.schedule_event(cmd.at_episode, edits)
    else:
        session.send_command("edit", {"edits": edits})

    return {"ok": True}


@app.get("/api/sessions/{session_id}/snapshot")
async def get_snapshot(session_id: str):
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return JSONResponse(_jsonable(session.get_snapshot()))


@app.post("/api/sessions/{session_id}/rollout")
async def rollout(session_id: str, req: RolloutRequest):
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return JSONResponse(_jsonable(session.run_rollout(req.agent_id, greedy=req.greedy, seed=req.seed)))


@app.post("/api/sessions/{session_id}/explain")
async def explain(session_id: str, req: ExplainRequest):
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return JSONResponse(_jsonable(session.run_explain(
        req.agent_id, req.s, req.a, req.r, req.s2, req.terminated, req.a2
    )))


@app.get("/api/sessions/{session_id}/export")
async def export_session(session_id: str):
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return JSONResponse(_jsonable(session.export_data()))


@app.websocket("/ws/sessions/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    session = session_manager.get(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return
    await ws_session_handler(websocket, session)


FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="static")
