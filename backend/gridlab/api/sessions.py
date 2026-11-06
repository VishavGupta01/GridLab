from __future__ import annotations

import asyncio
import copy
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import numpy as np

from ..core.agents.base import Agent
from ..core.agents.registry import create_agent
from ..core.env import GridWorldEnv
from ..core.grid import Grid, parse_grid
from ..core.metrics import OnlineMetrics, compute_eval_metrics
from ..core.solver import clear_cache, solve_grid
from ..core.trainer import LockstepTrainer, TrainerConfig

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "runs"


@dataclass
class Command:
    action: str
    data: dict = field(default_factory=dict)


class Session:
    def __init__(
        self,
        session_id: str,
        grid_data: dict,
        agent_configs: list[dict],
        master_seed: int = 42,
        budget: int = 2000,
        eval_every: int = 10,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.session_id = session_id
        self.grid_data = grid_data
        self.agent_configs = agent_configs
        self.master_seed = master_seed
        self.budget = budget
        self.eval_every = eval_every
        self.loop = loop

        self.grid = parse_grid(grid_data)
        self.env = GridWorldEnv(self.grid)

        self.agents: list[Agent] = []
        for cfg in agent_configs:
            cfg_copy = dict(cfg)
            cfg_copy["gamma"] = self.grid.get_setting("gamma")
            self.agents.append(create_agent(cfg_copy, self.env.n_states))

        trainer_config = TrainerConfig(
            budget=budget,
            eval_every=eval_every,
            master_seed=master_seed,
        )
        self.trainer = LockstepTrainer(self.grid, self.agents, trainer_config)

        self.cmd_queue: Queue[Command] = Queue()
        self.broadcast_queue: asyncio.Queue = asyncio.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._paused = False
        self._speed = 1
        self._state = "idle"
        self.last_activity = time.time()

        self.trainer.add_callback(self._on_trainer_event)

        self._run_dir = RESULTS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_id[:8]}"

    def _on_trainer_event(self, event_type: str, data: dict) -> None:
        self._broadcast(event_type, data)

    def _broadcast(self, msg_type: str, data: dict) -> None:
        msg = {"type": msg_type, **data}
        if self.loop:
            try:
                self.loop.call_soon_threadsafe(self.broadcast_queue.put_nowait, msg)
            except RuntimeError:
                pass

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._paused = False
        self._state = "running"
        self._broadcast("status", {"state": "running", "speed": self._speed, "episode": self.trainer.episode, "budget": self.budget})
        self._thread = threading.Thread(target=self._train_loop, daemon=True)
        self._thread.start()

    def _train_loop(self) -> None:
        while self._running and self.trainer.episode < self.budget:
            self._drain_commands()

            if self._paused:
                time.sleep(0.05)
                continue

            episodes_per_tick = self._speed
            for _ in range(episodes_per_tick):
                if not self._running or self._paused:
                    break
                if self.trainer.episode >= self.budget:
                    break
                self.trainer.run_episode()

            if self._speed < 10:
                time.sleep(max(0.01, 0.1 / self._speed))

            if self.trainer.episode % max(1, 10 // self._speed) == 0:
                self._send_snapshot()

        if self.trainer.episode >= self.budget:
            self._state = "finished"
            self._broadcast("status", {"state": "finished", "speed": self._speed, "episode": self.trainer.episode, "budget": self.budget})
        self._running = False

    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self.cmd_queue.get_nowait()
            except Empty:
                break
            self._handle_command(cmd)

    def _handle_command(self, cmd: Command) -> None:
        self.last_activity = time.time()
        if cmd.action == "pause":
            self._paused = True
            self._state = "paused"
            self._broadcast("status", {"state": "paused", "speed": self._speed, "episode": self.trainer.episode, "budget": self.budget})
        elif cmd.action == "resume":
            self._paused = False
            self._state = "running"
            self._broadcast("status", {"state": "running", "speed": self._speed, "episode": self.trainer.episode, "budget": self.budget})
        elif cmd.action == "reset":
            self._running = False
            self._paused = False
            self._reset()
        elif cmd.action == "set_speed":
            self._speed = max(1, min(10, cmd.data.get("speed", 1)))
        elif cmd.action == "edit":
            edits = cmd.data.get("edits", [])
            self.trainer.apply_edits(edits)
        elif cmd.action == "set_config":
            pass

    def _reset(self) -> None:
        self.grid = parse_grid(self.grid_data)
        self.env = GridWorldEnv(self.grid)
        self.agents = []
        for cfg in self.agent_configs:
            cfg_copy = dict(cfg)
            cfg_copy["gamma"] = self.grid.get_setting("gamma")
            self.agents.append(create_agent(cfg_copy, self.env.n_states))

        trainer_config = TrainerConfig(
            budget=self.budget,
            eval_every=self.eval_every,
            master_seed=self.master_seed,
        )
        self.trainer = LockstepTrainer(self.grid, self.agents, trainer_config)
        self.trainer.add_callback(self._on_trainer_event)
        self._state = "idle"
        self._broadcast("status", {"state": "idle", "speed": self._speed, "episode": 0, "budget": self.budget})

    def _send_snapshot(self) -> None:
        snapshot = self.trainer.get_snapshot()
        self._broadcast("snapshot", snapshot)

    def send_command(self, action: str, data: dict | None = None) -> None:
        self.last_activity = time.time()
        self.cmd_queue.put(Command(action=action, data=data or {}))

    def get_snapshot(self) -> dict:
        return self.trainer.get_snapshot()

    def run_rollout(self, agent_id: str, greedy: bool = True, seed: int = 0) -> dict:
        idx = next((i for i, a in enumerate(self.agents) if a.agent_id == agent_id), 0)
        result = self.trainer.run_rollout(idx, greedy=greedy, seed=seed)
        sol = self.trainer._solution
        trajectory = result["steps"]
        total_return = sum(step["r"] for step in trajectory)

        return {
            "agent_id": agent_id,
            "steps": trajectory,
            "total_return": total_return,
            "success": result["success"],
            "optimal_steps": sol["bfs_shortest"] if sol else None,
        }

    def run_explain(self, agent_id: str, s: int, a: int, r: float, s2: int, terminated: bool, a2: int | None = None) -> dict:
        results = {}
        for agent in self.agents:
            result = agent.explain_update(s, a, r, s2, terminated, a2)
            results[agent.agent_id] = result

        target_agent = next((ag for ag in self.agents if ag.agent_id == agent_id), self.agents[0])
        primary = target_agent.explain_update(s, a, r, s2, terminated, a2)
        primary["all_agents"] = results
        return primary

    def export_data(self) -> dict:
        return {
            "config": {
                "grid": self.grid_data,
                "agents": self.agent_configs,
                "master_seed": self.master_seed,
                "budget": self.budget,
            },
            "online": {
                aid: [
                    {"steps": m.steps, "return": m.total_return, "success": m.success,
                     "hazard_hits": m.hazard_hits, "eps": m.eps, "td": m.mean_abs_td_error}
                    for m in metrics
                ]
                for aid, metrics in self.trainer.metrics.online_history.items()
            },
            "eval": {
                aid: [
                    {"episode": ep, "gap": m.gap, "gap_pct": m.gap_pct,
                     "path_ratio": m.path_ratio, "pct_optimal": m.pct_optimal}
                    for ep, m in evals
                ]
                for aid, evals in self.trainer.metrics.eval_history.items()
            },
            "events": self.trainer.event_log,
        }

    @property
    def state(self) -> str:
        return self._state

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, grid_data: dict, agent_configs: list[dict],
               master_seed: int, budget: int, eval_every: int,
               loop: asyncio.AbstractEventLoop | None = None) -> Session:
        session_id = str(uuid.uuid4())[:8]
        session = Session(
            session_id=session_id,
            grid_data=grid_data,
            agent_configs=agent_configs,
            master_seed=master_seed,
            budget=budget,
            eval_every=eval_every,
            loop=loop,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def remove(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                session.stop()

    def cleanup_expired(self, max_age: float = 1800) -> None:
        now = time.time()
        expired = []
        with self._lock:
            for sid, session in self._sessions.items():
                if now - session.last_activity > max_age:
                    expired.append(sid)
        for sid in expired:
            self.remove(sid)
