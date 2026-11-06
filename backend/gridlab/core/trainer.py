from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .agents.base import Agent
from .env import GridWorldEnv
from .grid import Grid
from .metrics import MetricsTracker, OnlineMetrics, compute_eval_metrics
from .solver import solve_grid


@dataclass
class ScheduledEvent:
    at_episode: int
    edits: list[dict]
    applied: bool = False


@dataclass
class TrainerConfig:
    budget: int = 2000
    eval_every: int = 10
    master_seed: int = 42


class LockstepTrainer:
    def __init__(
        self,
        grid: Grid,
        agents: list[Agent],
        config: TrainerConfig | None = None,
    ):
        self.grid = grid
        self.agents = agents
        self.config = config or TrainerConfig()

        self.env = GridWorldEnv(grid)
        self.episode = 0
        self.grid_version = grid.version

        agent_ids = [a.agent_id for a in agents]
        self.metrics = MetricsTracker(agent_ids)
        self.scheduled_events: list[ScheduledEvent] = []
        self.event_log: list[dict] = []

        self._solution: dict | None = None
        self._solve()

        seq = np.random.SeedSequence(self.config.master_seed)
        child_seeds = seq.spawn(len(agents) + 1)
        self._env_rng = np.random.default_rng(child_seeds[0])
        self._agent_rngs = [np.random.default_rng(s) for s in child_seeds[1:]]

        self._callbacks: list[Callable] = []

    def _solve(self) -> None:
        model = self.env.model()
        self._solution = solve_grid(model, self.grid_version)

    def add_callback(self, cb: Callable) -> None:
        self._callbacks.append(cb)

    def _emit(self, event_type: str, data: dict) -> None:
        for cb in self._callbacks:
            cb(event_type, data)

    def schedule_event(self, at_episode: int, edits: list[dict]) -> None:
        self.scheduled_events.append(ScheduledEvent(at_episode=at_episode, edits=edits))

    def apply_edits(self, edits: list[dict]) -> None:
        self.grid = self.grid.apply_edits(edits)
        self.grid_version = self.grid.version
        self.env.update_grid(self.grid)
        self._solve()

        for agent in self.agents:
            agent.on_grid_change(self.env.n_states)

        self.metrics.on_grid_change(self.episode)

        event = {
            "episode": self.episode,
            "grid_version": self.grid_version,
            "edits": edits,
            "optimum_changed": True,
        }
        self.event_log.append(event)
        self._emit("event", event)

    def _check_scheduled_events(self) -> None:
        for evt in self.scheduled_events:
            if not evt.applied and evt.at_episode == self.episode:
                self.apply_edits(evt.edits)
                evt.applied = True

    def run_episode(self) -> dict[str, OnlineMetrics]:
        self._check_scheduled_events()

        model = self.env.model()
        sol = self._solution
        env_seed = int(self._env_rng.integers(0, 2**31))

        all_metrics = {}

        for i, agent in enumerate(self.agents):
            rng = self._agent_rngs[i]
            s, _ = self.env.reset(seed=env_seed)
            done = False
            total_return = 0.0
            steps = 0
            hazard_hits = 0
            td_errors = []

            a = agent.select_action(s, rng, explore=True)

            while not done:
                ns, r, terminated, truncated, info = self.env.step(a)
                done = terminated or truncated
                steps += 1
                total_return += r
                if info.get("hazard_hit"):
                    hazard_hits += 1

                a2 = agent.select_action(ns, rng, explore=True) if not done else 0

                if terminated:
                    td_error = agent.update(s, a, r, ns, True, a2)
                elif truncated:
                    td_error = agent.update(s, a, r, ns, False, a2)
                else:
                    td_error = agent.update(s, a, r, ns, False, a2)

                td_errors.append(abs(td_error))
                s = ns
                a = a2

            agent.end_episode()

            om = OnlineMetrics(
                steps=steps,
                total_return=total_return,
                success=info.get("hazard_hit", False) is False and terminated,
                hazard_hits=hazard_hits,
                eps=agent.exploration.get_eps(),
                mean_abs_td_error=float(np.mean(td_errors)) if td_errors else 0.0,
            )

            # Fix success: it's true if the episode terminated (reached goal)
            om.success = terminated and not truncated

            self.metrics.record_online(agent.agent_id, om)
            all_metrics[agent.agent_id] = om

        self._emit("metrics", {"episode": self.episode, "agents": all_metrics})

        if self.episode % self.config.eval_every == 0:
            self._run_eval()

        self.episode += 1
        return all_metrics

    def _run_eval(self) -> None:
        model = self.env.model()
        sol = self._solution
        eval_data = {}

        for agent in self.agents:
            em = compute_eval_metrics(
                agent, model,
                sol["V"], sol["Q"], sol["optimal_actions"],
                sol["bfs_shortest"],
            )
            self.metrics.record_eval(agent.agent_id, self.episode, em)
            eval_data[agent.agent_id] = em

        self._emit("eval", {"episode": self.episode, "agents": eval_data})

    def run(self, n_episodes: int | None = None) -> None:
        budget = n_episodes or self.config.budget
        for _ in range(budget):
            self.run_episode()

    def get_snapshot(self) -> dict:
        agents_data = {}
        for agent in self.agents:
            agents_data[agent.agent_id] = {
                "q": agent.q_table().tolist(),
                "visits": agent._visits.tolist(),
                "td_ema": agent._td_ema.tolist(),
                "greedy_policy": agent.greedy_policy().tolist(),
            }

        sol = self._solution
        return {
            "episode": self.episode,
            "agents": agents_data,
            "optimal": {
                "v": sol["V"].tolist(),
                "q": sol["Q"].tolist(),
                "policy": sol["optimal_actions"],
            },
        }

    def run_rollout(self, agent_idx: int, greedy: bool = True, seed: int = 0) -> dict:
        agent = self.agents[agent_idx]
        rng = np.random.default_rng(seed)

        env = GridWorldEnv(self.grid)
        s, _ = env.reset(seed=seed)
        trajectory = []
        done = False
        reached_goal = False

        while not done:
            if greedy:
                a = agent.select_action(s, rng, explore=False)
            else:
                a = agent.select_action(s, rng, explore=True)

            ns, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            reached_goal = terminated and not truncated
            trajectory.append({"s": s, "a": a, "r": float(r), "s2": ns, "done": done})
            s = ns

        return {"steps": trajectory, "success": reached_goal}
