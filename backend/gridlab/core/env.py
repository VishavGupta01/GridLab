from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .grid import (
    CELL_EMPTY,
    CELL_GOAL,
    CELL_HAZARD,
    CELL_MUD,
    CELL_START,
    CELL_WALL,
    Grid,
)

ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
ACTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}
PERPENDICULAR = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid: Grid, render_mode: str | None = None):
        super().__init__()
        self.grid = grid
        self.render_mode = render_mode
        self.n_states = grid.rows * grid.cols
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(4)

        self._step_cost = grid.get_setting("step_cost")
        self._hazard_penalty = grid.get_setting("hazard_penalty")
        self._mud_cost = grid.get_setting("mud_cost")
        self._goal_reward_default = grid.get_setting("goal_reward")
        self._slip_prob = grid.get_setting("slip_prob")
        self._max_steps = grid.get_setting("max_steps")
        self._gamma = grid.get_setting("gamma")

        self._start_state: int = 0
        self._goal_states: set[int] = set()

        self.next_state: np.ndarray = np.zeros(0)
        self.reward: np.ndarray = np.zeros(0)
        self.done: np.ndarray = np.zeros(0)

        self.compile()

        self._agent_state: int = self._start_state
        self._step_count: int = 0

    def _rc_to_s(self, r: int, c: int) -> int:
        return r * self.grid.cols + c

    def _s_to_rc(self, s: int) -> tuple[int, int]:
        return divmod(s, self.grid.cols)

    def compile(self) -> None:
        g = self.grid
        S, A = self.n_states, 4

        next_state = np.zeros((S, A), dtype=np.int32)
        reward = np.full((S, A), self._step_cost, dtype=np.float64)
        done = np.zeros((S, A), dtype=np.bool_)

        for r in range(g.rows):
            for c in range(g.cols):
                s = self._rc_to_s(r, c)
                cell = g.cells[r][c]

                if cell == CELL_WALL:
                    for a in range(A):
                        next_state[s, a] = s
                        reward[s, a] = 0.0
                        done[s, a] = False
                    continue

                if cell == CELL_GOAL:
                    for a in range(A):
                        next_state[s, a] = s
                        reward[s, a] = 0.0
                        done[s, a] = True
                    continue

                for a in range(A):
                    dr, dc = ACTIONS[a]
                    nr, nc = r + dr, c + dc

                    if nr < 0 or nr >= g.rows or nc < 0 or nc >= g.cols or g.cells[nr][nc] == CELL_WALL:
                        nr, nc = r, c

                    ns = self._rc_to_s(nr, nc)
                    next_cell = g.cells[nr][nc]

                    if next_cell == CELL_GOAL:
                        gr = g.goal_rewards.get((nr, nc), self._goal_reward_default)
                        reward[s, a] = gr
                        done[s, a] = True
                        next_state[s, a] = ns
                    elif next_cell == CELL_HAZARD:
                        reward[s, a] = self._hazard_penalty
                        done[s, a] = False
                        next_state[s, a] = self._rc_to_s(*g.start) if g.start else s
                    elif next_cell == CELL_MUD:
                        reward[s, a] = self._mud_cost
                        done[s, a] = False
                        next_state[s, a] = ns
                    else:
                        reward[s, a] = self._step_cost
                        done[s, a] = False
                        next_state[s, a] = ns

        self.next_state = next_state
        self.reward = reward
        self.done = done

        if g.start:
            self._start_state = self._rc_to_s(*g.start)
        self._goal_states = {self._rc_to_s(r, c) for r, c in g.goals}

        self._wall_mask = np.zeros(S, dtype=np.bool_)
        for r in range(g.rows):
            for c in range(g.cols):
                if g.cells[r][c] == CELL_WALL:
                    self._wall_mask[self._rc_to_s(r, c)] = True

    def model(self) -> dict:
        return {
            "next_state": self.next_state,
            "reward": self.reward,
            "done": self.done,
            "n_states": self.n_states,
            "n_actions": 4,
            "start": self._start_state,
            "goals": self._goal_states,
            "wall_mask": self._wall_mask,
            "gamma": self._gamma,
            "rows": self.grid.rows,
            "cols": self.grid.cols,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[int, dict]:
        super().reset(seed=seed)
        self._agent_state = self._start_state
        self._step_count = 0
        return self._agent_state, {"grid_version": self.grid.version}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        s = self._agent_state
        self._step_count += 1

        if self._slip_prob > 0 and self.np_random.random() < self._slip_prob:
            perp = PERPENDICULAR[action]
            action = perp[self.np_random.integers(2)]

        ns = int(self.next_state[s, action])
        r = float(self.reward[s, action])
        terminated = bool(self.done[s, action])
        truncated = not terminated and self._step_count >= self._max_steps

        self._agent_state = ns

        info: dict[str, Any] = {"grid_version": self.grid.version}
        if self.grid.cells[self._s_to_rc(ns)[0]][self._s_to_rc(ns)[1]] == CELL_HAZARD:
            info["hazard_hit"] = True

        return ns, r, terminated, truncated, info

    def update_grid(self, grid: Grid) -> None:
        self.grid = grid
        self.n_states = grid.rows * grid.cols
        self.observation_space = spaces.Discrete(self.n_states)
        self._step_cost = grid.get_setting("step_cost")
        self._hazard_penalty = grid.get_setting("hazard_penalty")
        self._mud_cost = grid.get_setting("mud_cost")
        self._goal_reward_default = grid.get_setting("goal_reward")
        self._slip_prob = grid.get_setting("slip_prob")
        self._max_steps = grid.get_setting("max_steps")
        self._gamma = grid.get_setting("gamma")
        self.compile()

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        lines = []
        ar, ac = self._s_to_rc(self._agent_state)
        for r in range(self.grid.rows):
            row = ""
            for c in range(self.grid.cols):
                if (r, c) == (ar, ac):
                    row += "A"
                else:
                    row += self.grid.cells[r][c]
            lines.append(row)
        return "\n".join(lines)
