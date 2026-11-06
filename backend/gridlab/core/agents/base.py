from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..exploration import ExplorationStrategy


class Agent(ABC):
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float,
        gamma: float,
        exploration: ExplorationStrategy,
        q_init: float = 0.0,
        eps_bump_on_change: bool = False,
        eps_bump_value: float = 0.3,
        agent_id: str = "",
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration
        self.eps_bump_on_change = eps_bump_on_change
        self.eps_bump_value = eps_bump_value
        self.agent_id = agent_id

        self._q = np.full((n_states, n_actions), q_init, dtype=np.float64)
        self._visits = np.zeros(n_states, dtype=np.int64)
        self._td_errors = np.zeros(n_states, dtype=np.float64)
        self._td_ema = np.zeros(n_states, dtype=np.float64)
        self._td_ema_alpha = 0.1

    def select_action(self, s: int, rng: np.random.Generator, explore: bool = True) -> int:
        if explore:
            return self.exploration.select_action(self._q[s], rng)
        return int(np.argmax(self._q[s]))

    @abstractmethod
    def update(self, s: int, a: int, r: float, s2: int, terminated: bool, a2: int | None = None) -> float:
        ...

    def end_episode(self) -> None:
        self.exploration.end_episode()

    def on_grid_change(self, new_n_states: int) -> None:
        if new_n_states != self.n_states:
            old_q = self._q
            old_visits = self._visits
            old_td_ema = self._td_ema
            self._q = np.zeros((new_n_states, self.n_actions), dtype=np.float64)
            self._visits = np.zeros(new_n_states, dtype=np.int64)
            self._td_ema = np.zeros(new_n_states, dtype=np.float64)
            min_s = min(self.n_states, new_n_states)
            self._q[:min_s] = old_q[:min_s]
            self._visits[:min_s] = old_visits[:min_s]
            self._td_ema[:min_s] = old_td_ema[:min_s]
            self.n_states = new_n_states

        if self.eps_bump_on_change:
            self.exploration.bump(self.eps_bump_value)

    def q_table(self) -> np.ndarray:
        return self._q.copy()

    def greedy_policy(self) -> np.ndarray:
        return np.argmax(self._q, axis=1).astype(np.int32)

    @abstractmethod
    def explain_update(self, s: int, a: int, r: float, s2: int, terminated: bool, a2: int | None = None) -> dict:
        ...

    def _record_td_error(self, s: int, td_error: float) -> None:
        self._visits[s] += 1
        abs_td = abs(td_error)
        self._td_ema[s] = (1 - self._td_ema_alpha) * self._td_ema[s] + self._td_ema_alpha * abs_td
