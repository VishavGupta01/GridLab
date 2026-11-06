from __future__ import annotations

import numpy as np

from .base import Agent


class SarsaAgent(Agent):
    def update(self, s: int, a: int, r: float, s2: int, terminated: bool, a2: int | None = None) -> float:
        if terminated:
            target = r
        else:
            target = r + self.gamma * self._q[s2, a2]

        td_error = target - self._q[s, a]
        self._q[s, a] += self.alpha * td_error
        self._record_td_error(s, td_error)
        return float(td_error)

    def explain_update(self, s: int, a: int, r: float, s2: int, terminated: bool, a2: int | None = None) -> dict:
        q_before = self._q[s].copy()
        if a2 is None:
            a2 = int(np.argmax(self._q[s2]))
        if terminated:
            target = r
            bootstrap_term = 0.0
            bootstrap_desc = "terminal (no bootstrap)"
        else:
            next_q = float(self._q[s2, a2])
            bootstrap_term = self.gamma * next_q
            target = r + bootstrap_term
            bootstrap_desc = f"Q(s',a') = Q(s',{a2}) = {next_q:.4f} (action actually taken)"

        td_error = target - self._q[s, a]

        return {
            "algo": "sarsa",
            "s": s, "a": a, "r": r, "s2": s2, "a2": a2, "terminated": terminated,
            "q_before": q_before.tolist(),
            "q_s_a_before": float(self._q[s, a]),
            "bootstrap": bootstrap_desc,
            "bootstrap_value": bootstrap_term,
            "target": target,
            "td_error": td_error,
            "q_s_a_after": float(self._q[s, a] + self.alpha * td_error),
            "explanation": f"SARSA bootstraps from the action actually taken ({bootstrap_desc}). Target = {r:.2f} + {self.gamma}*{bootstrap_term/self.gamma if not terminated else 0:.4f} = {target:.4f}",
        }
