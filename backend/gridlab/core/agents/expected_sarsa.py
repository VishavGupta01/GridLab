from __future__ import annotations

import numpy as np

from .base import Agent


class ExpectedSarsaAgent(Agent):
    def update(self, s: int, a: int, r: float, s2: int, terminated: bool, a2: int | None = None) -> float:
        if terminated:
            target = r
        else:
            probs = self.exploration.action_probs(self._q[s2])
            expected_q = float(np.dot(probs, self._q[s2]))
            target = r + self.gamma * expected_q

        td_error = target - self._q[s, a]
        self._q[s, a] += self.alpha * td_error
        self._record_td_error(s, td_error)
        return float(td_error)

    def explain_update(self, s: int, a: int, r: float, s2: int, terminated: bool, a2: int | None = None) -> dict:
        q_before = self._q[s].copy()
        if terminated:
            target = r
            bootstrap_term = 0.0
            bootstrap_desc = "terminal (no bootstrap)"
            expected_q = 0.0
        else:
            probs = self.exploration.action_probs(self._q[s2])
            expected_q = float(np.dot(probs, self._q[s2]))
            bootstrap_term = self.gamma * expected_q
            target = r + bootstrap_term
            bootstrap_desc = f"E[Q(s',a')] = {expected_q:.4f} (weighted by policy probs)"

        td_error = target - self._q[s, a]

        return {
            "algo": "expected_sarsa",
            "s": s, "a": a, "r": r, "s2": s2, "terminated": terminated,
            "q_before": q_before.tolist(),
            "q_s_a_before": float(self._q[s, a]),
            "bootstrap": bootstrap_desc,
            "bootstrap_value": bootstrap_term,
            "target": target,
            "td_error": td_error,
            "q_s_a_after": float(self._q[s, a] + self.alpha * td_error),
            "explanation": f"Expected SARSA bootstraps from the expected value under the policy ({bootstrap_desc}). Target = {r:.2f} + {self.gamma}*{expected_q:.4f} = {target:.4f}",
        }
