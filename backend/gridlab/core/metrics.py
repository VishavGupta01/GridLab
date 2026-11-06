from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .solver import exact_policy_evaluation, greedy_path, value_iteration


@dataclass
class OnlineMetrics:
    steps: int = 0
    total_return: float = 0.0
    success: bool = False
    hazard_hits: int = 0
    eps: float = 0.0
    mean_abs_td_error: float = 0.0


@dataclass
class EvalMetrics:
    gap: float = 0.0
    gap_pct: float = 0.0
    path_ratio: float | None = None
    pct_optimal: float = 0.0
    episodes_to_near_optimal: int | None = None


@dataclass
class AdaptationRecord:
    agent_id: str
    event_episode: int
    gap_spike: float = 0.0
    recovery_time: int | None = None
    adaptation_regret: float = 0.0


def compute_eval_metrics(
    agent,
    model: dict,
    v_star: np.ndarray,
    q_star: np.ndarray,
    optimal_actions: list[list[int]],
    bfs_shortest: int | None,
) -> EvalMetrics:
    policy = agent.greedy_policy()
    v_pi = exact_policy_evaluation(policy, model)
    start = model["start"]

    gap = float(v_star[start] - v_pi[start])
    v_star_s0 = float(v_star[start])
    gap_pct = (gap / abs(v_star_s0) * 100) if abs(v_star_s0) > 1e-10 else 0.0

    q_table = agent.q_table()
    path, reached = greedy_path(q_table, start, model)
    if reached and bfs_shortest is not None and bfs_shortest > 0:
        path_ratio = (len(path) - 1) / bfs_shortest
    else:
        path_ratio = None

    n_optimal = 0
    n_reachable = 0
    wall_mask = model["wall_mask"]
    goals = model["goals"]
    for s in range(model["n_states"]):
        if wall_mask[s] or s in goals:
            continue
        if not optimal_actions[s]:
            continue
        n_reachable += 1
        greedy_a = int(np.argmax(q_table[s]))
        if greedy_a in optimal_actions[s]:
            n_optimal += 1

    pct_optimal = n_optimal / n_reachable if n_reachable > 0 else 0.0

    return EvalMetrics(
        gap=gap,
        gap_pct=gap_pct,
        path_ratio=path_ratio,
        pct_optimal=pct_optimal,
    )


class MetricsTracker:
    def __init__(self, agent_ids: list[str]):
        self.agent_ids = agent_ids
        self.online_history: dict[str, list[OnlineMetrics]] = {aid: [] for aid in agent_ids}
        self.eval_history: dict[str, list[tuple[int, EvalMetrics]]] = {aid: [] for aid in agent_ids}
        self.adaptation_records: list[AdaptationRecord] = []

        self._near_optimal_streak: dict[str, int] = {aid: 0 for aid in agent_ids}
        self._episodes_to_near_optimal: dict[str, int | None] = {aid: None for aid in agent_ids}

    def record_online(self, agent_id: str, metrics: OnlineMetrics) -> None:
        self.online_history[agent_id].append(metrics)

    def record_eval(self, agent_id: str, episode: int, metrics: EvalMetrics) -> None:
        if self._episodes_to_near_optimal[agent_id] is not None:
            metrics.episodes_to_near_optimal = self._episodes_to_near_optimal[agent_id]

        if metrics.gap_pct < 5.0:
            self._near_optimal_streak[agent_id] += 1
            if self._near_optimal_streak[agent_id] >= 3 and self._episodes_to_near_optimal[agent_id] is None:
                self._episodes_to_near_optimal[agent_id] = episode
                metrics.episodes_to_near_optimal = episode
        else:
            self._near_optimal_streak[agent_id] = 0

        self.eval_history[agent_id].append((episode, metrics))

    def on_grid_change(self, episode: int) -> None:
        for aid in self.agent_ids:
            self._near_optimal_streak[aid] = 0

    def compute_adaptation(
        self,
        agent_id: str,
        event_episode: int,
        v_star_new_s0: float,
        window: int | None = None,
    ) -> AdaptationRecord:
        evals = self.eval_history[agent_id]
        post_evals = [(ep, m) for ep, m in evals if ep >= event_episode]

        gap_spike = 0.0
        if post_evals:
            gap_spike = post_evals[0][1].gap

        recovery_time = None
        streak = 0
        for ep, m in post_evals:
            if m.gap_pct < 5.0:
                streak += 1
                if streak >= 3:
                    recovery_time = ep - event_episode
                    break
            else:
                streak = 0

        adaptation_regret = 0.0
        for ep, m in post_evals:
            adaptation_regret += m.gap

        record = AdaptationRecord(
            agent_id=agent_id,
            event_episode=event_episode,
            gap_spike=gap_spike,
            recovery_time=recovery_time,
            adaptation_regret=adaptation_regret,
        )
        self.adaptation_records.append(record)
        return record
