from __future__ import annotations

from collections import deque

import numpy as np


def bfs_shortest_path(model: dict) -> tuple[int | None, np.ndarray]:
    S = model["n_states"]
    A = model["n_actions"]
    start = model["start"]
    goals = model["goals"]
    next_state = model["next_state"]
    wall_mask = model["wall_mask"]

    dist = np.full(S, -1, dtype=np.int32)
    dist[start] = 0
    queue = deque([start])
    reachable = np.zeros(S, dtype=np.bool_)
    reachable[start] = True

    while queue:
        s = queue.popleft()
        for a in range(A):
            ns = int(next_state[s, a])
            if ns != s and not wall_mask[ns] and dist[ns] == -1:
                dist[ns] = dist[s] + 1
                reachable[ns] = True
                queue.append(ns)

    shortest = None
    for g in goals:
        if dist[g] >= 0:
            if shortest is None or dist[g] < shortest:
                shortest = dist[g]

    return shortest, reachable


def value_iteration(
    model: dict,
    gamma: float | None = None,
    tol: float = 1e-9,
    max_iter: int = 10000,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    S = model["n_states"]
    A = model["n_actions"]
    next_state = model["next_state"]
    reward = model["reward"]
    done = model["done"]
    wall_mask = model["wall_mask"]
    if gamma is None:
        gamma = model["gamma"]

    V = np.zeros(S, dtype=np.float64)
    Q = np.zeros((S, A), dtype=np.float64)

    for iteration in range(max_iter):
        Q_new = np.copy(reward)
        not_done = ~done
        for a in range(A):
            ns = next_state[:, a]
            Q_new[:, a] += gamma * V[ns] * not_done[:, a]

        V_new = np.max(Q_new, axis=1)
        V_new[wall_mask] = 0.0

        delta = np.max(np.abs(V_new - V))
        V = V_new
        Q = Q_new

        if delta < tol:
            break

    optimal_actions: list[list[int]] = []
    for s in range(S):
        if wall_mask[s]:
            optimal_actions.append([])
            continue
        q_s = Q[s]
        max_q = np.max(q_s)
        actions = [int(a) for a in range(A) if abs(q_s[a] - max_q) < 1e-8]
        optimal_actions.append(actions)

    return V, Q, optimal_actions


def exact_policy_evaluation(
    policy: np.ndarray,
    model: dict,
    gamma: float | None = None,
) -> np.ndarray:
    S = model["n_states"]
    next_state = model["next_state"]
    reward_table = model["reward"]
    done_table = model["done"]
    wall_mask = model["wall_mask"]
    if gamma is None:
        gamma = model["gamma"]

    R_pi = np.zeros(S, dtype=np.float64)
    P_pi = np.zeros((S, S), dtype=np.float64)

    for s in range(S):
        if wall_mask[s]:
            continue
        a = int(policy[s])
        R_pi[s] = reward_table[s, a]
        if not done_table[s, a]:
            ns = int(next_state[s, a])
            P_pi[s, ns] = gamma

    A_mat = np.eye(S) - P_pi
    try:
        V_pi = np.linalg.solve(A_mat, R_pi)
    except np.linalg.LinAlgError:
        V_pi = np.linalg.lstsq(A_mat, R_pi, rcond=None)[0]

    V_pi[wall_mask] = 0.0
    return V_pi


def greedy_path(
    Q: np.ndarray,
    start: int,
    model: dict,
    max_len: int = 500,
) -> tuple[list[int], bool]:
    next_state = model["next_state"]
    done = model["done"]

    path = [start]
    visited = {start}
    s = start

    for _ in range(max_len):
        a = int(np.argmax(Q[s]))
        ns = int(next_state[s, a])
        path.append(ns)

        if done[s, a]:
            return path, True

        if ns in visited:
            return path, False

        visited.add(ns)
        s = ns

    return path, False


_cache: dict[int, tuple[np.ndarray, np.ndarray, list[list[int]], int | None]] = {}


def solve_grid(model: dict, grid_version: int) -> dict:
    if grid_version in _cache:
        V, Q, opt_actions, bfs_len = _cache[grid_version]
    else:
        V, Q, opt_actions = value_iteration(model)
        bfs_len, _ = bfs_shortest_path(model)
        _cache[grid_version] = (V, Q, opt_actions, bfs_len)

    start = model["start"]
    path, reached = greedy_path(Q, start, model)

    return {
        "V": V,
        "Q": Q,
        "optimal_actions": opt_actions,
        "bfs_shortest": bfs_len,
        "greedy_path": path,
        "greedy_reached_goal": reached,
        "v_star_s0": float(V[start]),
    }


def clear_cache():
    _cache.clear()
