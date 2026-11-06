import time

import numpy as np
import pytest

from gridlab.core.env import GridWorldEnv
from gridlab.core.grid import load_preset, parse_grid
from gridlab.core.solver import (
    bfs_shortest_path,
    exact_policy_evaluation,
    greedy_path,
    value_iteration,
)


def _make_model(rows, **kwargs):
    grid = parse_grid({"rows": rows, **kwargs})
    env = GridWorldEnv(grid)
    return env.model()


class TestBFS:
    def test_open_grid(self):
        model = _make_model(["S....", "....G"])
        length, reachable = bfs_shortest_path(model)
        assert length == 5  # right 4 + down 1

    def test_blocked(self):
        model = _make_model(["S....", "#####", "....G"])
        length, reachable = bfs_shortest_path(model)
        assert length is None

    def test_reachability_mask(self):
        model = _make_model(["S....", "#####", "....G"])
        _, reachable = bfs_shortest_path(model)
        # goal at (2,4) should not be reachable
        assert not reachable[2 * 5 + 4]


class TestValueIteration:
    def test_bellman_optimality(self):
        model = _make_model(["S....", "....G"])
        V, Q, _ = value_iteration(model)
        gamma = model["gamma"]

        for s in range(model["n_states"]):
            if model["wall_mask"][s]:
                continue
            max_q = -np.inf
            for a in range(4):
                ns = int(model["next_state"][s, a])
                r = model["reward"][s, a]
                d = model["done"][s, a]
                target = r + (0.0 if d else gamma * V[ns])
                max_q = max(max_q, target)

            assert abs(V[s] - max_q) < 1e-6, f"Bellman violated at state {s}"

    def test_hand_computed_3x3(self):
        # 3x3: S at (0,0), G at (0,2)
        # Optimal: right, right -> 2 steps
        # V*(0,1) = 10 (step right into goal, reward=+10, terminal)
        # V*(0,0) = -1 + gamma * V*(0,1) = -1 + 0.99 * 10 = 8.9
        model = _make_model(["S.G", "...", "..."])
        V, Q, opt = value_iteration(model)
        start = model["start"]

        expected = -1 + 0.99 * 10  # 8.9
        assert abs(V[start] - expected) < 1e-4, f"V*(start) = {V[start]}, expected {expected}"

        # Optimal action at start should be right (1)
        assert 1 in opt[start]

    def test_uniform_cost_vi_matches_bfs(self):
        model = _make_model(["S....", ".....", "....G"])
        V, Q, _ = value_iteration(model)
        bfs_len, _ = bfs_shortest_path(model)

        start = model["start"]
        path, reached = greedy_path(Q, start, model)
        vi_path_len = len(path) - 1

        assert reached
        assert vi_path_len == bfs_len

    def test_cliff_optimal_edge(self):
        grid = load_preset("cliff_4x12")
        env = GridWorldEnv(grid)
        model = env.model()
        V, Q, opt_actions = value_iteration(model)

        start = model["start"]  # (3,0)
        path, reached = greedy_path(Q, start, model)
        assert reached, "Optimal path should reach the goal"

        # Optimal path on cliff goes along the bottom edge
        # It goes up from start, across row 2, then down to goal
        # The path should be 13 steps (up, 11 right, down)
        rows, cols = model["rows"], model["cols"]
        path_cells = [(s // cols, s % cols) for s in path]

        # Should not step on hazards (row 3, cols 1-10)
        for r, c in path_cells:
            assert not (r == 3 and 1 <= c <= 10), f"Path steps on hazard at ({r},{c})"


class TestExactPolicyEval:
    def test_optimal_policy_matches_v_star(self):
        model = _make_model(["S.G", "...", "..."])
        V, Q, _ = value_iteration(model)

        policy = np.argmax(Q, axis=1).astype(np.int32)
        V_pi = exact_policy_evaluation(policy, model)

        np.testing.assert_allclose(V_pi, V, atol=1e-6)

    def test_suboptimal_policy(self):
        model = _make_model(["S.G", "...", "..."])
        V, Q, _ = value_iteration(model)

        # Always go down
        policy = np.full(model["n_states"], 2, dtype=np.int32)
        V_pi = exact_policy_evaluation(policy, model)

        start = model["start"]
        assert V_pi[start] < V[start]


class TestGreedyPath:
    def test_loop_detection(self):
        model = _make_model(["S..", "#.#", "..G"])
        # Create a Q-table that causes a loop
        Q = np.zeros((model["n_states"], 4))
        Q[0, 1] = 1.0  # right from (0,0)
        Q[1, 3] = 1.0  # left from (0,1) -> back to (0,0)

        path, reached = greedy_path(Q, 0, model)
        assert not reached

    def test_reaches_goal(self):
        model = _make_model(["S.G", "...", "..."])
        V, Q, _ = value_iteration(model)
        path, reached = greedy_path(Q, model["start"], model)
        assert reached
        assert path[-1] in model["goals"]


class TestPerformance:
    def test_20x20_under_half_second(self):
        rows = ["S" + "." * 19] + ["." * 20] * 18 + ["." * 19 + "G"]
        model = _make_model(rows)
        start = time.time()
        V, Q, _ = value_iteration(model)
        elapsed = time.time() - start
        assert elapsed < 0.5, f"20x20 solve took {elapsed:.2f}s"
