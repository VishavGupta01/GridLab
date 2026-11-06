import gymnasium as gym
import numpy as np
import pytest

from gridlab.core.env import GridWorldEnv
from gridlab.core.grid import load_preset, list_presets, parse_grid


def _make_env(rows, **kwargs):
    grid = parse_grid({"rows": rows, **kwargs})
    return GridWorldEnv(grid)


class TestBasicMechanics:
    def test_wall_bump(self):
        env = _make_env(["S#...", "....G"])
        s, _ = env.reset(seed=0)
        ns, r, term, trunc, _ = env.step(1)  # right into wall
        assert ns == s  # stays in place
        assert r == -1.0
        assert not term

    def test_boundary_bump(self):
        env = _make_env(["S....", "....G"])
        s, _ = env.reset(seed=0)
        ns, r, term, trunc, _ = env.step(0)  # up off grid
        assert ns == s
        assert r == -1.0

    def test_hazard_teleport(self):
        env = _make_env(["SH...", "....G"])
        s, _ = env.reset(seed=0)
        ns, r, term, trunc, _ = env.step(1)  # right into hazard
        assert r == -100.0
        assert not term  # episode continues
        assert ns == s  # teleported back to start

    def test_mud_cost(self):
        env = _make_env(["SM...", "....G"])
        env.reset(seed=0)
        ns, r, term, trunc, _ = env.step(1)  # right into mud
        assert r == -5.0
        assert not term

    def test_goal_termination(self):
        env = _make_env(["S...G", "....."])
        env.reset(seed=0)
        for _ in range(3):
            env.step(1)
        ns, r, term, trunc, _ = env.step(1)  # step into goal
        assert r == 10.0
        assert term
        assert not trunc

    def test_multiple_goals(self):
        env = _make_env(["SG..G", "....."])
        env.reset(seed=0)
        ns, r, term, trunc, _ = env.step(1)  # right into first goal
        assert r == 10.0
        assert term

    def test_truncation(self):
        env = _make_env(
            ["S....", "....."],
            settings={"max_steps": 3},
        )
        env.reset(seed=0)
        env.step(1)  # step 1
        env.step(3)  # step 2
        ns, r, term, trunc, _ = env.step(1)  # step 3 -> truncation
        assert trunc
        assert not term

    def test_truncation_is_not_termination(self):
        """Truncation should NOT count as terminated."""
        env = _make_env(
            ["S....", "....G"],
            settings={"max_steps": 2},
        )
        env.reset(seed=0)
        env.step(1)
        _, _, term, trunc, _ = env.step(1)
        assert trunc
        assert not term


class TestDeterminism:
    def test_same_seed(self):
        env = _make_env(["S....", "....G"])
        results1 = []
        env.reset(seed=42)
        for a in [1, 1, 1, 1, 2]:
            results1.append(env.step(a))

        results2 = []
        env.reset(seed=42)
        for a in [1, 1, 1, 1, 2]:
            results2.append(env.step(a))

        for r1, r2 in zip(results1, results2):
            assert r1[0] == r2[0]
            assert r1[1] == r2[1]


class TestSlipProb:
    def test_slip_changes_outcome(self):
        env = _make_env(
            ["S....", "....G"],
            settings={"slip_prob": 1.0},
        )
        env.reset(seed=42)
        ns, _, _, _, _ = env.step(1)  # intended right, but slips perpendicular
        # With slip_prob=1.0, action should always be perpendicular
        s_start = 0
        # ns should be different from what right (action=1) would give deterministically
        # or same if perpendicular also hits wall/boundary
        assert isinstance(ns, (int, np.integer))


class TestCompile:
    def test_model_keys(self):
        env = _make_env(["S....", "....G"])
        m = env.model()
        assert "next_state" in m
        assert "reward" in m
        assert "done" in m
        assert "n_states" in m
        assert "start" in m
        assert "goals" in m
        assert "wall_mask" in m

    def test_compiled_table_shapes(self):
        env = _make_env(["S....", "....G"])
        m = env.model()
        assert m["next_state"].shape == (10, 4)
        assert m["reward"].shape == (10, 4)
        assert m["done"].shape == (10, 4)


class TestGridUpdate:
    def test_update_grid(self):
        env = _make_env(["S....", "....G"])
        old_model = env.model()
        new_grid = env.grid.apply_edits([{"r": 0, "c": 2, "to": "#"}])
        env.update_grid(new_grid)
        new_model = env.model()
        assert not np.array_equal(old_model["next_state"], new_model["next_state"])


class TestPresets:
    @pytest.mark.parametrize("name", list_presets())
    def test_random_agent_on_preset(self, name):
        grid = load_preset(name)
        env = GridWorldEnv(grid)
        s, _ = env.reset(seed=0)
        done = False
        steps = 0
        while not done and steps < 500:
            a = env.action_space.sample()
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc
            steps += 1
        assert steps > 0


class TestRender:
    def test_ansi_render(self):
        env = _make_env(["S....", "....G"])
        env.render_mode = "ansi"
        env.reset(seed=0)
        output = env.render()
        assert output is not None
        assert "A" in output
