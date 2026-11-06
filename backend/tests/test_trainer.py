import numpy as np
import pytest

from gridlab.core.agents.registry import create_agent
from gridlab.core.env import GridWorldEnv
from gridlab.core.grid import load_preset, parse_grid
from gridlab.core.trainer import LockstepTrainer, TrainerConfig


def _make_trainer(rows, agent_configs=None, budget=100, eval_every=10, seed=42, **kwargs):
    grid = parse_grid({"rows": rows, **kwargs})
    env = GridWorldEnv(grid)
    n_states = env.n_states

    if agent_configs is None:
        agent_configs = [
            {"id": "q", "algo": "q_learning"},
            {"id": "s", "algo": "sarsa"},
            {"id": "e", "algo": "expected_sarsa"},
        ]

    agents = []
    for cfg in agent_configs:
        cfg_copy = dict(cfg)
        cfg_copy["gamma"] = grid.get_setting("gamma")
        agent = create_agent(cfg_copy, n_states)
        agents.append(agent)

    config = TrainerConfig(budget=budget, eval_every=eval_every, master_seed=seed)
    return LockstepTrainer(grid, agents, config)


class TestLockstepTrainer:
    def test_runs_without_error(self):
        trainer = _make_trainer(["S....", "....G"], budget=50)
        trainer.run()
        assert trainer.episode == 50

    def test_metrics_recorded(self):
        trainer = _make_trainer(["S....", "....G"], budget=50)
        trainer.run()
        for aid in ["q", "s", "e"]:
            assert len(trainer.metrics.online_history[aid]) == 50
            assert len(trainer.metrics.eval_history[aid]) > 0

    def test_deterministic(self):
        t1 = _make_trainer(["S....", "....G"], budget=50, seed=42)
        t1.run()
        q1 = t1.agents[0].q_table()

        t2 = _make_trainer(["S....", "....G"], budget=50, seed=42)
        t2.run()
        q2 = t2.agents[0].q_table()

        np.testing.assert_array_equal(q1, q2)

    def test_different_seeds_differ(self):
        t1 = _make_trainer(["S....", "....G"], budget=50, seed=42)
        t1.run()
        q1 = t1.agents[0].q_table()

        t2 = _make_trainer(["S....", "....G"], budget=50, seed=99)
        t2.run()
        q2 = t2.agents[0].q_table()

        assert not np.array_equal(q1, q2)


class TestCliffContrast:
    """The key acceptance test: reproduce Sutton & Barto cliff walking results."""

    def test_cliff_walking_contrast(self):
        grid = load_preset("cliff_4x12")
        env = GridWorldEnv(grid)
        n_states = env.n_states

        agent_configs = [
            {"id": "q", "algo": "q_learning", "exploration": {"type": "epsilon", "eps": 0.1}},
            {"id": "s", "algo": "sarsa", "exploration": {"type": "epsilon", "eps": 0.1}},
            {"id": "e", "algo": "expected_sarsa", "exploration": {"type": "epsilon", "eps": 0.1}},
        ]
        agents = []
        for cfg in agent_configs:
            cfg["gamma"] = grid.get_setting("gamma")
            agents.append(create_agent(cfg, n_states))

        config = TrainerConfig(budget=500, eval_every=10, master_seed=42)
        trainer = LockstepTrainer(grid, agents, config)
        trainer.run()

        # Get online returns (last 100 episodes)
        q_returns = [m.total_return for m in trainer.metrics.online_history["q"][-100:]]
        s_returns = [m.total_return for m in trainer.metrics.online_history["s"][-100:]]

        q_mean_return = np.mean(q_returns)
        s_mean_return = np.mean(s_returns)

        # SARSA should have higher online return (takes safer path)
        assert s_mean_return > q_mean_return, (
            f"SARSA online return ({s_mean_return:.1f}) should exceed "
            f"Q-learning ({q_mean_return:.1f})"
        )

        # Q-learning should have smaller greedy gap
        q_evals = trainer.metrics.eval_history["q"]
        s_evals = trainer.metrics.eval_history["s"]

        q_final_gap = q_evals[-1][1].gap
        s_final_gap = s_evals[-1][1].gap

        assert q_final_gap <= s_final_gap, (
            f"Q-learning gap ({q_final_gap:.2f}) should be <= "
            f"SARSA gap ({s_final_gap:.2f})"
        )

        # Verify Q-learning learns edge path (row 2, then along bottom)
        q_agent = trainer.agents[0]
        q_table = q_agent.q_table()
        cols = 12

        # Check that at state (2, 1-10), Q-learning's greedy action is right (action 1)
        wrong_cells = 0
        for c in range(1, 11):
            s = 2 * cols + c
            greedy_a = np.argmax(q_table[s])
            if greedy_a != 1:  # not right
                wrong_cells += 1
        # Allow some cells to be wrong (exploration noise)
        assert wrong_cells <= 3, f"Q-learning should mostly go right along row 2, but {wrong_cells} cells disagree"


class TestScheduledEvents:
    def test_scheduled_edit(self):
        trainer = _make_trainer(["S.#..", "....G"], budget=100)
        trainer.schedule_event(50, [{"r": 0, "c": 2, "to": "."}])
        trainer.run()

        assert len(trainer.event_log) == 1
        assert trainer.event_log[0]["episode"] == 50
        assert trainer.grid.cells[0][2] == "."


class TestSmallGridConvergence:
    def test_qlearning_converges(self):
        trainer = _make_trainer(
            ["S....", "....G"],
            agent_configs=[{"id": "q", "algo": "q_learning"}],
            budget=500,
        )
        trainer.run()

        evals = trainer.metrics.eval_history["q"]
        final_gap_pct = evals[-1][1].gap_pct
        assert final_gap_pct < 1.0, f"Q-learning gap is {final_gap_pct:.1f}%, expected < 1%"
