import numpy as np
import pytest

from gridlab.core.agents.expected_sarsa import ExpectedSarsaAgent
from gridlab.core.agents.qlearning import QLearningAgent
from gridlab.core.agents.registry import create_agent
from gridlab.core.agents.sarsa import SarsaAgent
from gridlab.core.exploration import DecayingEpsilon, EpsilonGreedy, Softmax


def _make_agent(cls, n_states=10, n_actions=4, alpha=0.1, gamma=0.99, eps=0.1):
    exploration = EpsilonGreedy(eps=eps)
    return cls(
        n_states=n_states,
        n_actions=n_actions,
        alpha=alpha,
        gamma=gamma,
        exploration=exploration,
        agent_id="test",
    )


class TestQLearning:
    def test_hand_computed_update(self):
        agent = _make_agent(QLearningAgent, n_states=5, alpha=0.5, gamma=0.9)
        # Q is all zeros. s=0, a=1, r=1.0, s2=2, not terminated
        # target = r + gamma * max Q(s2) = 1.0 + 0.9 * 0.0 = 1.0
        # td = 1.0 - 0.0 = 1.0
        # Q(0,1) = 0 + 0.5 * 1.0 = 0.5
        td = agent.update(0, 1, 1.0, 2, False)
        assert abs(td - 1.0) < 1e-10
        assert abs(agent._q[0, 1] - 0.5) < 1e-10

    def test_terminal_update(self):
        agent = _make_agent(QLearningAgent, alpha=1.0)
        td = agent.update(0, 0, 10.0, 1, True)
        assert abs(agent._q[0, 0] - 10.0) < 1e-10


class TestSarsa:
    def test_hand_computed_update(self):
        agent = _make_agent(SarsaAgent, n_states=5, alpha=0.5, gamma=0.9)
        # Set Q(2, 3) = 2.0 so SARSA bootstraps from it
        agent._q[2, 3] = 2.0
        # s=0, a=1, r=1.0, s2=2, a2=3, not terminated
        # target = 1.0 + 0.9 * Q(2, 3) = 1.0 + 0.9 * 2.0 = 2.8
        # td = 2.8 - 0.0 = 2.8
        # Q(0,1) = 0 + 0.5 * 2.8 = 1.4
        td = agent.update(0, 1, 1.0, 2, False, a2=3)
        assert abs(td - 2.8) < 1e-10
        assert abs(agent._q[0, 1] - 1.4) < 1e-10

    def test_uses_actual_next_action(self):
        agent = _make_agent(SarsaAgent, alpha=0.5, gamma=0.9)
        agent._q[2, 0] = 10.0  # best action
        agent._q[2, 3] = -5.0  # bad action
        # SARSA uses a2=3, not the best
        td = agent.update(0, 0, 0.0, 2, False, a2=3)
        expected_target = 0.0 + 0.9 * (-5.0)
        assert abs(td - expected_target) < 1e-10


class TestExpectedSarsa:
    def test_hand_computed_update(self):
        agent = _make_agent(ExpectedSarsaAgent, n_states=5, alpha=0.5, gamma=0.9, eps=0.0)
        # With eps=0, expected SARSA = Q-learning (all weight on max)
        agent._q[2] = [1.0, 3.0, 2.0, 0.0]
        td = agent.update(0, 0, 1.0, 2, False)
        # target = 1.0 + 0.9 * 3.0 = 3.7 (all weight on action 1)
        assert abs(td - 3.7) < 1e-6

    def test_with_exploration(self):
        agent = _make_agent(ExpectedSarsaAgent, n_states=5, alpha=0.5, gamma=0.9, eps=0.4)
        agent._q[2] = [1.0, 3.0, 2.0, 0.0]
        # eps=0.4: probs = [0.1, 0.7, 0.1, 0.1]
        # E[Q] = 0.1*1 + 0.7*3 + 0.1*2 + 0.1*0 = 0.1 + 2.1 + 0.2 + 0 = 2.4
        # target = 1.0 + 0.9 * 2.4 = 3.16
        td = agent.update(0, 0, 1.0, 2, False)
        expected_target = 1.0 + 0.9 * 2.4
        assert abs(td - expected_target) < 1e-6


class TestExplainUpdate:
    def test_explain_no_mutation(self):
        agent = _make_agent(QLearningAgent)
        q_before = agent.q_table()
        agent.explain_update(0, 0, 1.0, 1, False)
        q_after = agent.q_table()
        np.testing.assert_array_equal(q_before, q_after)

    def test_explain_returns_expected_keys(self):
        agent = _make_agent(QLearningAgent)
        result = agent.explain_update(0, 0, 1.0, 1, False)
        assert "target" in result
        assert "td_error" in result
        assert "explanation" in result


class TestExploration:
    def test_epsilon_greedy(self):
        exp = EpsilonGreedy(eps=0.0)
        rng = np.random.default_rng(42)
        q = np.array([1.0, 5.0, 2.0, 0.0])
        for _ in range(100):
            assert exp.select_action(q, rng) == 1

    def test_decaying_epsilon(self):
        exp = DecayingEpsilon(eps0=1.0, eps_min=0.1, decay=0.5)
        assert exp.get_eps() == 1.0
        exp.end_episode()
        assert abs(exp.get_eps() - 0.5) < 1e-10
        exp.end_episode()
        assert abs(exp.get_eps() - 0.25) < 1e-10

    def test_softmax(self):
        exp = Softmax(tau=0.01)
        rng = np.random.default_rng(42)
        q = np.array([1.0, 5.0, 2.0, 0.0])
        # Very low tau → almost always picks best
        actions = [exp.select_action(q, rng) for _ in range(100)]
        assert sum(a == 1 for a in actions) > 95

    def test_bump(self):
        exp = DecayingEpsilon(eps0=1.0, eps_min=0.05, decay=0.99)
        for _ in range(100):
            exp.end_episode()
        old_eps = exp.get_eps()
        exp.bump(0.5)
        assert exp.get_eps() == 0.5


class TestRegistry:
    def test_create_all_types(self):
        for algo in ["q_learning", "sarsa", "expected_sarsa"]:
            agent = create_agent({"id": "test", "algo": algo}, n_states=10)
            assert agent is not None

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            create_agent({"id": "test", "algo": "unknown"}, n_states=10)
