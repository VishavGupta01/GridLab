from __future__ import annotations

from ..exploration import create_exploration
from .base import Agent
from .expected_sarsa import ExpectedSarsaAgent
from .qlearning import QLearningAgent
from .sarsa import SarsaAgent

AGENT_CLASSES = {
    "q_learning": QLearningAgent,
    "sarsa": SarsaAgent,
    "expected_sarsa": ExpectedSarsaAgent,
}


def create_agent(config: dict, n_states: int, n_actions: int = 4) -> Agent:
    algo = config["algo"]
    cls = AGENT_CLASSES.get(algo)
    if cls is None:
        raise ValueError(f"Unknown algorithm: {algo}. Available: {list(AGENT_CLASSES)}")

    exploration_config = config.get("exploration", {"type": "epsilon", "eps": 0.1})
    if isinstance(exploration_config, dict):
        exploration = create_exploration(exploration_config)
    else:
        exploration = exploration_config

    return cls(
        n_states=n_states,
        n_actions=n_actions,
        alpha=config.get("alpha", 0.1),
        gamma=config.get("gamma", 0.99),
        exploration=exploration,
        q_init=config.get("q_init", 0.0),
        eps_bump_on_change=config.get("eps_bump_on_change", False),
        eps_bump_value=config.get("eps_bump_value", 0.3),
        agent_id=config.get("id", ""),
    )
