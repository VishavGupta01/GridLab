from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ExplorationStrategy(ABC):
    @abstractmethod
    def select_action(self, q_values: np.ndarray, rng: np.random.Generator) -> int: ...

    @abstractmethod
    def get_eps(self) -> float: ...

    def end_episode(self) -> None:
        pass

    def bump(self, eps_value: float) -> None:
        pass

    def action_probs(self, q_values: np.ndarray) -> np.ndarray:
        n = len(q_values)
        eps = self.get_eps()
        probs = np.full(n, eps / n)
        probs[np.argmax(q_values)] += 1.0 - eps
        return probs


class EpsilonGreedy(ExplorationStrategy):
    def __init__(self, eps: float = 0.1):
        self.eps = eps

    def select_action(self, q_values: np.ndarray, rng: np.random.Generator) -> int:
        if rng.random() < self.eps:
            return int(rng.integers(len(q_values)))
        return int(np.argmax(q_values))

    def get_eps(self) -> float:
        return self.eps

    def bump(self, eps_value: float) -> None:
        self.eps = eps_value


class DecayingEpsilon(ExplorationStrategy):
    def __init__(self, eps0: float = 1.0, eps_min: float = 0.05, decay: float = 0.995):
        self.eps0 = eps0
        self.eps_min = eps_min
        self.decay = decay
        self.eps = eps0

    def select_action(self, q_values: np.ndarray, rng: np.random.Generator) -> int:
        if rng.random() < self.eps:
            return int(rng.integers(len(q_values)))
        return int(np.argmax(q_values))

    def get_eps(self) -> float:
        return self.eps

    def end_episode(self) -> None:
        self.eps = max(self.eps_min, self.eps * self.decay)

    def bump(self, eps_value: float) -> None:
        self.eps = eps_value


class Softmax(ExplorationStrategy):
    def __init__(self, tau: float = 1.0):
        self.tau = tau

    def select_action(self, q_values: np.ndarray, rng: np.random.Generator) -> int:
        probs = self.action_probs(q_values)
        return int(rng.choice(len(q_values), p=probs))

    def get_eps(self) -> float:
        return 0.0

    def action_probs(self, q_values: np.ndarray) -> np.ndarray:
        scaled = q_values / max(self.tau, 1e-10)
        scaled -= np.max(scaled)
        exp_q = np.exp(scaled)
        return exp_q / np.sum(exp_q)


def create_exploration(config: dict) -> ExplorationStrategy:
    etype = config.get("type", "epsilon")
    if etype == "epsilon":
        return EpsilonGreedy(eps=config.get("eps", 0.1))
    elif etype == "decaying_epsilon":
        return DecayingEpsilon(
            eps0=config.get("eps0", 1.0),
            eps_min=config.get("eps_min", 0.05),
            decay=config.get("decay", 0.995),
        )
    elif etype == "softmax":
        return Softmax(tau=config.get("tau", 1.0))
    else:
        raise ValueError(f"Unknown exploration type: {etype}")
