from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GridSettings(BaseModel):
    step_cost: float = -1.0
    hazard_penalty: float = -100.0
    mud_cost: float = -5.0
    goal_reward: float = 10.0
    slip_prob: float = 0.0
    max_steps: int = 200
    gamma: float = 0.99


class GridJSON(BaseModel):
    version: int = 1
    name: str = ""
    rows: list[str]
    goal_rewards: dict[str, float] = Field(default_factory=dict)
    settings: GridSettings = Field(default_factory=GridSettings)


class ExplorationConfig(BaseModel):
    type: Literal["epsilon", "decaying_epsilon", "softmax"] = "epsilon"
    eps: float = 0.1
    eps0: float = 1.0
    eps_min: float = 0.05
    decay: float = 0.995
    tau: float = 1.0


class AgentConfig(BaseModel):
    id: str
    algo: Literal["q_learning", "sarsa", "expected_sarsa"]
    alpha: float = 0.1
    exploration: ExplorationConfig = Field(default_factory=ExplorationConfig)
    q_init: float = 0.0
    eps_bump_on_change: bool = False
    eps_bump_value: float = 0.3


class SessionCreate(BaseModel):
    grid: GridJSON
    agents: list[AgentConfig]
    master_seed: int = 42
    budget: int = 2000
    eval_every: int = 10


class ControlCommand(BaseModel):
    action: Literal["start", "pause", "resume", "reset", "set_speed"]
    speed: int | None = None


class CellEdit(BaseModel):
    r: int
    c: int
    to: str


class EditCommand(BaseModel):
    edits: list[CellEdit]
    at_episode: int | None = None


class ExplainRequest(BaseModel):
    agent_id: str
    s: int
    a: int
    r: float
    s2: int
    terminated: bool
    a2: int | None = None


class RolloutRequest(BaseModel):
    agent_id: str
    greedy: bool = True
    seed: int = 0


# --- WebSocket message types (server → client) ---


class AgentOnlineMetrics(BaseModel):
    steps: int
    total_return: float = Field(alias="return")
    success: bool
    hazard_hits: int = 0
    eps: float = 0.0
    td: float = 0.0

    model_config = {"populate_by_name": True}


class MetricsBatch(BaseModel):
    type: Literal["metrics"] = "metrics"
    episode: int
    agents: dict[str, AgentOnlineMetrics]


class AgentEvalMetrics(BaseModel):
    gap: float
    gap_pct: float
    path_ratio: float | None = None
    pct_optimal: float
    episodes_to_near_optimal: int | None = None


class EvalBatch(BaseModel):
    type: Literal["eval"] = "eval"
    episode: int
    agents: dict[str, AgentEvalMetrics]


class AgentSnapshot(BaseModel):
    q: list[list[float]]
    visits: list[int]
    td_ema: list[float]
    greedy_policy: list[int]


class OptimalSnapshot(BaseModel):
    v: list[float]
    q: list[list[float]]
    policy: list[list[int]]


class SnapshotMsg(BaseModel):
    type: Literal["snapshot"] = "snapshot"
    episode: int
    agents: dict[str, AgentSnapshot]
    optimal: OptimalSnapshot


class EventMsg(BaseModel):
    type: Literal["event"] = "event"
    episode: int
    grid_version: int
    edits: list[CellEdit]
    optimum_changed: bool


class StatusMsg(BaseModel):
    type: Literal["status"] = "status"
    state: Literal["idle", "running", "paused", "finished"]
    speed: int = 1
    episode: int = 0
    budget: int = 2000


class GridValidation(BaseModel):
    valid: bool
    solvable: bool = False
    issues: list[str] = Field(default_factory=list)
    optimal_path_length: int | None = None
    v_star_s0: float | None = None


class RolloutStep(BaseModel):
    s: int
    a: int
    r: float
    s2: int
    done: bool


class RolloutResult(BaseModel):
    agent_id: str
    steps: list[RolloutStep]
    total_return: float
    success: bool
    optimal_steps: int | None = None


class ExplainResult(BaseModel):
    agent_id: str
    s: int
    a: int
    r: float
    s2: int
    terminated: bool
    a2: int | None = None
    q_before: list[float]
    q_after: list[float]
    targets: dict[str, float]
    td_errors: dict[str, float]
    explanation: str


class AdaptationMetrics(BaseModel):
    agent_id: str
    event_episode: int
    gap_spike: float
    recovery_time: int | None = None
    adaptation_regret: float
