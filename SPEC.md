# SPEC: GridLab, an Interactive Arena for Comparing TD-Learning Agents

| | |
|---|---|
| **Course** | UCS760 Reinforcement Learning |
| **Project** | A web dashboard where you draw a grid world (walls, goals, hazards, mud), let 3 RL agents learn on it side by side, edit the grid **while they train**, and see **how far each agent is from the true optimum** and **why** it behaves the way it does |
| **Syllabus topics** | Temporal-Difference Methods, OpenAI Gym (Gymnasium) |
| **Type** | Software only. Runs locally in the browser, no GPU, no hardware |
| **Stack** | Python (FastAPI, NumPy, Gymnasium) backend + React (Vite) frontend. **No Streamlit.** |
| **Effort** | About 50-55 focused hours in total (see Section 11 for schedule options and what to cut) |

> The name "GridLab" is a placeholder; rename freely.

---

## 1. Summary

Most gridworld demos show one agent on one fixed map and only show that "the agent reaches the goal." GridLab turns that into an **experiment bench**:

1. **Build** any grid with a click-and-drag editor.
2. **Race** three agents (Q-learning, SARSA, Expected SARSA by default) trained in lockstep on the same grid with the same seeds.
3. **Measure** each agent against the mathematically optimal solution, not just against each other.
4. **Disrupt** the world mid-training (open a shortcut, block a corridor, move the goal) and measure how fast each agent re-adapts.
5. **Explain** each agent's behaviour with value heatmaps, policy arrows, error maps and a step-by-step view of the TD update.

The RL analysis is the core of the project. The dashboard is the instrument that makes the analysis visible.

---

## 2. Novelty (three claims, stated honestly)

None of these are new RL algorithms. The novelty is in the **combination** and in the **analysis tooling**. The closest well-known prior work I know of is Sutton & Barto's Cliff Walking example (the Q-learning vs. SARSA contrast) and Karpathy's browser-based REINFORCEjs gridworld demos (a single editable agent). I have not done a systematic literature search, so cite those as related work and state precisely what GridLab adds.

| # | Novelty | What it adds | Why it matters |
|---|---|---|---|
| **N1** | **Live edits during training, with adaptation metrics** | The grid can be modified while the agents are learning, and the system measures *recovery time* and *adaptation regret* per agent. Edits can also be scripted so the same disruption is repeated across seeds. | Most demos train on a fixed map. Non-stationarity is where TD methods differ in interesting ways (stale values, vanished exploration). |
| **N2** | **Every agent is compared to the exact optimum** | The grid is solved exactly (BFS + value iteration). Each agent's greedy policy is evaluated exactly, giving an *optimality gap*, a *path-length ratio*, *% optimal actions* and *episodes-to-near-optimal*. | Replaces "it reached the goal" with "it is 3.2% below optimal and still wrong in 7 cells." Also separates online performance from learned-policy quality. |
| **N3** | **"Why did it do that?" explainability** | Per-action Q-values on hover, TD-error maps (where is it still learning?), policy disagreement overlay, and an **Update Inspector** that shows the numeric TD update for the same transition under three different rules. | Makes the difference between max (Q-learning), sampled next action (SARSA) and expectation (Expected SARSA) visible on real numbers. |

---

## 3. Scope

**Must**
- Custom Gymnasium grid environment with walls, start, one or more goals, hazards
- 3 tabular agents sharing one interface: Q-learning, SARSA, Expected SARSA
- Exact solver (BFS + value iteration + exact policy evaluation)
- Lockstep trainer with greedy evaluation and metric logging
- Dashboard: grid editor, run controls, learning curves, Q-heatmap and policy arrows, race view with step counters, scorecard
- N1, N2 and N3 as defined in Section 2
- Headless experiment runner with multi-seed results and figures for the report

**Should**
- Mud cells (higher step cost)
- Exploration strategies: fixed ε, decaying ε, softmax
- ε-bump on grid change (per-agent toggle)
- Save/load grids as JSON, preset library, run export (CSV/JSON)

**Could**
- Slippery transitions (`slip_prob`)
- Dyna-Q as a swappable fourth agent
- Dark mode, keyboard shortcuts, live hyperparameter edits
- "Same Q-table for all rules" toggle in the Update Inspector

**Won't (future work)**
- Deep RL, continuous spaces, multi-agent interaction, accounts or deployment to the internet

---

## 4. RL formulation

| Element | Definition |
|---|---|
| **State** | Agent's cell, indexed `s = row * cols + col` (`Discrete(rows*cols)`) |
| **Actions** | 4 discrete: up, right, down, left |
| **Cell types** | `.` empty, `#` wall, `S` start (exactly one), `G` goal (one or more), `H` hazard, `M` mud |
| **Transition** | Move one cell. Moving into a wall or off the grid leaves the agent in place. Optional slip: with probability `slip_prob` a perpendicular move happens instead. |
| **Reward** | Step cost `-1` per move; entering mud costs `-5`; entering a goal gives `+10` (configurable per goal); entering a hazard gives `-100` **and teleports the agent back to start, the episode continues** (Sutton & Barto cliff walking) |
| **Termination** | Reaching any goal (`terminated`). Hitting `max_steps` (default 200) gives `truncated`. |
| **Discount** | `gamma = 0.99` |

Implementation rules worth stating in the report:
- **Truncation is not termination.** On truncation the TD target still bootstraps from `Q(s',.)`; only true termination uses target `= r`.
- The environment **compiles** a transition table whenever the grid changes (`next_state[s,a]`, `reward[s,a]`, `done[s,a]`, plus the slip distribution). `step()` reads the table, and the exact solver reads the same table. One source of truth, and fast stepping.
- Q-tables start at `0`, which is optimistic because most returns are negative. This encourages exploration and is a parameter (`q_init`) worth noting in experiments.

---

## 5. Algorithms and hyperparameters

```
Q-learning        : Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a')            - Q(s,a) ]
SARSA             : Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s',a')                   - Q(s,a) ]
Expected SARSA    : Q(s,a) <- Q(s,a) + alpha * [ r + gamma * sum_a' pi(a'|s') Q(s',a')  - Q(s,a) ]
terminal transition: target = r
```

`pi` in Expected SARSA is the agent's current behaviour policy (ε-greedy or softmax).

**Exploration options:** fixed ε-greedy (`eps=0.1`), decaying ε (`eps0=1.0`, `eps_min=0.05`, `decay=0.995` per episode), softmax (`tau` configurable).

**Defaults:** `alpha=0.1` (constant), `gamma=0.99`, `eps=0.1`, 2,000 episodes for live demos, 3,000-5,000 for experiments.

**Fairness rules for comparisons**
- All agents get the same grid, the same hyperparameters (unless the experiment varies one), and the same environment RNG stream.
- Agents train in **lockstep**: episode `i` of every agent runs before episode `i+1` of any agent, so edits land at the same episode index for everyone.
- Per-agent RNGs come from `np.random.SeedSequence(master_seed).spawn(n)`. Every result is reproducible from `(grid, configs, master_seed)`.

---

## 6. Metrics (the analysis backbone)

**Online metrics (per episode, per agent):** steps taken, return, success flag (reached a goal), hazard hits, current ε, mean absolute TD error.

**Greedy-evaluation metrics (every `eval_every = 10` episodes, learning off).** The agent's greedy policy is extracted (ties broken by lowest action index) and evaluated **exactly** by solving `(I - gamma * P_pi) V_pi = R_pi`.

| Metric | Definition |
|---|---|
| **Optimality gap** | `V*(s0) - V_pi(s0)` (and the same as a % of `|V*(s0)|`) |
| **Path-length ratio** | Steps along the greedy path ÷ BFS shortest path. Marked "fails" if the greedy path loops or never reaches a goal. |
| **% optimal actions** | Share of reachable non-terminal states where the greedy action is in the optimal action set from `Q*` |
| **Episodes-to-near-optimal** | First episode at which the gap is under 5% and stays so for 3 consecutive evaluations |

**Adaptation metrics (per grid-change event, per agent)**

| Metric | Definition |
|---|---|
| **Gap spike** | Optimality gap right after the change, measured against the *new* optimum |
| **Recovery time** | Episodes from the change until the gap is under 5% (stays for 3 evaluations). "Did not recover" if the budget ends first. |
| **Adaptation regret** | Area between `V*_new(s0)` and `V_pi(s0)` over the episodes since the change |

**Important nuance to explain in the report:** with a fixed ε, SARSA's policy is optimal for the *ε-greedy behaviour*, not for the greedy objective. On the cliff map it has a better **online return** but a worse **greedy optimality gap** than Q-learning. The dashboard shows both, so the contrast is visible rather than argued.

---

## 7. System architecture

```
┌─────────────────────────── Browser (React + Vite) ───────────────────────────┐
│  Toolbox │ Grid canvas + overlays │ Agent cards │ Charts │ Race │ Inspector   │
└───────────────▲──────────────────────────────────────────┬───────────────────┘
        WebSocket (live metrics, snapshots, events)        │ REST (commands, grids, explain, rollout)
┌───────────────┴──────────────────────────────────────────▼───────────────────┐
│ FastAPI: api/  (schemas, session manager, ws)                                  │
│   Session = grid + agents + trainer thread + command queue                     │
├────────────────────────────────────────────────────────────────────────────────┤
│ core/ (pure Python, no web imports, usable from CLI and tests)                 │
│   grid.py  env.py  solver.py  agents/  exploration.py  trainer.py  metrics.py  │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 7.1 Stack and rationale

- **FastAPI + Uvicorn:** typed schemas (Pydantic), native WebSocket support.
- **NumPy + Gymnasium:** tabular RL plus the standard Gym API (the syllabus topic).
- **React + Vite:** component-based dashboard. Grid drawn on an HTML `<canvas>`; charts with Recharts; state with React Context or Zustand; styling with Tailwind or plain CSS.
- **Fallback if you do not know React:** Vite + plain TypeScript/JS with Chart.js or uPlot. Backend and API stay identical.

### 7.2 Concurrency model

- Each session runs training in its own **background thread**. Only that thread mutates the environment and agents.
- API handlers never touch agents directly. They push **commands** (`start`, `pause`, `reset`, `edit`, `set_speed`, `set_config`) onto a `queue.Queue`.
- The trainer drains the queue **at episode boundaries**, so an edit is never applied mid-episode and every agent sees it at the same episode index.
- The trainer publishes batches to an `asyncio.Queue` via `loop.call_soon_threadsafe`; the WebSocket handler forwards them.
- Backpressure: snapshots (large) may be dropped or coalesced; metric batches and events are never dropped.
- Speed control: the trainer runs K episodes per tick, then sleeps according to the speed slider.
- Sessions expire after 30 minutes of inactivity. Every run is also written to `results/runs/<timestamp>/` for reproducibility.

### 7.3 Data contracts

**Grid JSON** (compact row strings; also used for presets and tests)
```json
{
  "version": 1,
  "name": "cliff_4x12",
  "rows": ["............",
           "............",
           "............",
           "SHHHHHHHHHHG"],
  "goal_rewards": {"3,11": 10},
  "settings": {"step_cost": -1, "hazard_penalty": -100, "mud_cost": -5,
               "goal_reward": 10, "slip_prob": 0.0, "max_steps": 200, "gamma": 0.99}
}
```

**Agent config**
```json
{ "id": "a1", "algo": "q_learning", "alpha": 0.1,
  "exploration": {"type": "epsilon", "eps": 0.1},
  "q_init": 0.0, "eps_bump_on_change": false }
```

**REST endpoints**

| Method + path | Purpose |
|---|---|
| `GET /api/health` | Liveness check |
| `GET /api/presets`, `GET /api/presets/{name}` | Preset grids |
| `POST /api/grids/validate` | Grid check: has start and goal, solvable, optimal path length, `V*(s0)` |
| `POST /api/sessions` | Create session from `{grid, agents[], master_seed, budget}` |
| `POST /api/sessions/{id}/control` | `start`, `pause`, `resume`, `reset`, `set_speed` |
| `POST /api/sessions/{id}/edit` | Apply a batch of cell edits at the next episode boundary (also schedulable: `at_episode`) |
| `GET /api/sessions/{id}/snapshot` | Full current state (Q-tables, visits, TD-error map, policies, optimum) |
| `POST /api/sessions/{id}/rollout` | Run one episode on a **copy** (greedy or behaviour policy, given seed) and return the trajectory for the race animation |
| `POST /api/sessions/{id}/explain` | Compute a TD update breakdown on a **copy** (never mutates training state) |
| `GET /api/sessions/{id}/export` | Metrics CSV + events JSON + configs |

**WebSocket `/ws/sessions/{id}` (server to client)**
```json
{"type": "metrics",  "episode": 420, "agents": {"a1": {"steps": 17, "return": -9.0, "success": true, "eps": 0.1, "td": 0.21}}}
{"type": "eval",     "episode": 420, "agents": {"a1": {"gap": 0.8, "gap_pct": 4.2, "path_ratio": 1.0, "pct_optimal": 0.93}}}
{"type": "snapshot", "episode": 420, "agents": {"a1": {"q": [[...]], "visits": [...], "td_ema": [...]}}, "optimal": {"v": [...], "q": [[...]]}}
{"type": "event",    "episode": 500, "grid_version": 2, "edits": [{"r": 3, "c": 5, "to": "."}], "optimum_changed": true}
{"type": "status",   "state": "running", "speed": 3}
```

---

## 8. Dashboard UX specification

**Layout (designed for 1280 px and wider; mobile is out of scope)**

```
┌──────────────────────────── Top bar: Train ▶ Pause ⏸ Reset ↺ │ Speed │ Episode 420/2000 │ Grid v2 ───────────────┐
│ Toolbox        │               Grid canvas (main)                     │  Agent cards (3)                         │
│ ▢ Wall         │   overlays: Q-heatmap | policy arrows | visits |     │  Algo, exploration, α, γ, ε, q_init,     │
│ ▢ Start/Goal   │   TD-error | optimal policy | disagreement           │  ε-bump toggle, color                    │
│ ▢ Hazard/Mud   │                                                      │                                          │
│ ▢ Erase        │                                                      │                                          │
│ Presets/Save   │                                                      │                                          │
├────────────────┴──────────────────────────────────────────────────────┴──────────────────────────────────────────┤
│ Tabs:  Learning curves │ Race │ Scorecard │ Adaptation │ Inspector │ Export                                       │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Grid editor**
- Brushes: wall, start, goal, hazard, mud, erase. Click and drag to paint. Start is unique (placing it moves it). Goals can be many.
- Resize (5×5 up to 20×20), undo/redo, clear, preset dropdown, import/export JSON.
- Live validation badge: "solvable, optimal path = 14 steps, V* = 6.2" or a clear warning.
- **While training, edits are staged** (shown as ghost cells) and committed with one **Apply at next episode** button (Enter). This produces one clean event for the metrics.

**Overlays (per agent, selectable)**
- Value heatmap of `max_a Q(s,a)` with a diverging colour scale
- Policy arrows; with "compare to optimal" on, an arrow is green if it is optimal and red if not
- Visit-count heatmap (cumulative or recent window)
- TD-error map (moving average of absolute TD error per state)
- Disagreement overlay: cells where the three agents' greedy actions differ
- Hover a cell: the four Q-values for each agent in a popover

**Charts (all with smoothing slider, downsampling for long runs, and dashed vertical lines at grid-change events)**
steps per episode, return per episode, optimality gap, success rate, ε

**Race view:** three mini-grids side by side. Each animates one episode (greedy or ε-greedy, user-selectable) from a rollout, with a **step counter, return, and status below each grid** and the optimal step count as a reference. Shows finish order.

**Scorecard:** one row per agent with final gap, path ratio, % optimal actions, episodes-to-near-optimal, online return (last 100 episodes), hazard hits.

**Adaptation tab:** timeline of events, and per event/agent the gap spike, recovery time (or "did not recover") and adaptation regret.

**Update Inspector (N3)**
- *Follow-an-agent mode:* step through one agent's transitions. Shows `Q(s,·)` before, chosen action (greedy or exploratory, and ε), next state, reward, next action, the target broken down by term, the TD error, and `Q(s,a)` before and after.
- *Same-transition-three-rules mode:* pick a transition and see Q-learning, SARSA and Expected SARSA targets side by side, with a one-line plain-English explanation. Example: "Q-learning bootstraps from the best next action (value -3.2). SARSA bootstraps from the action actually taken, an exploratory move toward the hazard (value -41)."

**Quality bar:** colour-blind-safe palette, legible labels, loading/empty/error states, no UI jank at 20×20 with three overlays, keyboard shortcuts (1-6 brushes, Space play/pause, Ctrl+Z undo, Enter apply).

---

## 9. Repository layout

```
gridlab/
├── SPEC.md
├── README.md
├── Makefile                      # dev, test, results, build
├── backend/
│   ├── pyproject.toml / requirements.txt
│   ├── gridlab/
│   │   ├── core/
│   │   │   ├── grid.py           # parse, validate, edit, presets
│   │   │   ├── env.py            # GridWorldEnv(gym.Env), compile(), model()
│   │   │   ├── solver.py         # BFS, value iteration, exact policy evaluation
│   │   │   ├── exploration.py
│   │   │   ├── agents/           # base.py, qlearning.py, sarsa.py, expected_sarsa.py, registry.py
│   │   │   ├── trainer.py        # LockstepTrainer, eval hooks, scheduled events
│   │   │   └── metrics.py        # optimality + adaptation metrics
│   │   ├── api/                  # main.py, schemas.py, sessions.py, ws.py
│   │   └── cli/run_experiment.py
│   └── tests/
├── frontend/
│   └── src/ { api/, store/, components/{GridCanvas, Toolbox, AgentCard, RunBar,
│              Charts, RaceView, Scorecard, AdaptationPanel, Inspector}, App.tsx }
├── presets/grids/*.json
├── experiments/                  # YAML configs, one per experiment
├── results/                      # runs/, csv/, figs/
└── docs/                         # architecture diagram, screenshots, demo script
```

**Agent interface**
```python
class Agent(ABC):
    def select_action(self, s: int, explore: bool = True) -> int: ...
    def update(self, s, a, r, s2, terminated, a2=None) -> float: ...   # returns TD error
    def end_episode(self) -> None: ...                                  # decay eps, etc.
    def on_grid_change(self, event) -> None: ...                        # e.g. eps bump
    def q_table(self) -> np.ndarray: ...
    def greedy_policy(self) -> np.ndarray: ...
    def explain_update(self, s, a, r, s2, terminated, a2) -> dict: ...  # pure, no mutation
```
The trainer always selects `a2` before calling `update`, so SARSA works in the same loop as the others (Q-learning and Expected SARSA ignore it).

**Presets** (each has a purpose)

| Preset | Purpose |
|---|---|
| `open_10x10` | Baseline, no obstacles |
| `cliff_4x12` | Classic Q-learning vs. SARSA contrast (edge path vs. safe path) |
| `maze_10x10` | Walls, long paths, exploration difficulty |
| `two_goals` | Near small reward vs. far large reward |
| `shortcut` | Long safe route plus a wall that, once removed, opens a short route (for N1) |
| `rush_hour` | Mud corridors that force trade-offs |
| `blocked` | Unsolvable grid, used to test validation |

---

## 10. Phases

Each phase has a goal, tasks, deliverables and **acceptance criteria**. Do not start a phase until the previous one passes its criteria. Effort estimates are focused hours.

### Phase 0: Project initialization and walking skeleton (about 2 h)

**Goal:** a repo where backend and frontend talk to each other, with schemas frozen.

**Tasks**
```bash
mkdir gridlab && cd gridlab && git init
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install fastapi "uvicorn[standard]" numpy gymnasium pydantic pytest pandas matplotlib pyyaml httpx websockets
pip freeze > backend/requirements.lock

npm create vite@latest frontend -- --template react-ts
cd frontend && npm install recharts zustand       # add Tailwind via its current Vite guide, or use plain CSS
```
- Create the folder structure from Section 9.
- Add a Vite dev proxy for `/api` and `/ws` to `localhost:8000`.
- Write `GET /api/health`; show its result on the frontend home page.
- Write `docs/api.md` containing the grid JSON, agent config and WebSocket message formats from Section 7.3. **Freeze them now.**
- Add one smoke test and a `Makefile` (`make dev`, `make test`).

**Run:** `uvicorn gridlab.api.main:app --reload --port 8000` and `npm run dev`.

**Acceptance:** frontend displays "backend: ok"; `pytest` passes; schemas are committed.

---

### Phase 1: Grid and environment (about 3 h)

**Goal:** a correct, fast, Gymnasium-compatible environment.

**Tasks**
- `grid.py`: parse row strings, validate (exactly one start, at least one goal, no start/goal on walls), serialize, apply cell edits, load presets.
- `env.py`: `GridWorldEnv(gym.Env)` with `Discrete` spaces, `reset(seed)`, `step(a)`, truncation at `max_steps`. `compile()` builds `next_state`, `reward`, `done` and the slip distribution. Register as `GridLab-v0`.
- Create all presets from Section 9.
- Tests: wall bump, boundary bump, hazard teleport-to-start with continuing episode, mud cost, multiple goals, termination vs. truncation, determinism under a seed.

**Acceptance:** `gymnasium.utils.env_checker.check_env` passes; a random agent runs on every preset; all unit tests pass.

---

### Phase 2: Exact solver (about 3 h) (foundation of N2)

**Goal:** know the true answer for any grid.

**Tasks**
- BFS shortest path (steps) with reachability check.
- Value iteration on the compiled model, returning `V*`, `Q*`, and the set of optimal actions per state (tolerance `1e-9`).
- Exact policy evaluation for any deterministic policy, solving the linear system.
- Greedy path extraction with loop detection.
- Cache results; recompute only when the grid version changes.

**Acceptance (tests)**
- On uniform-cost grids, value-iteration path length equals BFS length.
- `V*` satisfies the Bellman optimality equation within `1e-6`.
- A hand-computed 3×3 example matches exactly.
- On `cliff_4x12` the optimal path runs along the cliff edge.
- Solving a 20×20 grid takes under about 0.5 s.

---

### Phase 3: Agents, lockstep trainer and headless experiments (about 5 h)

**Goal:** a trustworthy RL engine, usable without any UI.

**Tasks**
- Implement the `Agent` base class and the three agents; exploration strategies; `agent.explain_update`.
- `LockstepTrainer`: runs agents episode by episode in lockstep, logs online metrics, runs greedy evaluation every `eval_every` episodes, supports **scheduled grid events** (`at_episode`) and `on_grid_change` hooks.
- `metrics.py`: optimality gap, path ratio, % optimal actions, episodes-to-near-optimal, adaptation metrics from Section 6.
- CLI: `python -m gridlab.cli.run_experiment experiments/cliff.yaml --seeds 20` writes CSVs.
- Measure steps per second; if below about 20k agent-steps/s, optimise `step()` using the compiled tables.

**Acceptance**
- On a small open grid, Q-learning's greedy gap falls below 1% within the budget.
- On `cliff_4x12` with `eps=0.1`: Q-learning learns the edge path, SARSA the safer path, SARSA has the higher online return, Q-learning the smaller greedy gap. (If you do not reproduce this, debug before moving on.)
- Same seed gives identical CSVs; different seeds differ.
- **Gate A:** the engine works headlessly. Do not start the UI before this.

---

### Phase 4: Backend API and live sessions (about 5 h)

**Goal:** a stable service that streams training to any client.

**Tasks**
- Pydantic schemas; session manager; trainer thread; command queue; episode-boundary edit application; WebSocket streaming with the message types in Section 7.3.
- Endpoints: validate, presets, create session, control, edit (with `at_episode`), snapshot, rollout, explain, export.
- Write every run to `results/runs/<timestamp>/`.

**Acceptance**
- A Python test script creates a session, starts it, receives metric batches at 5 Hz or faster, sends an edit and receives the matching `event`, pauses, resumes and resets.
- Three simultaneous WebSocket clients all work.
- `explain` and `rollout` leave the training state unchanged (checksum test on the Q-tables).
- A 10-minute run has no exceptions, no thread leaks, and no unbounded memory growth.

---

### Phase 5: Dashboard shell and grid editor (about 7 h)

**Goal:** a good-looking editor you can actually build maps in.

**Tasks**
- App layout and theme; `GridCanvas` with painting, drag-paint, hover cell; `Toolbox`; unique start; resize; undo/redo; clear.
- Preset picker; JSON import/export; call `validate` and show the badge.
- `AgentCard` x3 with algorithm, exploration, `alpha`, `gamma`, ε settings; `RunBar` (train, pause, reset, speed) wired to the backend.

**Acceptance**
- You can recreate the cliff map by hand in under a minute.
- Save and reload round-trips identically.
- A missing start or goal is blocked with a clear message; an unsolvable map shows a warning.
- Usable at 1280×720.

---

### Phase 6: Live training views and the optimum comparison (N2) (about 7 h)

**Goal:** watch three agents learn, and see how far from optimal they are.

**Tasks**
- Frontend WebSocket client and state store (batching, throttled rendering).
- Charts: steps, return, optimality gap, success rate, ε. Smoothing and downsampling.
- Overlays: Q-heatmap, policy arrows, visit counts; optimal overlay with green/red match colouring; per-cell hover popover.
- **Race view** with step counters below each grid, using the `rollout` endpoint.
- **Scorecard** table with the Section 6 metrics.

**Acceptance**
- All three agents visibly learn in real time; UI stays smooth at 20×20 with 5,000 episodes of history.
- Scorecard numbers match the headless CLI output for the same seed (automated cross-check).
- Race animation matches the rollout trajectory step for step.
- **Gate B (minimum viable project):** Phases 0-6 pass. If you are short on time, you can submit from here with a strong report.

---

### Phase 7: Live edits and adaptation (N1) (about 5 h)

**Goal:** disrupt the world mid-training and measure recovery.

**Tasks**
- Staged edits with "Apply at next episode"; backend applies at the episode boundary, bumps `grid_version`, recomputes the optimum, emits an `event`.
- Optional per-agent **ε-bump on change** (set ε back to a chosen value and decay again).
- Event markers on every chart; a banner showing the current grid version.
- **Scripted disruptions** (one-click "Open shortcut", "Block corridor", "Move goal" at a chosen episode) so the same disruption can be replayed across seeds.
- Adaptation tab with the Section 6 adaptation metrics.

**Acceptance**
- After an edit, all agents run the new grid from the same episode index and the gap chart jumps against the new optimum.
- A scripted event through the UI and through the CLI give identical results for the same seed.
- Edits that make the grid unsolvable ask for confirmation.
- The TD-error map lights up around the edit and fades as agents re-learn.

---

### Phase 8: Explainability (N3) (about 5 h)

**Goal:** show why each agent does what it does.

**Tasks**
- TD-error moving average per state (collected by the trainer); disagreement overlay; four-action Q popover.
- `explain` endpoint and the **Update Inspector** in both modes (follow-an-agent; same transition, three rules), with the plain-English summary line.
- Optional ("Could"): shared-Q toggle to isolate the effect of the update rule.

**Acceptance**
- Inspector numbers equal a hand calculation on a toy transition (unit test).
- Near the cliff edge, the same transition shows clearly different targets for Q-learning and SARSA.
- Calling `explain` never changes training state.

---

### Phase 9: Experiments and analysis (about 6 h)

**Goal:** turn the tool into evidence. This is where most of the marks are.

**Setup:** each experiment is a YAML file, run headlessly with **20 seeds**, reporting mean ± 95% confidence interval (`1.96 * standard error` or bootstrap). `make results` regenerates every figure.

| ID | Experiment | Question |
|---|---|---|
| **E1** | Cliff Walking, `eps=0.1` fixed | Do we reproduce the Q-learning/SARSA contrast, and does Expected SARSA sit between or above? Compare online return vs. greedy gap. |
| **E2** | Learning-rate sensitivity | Sweep `alpha` in {0.05, 0.1, 0.3, 0.5, 1.0} per algorithm: episodes-to-near-optimal and stability. |
| **E3** | Exploration strategy | Fixed ε vs. decaying ε vs. softmax, same algorithm. |
| **E4** | Map difficulty | Open, maze, two-goals, rush-hour: how does time-to-optimal scale? |
| **E5** | Adaptation | Three disruptions: (a) shortcut opens, (b) corridor blocked, (c) goal moves. With and without ε-bump; constant vs. decaying `alpha`. Measure recovery time and adaptation regret. |
| **E6** (Could) | Slippery transitions | Do the algorithms' rankings change when moves are stochastic? |

**Hypotheses (test, do not assume)**
- **H1.** On the cliff, Q-learning has the smaller greedy gap but the lower online return than SARSA.
- **H2.** Expected SARSA has an online return at least as good as SARSA, with lower variance across seeds.
- **H3.** After a *shortcut opens*, agents whose ε has decayed may fail to discover it in the budget unless ε is bumped. After a *corridor is blocked*, all agents recover because the negative experience is immediate. (An asymmetry you can measure.)
- **H4.** Larger `alpha` adapts faster after a change but is noisier.

**Acceptance:** at least E1, E2, E3 and E5 complete with figures and tables saved in `results/`; every figure is reproducible by one command.

---

### Phase 10: Polish, documentation, demo and report (about 5 h)

**Tasks**
- One-command demo: build the frontend and serve it from FastAPI (`make demo`).
- README with setup, architecture diagram, screenshots or a GIF, and exact reproduction commands.
- Empty, loading and error states; keyboard shortcuts; consistent styling.
- **Demo script (5 minutes), three stories:**
  1. *Cliff:* build it live, run it, show Q-learning's edge path vs. SARSA's safe path, then the scorecard (online return vs. greedy gap).
  2. *Shortcut opens:* edit mid-training, show which agents recover, and use the ε-bump toggle.
  3. *Inspector:* pick a transition near the cliff and show the three different targets.
- Write the report (Section 14) and rehearse the viva.

**Acceptance:** a stranger can clone the repo, follow the README, and reproduce a figure.

---

## 11. Schedule options and what to cut

| Phase | Hours |
|---|---|
| 0 Init | 2 |
| 1 Env | 3 |
| 2 Solver | 3 |
| 3 Agents + trainer | 5 |
| 4 API | 5 |
| 5 Editor | 7 |
| 6 Live views + N2 | 7 |
| 7 Live edits + N1 | 5 |
| 8 Explainability + N3 | 5 |
| 9 Experiments | 6 |
| 10 Polish + report | 5 |
| **Total** | **about 53** |

**Option A, 7 days (about 8 h/day, aggressive):** D1 Phases 0-2; D2 Phase 3 and start 4; D3 finish 4 and start 5; D4 Phase 5 and 6; D5 Phases 6-7; D6 Phases 8-9; D7 Phase 10 and buffer.

**Option B, 12-14 days (about 4-5 h/day, comfortable):** roughly one phase per day, with Phase 9 getting two.

**If you run behind, cut in this order:** (1) the "same-transition" mode of the Inspector, (2) softmax and extra exploration, (3) slippery moves (E6), (4) mud cells, (5) dark mode and shortcuts, (6) the 10-minute soak test.

**Never cut:** Phases 0-6 and the scripted shortcut demo from Phase 7. A working editor, three agents, the optimum comparison and one live disruption already make a strong project.

---

## 12. Testing strategy

- **Unit tests (pytest):** environment rules, parser, solver, agents (one hand-computed update per rule), metrics.
- **Property tests:** value iteration vs. BFS on random uniform-cost grids; exact policy evaluation vs. Monte Carlo rollouts (agree within noise).
- **Golden tests:** fixed seed on `cliff_4x12` produces known CSVs.
- **Integration tests:** WebSocket flow (create, start, edit, pause, reset) with a Python client.
- **Invariants:** `explain` and `rollout` never mutate training state; every agent sees the same grid version at the same episode.
- **Frontend:** a manual checklist per phase (and optionally Vitest for the state store).

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| UI work swallows the RL work | Phase 3 must pass first; hours for Phase 9 are protected; the core library never imports web code. |
| Race conditions on live edits | Single trainer thread owns all state; edits only via the command queue at episode boundaries. |
| Frontend chart lag | Downsample on the server or client; throttle snapshot rate; draw the grid on canvas, not DOM nodes. |
| "Unfair" agent comparison | Lockstep training, shared seeds, shared hyperparameters, 20 seeds in experiments. |
| Hypotheses do not hold | Report honestly and explain; a surprising result is still a result. |
| Bugs in the RL update | Hand-computed unit tests, plus the cliff reproduction in Phase 3 as a sanity check. |
| React is new to you | Start with the Phase 0 skeleton early; use the vanilla-JS fallback if blocked after about half a day. |
| Scope creep | Section 3 tiers and Section 11 cut order are the contract. |

---

## 14. Report, viva and definition of done

**Report outline**
1. Abstract
2. Introduction and motivation; the three novelty claims with related work (Sutton & Barto cliff walking; Karpathy's REINFORCEjs gridworld)
3. Background: MDPs, TD(0), Q-learning, SARSA, Expected SARSA, exploration, on-policy vs. off-policy
4. System design: environment, solver, trainer, dashboard architecture
5. Metrics and methodology (Section 6), including the fairness rules
6. Experiments E1-E5 (and E6 if done): setup, results with confidence intervals, discussion of H1-H4
7. Dashboard walkthrough with screenshots
8. Limitations: tabular only, small grids, deterministic bias, limited seeds for UI runs
9. Future work: Dyna-Q, n-step TD, function approximation, continuous state
10. References: Sutton & Barto *Reinforcement Learning: An Introduction* (2nd ed.); Gymnasium documentation

**Viva questions to prepare**
- What is the difference between on-policy and off-policy? Which of your agents is which?
- Why does SARSA pick the safer path on the cliff while Q-learning does not? (Answer with your measured numbers.)
- Why does SARSA have the higher online return but a worse greedy optimality gap?
- What is a TD error, and what does your TD-error map show?
- Why does truncation bootstrap but termination does not?
- How did you compute the optimum, and why is it exact?
- Why lockstep training, and why 20 seeds?
- What happened when the shortcut opened, and why?
- What does Expected SARSA change, and why might its variance be lower?

**Definition of done**
- [ ] Phases 0-6 pass their acceptance criteria (Gate B)
- [ ] N1, N2 and N3 each demonstrated live in the dashboard
- [ ] E1, E2, E3, E5 run with 20 seeds, figures regenerated by `make results`
- [ ] README lets a new user run the demo and reproduce a figure
- [ ] Report complete, with honest limitations
- [ ] 5-minute demo script rehearsed; viva questions answered from your own results

---

## 15. Teammate work packages (parallel-safe, optional)

None of these block the core pipeline. They depend only on the frozen schemas from Phase 0.

1. **Preset and test grids:** design extra presets and "evil" test grids in the Grid JSON format.
2. **Experiment configs:** write the YAML files for E1-E5 from the table in Phase 9.
3. **Report background sections:** MDPs, TD methods, exploration, related work.
4. **Demo and docs:** record the demo GIF/screenshots, proofread the README.
5. **UI review:** manually run the Phase 5 and 6 checklists and report bugs.
