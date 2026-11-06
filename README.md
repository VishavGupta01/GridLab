# GridLab

A web dashboard for racing TD-learning agents (Q-learning, SARSA, Expected SARSA) on custom grid worlds, comparing them to the mathematically exact optimum, with live grid editing and explainability.

**UCS760 Reinforcement Learning Course Project**

## Features

- **Grid Editor** — Draw grid worlds with walls, hazards, mud, goals; 7 presets included
- **3 TD Agents** — Q-learning, SARSA, Expected SARSA training in lockstep
- **Exact Optimum (N2)** — BFS + Value Iteration + linear-system policy evaluation; compare agent performance to V\*/Q\*
- **Live Edits (N1)** — Edit the grid mid-training; agents adapt in real time; adaptation metrics tracked
- **Update Inspector (N3)** — Step through TD updates; compare how all 3 agents handle the same transition
- **6 Overlays** — Q-value heatmap, policy arrows, visit counts, TD-error, optimal V\*, disagreement
- **Learning Curves** — Steps, return, gap, success rate, epsilon, % optimal actions with EMA smoothing
- **Race View** — Animate greedy rollouts side by side
- **Experiments** — 8 experiments (E1-E6) with 20 seeds, mean ± 95% CI

## Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, NumPy, Gymnasium |
| Frontend | React, TypeScript, Vite, Tailwind CSS, Zustand, Recharts |
| Communication | REST + WebSocket |
| Experiments | YAML configs, pandas, matplotlib |

## Quick Start

```bash
# Clone
git clone https://github.com/VishavGupta01/GridLab.git
cd GridLab

# Backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e backend/

# Frontend
cd frontend && npm install && cd ..

# Run (two terminals)
make dev-backend    # Terminal 1: http://localhost:8000
make dev-frontend   # Terminal 2: http://localhost:5173
```

### Production Build

```bash
make demo   # Builds frontend, serves everything from FastAPI on :8000
```

## Experiments

Run all experiments with 20 seeds:

```bash
make results
```

This runs E1-E6 and generates figures in `results/figs/`.

| ID | Experiment | Grid | Key Variable |
|---|---|---|---|
| E1 | Cliff Walking contrast | cliff_4x12 | 3 algos, eps=0.1 |
| E2 | Learning rate sensitivity | cliff_4x12 | alpha ∈ {0.05, 0.1, 0.3, 0.5, 1.0} |
| E3 | Exploration strategy | maze_10x10 | fixed ε vs decaying ε vs softmax |
| E4 | Map difficulty | open/maze/two_goals | per-map convergence |
| E5 | Adaptation | shortcut | wall removed at ep 500, ±ε-bump |
| E6 | Slippery transitions | cliff_4x12 | slip_prob=0.1 |

## Architecture

```
GridLab/
├── backend/gridlab/
│   ├── core/           # Grid, Env, Solver, Agents, Trainer, Metrics
│   ├── api/            # FastAPI endpoints, sessions, WebSocket
│   └── cli/            # Experiment runner, plotter
├── frontend/src/
│   ├── components/     # GridCanvas, Charts, RunBar, Toolbox, Inspector, ...
│   ├── store/          # Zustand stores (grid, session, UI)
│   └── api/            # REST + WS client
├── presets/grids/      # 7 preset grid JSONs
├── experiments/        # YAML experiment configs
└── results/            # Generated CSVs and figures
```

## Tests

```bash
make test   # 90+ backend tests
```

## Key Results

- **Cliff Walking (E1)**: Q-learning achieves ~97% optimal actions (learns cliff-edge path); SARSA ~80% (learns safe path). Classic on-policy vs off-policy contrast.
- **Adaptation (E5)**: ε-bump agents discover new shortcuts faster after grid edits; without bump, decayed-ε agents may never find them.
- **Explainability (N3)**: On the same cliff-edge transition, Q-learning bootstraps from the best next action, SARSA from the (possibly exploratory) action taken — visible difference in the Inspector.
