from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gridlab.core.agents.registry import create_agent
from gridlab.core.env import GridWorldEnv
from gridlab.core.grid import load_preset, parse_grid
from gridlab.core.trainer import LockstepTrainer, ScheduledEvent, TrainerConfig


def run_single_seed(
    grid_data: dict,
    agent_configs: list[dict],
    trainer_config: TrainerConfig,
    scheduled_events: list[dict],
    seed: int,
) -> dict:
    if "preset" in grid_data:
        grid = load_preset(grid_data["preset"])
    else:
        grid = parse_grid(grid_data)

    env = GridWorldEnv(grid)
    n_states = env.n_states

    agents = []
    for cfg in agent_configs:
        cfg_copy = dict(cfg)
        cfg_copy["gamma"] = trainer_config.__dict__.get("gamma", 0.99)
        if "gamma" not in cfg_copy:
            cfg_copy["gamma"] = grid.get_setting("gamma")
        agent = create_agent(cfg_copy, n_states)
        agents.append(agent)

    config = TrainerConfig(
        budget=trainer_config.budget,
        eval_every=trainer_config.eval_every,
        master_seed=seed,
    )

    trainer = LockstepTrainer(grid, agents, config)

    for evt in scheduled_events:
        trainer.schedule_event(evt["at_episode"], evt["edits"])

    trainer.run()

    return {
        "seed": seed,
        "online": trainer.metrics.online_history,
        "eval": trainer.metrics.eval_history,
        "events": trainer.event_log,
    }


def results_to_dataframes(all_results: list[dict], agent_configs: list[dict]) -> dict[str, pd.DataFrame]:
    agent_ids = [c["id"] for c in agent_configs]

    online_rows = []
    for result in all_results:
        seed = result["seed"]
        for aid in agent_ids:
            for ep, om in enumerate(result["online"][aid]):
                online_rows.append({
                    "seed": seed,
                    "agent": aid,
                    "episode": ep,
                    "steps": om.steps,
                    "return": om.total_return,
                    "success": om.success,
                    "hazard_hits": om.hazard_hits,
                    "eps": om.eps,
                    "td": om.mean_abs_td_error,
                })

    eval_rows = []
    for result in all_results:
        seed = result["seed"]
        for aid in agent_ids:
            for ep, em in result["eval"][aid]:
                eval_rows.append({
                    "seed": seed,
                    "agent": aid,
                    "episode": ep,
                    "gap": em.gap,
                    "gap_pct": em.gap_pct,
                    "path_ratio": em.path_ratio,
                    "pct_optimal": em.pct_optimal,
                    "episodes_to_near_optimal": em.episodes_to_near_optimal,
                })

    return {
        "online": pd.DataFrame(online_rows),
        "eval": pd.DataFrame(eval_rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Run a GridLab experiment")
    parser.add_argument("config", help="YAML experiment config file")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        exp = yaml.safe_load(f)

    grid_data = exp["grid"]
    agent_configs = exp["agents"]
    budget = exp.get("budget", 2000)
    eval_every = exp.get("eval_every", 10)
    scheduled_events = exp.get("scheduled_events", [])

    trainer_config = TrainerConfig(budget=budget, eval_every=eval_every)

    output_dir = args.output
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = Path(args.config).stem
        output_dir = str(Path("../results/runs") / f"{exp_name}_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Running {args.seeds} seeds for {Path(args.config).stem}...")
    all_results = []
    for seed in range(args.seeds):
        print(f"  Seed {seed}/{args.seeds}", end="\r")
        result = run_single_seed(grid_data, agent_configs, trainer_config, scheduled_events, seed)
        all_results.append(result)

    print(f"\nDone. Saving results to {output_dir}")

    dfs = results_to_dataframes(all_results, agent_configs)
    dfs["online"].to_csv(f"{output_dir}/online.csv", index=False)
    dfs["eval"].to_csv(f"{output_dir}/eval.csv", index=False)

    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(exp, f, indent=2)

    print(f"Saved: online.csv, eval.csv, config.json")


if __name__ == "__main__":
    main()
