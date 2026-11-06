from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AGENT_COLORS = {"q": "#3b82f6", "s": "#ef4444", "e": "#22c55e"}
AGENT_LABELS = {"q": "Q-Learning", "s": "SARSA", "e": "Expected SARSA"}


def ci95(data: np.ndarray) -> float:
    return 1.96 * np.std(data, ddof=1) / np.sqrt(len(data)) if len(data) > 1 else 0.0


def plot_learning_curve(
    df: pd.DataFrame, metric: str, title: str, ylabel: str,
    ax: plt.Axes, smoothing: int = 20, events: list[int] | None = None
):
    agents = df["agent"].unique()
    for aid in sorted(agents):
        sub = df[df["agent"] == aid]
        grouped = sub.groupby("episode")[metric]
        mean = grouped.mean()
        if smoothing > 1:
            mean = mean.rolling(smoothing, min_periods=1).mean()
        se = grouped.apply(lambda x: ci95(x.values))
        if smoothing > 1:
            se = se.rolling(smoothing, min_periods=1).mean()
        episodes = mean.index

        color = AGENT_COLORS.get(aid, None)
        label = AGENT_LABELS.get(aid, aid)
        ax.plot(episodes, mean.values, label=label, color=color, linewidth=1.2)
        ax.fill_between(
            episodes,
            (mean - se).values,
            (mean + se).values,
            alpha=0.15,
            color=color,
        )

    if events:
        for ep in events:
            ax.axvline(ep, color="#94a3b8", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_eval_curve(
    df: pd.DataFrame, metric: str, title: str, ylabel: str,
    ax: plt.Axes, events: list[int] | None = None
):
    agents = df["agent"].unique()
    for aid in sorted(agents):
        sub = df[df["agent"] == aid]
        grouped = sub.groupby("episode")[metric]
        mean = grouped.mean()
        se = grouped.apply(lambda x: ci95(x.values))
        episodes = mean.index

        color = AGENT_COLORS.get(aid, None)
        label = AGENT_LABELS.get(aid, aid)
        ax.plot(episodes, mean.values, label=label, color=color, linewidth=1.2)
        ax.fill_between(
            episodes,
            (mean - se).values,
            (mean + se).values,
            alpha=0.15,
            color=color,
        )

    if events:
        for ep in events:
            ax.axvline(ep, color="#94a3b8", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_experiment(run_dir: str, output_dir: str):
    run_path = Path(run_dir)
    online = pd.read_csv(run_path / "online.csv")
    eval_df = pd.read_csv(run_path / "eval.csv")

    import json
    config_path = run_path / "config.json"
    events = []
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        for evt in cfg.get("scheduled_events", []):
            events.append(evt["at_episode"])

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(run_path.name, fontsize=13, fontweight="bold")

    plot_learning_curve(online, "return", "Return per Episode", "Return", axes[0, 0], events=events)
    plot_learning_curve(online, "steps", "Steps per Episode", "Steps", axes[0, 1], events=events)
    plot_learning_curve(online, "success", "Success Rate", "Success", axes[0, 2], events=events)

    if not eval_df.empty:
        plot_eval_curve(eval_df, "gap", "Optimality Gap", "Gap (V* - V_pi)", axes[1, 0], events=events)
        plot_eval_curve(eval_df, "gap_pct", "Gap %", "Gap %", axes[1, 1], events=events)
        plot_eval_curve(eval_df, "pct_optimal", "% Optimal Actions", "% Optimal", axes[1, 2], events=events)

    fig.tight_layout()
    fig.savefig(f"{output_dir}/{run_path.name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir}/{run_path.name}.png")

    summary = summarize_final(online, eval_df)
    summary.to_csv(f"{output_dir}/{run_path.name}_summary.csv", index=False)
    print(f"  Saved: {output_dir}/{run_path.name}_summary.csv")


def summarize_final(online: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    agents = online["agent"].unique()
    rows = []
    for aid in sorted(agents):
        sub_online = online[online["agent"] == aid]
        seed_means = []
        for seed in sub_online["seed"].unique():
            seed_data = sub_online[sub_online["seed"] == seed]
            seed_means.append(seed_data.tail(100)["return"].mean())
        seed_means = np.array(seed_means)
        label = AGENT_LABELS.get(aid, aid)

        row = {
            "agent": label,
            "avg_return_last100": f"{seed_means.mean():.1f} +/- {ci95(seed_means):.1f}",
        }

        if not eval_df.empty:
            sub_eval = eval_df[eval_df["agent"] == aid]
            if len(sub_eval) > 0:
                last_per_seed = sub_eval.loc[sub_eval.groupby("seed")["episode"].idxmax()]
                gaps = last_per_seed["gap"].values.astype(float)
                pct_opt = last_per_seed["pct_optimal"].values.astype(float)
                row["final_gap"] = f"{np.mean(gaps):.2f} +/- {ci95(gaps):.2f}"
                row["final_pct_optimal"] = f"{np.mean(pct_opt):.3f} +/- {ci95(pct_opt):.3f}"
                eto = last_per_seed["episodes_to_near_optimal"].dropna().values.astype(float)
                if len(eto) > 0:
                    row["eps_to_near_optimal"] = f"{np.mean(eto):.0f} +/- {ci95(eto):.0f}"
                else:
                    row["eps_to_near_optimal"] = "N/A"

        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("run_dirs", nargs="+", help="Run directories or 'all' for latest of each")
    parser.add_argument("--output", default="../results/figs", help="Output directory for figures")
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        print(f"Plotting: {run_dir}")
        plot_experiment(run_dir, args.output)


if __name__ == "__main__":
    main()
