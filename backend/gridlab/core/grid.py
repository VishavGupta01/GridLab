from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CELL_TYPES = frozenset(".#SGHMsghm")
PRESET_DIR = Path(__file__).resolve().parents[3] / "presets" / "grids"

CELL_EMPTY = "."
CELL_WALL = "#"
CELL_START = "S"
CELL_GOAL = "G"
CELL_HAZARD = "H"
CELL_MUD = "M"


@dataclass
class Grid:
    cells: list[list[str]]
    goal_rewards: dict[tuple[int, int], float] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    version: int = 1

    @property
    def rows(self) -> int:
        return len(self.cells)

    @property
    def cols(self) -> int:
        return len(self.cells[0]) if self.cells else 0

    @property
    def start(self) -> tuple[int, int] | None:
        for r in range(self.rows):
            for c in range(self.cols):
                if self.cells[r][c] == CELL_START:
                    return (r, c)
        return None

    @property
    def goals(self) -> list[tuple[int, int]]:
        result = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.cells[r][c] == CELL_GOAL:
                    result.append((r, c))
        return result

    def get_setting(self, key: str, default: Any = None) -> Any:
        defaults = {
            "step_cost": -1.0,
            "hazard_penalty": -100.0,
            "mud_cost": -5.0,
            "goal_reward": 10.0,
            "slip_prob": 0.0,
            "max_steps": 200,
            "gamma": 0.99,
        }
        return self.settings.get(key, defaults.get(key, default))

    def cell_at(self, r: int, c: int) -> str:
        return self.cells[r][c]

    def apply_edits(self, edits: list[dict]) -> Grid:
        new_grid = self.copy()
        for edit in edits:
            r, c, to = edit["r"], edit["c"], edit["to"]
            if 0 <= r < new_grid.rows and 0 <= c < new_grid.cols:
                new_grid.cells[r][c] = to.upper()
                if to.upper() == CELL_GOAL and (r, c) not in new_grid.goal_rewards:
                    new_grid.goal_rewards[(r, c)] = new_grid.get_setting("goal_reward")
                if to.upper() != CELL_GOAL and (r, c) in new_grid.goal_rewards:
                    del new_grid.goal_rewards[(r, c)]
        new_grid.version += 1
        return new_grid

    def copy(self) -> Grid:
        return Grid(
            cells=[row[:] for row in self.cells],
            goal_rewards=dict(self.goal_rewards),
            settings=dict(self.settings),
            name=self.name,
            version=self.version,
        )

    def to_json(self) -> dict:
        goal_rewards_str = {
            f"{r},{c}": v for (r, c), v in self.goal_rewards.items()
        }
        return {
            "version": self.version,
            "name": self.name,
            "rows": ["".join(row) for row in self.cells],
            "goal_rewards": goal_rewards_str,
            "settings": self.settings,
        }

    def resize(self, new_rows: int, new_cols: int) -> Grid:
        new_rows = max(5, min(20, new_rows))
        new_cols = max(5, min(20, new_cols))
        new_cells = []
        for r in range(new_rows):
            row = []
            for c in range(new_cols):
                if r < self.rows and c < self.cols:
                    row.append(self.cells[r][c])
                else:
                    row.append(CELL_EMPTY)
            new_cells.append(row)
        new_goal_rewards = {
            (r, c): v
            for (r, c), v in self.goal_rewards.items()
            if r < new_rows and c < new_cols
        }
        return Grid(
            cells=new_cells,
            goal_rewards=new_goal_rewards,
            settings=dict(self.settings),
            name=self.name,
            version=self.version + 1,
        )


def parse_grid(data: dict) -> Grid:
    row_strings = data["rows"]
    cells = [list(row.upper()) for row in row_strings]
    if not cells:
        raise ValueError("Grid has no rows")
    col_count = len(cells[0])
    for i, row in enumerate(cells):
        if len(row) != col_count:
            raise ValueError(f"Row {i} has {len(row)} cols, expected {col_count}")
        for c_idx, ch in enumerate(row):
            if ch not in CELL_TYPES and ch not in ".#SGHM":
                raise ValueError(f"Invalid cell '{ch}' at ({i},{c_idx})")

    goal_rewards: dict[tuple[int, int], float] = {}
    for key, val in data.get("goal_rewards", {}).items():
        parts = key.split(",")
        goal_rewards[(int(parts[0]), int(parts[1]))] = float(val)

    settings = data.get("settings", {})
    name = data.get("name", "")

    grid = Grid(
        cells=cells,
        goal_rewards=goal_rewards,
        settings=settings,
        name=name,
        version=data.get("version", 1),
    )

    for r, c in grid.goals:
        if (r, c) not in grid.goal_rewards:
            grid.goal_rewards[(r, c)] = grid.get_setting("goal_reward")

    return grid


def validate_grid(grid: Grid) -> list[str]:
    issues = []
    starts = []
    goals = []
    for r in range(grid.rows):
        for c in range(grid.cols):
            ch = grid.cells[r][c]
            if ch == CELL_START:
                starts.append((r, c))
            elif ch == CELL_GOAL:
                goals.append((r, c))

    if len(starts) == 0:
        issues.append("No start cell")
    elif len(starts) > 1:
        issues.append(f"Multiple start cells: {starts}")
    if len(goals) == 0:
        issues.append("No goal cell")

    if grid.rows > 20:
        issues.append(f"Rows must be at most 20, got {grid.rows}")
    if grid.cols > 20:
        issues.append(f"Cols must be at most 20, got {grid.cols}")

    return issues


def is_reachable(grid: Grid) -> tuple[bool, set[tuple[int, int]]]:
    start = grid.start
    if start is None:
        return False, set()

    visited: set[tuple[int, int]] = set()
    queue = [start]
    visited.add(start)
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    while queue:
        r, c = queue.pop(0)
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < grid.rows
                and 0 <= nc < grid.cols
                and (nr, nc) not in visited
                and grid.cells[nr][nc] != CELL_WALL
            ):
                visited.add((nr, nc))
                queue.append((nr, nc))

    goals = grid.goals
    all_reachable = all(g in visited for g in goals)
    return all_reachable, visited


def load_preset(name: str) -> Grid:
    path = PRESET_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Preset '{name}' not found at {path}")
    with open(path) as f:
        data = json.load(f)
    return parse_grid(data)


def list_presets() -> list[str]:
    if not PRESET_DIR.exists():
        return []
    return sorted(p.stem for p in PRESET_DIR.glob("*.json"))
