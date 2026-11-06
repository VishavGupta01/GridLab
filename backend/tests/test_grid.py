import pytest

from gridlab.core.grid import (
    CELL_EMPTY,
    CELL_GOAL,
    CELL_HAZARD,
    CELL_MUD,
    CELL_START,
    CELL_WALL,
    Grid,
    is_reachable,
    list_presets,
    load_preset,
    parse_grid,
    validate_grid,
)


def _make_grid(rows: list[str], **kwargs) -> Grid:
    return parse_grid({"rows": rows, **kwargs})


class TestParseGrid:
    def test_basic_parse(self):
        g = _make_grid(["S....", "....G"])
        assert g.rows == 2
        assert g.cols == 5
        assert g.start == (0, 0)
        assert g.goals == [(1, 4)]

    def test_all_cell_types(self):
        g = _make_grid(["S.#HM", "....G"])
        assert g.cell_at(0, 0) == CELL_START
        assert g.cell_at(0, 1) == CELL_EMPTY
        assert g.cell_at(0, 2) == CELL_WALL
        assert g.cell_at(0, 3) == CELL_HAZARD
        assert g.cell_at(0, 4) == CELL_MUD
        assert g.cell_at(1, 4) == CELL_GOAL

    def test_lowercase_normalized(self):
        g = _make_grid(["s...g"])
        assert g.cell_at(0, 0) == CELL_START
        assert g.cell_at(0, 4) == CELL_GOAL

    def test_uneven_rows_raises(self):
        with pytest.raises(ValueError, match="cols"):
            _make_grid(["S..", "..G."])

    def test_empty_grid_raises(self):
        with pytest.raises(ValueError):
            _make_grid([])


class TestValidateGrid:
    def test_valid_grid(self):
        g = _make_grid(["S....", "....G"])
        assert validate_grid(g) == []

    def test_no_start(self):
        g = _make_grid([".....", "....G"])
        issues = validate_grid(g)
        assert any("start" in i.lower() for i in issues)

    def test_no_goal(self):
        g = _make_grid(["S....", "....."])
        issues = validate_grid(g)
        assert any("goal" in i.lower() for i in issues)

    def test_multiple_starts(self):
        g = _make_grid(["S...S", "....G"])
        issues = validate_grid(g)
        assert any("multiple" in i.lower() or "start" in i.lower() for i in issues)

    def test_multiple_goals_ok(self):
        g = _make_grid(["S...G", "....G"])
        assert validate_grid(g) == []


class TestReachability:
    def test_reachable(self):
        g = _make_grid(["S....", "....G"])
        reachable, visited = is_reachable(g)
        assert reachable
        assert (1, 4) in visited

    def test_blocked(self):
        g = _make_grid(["S....", "#####", "....G"])
        reachable, visited = is_reachable(g)
        assert not reachable

    def test_no_start(self):
        g = _make_grid([".....", "....G"])
        reachable, visited = is_reachable(g)
        assert not reachable


class TestGridEdits:
    def test_apply_edit(self):
        g = _make_grid(["S....", "....G"])
        g2 = g.apply_edits([{"r": 0, "c": 2, "to": "#"}])
        assert g2.cell_at(0, 2) == CELL_WALL
        assert g2.version == g.version + 1
        assert g.cell_at(0, 2) == CELL_EMPTY

    def test_add_goal(self):
        g = _make_grid(["S....", "....G"])
        g2 = g.apply_edits([{"r": 0, "c": 4, "to": "G"}])
        assert g2.cell_at(0, 4) == CELL_GOAL
        assert (0, 4) in g2.goal_rewards

    def test_remove_goal(self):
        g = _make_grid(["S....", "....G"])
        g2 = g.apply_edits([{"r": 1, "c": 4, "to": "."}])
        assert (1, 4) not in g2.goal_rewards


class TestSerialization:
    def test_round_trip(self):
        g = _make_grid(["S.#..", "..H.G"], goal_rewards={"1,4": 10})
        data = g.to_json()
        g2 = parse_grid(data)
        assert g2.rows == g.rows
        assert g2.cols == g.cols
        assert g2.cells == g.cells
        assert g2.goal_rewards == g.goal_rewards


class TestResize:
    def test_grow(self):
        g = _make_grid(["S....", "....G"])
        g2 = g.resize(6, 6)
        assert g2.rows == 6
        assert g2.cols == 6
        assert g2.cell_at(0, 0) == CELL_START

    def test_shrink(self):
        g = _make_grid(["S.........", ".........G"])
        g2 = g.resize(5, 5)
        assert g2.rows == 5
        assert g2.cols == 5

    def test_clamp_min(self):
        g = _make_grid(["S....", "....G"])
        g2 = g.resize(2, 2)
        assert g2.rows == 5
        assert g2.cols == 5


class TestPresets:
    def test_list_presets(self):
        presets = list_presets()
        assert "cliff_4x12" in presets
        assert "open_10x10" in presets

    def test_load_all_presets(self):
        for name in list_presets():
            g = load_preset(name)
            issues = validate_grid(g)
            if name != "blocked":
                assert issues == [], f"Preset {name} has issues: {issues}"

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_preset("nonexistent")
