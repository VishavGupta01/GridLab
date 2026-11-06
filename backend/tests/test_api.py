import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gridlab.api.main import app, session_manager

client = TestClient(app)

CLIFF_GRID = {
    "version": 1,
    "name": "cliff_4x12",
    "rows": ["............", "............", "............", "SHHHHHHHHHHG"],
    "goal_rewards": {"3,11": 10.0},
    "settings": {"step_cost": -1, "hazard_penalty": -100, "mud_cost": -5,
                 "goal_reward": 10, "slip_prob": 0.0, "max_steps": 200, "gamma": 0.99},
}

AGENTS = [
    {"id": "q", "algo": "q_learning", "alpha": 0.1,
     "exploration": {"type": "epsilon", "eps": 0.1}, "q_init": 0.0, "eps_bump_on_change": False, "eps_bump_value": 0.3},
    {"id": "s", "algo": "sarsa", "alpha": 0.1,
     "exploration": {"type": "epsilon", "eps": 0.1}, "q_init": 0.0, "eps_bump_on_change": False, "eps_bump_value": 0.3},
    {"id": "e", "algo": "expected_sarsa", "alpha": 0.1,
     "exploration": {"type": "epsilon", "eps": 0.1}, "q_init": 0.0, "eps_bump_on_change": False, "eps_bump_value": 0.3},
]


class TestHealth:
    def test_health(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestPresets:
    def test_list_presets(self):
        resp = client.get("/api/presets")
        assert resp.status_code == 200
        presets = resp.json()["presets"]
        assert "cliff_4x12" in presets

    def test_get_preset(self):
        resp = client.get("/api/presets/cliff_4x12")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "cliff_4x12"

    def test_preset_not_found(self):
        resp = client.get("/api/presets/nonexistent")
        assert resp.status_code == 404


class TestValidation:
    def test_valid_grid(self):
        resp = client.post("/api/grids/validate", json=CLIFF_GRID)
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"]
        assert data["solvable"]
        assert data["v_star_s0"] is not None

    def test_invalid_grid(self):
        bad = dict(CLIFF_GRID, rows=[".....", "#####", "....G"])
        resp = client.post("/api/grids/validate", json=bad)
        data = resp.json()
        assert data["valid"] is False or not data["solvable"]


class TestSessionLifecycle:
    def test_create_and_snapshot(self):
        resp = client.post("/api/sessions", json={
            "grid": CLIFF_GRID,
            "agents": AGENTS,
            "master_seed": 42,
            "budget": 100,
            "eval_every": 10,
        })
        assert resp.status_code == 200
        sid = resp.json()["session_id"]

        resp = client.get(f"/api/sessions/{sid}/snapshot")
        assert resp.status_code == 200
        snap = resp.json()
        assert "agents" in snap
        assert "optimal" in snap

    def test_control(self):
        resp = client.post("/api/sessions", json={
            "grid": CLIFF_GRID,
            "agents": AGENTS,
            "master_seed": 42,
            "budget": 50,
        })
        sid = resp.json()["session_id"]

        resp = client.post(f"/api/sessions/{sid}/control", json={"action": "start"})
        assert resp.status_code == 200

        time.sleep(0.5)

        resp = client.post(f"/api/sessions/{sid}/control", json={"action": "pause"})
        assert resp.status_code == 200

        resp = client.post(f"/api/sessions/{sid}/control", json={"action": "resume"})
        assert resp.status_code == 200

    def test_edit(self):
        resp = client.post("/api/sessions", json={
            "grid": CLIFF_GRID,
            "agents": AGENTS,
            "master_seed": 42,
            "budget": 50,
        })
        sid = resp.json()["session_id"]

        resp = client.post(f"/api/sessions/{sid}/edit", json={
            "edits": [{"r": 0, "c": 0, "to": "#"}],
        })
        assert resp.status_code == 200


class TestRolloutAndExplain:
    def _create_trained_session(self):
        resp = client.post("/api/sessions", json={
            "grid": CLIFF_GRID,
            "agents": AGENTS,
            "master_seed": 42,
            "budget": 50,
        })
        sid = resp.json()["session_id"]
        session = session_manager.get(sid)
        session.trainer.run(50)
        return sid, session

    def test_rollout(self):
        sid, _ = self._create_trained_session()
        resp = client.post(f"/api/sessions/{sid}/rollout", json={
            "agent_id": "q",
            "greedy": True,
            "seed": 0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "steps" in data
        assert len(data["steps"]) > 0

    def test_explain_no_mutation(self):
        sid, session = self._create_trained_session()
        q_before = session.agents[0].q_table().copy()

        resp = client.post(f"/api/sessions/{sid}/explain", json={
            "agent_id": "q",
            "s": 0, "a": 1, "r": -1.0, "s2": 1,
            "terminated": False,
        })
        assert resp.status_code == 200

        q_after = session.agents[0].q_table()
        np.testing.assert_array_equal(q_before, q_after)

    def test_export(self):
        sid, _ = self._create_trained_session()
        resp = client.get(f"/api/sessions/{sid}/export")
        assert resp.status_code == 200
        data = resp.json()
        assert "config" in data
        assert "online" in data
        assert "eval" in data
