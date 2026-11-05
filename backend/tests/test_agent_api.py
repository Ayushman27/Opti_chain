"""
Unit and integration tests for FastAPI backend and reasoning agent.
"""

from fastapi.testclient import TestClient
from optichain.agent.service import app
from optichain.agent.reasoning_agent import SupplyChainReasoningAgent


client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_config_endpoint():
    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.json()
    assert "n_warehouses" in data
    assert "n_products" in data


def test_simulation_endpoint():
    payload = {
        "policy": "min_max",
        "n_warehouses": 1,
        "n_products": 2,
        "n_suppliers": 1,
        "episode_days": 5,
        "random_seed": 42,
    }
    res = client.post("/api/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "kpis" in data
    assert "step_history" in data
    assert len(data["step_history"]) == 5
    assert "cost_breakdown" in data


def test_agent_chat_endpoint():
    payload = {
        "prompt": "Why did we experience disruptions and stockouts?",
        "context_kpis": {"fill_rate_percent": 88.0, "net_profit": 50000.0},
    }
    res = client.post("/api/agent/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "response" in data
    assert "suggested_actions" in data


def test_reasoning_agent_disruption_diagnostics():
    agent = SupplyChainReasoningAgent()
    step_history = [
        {"step": 0, "unmet": 0, "fill_rate": 100.0},
        {"step": 10, "unmet": 50, "fill_rate": 60.0},
    ]
    disruptions = [
        {
            "id": "D1",
            "name": "Port Congestion",
            "type": "PORT_CONGESTION",
            "start_step": 8,
            "duration_steps": 5,
        }
    ]
    findings = agent.diagnose_disruptions(step_history, disruptions)
    assert len(findings) == 1
    assert findings[0]["step"] == 10
    assert "Port Congestion" in findings[0]["root_cause"]
