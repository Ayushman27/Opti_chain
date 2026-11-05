"""
Unit tests for OptiChain AI cost engine and financial accounting.
"""

from optichain.core.cost_engine import CostEngine, CostBreakdown


def test_cost_breakdown_properties():
    bd = CostBreakdown(
        step=0,
        holding_cost=50.0,
        stockout_penalty=100.0,
        fixed_order_cost=20.0,
        procurement_cost=200.0,
        transport_cost=30.0,
        operating_cost=500.0,
        revenue=1200.0,
    )
    assert bd.total_cost == (50 + 100 + 20 + 200 + 30 + 500)
    assert bd.net_profit == 1200.0 - 900.0


def test_cost_engine_accumulation():
    engine = CostEngine()
    engine.record_step_cost(
        step=0,
        holding_cost=10.0,
        stockout_penalty=0.0,
        fixed_order_cost=50.0,
        procurement_cost=100.0,
        transport_cost=10.0,
        operating_cost=100.0,
        revenue=300.0,
    )
    engine.record_step_cost(
        step=1,
        holding_cost=15.0,
        stockout_penalty=20.0,
        fixed_order_cost=0.0,
        procurement_cost=0.0,
        transport_cost=0.0,
        operating_cost=100.0,
        revenue=250.0,
    )

    summary = engine.get_summary()
    assert summary["holding_cost"] == 25.0
    assert summary["stockout_penalty"] == 20.0
    assert summary["revenue"] == 550.0
    assert summary["total_cost"] == (10 + 50 + 100 + 10 + 100 + 15 + 20 + 100)
    assert summary["net_profit"] == summary["revenue"] - summary["total_cost"]


def test_cost_engine_reset():
    engine = CostEngine()
    engine.record_step_cost(
        step=0,
        holding_cost=10.0,
        stockout_penalty=0.0,
        fixed_order_cost=0.0,
        procurement_cost=0.0,
        transport_cost=0.0,
        operating_cost=0.0,
        revenue=50.0,
    )
    assert len(engine.history) == 1
    engine.reset()
    assert len(engine.history) == 0
    assert engine.cumulative_total_cost == 0.0
