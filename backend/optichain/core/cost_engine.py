"""
OptiChain AI - Financial & Cost Accounting Engine
Computes granular daily and cumulative operational and economic metrics
including holding, stockout penalties, procurement, transport, and fixed setup costs.
"""

from __future__ import annotations
from typing import Dict, List
from pydantic import BaseModel, Field


class CostBreakdown(BaseModel):
    step: int
    holding_cost: float = Field(default=0.0, ge=0.0)
    stockout_penalty: float = Field(default=0.0, ge=0.0)
    fixed_order_cost: float = Field(default=0.0, ge=0.0)
    procurement_cost: float = Field(default=0.0, ge=0.0)
    transport_cost: float = Field(default=0.0, ge=0.0)
    operating_cost: float = Field(default=0.0, ge=0.0)
    revenue: float = Field(default=0.0, ge=0.0)

    @property
    def total_cost(self) -> float:
        return (
            self.holding_cost
            + self.stockout_penalty
            + self.fixed_order_cost
            + self.procurement_cost
            + self.transport_cost
            + self.operating_cost
        )

    @property
    def net_profit(self) -> float:
        return self.revenue - self.total_cost


class CostEngine:
    """
    Manages and tallies multi-echelon supply chain financial balances.
    Maintains time-series history for telemetry and auditability.
    """

    def __init__(self):
        self.history: List[CostBreakdown] = []

    def reset(self) -> None:
        self.history.clear()

    def record_step_cost(
        self,
        step: int,
        holding_cost: float,
        stockout_penalty: float,
        fixed_order_cost: float,
        procurement_cost: float,
        transport_cost: float,
        operating_cost: float,
        revenue: float,
    ) -> CostBreakdown:
        breakdown = CostBreakdown(
            step=step,
            holding_cost=round(holding_cost, 4),
            stockout_penalty=round(stockout_penalty, 4),
            fixed_order_cost=round(fixed_order_cost, 4),
            procurement_cost=round(procurement_cost, 4),
            transport_cost=round(transport_cost, 4),
            operating_cost=round(operating_cost, 4),
            revenue=round(revenue, 4),
        )
        self.history.append(breakdown)
        return breakdown

    @property
    def cumulative_holding_cost(self) -> float:
        return sum(b.holding_cost for b in self.history)

    @property
    def cumulative_stockout_penalty(self) -> float:
        return sum(b.stockout_penalty for b in self.history)

    @property
    def cumulative_fixed_order_cost(self) -> float:
        return sum(b.fixed_order_cost for b in self.history)

    @property
    def cumulative_procurement_cost(self) -> float:
        return sum(b.procurement_cost for b in self.history)

    @property
    def cumulative_transport_cost(self) -> float:
        return sum(b.transport_cost for b in self.history)

    @property
    def cumulative_operating_cost(self) -> float:
        return sum(b.operating_cost for b in self.history)

    @property
    def cumulative_total_cost(self) -> float:
        return sum(b.total_cost for b in self.history)

    @property
    def cumulative_revenue(self) -> float:
        return sum(b.revenue for b in self.history)

    @property
    def cumulative_net_profit(self) -> float:
        return self.cumulative_revenue - self.cumulative_total_cost

    def get_summary(self) -> Dict[str, float]:
        steps = max(1, len(self.history))
        return {
            "total_cost": round(self.cumulative_total_cost, 2),
            "holding_cost": round(self.cumulative_holding_cost, 2),
            "stockout_penalty": round(self.cumulative_stockout_penalty, 2),
            "fixed_order_cost": round(self.cumulative_fixed_order_cost, 2),
            "procurement_cost": round(self.cumulative_procurement_cost, 2),
            "transport_cost": round(self.cumulative_transport_cost, 2),
            "operating_cost": round(self.cumulative_operating_cost, 2),
            "revenue": round(self.cumulative_revenue, 2),
            "net_profit": round(self.cumulative_net_profit, 2),
            "mean_daily_cost": round(self.cumulative_total_cost / steps, 2),
            "mean_daily_revenue": round(self.cumulative_revenue / steps, 2),
            "mean_daily_profit": round(self.cumulative_net_profit / steps, 2),
        }
