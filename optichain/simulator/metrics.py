"""
OptiChain AI - Performance & Supply Chain KPI Metrics Engine
Computes industry-standard operational KPIs: Fill Rate (OTIF), Service Level (alpha/beta),
Inventory Turnover, Stockout Frequency, and Financial Returns.
"""

from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
from pydantic import BaseModel, Field


class EpisodeKPIs(BaseModel):
    total_demand: int = Field(default=0)
    total_fulfilled: int = Field(default=0)
    total_unmet_demand: int = Field(default=0)
    fill_rate_percent: float = Field(default=100.0)
    alpha_service_level_percent: float = Field(
        default=100.0, description="Percentage of days without any stockout"
    )
    stockout_incident_count: int = Field(default=0)
    average_on_hand_inventory: float = Field(default=0.0)
    inventory_turnover: float = Field(default=0.0)
    total_cost: float = Field(default=0.0)
    total_revenue: float = Field(default=0.0)
    net_profit: float = Field(default=0.0)
    cost_breakdown: Dict[str, float] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class MetricsTracker:
    """
    Collects step-by-step telemetry and aggregates episode KPIs.
    """

    def __init__(self):
        self.daily_demand: List[int] = []
        self.daily_fulfilled: List[int] = []
        self.daily_unmet: List[int] = []
        self.daily_inventory_units: List[float] = []
        self.daily_inventory_value: List[float] = []
        self.daily_cogs: List[float] = []

    def reset(self) -> None:
        self.daily_demand.clear()
        self.daily_fulfilled.clear()
        self.daily_unmet.clear()
        self.daily_inventory_units.clear()
        self.daily_inventory_value.clear()
        self.daily_cogs.clear()

    def record_step(
        self,
        demand: int,
        fulfilled: int,
        unmet: int,
        on_hand_units: float,
        on_hand_value: float,
        cogs: float,
    ) -> None:
        self.daily_demand.append(int(demand))
        self.daily_fulfilled.append(int(fulfilled))
        self.daily_unmet.append(int(unmet))
        self.daily_inventory_units.append(float(on_hand_units))
        self.daily_inventory_value.append(float(on_hand_value))
        self.daily_cogs.append(float(cogs))

    def compute_kpis(self, financial_summary: Dict[str, float]) -> EpisodeKPIs:
        total_demand = sum(self.daily_demand)
        total_fulfilled = sum(self.daily_fulfilled)
        total_unmet = sum(self.daily_unmet)

        fill_rate = (
            (total_fulfilled / total_demand * 100.0)
            if total_demand > 0
            else 100.0
        )

        n_days = max(1, len(self.daily_unmet))
        no_stockout_days = sum(1 for unmet in self.daily_unmet if unmet == 0)
        alpha_service_level = (no_stockout_days / n_days) * 100.0
        stockout_incidents = sum(1 for unmet in self.daily_unmet if unmet > 0)

        avg_inventory_units = (
            float(np.mean(self.daily_inventory_units))
            if self.daily_inventory_units
            else 0.0
        )
        avg_inventory_val = (
            float(np.mean(self.daily_inventory_value))
            if self.daily_inventory_value
            else 1.0
        )
        total_cogs = sum(self.daily_cogs)
        inventory_turnover = (
            total_cogs / avg_inventory_val if avg_inventory_val > 0 else 0.0
        )

        return EpisodeKPIs(
            total_demand=total_demand,
            total_fulfilled=total_fulfilled,
            total_unmet_demand=total_unmet,
            fill_rate_percent=round(fill_rate, 2),
            alpha_service_level_percent=round(alpha_service_level, 2),
            stockout_incident_count=stockout_incidents,
            average_on_hand_inventory=round(avg_inventory_units, 2),
            inventory_turnover=round(inventory_turnover, 2),
            total_cost=financial_summary.get("total_cost", 0.0),
            total_revenue=financial_summary.get("revenue", 0.0),
            net_profit=financial_summary.get("net_profit", 0.0),
            cost_breakdown=financial_summary,
        )
