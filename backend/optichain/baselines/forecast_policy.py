"""
OptiChain AI - ML Forecast-Driven Dynamic Inventory Policy
Uses LightGBM forward demand projections across lead-time horizons to dynamically
adjust reorder points and order-up-to target levels in real time.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Any, Dict, Tuple
import numpy as np

from optichain.baselines.base_policy import BasePolicy
from optichain.forecasting.models import ForecastEngine

if TYPE_CHECKING:
    from optichain.simulator.environment import DigitalTwinSimulator


class MLForecastPolicy(BasePolicy):
    """
    Dynamic Base-Stock Policy governed by Time-Series ML forecasts.
    """

    def __init__(
        self,
        name: str = "ML-Forecast Dynamic Base-Stock",
        service_factor_z: float = 1.65,
    ):
        super().__init__(name=name)
        self.service_factor_z = service_factor_z
        self.engine = ForecastEngine()
        self.best_suppliers: Dict[Tuple[int, int], Tuple[str, int, float, float]] = {}

    def fit(self, sim: DigitalTwinSimulator) -> None:
        # Fit forecasting model on simulator network & demand
        self.engine.train_and_backtest(
            sim.demand_matrix,
            sim.products,
            sim.warehouses,
            sim.disruptions,
            train_split_ratio=0.70,
        )

        demand = sim.demand_matrix
        T, W, P = demand.shape

        for w_idx in range(W):
            for p_idx in range(P):
                prod = sim.products[p_idx]
                d_series = demand[:, w_idx, p_idx]
                mean_d = float(np.mean(d_series))

                eligible_sups = [s for s in sim.suppliers if prod.id in s.catalog]
                if not eligible_sups:
                    eligible_sups = sim.suppliers

                best_sup = min(
                    eligible_sups,
                    key=lambda s: s.catalog.get(prod.id, prod.unit_cost),
                )

                L = best_sup.base_lead_time_days
                # EOQ lot target
                annual_d = mean_d * 365.0
                K = best_sup.fixed_order_cost
                h = max(0.01, prod.holding_cost_rate * 365.0)
                eoq = math.sqrt((2.0 * annual_d * K) / h)

                self.best_suppliers[(w_idx, p_idx)] = (
                    best_sup.id,
                    L,
                    eoq,
                    mean_d,
                )

    def get_actions(
        self, state: Dict[str, Any], sim: DigitalTwinSimulator
    ) -> Dict[Tuple[str, str, str], int]:
        orders: Dict[Tuple[str, str, str], int] = {}
        t = state["current_step"]
        effective_inv = state["effective_inventory"]

        for (w_idx, p_idx), (sup_id, L, eoq, mean_d) in self.best_suppliers.items():
            wh = sim.warehouses[w_idx]
            prod = sim.products[p_idx]

            # Dynamic demand prediction for upcoming lead time window [t, t+L]
            pred_lead_time_demand = self.engine.predict_next_lead_time_demand(
                w_id=wh.id,
                p_id=prod.id,
                current_step=t,
                lead_time_days=L,
                fallback_mean=mean_d,
            )

            # Dynamic safety stock based on forecast horizon
            safety_stock = self.service_factor_z * math.sqrt(max(1.0, pred_lead_time_demand * 0.25))
            dynamic_s = pred_lead_time_demand + safety_stock
            dynamic_S = dynamic_s + eoq

            current_effective = effective_inv[w_idx, p_idx]
            if current_effective <= dynamic_s:
                order_qty = int(math.ceil(dynamic_S - current_effective))
                if order_qty > 0:
                    orders[(wh.id, prod.id, sup_id)] = order_qty

        return orders
