"""
OptiChain AI - Periodic Review (R, S) Inventory Policy
Inspects inventory position at fixed review intervals of R days.
At each review epoch, orders up to base stock level S to cover demand over (R + L).
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Any, Dict, Tuple
import numpy as np

from optichain.baselines.base_policy import BasePolicy

if TYPE_CHECKING:
    from optichain.simulator.environment import DigitalTwinSimulator


class PeriodicReviewPolicy(BasePolicy):
    """
    Fixed-interval (R, S) Periodic Review replenishment policy.
    Orders only on review days (e.g., every 7 days) up to target S.
    """

    def __init__(
        self,
        name: str = "Periodic Review (R, S) Policy",
        review_period_days: int = 7,
        service_factor_z: float = 1.65,
    ):
        super().__init__(name=name)
        self.review_period = review_period_days
        self.service_factor_z = service_factor_z
        # Precomputed parameters: (w_idx, p_idx) -> (target_S, supplier_id)
        self.params: Dict[Tuple[int, int], Tuple[float, str]] = {}

    def fit(self, sim: DigitalTwinSimulator) -> None:
        self.params.clear()
        demand = sim.demand_matrix
        T, W, P = demand.shape

        for w_idx in range(W):
            for p_idx in range(P):
                prod = sim.products[p_idx]
                d_series = demand[:, w_idx, p_idx]
                mean_d = float(np.mean(d_series))
                std_d = float(np.std(d_series)) if len(d_series) > 1 else 1.0

                eligible_sups = [s for s in sim.suppliers if prod.id in s.catalog]
                if not eligible_sups:
                    eligible_sups = sim.suppliers

                best_sup = min(
                    eligible_sups,
                    key=lambda s: s.catalog.get(prod.id, prod.unit_cost),
                )

                L = float(best_sup.base_lead_time_days)
                # Demand over protection interval (R + L)
                protection_interval = self.review_period + L
                interval_demand = mean_d * protection_interval
                interval_std = std_d * math.sqrt(protection_interval)

                safety_stock = self.service_factor_z * interval_std
                S = interval_demand + safety_stock

                self.params[(w_idx, p_idx)] = (S, best_sup.id)

    def get_actions(
        self, state: Dict[str, Any], sim: DigitalTwinSimulator
    ) -> Dict[Tuple[str, str, str], int]:
        orders: Dict[Tuple[str, str, str], int] = {}
        step = state["current_step"]

        # Only place replenishment orders on review epochs
        if step % self.review_period != 0:
            return orders

        effective_inv = state["effective_inventory"]

        for (w_idx, p_idx), (S, sup_id) in self.params.items():
            current_effective = effective_inv[w_idx, p_idx]
            if current_effective < S:
                order_qty = int(math.ceil(S - current_effective))
                if order_qty > 0:
                    wh = sim.warehouses[w_idx]
                    prod = sim.products[p_idx]
                    orders[(wh.id, prod.id, sup_id)] = order_qty

        return orders
