"""
OptiChain AI - Reorder Point (ROP) & Economic Order Quantity (EOQ) Policy
Implements continuous review fixed-order-quantity replenishment.
When effective inventory falls to or below ROP, order a fixed lot size Q = EOQ.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Any, Dict, Tuple
import numpy as np

from optichain.baselines.base_policy import BasePolicy

if TYPE_CHECKING:
    from optichain.simulator.environment import DigitalTwinSimulator


class ROPPolicy(BasePolicy):
    """
    Classic (s, Q) Continuous Review Reorder Point Policy.
    Orders fixed Economic Order Quantity Q whenever inventory drops below ROP.
    """

    def __init__(
        self,
        name: str = "Reorder Point (ROP / EOQ) Policy",
        service_factor_z: float = 1.65,
    ):
        super().__init__(name=name)
        self.service_factor_z = service_factor_z
        # Precomputed parameters: (w_idx, p_idx) -> (rop, eoq_lot_size, supplier_id)
        self.params: Dict[Tuple[int, int], Tuple[float, int, str]] = {}

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
                L_std = float(best_sup.lead_time_std)

                # Demand during lead time + Safety Stock
                demand_during_lt = mean_d * L
                lt_demand_var = (L * (std_d**2)) + ((mean_d**2) * (L_std**2))
                safety_stock = self.service_factor_z * math.sqrt(max(0.1, lt_demand_var))
                rop = demand_during_lt + safety_stock

                # EOQ = sqrt(2 * D * K / h)
                annual_d = mean_d * 365.0
                K = best_sup.fixed_order_cost
                h = max(0.01, prod.holding_cost_rate * 365.0)
                eoq = int(round(math.sqrt((2.0 * annual_d * K) / h)))
                lot_size = max(1, eoq)

                self.params[(w_idx, p_idx)] = (rop, lot_size, best_sup.id)

    def get_actions(
        self, state: Dict[str, Any], sim: DigitalTwinSimulator
    ) -> Dict[Tuple[str, str, str], int]:
        orders: Dict[Tuple[str, str, str], int] = {}
        effective_inv = state["effective_inventory"]

        for (w_idx, p_idx), (rop, lot_size, sup_id) in self.params.items():
            if effective_inv[w_idx, p_idx] <= rop:
                wh = sim.warehouses[w_idx]
                prod = sim.products[p_idx]
                orders[(wh.id, prod.id, sup_id)] = lot_size

        return orders
