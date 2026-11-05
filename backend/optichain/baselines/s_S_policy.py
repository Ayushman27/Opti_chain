"""
OptiChain AI - Min-Max (s, S) Inventory Replenishment Policy
Implements Scarf's classic (s, S) inventory control rule:
When effective inventory (on-hand + in-transit - backlog) drops below reorder point s,
order up to base-stock target level S.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Any, Dict, Tuple
import numpy as np

from optichain.baselines.base_policy import BasePolicy

if TYPE_CHECKING:
    from optichain.simulator.environment import DigitalTwinSimulator


class MinMaxPolicy(BasePolicy):
    """
    Continuous review (s, S) policy calibrated using service-level safety factor
    and Economic Order Quantity target levels.
    """

    def __init__(
        self,
        name: str = "Min-Max (s, S) Policy",
        service_factor_z: float = 1.65,  # ~95% cycle service level
        eoq_multiplier: float = 1.0,
    ):
        super().__init__(name=name)
        self.service_factor_z = service_factor_z
        self.eoq_multiplier = eoq_multiplier
        # Precomputed parameters: (w_idx, p_idx) -> (s, S, primary_supplier_id)
        self.params: Dict[Tuple[int, int], Tuple[float, float, str]] = {}

    def fit(self, sim: DigitalTwinSimulator) -> None:
        """
        Calibrates reorder point s and order-up-to level S for each warehouse-product pair.
        """
        self.params.clear()
        demand = sim.demand_matrix  # Shape: (T, W, P)
        T, W, P = demand.shape

        for w_idx, wh in enumerate(sim.warehouses):
            for p_idx, prod in enumerate(sim.products):
                # Historical demand statistics for this item & location
                d_series = demand[:, w_idx, p_idx]
                mean_d = float(np.mean(d_series))
                std_d = float(np.std(d_series)) if len(d_series) > 1 else 1.0

                # Select best supplier for this product (cheapest or shortest lead time)
                eligible_sups = [
                    s for s in sim.suppliers if prod.id in s.catalog
                ]
                if not eligible_sups:
                    eligible_sups = sim.suppliers

                # Pick supplier with lowest combined procurement + setup amortized cost
                best_sup = min(
                    eligible_sups,
                    key=lambda s: s.catalog.get(prod.id, prod.unit_cost)
                    + (s.fixed_order_cost / max(10, mean_d * 10)),
                )

                L = float(best_sup.base_lead_time_days)
                L_std = float(best_sup.lead_time_std)

                # Combined lead-time demand variance: Var(D_L) = L * Var(D) + D^2 * Var(L)
                demand_during_lt = mean_d * L
                lt_demand_var = (L * (std_d**2)) + ((mean_d**2) * (L_std**2))
                safety_stock = self.service_factor_z * math.sqrt(max(0.1, lt_demand_var))

                # Reorder point s
                s = demand_during_lt + safety_stock

                # Economic Order Quantity EOQ = sqrt(2 * D * K / h)
                annual_d = mean_d * 365.0
                K = best_sup.fixed_order_cost
                h = max(0.01, prod.holding_cost_rate * 365.0)
                eoq = math.sqrt((2.0 * annual_d * K) / h) * self.eoq_multiplier

                # Order-up-to target S = s + EOQ
                S = s + max(prod.unit_cost * 0.1, eoq)

                self.params[(w_idx, p_idx)] = (s, S, best_sup.id)

    def get_actions(
        self, state: Dict[str, Any], sim: DigitalTwinSimulator
    ) -> Dict[Tuple[str, str, str], int]:
        orders: Dict[Tuple[str, str, str], int] = {}
        effective_inv = state["effective_inventory"]  # Shape (W, P)

        for (w_idx, p_idx), (s, S, sup_id) in self.params.items():
            current_effective = effective_inv[w_idx, p_idx]
            if current_effective <= s:
                order_qty = int(math.ceil(S - current_effective))
                if order_qty > 0:
                    wh = sim.warehouses[w_idx]
                    prod = sim.products[p_idx]
                    orders[(wh.id, prod.id, sup_id)] = order_qty

        return orders
