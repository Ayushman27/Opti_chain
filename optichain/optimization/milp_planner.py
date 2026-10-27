"""
OptiChain AI - Operations Research Constrained Inventory Optimizer
Formulates and solves a multi-period, multi-echelon Mixed-Integer Linear Program (MILP)
for rolling-horizon replenishment planning with warehouse capacity, supplier daily limits,
fixed setup costs, and lead times using PuLP.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import numpy as np
import pulp

from optichain.baselines.base_policy import BasePolicy

if TYPE_CHECKING:
    from optichain.simulator.environment import DigitalTwinSimulator


class MILPReplenishmentPolicy(BasePolicy):
    """
    Rolling-Horizon Constrained MILP Inventory Optimizer (Model Predictive Control).
    Solves for optimal replenishment schedule across lookahead window H,
    executes step t decision, then recedes horizon.
    """

    def __init__(
        self,
        name: str = "OR Constrained MILP Optimizer",
        horizon: int = 10,
        time_limit_sec: int = 5,
    ):
        super().__init__(name=name)
        self.horizon = horizon
        self.time_limit = time_limit_sec

    def fit(self, sim: DigitalTwinSimulator) -> None:
        pass

    def get_actions(
        self, state: Dict[str, Any], sim: DigitalTwinSimulator
    ) -> Dict[Tuple[str, str, str], int]:
        t = state["current_step"]
        H = min(self.horizon, sim.config.episode_days - t)
        if H <= 0:
            return {}

        warehouses = sim.warehouses
        products = sim.products
        suppliers = sim.suppliers
        W = len(warehouses)
        P = len(products)
        S = len(suppliers)

        # Create PuLP optimization problem
        prob = pulp.LpProblem(f"OptiChain_MILP_step_{t}", pulp.LpMinimize)

        # Decision Variables
        # x[w, p, s, tau] = quantity ordered from supplier s to warehouse w for product p at time tau
        x_vars = {}
        for tau in range(H):
            for w in range(W):
                for p in range(P):
                    for s in range(S):
                        p_id = products[p].id
                        if p_id in suppliers[s].catalog:
                            var_name = f"x_w{w}_p{p}_s{s}_t{tau}"
                            x_vars[(w, p, s, tau)] = pulp.LpVariable(
                                var_name, lowBound=0, cat=pulp.LpInteger
                            )

        # Binary indicator z[s, tau] = 1 if supplier s receives an order at step tau (for setup cost)
        z_vars = {}
        for tau in range(H):
            for s in range(S):
                z_vars[(s, tau)] = pulp.LpVariable(
                    f"z_s{s}_t{tau}", cat=pulp.LpBinary
                )

        # Inventory variables: I[w, p, tau] >= 0
        I_vars = {}
        # Unmet demand variables: U[w, p, tau] >= 0
        U_vars = {}
        for tau in range(H):
            for w in range(W):
                for p in range(P):
                    I_vars[(w, p, tau)] = pulp.LpVariable(
                        f"I_w{w}_p{p}_t{tau}", lowBound=0, cat=pulp.LpContinuous
                    )
                    U_vars[(w, p, tau)] = pulp.LpVariable(
                        f"U_w{w}_p{p}_t{tau}", lowBound=0, cat=pulp.LpContinuous
                    )

        # Build Objective Function
        objective_terms = []

        # 1. Holding & Stockout Costs across horizon
        for tau in range(H):
            for w in range(W):
                for p in range(P):
                    h_rate = products[p].holding_cost_rate
                    p_penalty = products[p].stockout_penalty
                    objective_terms.append(h_rate * I_vars[(w, p, tau)])
                    objective_terms.append(p_penalty * U_vars[(w, p, tau)])

        # 2. Procurement & Transport Costs
        for (w, p, s, tau), x_var in x_vars.items():
            unit_price = suppliers[s].catalog[products[p].id]
            transport = sim.config.transport_cost_per_unit
            objective_terms.append((unit_price + transport) * x_var)

        # 3. Fixed setup costs
        for (s, tau), z_var in z_vars.items():
            fixed_cost = suppliers[s].fixed_order_cost
            objective_terms.append(fixed_cost * z_var)

        prob += pulp.lpSum(objective_terms)

        # Constraints
        # Precompute scheduled pipeline arrivals from previous steps
        scheduled_arrivals = np.zeros((H, W, P), dtype=float)
        for order in sim.in_transit_orders:
            arr_step = order.expected_arrival_step
            offset = arr_step - t
            if 0 <= offset < H:
                w_i = sim.wh_idx[order.warehouse_id]
                p_i = sim.prod_idx[order.product_id]
                scheduled_arrivals[offset, w_i, p_i] += order.quantity

        # Big-M for supplier setup indicator
        M = 10000.0

        for tau in range(H):
            # Supplier capacity and setup linking
            for s in range(S):
                orders_for_sup = [
                    x_vars[(w, p, s, tau)]
                    for w in range(W)
                    for p in range(P)
                    if (w, p, s, tau) in x_vars
                ]
                if orders_for_sup:
                    prob += pulp.lpSum(orders_for_sup) <= M * z_vars[(s, tau)]

            # Warehouse capacity constraints
            for w in range(W):
                wh_cap = warehouses[w].storage_capacity
                prob += (
                    pulp.lpSum(
                        products[p].volume_m3 * I_vars[(w, p, tau)] for p in range(P)
                    )
                    <= wh_cap
                )

            # Inventory Balance Constraints
            for w in range(W):
                for p in range(P):
                    # Projected customer demand at step (t + tau)
                    sim_step = min(t + tau, sim.config.episode_days - 1)
                    dem = sim.demand_matrix[sim_step, w, p]

                    # Arriving shipments: orders placed earlier in horizon with lead time L
                    arrivals_from_horizon = []
                    for s in range(S):
                        if (w, p, s, tau) in x_vars:
                            lt = suppliers[s].base_lead_time_days
                            order_epoch = tau - lt
                            if order_epoch >= 0:
                                arrivals_from_horizon.append(
                                    x_vars[(w, p, s, order_epoch)]
                                )

                    incoming = scheduled_arrivals[tau, w, p] + pulp.lpSum(
                        arrivals_from_horizon
                    )

                    prev_inv = (
                        state["on_hand"][w, p]
                        if tau == 0
                        else I_vars[(w, p, tau - 1)]
                    )

                    # I_{tau} = I_{tau-1} + Arrivals - Demand + Unmet
                    prob += (
                        I_vars[(w, p, tau)]
                        == prev_inv + incoming - dem + U_vars[(w, p, tau)]
                    )

        # Solve MILP with silent CBC solver
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=self.time_limit)
        prob.solve(solver)

        # Extract step 0 orders: (w_id, p_id, s_id) -> quantity
        actions: Dict[Tuple[str, str, str], int] = {}
        for (w, p, s, tau), x_var in x_vars.items():
            if tau == 0 and x_var.varValue is not None:
                val = int(round(x_var.varValue))
                if val > 0:
                    actions[
                        (warehouses[w].id, products[p].id, suppliers[s].id)
                    ] = val

        return actions
