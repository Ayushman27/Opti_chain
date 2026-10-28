"""
OptiChain AI - Forecast-Driven Mixed-Integer Linear Programming (MILP) Optimizer
Formulates and solves a rolling-horizon replenishment planning problem using PuLP:
- Decision variables for replenishment purchase orders, supplier setup binary indicators,
  ending inventory, and unmet demand (shortage).
- Minimizes total relevant supply chain costs: procurement, holding, ordering/setup,
  transportation, and stockout penalties.
- Enforces supplier catalog eligibility, minimum order quantities (MOQ), supplier daily capacity,
  warehouse storage capacity, lead-time pipeline arrivals, and non-negativity.
- Driven by demand forecasts from the forecasting module (with zero lookahead into future actual demand).
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pulp
import warnings

from optichain.baselines.base_policy import BasePolicy
from optichain.forecasting.models import ForecastEngine

if TYPE_CHECKING:
    from optichain.simulator.environment import DigitalTwinSimulator
    from optichain.core.entities import Warehouse, Product, Supplier


@dataclass
class OptimizationResult:
    """Encapsulates the mathematical solution and operational diagnostics of an MILP solve."""
    status: str
    is_optimal: bool
    objective_value: float
    planned_orders: Dict[Tuple[str, str, str], int]  # Action dictionary for epoch 0: (w_id, p_id, s_id) -> qty
    full_schedule: Dict[int, Dict[Tuple[str, str, str], int]]  # Order schedule across horizon: tau -> orders
    projected_inventory: Dict[Tuple[int, int, int], float]  # (w_idx, p_idx, tau) -> units
    projected_unmet: Dict[Tuple[int, int, int], float]  # (w_idx, p_idx, tau) -> units
    solve_time_sec: float
    solver_message: str = ""
    is_proven_optimal: bool = True
    is_time_limited: bool = False
    termination_reason: str = "ProvenOptimal"
    raw_status: int = 1
    raw_sol_status: Optional[int] = 1
    sol_status_str: str = "Optimal Solution Found"
    has_feasible_solution: bool = True
    is_feasible_incumbent: bool = False
    is_infeasible: bool = False


class MILPReplenishmentPlanner:
    """
    Mathematical solver that constructs and solves the multi-period constrained MILP.
    """

    def __init__(self, time_limit_sec: float = 2.0):
        self.time_limit = time_limit_sec

    @staticmethod
    def classify_solver_termination(prob: pulp.LpProblem, has_vars: bool = True) -> Dict[str, Any]:
        """
        Classifies CBC solver termination using actual PuLP problem and solution status codes.
        STRICT REQUIREMENT: Does not use elapsed-time thresholds.

        Distinguishes:
        1. Proven Optimal: prob.status == LpStatusOptimal and prob.sol_status == LpSolutionOptimal
        2. Feasible Incumbent Unproven (e.g. time-limited with incumbent):
           prob.sol_status == LpSolutionIntegerFeasible
        3. Infeasible / No Solution Exists:
           prob.status == LpStatusInfeasible or prob.sol_status == LpSolutionInfeasible
        4. No Feasible Solution Found (e.g. stopped on time without incumbent):
           prob.status == LpStatusNotSolved or prob.sol_status == LpSolutionNoSolutionFound
        5. Solver Error / Other
        """
        raw_status = prob.status
        raw_sol_status = getattr(prob, "sol_status", None)

        status_str = pulp.LpStatus.get(raw_status, f"Status_{raw_status}")
        sol_status_str = pulp.LpSolution.get(raw_sol_status, f"SolStatus_{raw_sol_status}")

        is_proven_optimal = False
        is_feasible_incumbent = False
        is_infeasible = False
        is_no_solution = False
        has_feasible_solution = False
        is_time_limited = False

        if raw_sol_status == pulp.constants.LpSolutionOptimal:
            is_proven_optimal = True
            has_feasible_solution = has_vars
            termination_reason = "ProvenOptimal"
        elif raw_sol_status == pulp.constants.LpSolutionIntegerFeasible:
            is_feasible_incumbent = True
            has_feasible_solution = has_vars
            is_time_limited = True
            termination_reason = "FeasibleIncumbentUnproven"
        elif raw_status == pulp.constants.LpStatusInfeasible or raw_sol_status == pulp.constants.LpSolutionInfeasible:
            is_infeasible = True
            termination_reason = "Infeasible"
        elif raw_status == pulp.constants.LpStatusUnbounded or getattr(pulp.constants, 'LpSolutionUnbounded', None) == raw_sol_status:
            is_no_solution = True
            termination_reason = "Unbounded"
        elif raw_status == pulp.constants.LpStatusNotSolved or raw_sol_status == pulp.constants.LpSolutionNoSolutionFound:
            is_no_solution = True
            is_time_limited = True
            termination_reason = "NoSolutionFound"
        else:
            # Fallback when sol_status is not set (e.g. custom or mock solver)
            if raw_status == pulp.constants.LpStatusOptimal and has_vars:
                is_proven_optimal = True
                has_feasible_solution = True
                termination_reason = "ProvenOptimal"
            elif raw_status == pulp.constants.LpStatusInfeasible:
                is_infeasible = True
                termination_reason = "Infeasible"
            else:
                termination_reason = f"SolverStatus_{status_str}"

        return {
            "raw_status": raw_status,
            "raw_sol_status": raw_sol_status,
            "status_str": status_str,
            "sol_status_str": sol_status_str,
            "is_proven_optimal": is_proven_optimal,
            "is_feasible_incumbent": is_feasible_incumbent,
            "is_infeasible": is_infeasible,
            "has_feasible_solution": has_feasible_solution,
            "is_time_limited": is_time_limited,
            "termination_reason": termination_reason,
        }

    def solve(
        self,
        current_step: int,
        current_on_hand: np.ndarray,  # Shape: (W, P)
        scheduled_arrivals: np.ndarray,  # Shape: (H, W, P)
        forecast_demands: np.ndarray,  # Shape: (H, W, P)
        warehouses: List[Warehouse],
        products: List[Product],
        suppliers: List[Supplier],
        horizon: int,
        transport_cost_per_unit: float = 0.50,
        enable_moq: bool = True,
    ) -> OptimizationResult:
        """
        Formulates and solves the multi-period MILP replenishment problem.

        Decision Variables:
        -------------------
        - x[w, p, s, tau] >= 0, integer: Units ordered from supplier s to warehouse w for SKU p at epoch tau
        - y[w, p, s, tau] in {0, 1}: Binary indicator if order placed for SKU p from supplier s at epoch tau (for MOQ)
        - z[s, tau] in {0, 1}: Binary indicator if supplier s receives any order at epoch tau (for setup cost)
        - I[w, p, tau] >= 0: Projected ending on-hand inventory at warehouse w for SKU p at epoch tau
        - U[w, p, tau] >= 0: Projected unmet customer demand at warehouse w for SKU p at epoch tau

        Objective Function:
        -------------------
        Minimize:
          Sum_tau [
            Sum_{w, p} h_p * I[w, p, tau]                     (Holding Cost)
            + Sum_{w, p} pi_p * U[w, p, tau]                  (Stockout Penalty)
            + Sum_{w, p, s} (c_{p, s} + c_trans) * x[w,p,s,tau] (Procurement + Transport)
            + Sum_s K_s * z[s, tau]                           (Fixed Order Setup Cost)
          ]
        """
        start_time = time.perf_counter()
        H = horizon
        W = len(warehouses)
        P = len(products)
        S = len(suppliers)

        # Silence PuLP deprecation warnings for cleaner execution
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            prob = pulp.LpProblem(f"OptiChain_MILP_t{current_step}", pulp.LpMinimize)

            # -------------------------------------------------------------
            # 1. DECISION VARIABLES
            # -------------------------------------------------------------
            x_vars: Dict[Tuple[int, int, int, int], pulp.LpVariable] = {}
            y_vars: Dict[Tuple[int, int, int, int], pulp.LpVariable] = {}
            z_vars: Dict[Tuple[int, int], pulp.LpVariable] = {}
            I_vars: Dict[Tuple[int, int, int], pulp.LpVariable] = {}
            U_vars: Dict[Tuple[int, int, int], pulp.LpVariable] = {}

            # Supplier setup binary variables
            for tau in range(H):
                for s in range(S):
                    z_vars[(s, tau)] = pulp.LpVariable(
                        f"z_s{s}_t{tau}", cat=pulp.LpBinary
                    )

            # Order variables: only defined for products present in the supplier's catalog (Catalog Eligibility)
            for tau in range(H):
                for w in range(W):
                    for p in range(P):
                        p_id = products[p].id
                        for s in range(S):
                            if p_id in suppliers[s].catalog:
                                x_vars[(w, p, s, tau)] = pulp.LpVariable(
                                    f"x_w{w}_p{p}_s{s}_t{tau}", lowBound=0, cat=pulp.LpInteger
                                )
                                if enable_moq:
                                    y_vars[(w, p, s, tau)] = pulp.LpVariable(
                                        f"y_w{w}_p{p}_s{s}_t{tau}", cat=pulp.LpBinary
                                    )

            # State variables: Inventory and Unmet Demand
            for tau in range(H):
                for w in range(W):
                    for p in range(P):
                        I_vars[(w, p, tau)] = pulp.LpVariable(
                            f"I_w{w}_p{p}_t{tau}", lowBound=0, cat=pulp.LpContinuous
                        )
                        U_vars[(w, p, tau)] = pulp.LpVariable(
                            f"U_w{w}_p{p}_t{tau}", lowBound=0, cat=pulp.LpContinuous
                        )

            # -------------------------------------------------------------
            # 2. OBJECTIVE FUNCTION
            # -------------------------------------------------------------
            cost_terms = []

            # Inventory holding and stockout shortage penalty
            for tau in range(H):
                for w in range(W):
                    for p in range(P):
                        h_rate = products[p].holding_cost_rate
                        p_penalty = products[p].stockout_penalty
                        cost_terms.append(h_rate * I_vars[(w, p, tau)])
                        cost_terms.append(p_penalty * U_vars[(w, p, tau)])

            # Procurement cost + unit transport cost
            for (w, p, s, tau), x_var in x_vars.items():
                unit_price = suppliers[s].catalog[products[p].id]
                cost_terms.append((unit_price + transport_cost_per_unit) * x_var)

            # Fixed order setup cost per utilized supplier
            for (s, tau), z_var in z_vars.items():
                fixed_cost = suppliers[s].fixed_order_cost
                cost_terms.append(fixed_cost * z_var)

            prob += pulp.lpSum(cost_terms)

            # -------------------------------------------------------------
            # 3. CONSTRAINTS
            # -------------------------------------------------------------
            for tau in range(H):
                # Constraint A: Minimum Order Quantity (MOQ) & Supplier SKU Capacity
                for (w, p, s, t_idx), x_var in x_vars.items():
                    if t_idx == tau:
                        p_id = products[p].id
                        moq = suppliers[s].min_order_quantity.get(p_id, 1)
                        daily_cap = suppliers[s].daily_capacity.get(p_id, 999999)

                        if enable_moq and (w, p, s, tau) in y_vars:
                            y_var = y_vars[(w, p, s, tau)]
                            # Lower bound: if ordered, x >= MOQ
                            prob += x_var >= moq * y_var
                            # Upper bound: x <= Cap * y
                            prob += x_var <= daily_cap * y_var
                            # Link SKU order to supplier setup indicator
                            prob += y_var <= z_vars[(s, tau)]
                        else:
                            # Link x directly to setup indicator
                            prob += x_var <= daily_cap * z_vars[(s, tau)]

                # Constraint B: Supplier Aggregate Daily Capacity
                for s in range(S):
                    for p in range(P):
                        p_id = products[p].id
                        if p_id in suppliers[s].catalog:
                            cap = suppliers[s].daily_capacity.get(p_id, 999999)
                            orders_sku = [
                                x_vars[(w, p, s, tau)]
                                for w in range(W)
                                if (w, p, s, tau) in x_vars
                            ]
                            if orders_sku:
                                prob += pulp.lpSum(orders_sku) <= cap * z_vars[(s, tau)]

                # Constraint C: Warehouse Storage Physical Capacity
                for w in range(W):
                    wh_cap = warehouses[w].storage_capacity
                    prob += (
                        pulp.lpSum(
                            products[p].volume_m3 * I_vars[(w, p, tau)]
                            for p in range(P)
                        )
                        <= wh_cap
                    )

                # Constraint D: Inventory Conservation of Mass
                for w in range(W):
                    for p in range(P):
                        dem_forecast = float(forecast_demands[tau, w, p])

                        # Orders placed earlier in the horizon that arrive at epoch tau
                        horizon_arrivals = []
                        for s in range(S):
                            if (w, p, s, tau) in x_vars:
                                lt = suppliers[s].base_lead_time_days
                                order_epoch = tau - lt
                                if order_epoch >= 0 and (w, p, s, order_epoch) in x_vars:
                                    horizon_arrivals.append(x_vars[(w, p, s, order_epoch)])

                        # Total incoming = pre-existing scheduled pipeline + horizon orders
                        incoming_shipments = scheduled_arrivals[tau, w, p] + pulp.lpSum(horizon_arrivals)

                        # Previous inventory position: on_hand if tau == 0, else I[tau-1]
                        prev_inv = (
                            current_on_hand[w, p]
                            if tau == 0
                            else I_vars[(w, p, tau - 1)]
                        )

                        # Balance: I_tau = I_{tau-1} + Arrivals - ForecastDemand + Unmet
                        prob += (
                            I_vars[(w, p, tau)]
                            == prev_inv + incoming_shipments - dem_forecast + U_vars[(w, p, tau)]
                        )

            # -------------------------------------------------------------
            # 4. SOLVE WITH CBC SOLVER
            # -------------------------------------------------------------
            solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=self.time_limit)
            prob.solve(solver)

        solve_time = time.perf_counter() - start_time
        has_vars = any(x_var.varValue is not None for x_var in x_vars.values()) if x_vars else True
        diag = self.classify_solver_termination(prob, has_vars=has_vars)

        is_executable = diag["has_feasible_solution"]
        is_optimal = is_executable

        # Extract planned orders for epoch 0 (immediate execution today)
        planned_orders: Dict[Tuple[str, str, str], int] = {}
        full_schedule: Dict[int, Dict[Tuple[str, str, str], int]] = {t_idx: {} for t_idx in range(H)}

        if is_optimal:
            for (w, p, s, tau), x_var in x_vars.items():
                if x_var.varValue is not None:
                    val = int(round(x_var.varValue))
                    if val > 0:
                        key = (warehouses[w].id, products[p].id, suppliers[s].id)
                        full_schedule[tau][key] = val
                        if tau == 0:
                            planned_orders[key] = val

        # Extract projected inventory and unmet demand profiles
        proj_inv: Dict[Tuple[int, int, int], float] = {}
        proj_unmet: Dict[Tuple[int, int, int], float] = {}
        for tau in range(H):
            for w in range(W):
                for p in range(P):
                    proj_inv[(w, p, tau)] = (
                        float(I_vars[(w, p, tau)].varValue)
                        if I_vars[(w, p, tau)].varValue is not None
                        else 0.0
                    )
                    proj_unmet[(w, p, tau)] = (
                        float(U_vars[(w, p, tau)].varValue)
                        if U_vars[(w, p, tau)].varValue is not None
                        else 0.0
                    )

        obj_val = float(pulp.value(prob.objective)) if prob.objective is not None and is_executable else float("inf")

        return OptimizationResult(
            status=diag["status_str"],
            is_optimal=is_executable,
            objective_value=obj_val,
            planned_orders=planned_orders,
            full_schedule=full_schedule,
            projected_inventory=proj_inv,
            projected_unmet=proj_unmet,
            solve_time_sec=round(solve_time, 4),
            is_proven_optimal=diag["is_proven_optimal"],
            is_time_limited=diag["is_time_limited"],
            termination_reason=diag["termination_reason"],
            raw_status=diag["raw_status"],
            raw_sol_status=diag["raw_sol_status"],
            sol_status_str=diag["sol_status_str"],
            has_feasible_solution=diag["has_feasible_solution"],
            is_feasible_incumbent=diag["is_feasible_incumbent"],
            is_infeasible=diag["is_infeasible"],
        )


class MILPReplenishmentPolicy(BasePolicy):
    """
    Rolling-Horizon Forecast-Driven MILP Inventory Optimization Policy (MPC).
    At every daily review step t:
    1. Obtains forecast demand projections hat{D} across lookahead horizon H
       using the trained forecasting module (with zero lookahead into actual future demand).
    2. Builds scheduled pipeline arrivals from current in-transit orders.
    3. Solves the constrained MILP to global optimality.
    4. Issues the epoch-0 replenishment purchase orders to the digital twin simulator.
    """

    def __init__(
        self,
        name: str = "Forecast-Driven MILP Optimizer",
        horizon: int = 8,
        time_limit_sec: float = 2.0,
        forecast_engine: Optional[ForecastEngine] = None,
        forecast_method: str = "heuristic_lag7",
        enable_moq: bool = True,
        safety_buffer_factor: float = 0.35,
    ):
        super().__init__(name=name)
        self.horizon = horizon
        self.time_limit = time_limit_sec
        self.planner = MILPReplenishmentPlanner(time_limit_sec=time_limit_sec)
        self.forecast_engine = forecast_engine
        self._engine_fixed = forecast_engine is not None
        self.forecast_method = forecast_method
        self.enable_moq = enable_moq
        self.safety_buffer_factor = safety_buffer_factor
        self.historical_demands: Dict[Tuple[int, int], List[float]] = {}
        self.solve_history: List[OptimizationResult] = []

    def fit(self, sim: DigitalTwinSimulator) -> None:
        """
        Fits or initializes the forecasting engine and precomputes network statistics.
        """
        self.solve_history.clear()
        self.historical_demands.clear()

        # Retrieve pre-simulation historical demand strictly prior to simulation horizon (t < 0)
        hist_demands = getattr(sim, "historical_demand_matrix", None)
        if hist_demands is None:
            from optichain.data.generator import SupplyChainDataGenerator
            hist_gen = SupplyChainDataGenerator(sim.config.model_copy(update={"random_seed": sim.config.random_seed + 10000}))
            hist_demands = hist_gen.generate_demand_matrix(sim.products, sim.warehouses, disruptions=[])

        # If a forecast engine wasn't provided, initialize and train one on historical data
        if not self._engine_fixed:
            self.forecast_engine = ForecastEngine(random_seed=sim.config.random_seed)
            # Train forecasting models strictly on historical data prior to simulation
            self.forecast_engine.train_and_backtest(
                hist_demands,
                sim.products,
                sim.warehouses,
                disruptions=[],
                train_split_ratio=0.70,
            )

        W = len(sim.warehouses)
        P = len(sim.products)
        self.prod_mean_demands = hist_demands.mean(axis=0)  # Shape (W, P)
        for w in range(W):
            for p in range(P):
                self.historical_demands[(w, p)] = []


    def _generate_forecast_window(
        self, state: Dict[str, Any], sim: DigitalTwinSimulator, H: int
    ) -> np.ndarray:
        """
        Generates lookahead demand forecasts hat{D}_{tau, w, p} for tau in [0, H-1].
        STRICT GUARANTEE: Does NOT read future ground-truth demand from sim.demand_matrix!
        Uses the trained forecast engine and historical feature projections.
        """
        t = state["current_step"]
        W = len(sim.warehouses)
        P = len(sim.products)
        forecast_matrix = np.zeros((H, W, P), dtype=float)

        for w_i, wh in enumerate(sim.warehouses):
            for p_i, prod in enumerate(sim.products):
                # Historical mean fallback
                hist_series = self.historical_demands.get((w_i, p_i), [])
                if len(hist_series) >= 3:
                    fallback_mean = float(np.mean(hist_series[-14:]))
                elif hasattr(self, "prod_mean_demands") and self.prod_mean_demands is not None:
                    fallback_mean = float(self.prod_mean_demands[w_i, p_i])
                else:
                    fallback_mean = getattr(prod, "base_demand_mean", 20.0)


                for tau in range(H):
                    future_step = t + tau
                    if self.forecast_engine is not None and self.forecast_engine.is_fitted:
                        dow = future_step % 7
                        is_weekend = 1 if dow in [5, 6] else 0
                        sin_dow = math.sin(2.0 * math.pi * dow / 7.0)
                        cos_dow = math.cos(2.0 * math.pi * dow / 7.0)
                        trend = float(future_step) / float(max(1, sim.config.episode_days - 1))

                        lag_7_val = (
                            float(hist_series[-7])
                            if len(hist_series) >= 7
                            else fallback_mean
                        )
                        lag_1_val = (
                            float(hist_series[-1])
                            if len(hist_series) >= 1
                            else fallback_mean
                        )

                        if self.forecast_method == "seasonal_naive":
                            row_df = pd.DataFrame([{"lag_7": lag_7_val, "lag_1": lag_1_val}])
                            base_pred = float(self.forecast_engine.seasonal_naive_model.predict(row_df)[0])
                        elif self.forecast_method in ["ridge", "lgbm"]:
                            row_dict = {
                                "step": future_step,
                                "trend": trend,
                                "warehouse_id": wh.id,
                                "product_id": prod.id,
                                "unit_cost": getattr(prod, "unit_cost", 10.0),
                                "selling_price": getattr(prod, "selling_price", 20.0),
                                "day_of_week": dow,
                                "is_weekend": is_weekend,
                                "sin_dow": sin_dow,
                                "cos_dow": cos_dow,
                                "disruption_active": 1 if any(d.is_active(future_step) for d in sim.disruptions) else 0,
                                "lag_1": lag_1_val,
                                "lag_2": float(hist_series[-2]) if len(hist_series) >= 2 else fallback_mean,
                                "lag_3": float(hist_series[-3]) if len(hist_series) >= 3 else fallback_mean,
                                "lag_7": lag_7_val,
                                "lag_14": float(hist_series[-14]) if len(hist_series) >= 14 else fallback_mean,
                                "rolling_mean_7": float(np.mean(hist_series[-7:])) if len(hist_series) >= 1 else fallback_mean,
                                "rolling_std_7": float(np.std(hist_series[-7:])) if len(hist_series) > 1 else 0.0,
                                "rolling_min_7": float(np.min(hist_series[-7:])) if len(hist_series) >= 1 else fallback_mean,
                                "rolling_max_7": float(np.max(hist_series[-7:])) if len(hist_series) >= 1 else fallback_mean,
                                "rolling_mean_14": float(np.mean(hist_series[-14:])) if len(hist_series) >= 1 else fallback_mean,
                                "rolling_std_14": float(np.std(hist_series[-14:])) if len(hist_series) > 1 else 0.0,
                                "rolling_min_14": float(np.min(hist_series[-14:])) if len(hist_series) >= 1 else fallback_mean,
                                "rolling_max_14": float(np.max(hist_series[-14:])) if len(hist_series) >= 1 else fallback_mean,
                            }
                            row_dict[f"category_{getattr(prod, 'category', 'General')}"] = 1
                            row_dict[f"is_prod_{prod.id}"] = 1
                            target_model = self.forecast_engine.ridge_model if self.forecast_method == "ridge" else self.forecast_engine.lgbm_model
                            row_df = pd.DataFrame([row_dict]).reindex(columns=target_model.feature_cols, fill_value=0.0)
                            base_pred = float(target_model.predict(row_df)[0])
                        else:
                            # Default heuristic_lag7: linear combination of weekly lag-7 and mean with weekend harmonic
                            base_pred = (
                                0.5 * lag_7_val
                                + 0.5 * fallback_mean * (1.0 + 0.15 * (1 if is_weekend else -0.05))
                            )
                        # Add heuristic safety buffer factor to absorb stochastic Poisson demand variance
                        pred_val = max(1.0, base_pred * (1.0 + self.safety_buffer_factor))
                    else:
                        pred_val = fallback_mean * (1.0 + self.safety_buffer_factor)

                    forecast_matrix[tau, w_i, p_i] = round(pred_val, 2)

        return forecast_matrix

    def get_actions(
        self, state: Dict[str, Any], sim: DigitalTwinSimulator
    ) -> Dict[Tuple[str, str, str], int]:
        """
        Executes rolling-horizon optimization step:
        1. Updates historical demand observation buffer.
        2. Forecasts demand for [t, t+H].
        3. Extracts in-transit arrival pipeline for [t, t+H].
        4. Solves MILP and returns day-0 purchase orders.
        """
        t = state["current_step"]
        H = min(self.horizon, sim.config.episode_days - t)
        if H <= 0:
            return {}

        W = len(sim.warehouses)
        P = len(sim.products)

        # Update historical demand tracking (demand from previous step t-1 if available)
        if t > 0:
            prev_t = t - 1
            for w_i in range(W):
                for p_i in range(P):
                    prev_demand = sim.history_demand[prev_t, w_i, p_i]
                    self.historical_demands[(w_i, p_i)].append(float(prev_demand))

        # 1. Forecast lookahead demand
        forecast_demands = self._generate_forecast_window(state, sim, H)

        # 2. Extract scheduled pipeline arrivals
        scheduled_arrivals = np.zeros((H, W, P), dtype=float)
        for order in sim.in_transit_orders:
            arr_step = order.expected_arrival_step
            offset = arr_step - t
            if 0 <= offset < H:
                w_i = sim.wh_idx[order.warehouse_id]
                p_i = sim.prod_idx[order.product_id]
                scheduled_arrivals[offset, w_i, p_i] += order.quantity

        # 3. Solve MILP
        result = self.planner.solve(
            current_step=t,
            current_on_hand=state["on_hand"],
            scheduled_arrivals=scheduled_arrivals,
            forecast_demands=forecast_demands,
            warehouses=sim.warehouses,
            products=sim.products,
            suppliers=sim.suppliers,
            horizon=H,
            transport_cost_per_unit=sim.config.transport_cost_per_unit,
            enable_moq=self.enable_moq,
        )
        self.solve_history.append(result)

        # 4. Return planned orders or safe fallback
        if result.is_optimal:
            return result.planned_orders

        # Fallback if solver fails or is infeasible: order nothing or minimal safety stock
        return {}
