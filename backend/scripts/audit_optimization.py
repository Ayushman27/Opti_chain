"""
OptiChain AI - Phase 4 Independent Audit Script
Runs individual seed evaluations (42, 101, 777) for all 4 policies,
records granular per-seed KPIs, computes mean +- std,
and audits solver statuses across all solves.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import json
import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from optichain.config import get_default_config
from optichain.simulator.environment import DigitalTwinSimulator
from optichain.simulator.evaluator import PolicyEvaluator
from optichain.baselines.rop_policy import ROPPolicy
from optichain.baselines.s_S_policy import MinMaxPolicy
from optichain.baselines.periodic_review import PeriodicReviewPolicy
from optichain.optimization.milp_planner import MILPReplenishmentPolicy, OptimizationResult


def run_audit():
    print("Initializing Phase 4 Audit...", flush=True)
    config = get_default_config()
    seeds = [42, 101, 777]

    policies = [
        ("Periodic Review (R, S) 7-day", PeriodicReviewPolicy(
            name="Periodic Review (R, S) 7-day", review_period_days=7, service_factor_z=1.65
        )),
        ("Min-Max (s, S) Policy", MinMaxPolicy(
            name="Min-Max (s, S) Policy", service_factor_z=1.65
        )),
        ("ROP Order-Up-To Target", ROPPolicy(
            name="ROP Order-Up-To Target", service_factor_z=1.65
        )),
        ("Forecast-Driven MILP Optimizer", MILPReplenishmentPolicy(
            name="Forecast-Driven MILP Optimizer",
            horizon=8,
            time_limit_sec=2.0,
            enable_moq=True,
            safety_buffer_factor=0.35,
        )),
    ]

    evaluator = PolicyEvaluator(config)

    # Pre-instantiate identical simulated environments per seed
    seed_environments = {}
    for s in seeds:
        sim_config = config.model_copy(update={"random_seed": s})
        seed_environments[s] = DigitalTwinSimulator(config=sim_config)

    per_seed_data = {}
    solver_status_records = []

    for pol_name, policy_inst in policies:
        per_seed_data[pol_name] = {}
        for s in seeds:
            print(f"Evaluating {pol_name} on Seed {s}...", flush=True)
            base_sim = seed_environments[s]
            eval_sim = DigitalTwinSimulator(
                config=base_sim.config,
                products=base_sim.products,
                warehouses=base_sim.warehouses,
                suppliers=base_sim.suppliers,
                disruptions=base_sim.disruptions,
                demand_matrix=base_sim.demand_matrix.copy(),
                historical_demand_matrix=base_sim.historical_demand_matrix.copy(),
            )
            ep_res = evaluator.run_episode(policy_inst, eval_sim, seed=s)
            kpi = ep_res["kpis"]
            breakdown = ep_res["cost_breakdown"]

            per_seed_data[pol_name][s] = {
                "fill_rate": kpi.fill_rate_percent,
                "service_level": kpi.alpha_service_level_percent,
                "total_cost": kpi.total_cost,
                "stockout_units": kpi.total_unmet_demand,
                "avg_inventory": kpi.average_on_hand_inventory,
                "holding_cost": breakdown.get("holding_cost", 0.0),
                "stockout_cost": breakdown.get("stockout_penalty", 0.0),
                "procurement_cost": breakdown.get("procurement_cost", 0.0),
                "fixed_cost": breakdown.get("fixed_order_cost", 0.0),
                "transport_cost": breakdown.get("transport_cost", 0.0),
                "operating_cost": breakdown.get("operating_cost", 0.0),
                "revenue": breakdown.get("revenue", 0.0),
                "net_profit": kpi.net_profit,
                "inventory_turnover": kpi.inventory_turnover,
            }

            if pol_name == "Forecast-Driven MILP Optimizer":
                for step_idx, solve_res in enumerate(policy_inst.solve_history[-90:]):
                    solver_status_records.append({
                        "seed": s,
                        "step": step_idx,
                        "status": solve_res.status,
                        "raw_status": getattr(solve_res, "raw_status", 1),
                        "raw_sol_status": getattr(solve_res, "raw_sol_status", 1),
                        "sol_status_str": getattr(solve_res, "sol_status_str", "Optimal Solution Found"),
                        "is_optimal": solve_res.is_optimal,
                        "has_feasible_solution": getattr(solve_res, "has_feasible_solution", True),
                        "is_proven_optimal": solve_res.is_proven_optimal,
                        "is_feasible_incumbent": getattr(solve_res, "is_feasible_incumbent", False),
                        "is_time_limited": solve_res.is_time_limited,
                        "is_infeasible": getattr(solve_res, "is_infeasible", False),
                        "termination_reason": solve_res.termination_reason,
                        "obj_val": solve_res.objective_value,
                        "solve_time_sec": solve_res.solve_time_sec,
                    })

    # Output detailed audit summary
    print("\n" + "=" * 95, flush=True)
    print("OPTICHAIN AI - PHASE 4 INDEPENDENT AUDIT RESULTS PER SEED", flush=True)
    print("=" * 95, flush=True)

    audit_summary = []
    for pol_name, _ in policies:
        print(f"\n--- Policy: {pol_name} ---", flush=True)
        seed_dicts = [per_seed_data[pol_name][s] for s in seeds]
        for s in seeds:
            d = per_seed_data[pol_name][s]
            print(f"  Seed {s:3d}: Total Cost = ${d['total_cost']:10.2f} | Fill Rate = {d['fill_rate']:6.2f}% | "
                  f"Stockout Units = {d['stockout_units']:4d} | Avg Inv = {d['avg_inventory']:6.2f} | "
                  f"Net Profit = ${d['net_profit']:9.2f} | Turnover = {d['inventory_turnover']:5.2f}x", flush=True)

        kpis_to_aggregate = list(seed_dicts[0].keys())
        agg_mean = {k: float(np.mean([d[k] for d in seed_dicts])) for k in kpis_to_aggregate}
        agg_std = {k: float(np.std([d[k] for d in seed_dicts], ddof=1)) for k in kpis_to_aggregate}

        print(f"  MEAN +- STD across seeds [42, 101, 777]:", flush=True)
        print(f"    Total Cost   : ${agg_mean['total_cost']:.2f} +- ${agg_std['total_cost']:.2f}", flush=True)
        print(f"    Fill Rate    : {agg_mean['fill_rate']:.2f}% +- {agg_std['fill_rate']:.2f}%", flush=True)
        print(f"    Service Level: {agg_mean['service_level']:.2f}% +- {agg_std['service_level']:.2f}%", flush=True)
        print(f"    Stockout Pcs : {agg_mean['stockout_units']:.1f} +- {agg_std['stockout_units']:.1f}", flush=True)
        print(f"    Avg Inventory: {agg_mean['avg_inventory']:.2f} +- {agg_std['avg_inventory']:.2f}", flush=True)
        print(f"    Net Profit   : ${agg_mean['net_profit']:.2f} +- ${agg_std['net_profit']:.2f}", flush=True)
        print(f"    Turnover     : {agg_mean['inventory_turnover']:.2f}x +- {agg_std['inventory_turnover']:.2f}x", flush=True)

        audit_summary.append({
            "policy": pol_name,
            "mean": agg_mean,
            "std": agg_std,
            "per_seed": per_seed_data[pol_name],
        })

    # Solver status analysis for MILP
    print("\n" + "=" * 95, flush=True)
    print("MILP SOLVER STATUS AUDIT (Across 90 steps x 3 seeds = 270 solves)", flush=True)
    print("=" * 95, flush=True)
    status_counts = {}
    sol_status_counts = {}
    term_counts = {}
    solve_times = []
    for r in solver_status_records:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
        sol_status_counts[r["sol_status_str"]] = sol_status_counts.get(r["sol_status_str"], 0) + 1
        term_counts[r["termination_reason"]] = term_counts.get(r["termination_reason"], 0) + 1
        solve_times.append(r["solve_time_sec"])

    proven_optimal_count = sum(1 for r in solver_status_records if r.get("is_proven_optimal", False))
    feasible_incumbent_count = sum(1 for r in solver_status_records if r.get("is_feasible_incumbent", False))
    time_limited_count = sum(1 for r in solver_status_records if r.get("is_time_limited", False))
    executable_count = sum(1 for r in solver_status_records if r.get("has_feasible_solution", False))
    infeasible_count = sum(1 for r in solver_status_records if r.get("is_infeasible", False))

    print(f"Total Solves Audited           : {len(solver_status_records)}", flush=True)
    print(f"PuLP Status Distribution       : {status_counts}", flush=True)
    print(f"PuLP Solution Status Dist.     : {sol_status_counts}", flush=True)
    print(f"Termination Reason Breakdown   : {term_counts}", flush=True)
    print(f"Proven Optimal (Optimality proof)    : {proven_optimal_count} / {len(solver_status_records)} ({proven_optimal_count/len(solver_status_records)*100:.2f}%)", flush=True)
    print(f"Feasible Incumbent (Unproven)        : {feasible_incumbent_count} / {len(solver_status_records)} ({feasible_incumbent_count/len(solver_status_records)*100:.2f}%)", flush=True)
    print(f"Time-Limited Incumbents              : {time_limited_count} / {len(solver_status_records)} ({time_limited_count/len(solver_status_records)*100:.2f}%)", flush=True)
    print(f"Executable Incumbents Returned       : {executable_count} / {len(solver_status_records)} ({executable_count/len(solver_status_records)*100:.2f}%)", flush=True)
    print(f"Infeasible Solves                    : {infeasible_count}", flush=True)
    print(f"Mean Solve Time (s)                  : {np.mean(solve_times):.4f}s (Min: {np.min(solve_times):.4f}s, Max: {np.max(solve_times):.4f}s)", flush=True)

    # Save detailed audit data to json
    audit_output_path = root_dir / "docs" / "optimization_audit_per_seed.json"
    audit_payload = {
        "audit_summary": audit_summary,
        "solver_diagnostics": {
            "total_solves": len(solver_status_records),
            "status_distribution": status_counts,
            "solution_status_distribution": sol_status_counts,
            "termination_distribution": term_counts,
            "proven_optimal_solves": proven_optimal_count,
            "feasible_incumbent_solves": feasible_incumbent_count,
            "time_limited_solves": time_limited_count,
            "infeasible_solves": infeasible_count,
            "executable_solves": executable_count,
            "mean_solve_time_sec": float(np.mean(solve_times)),
            "max_solve_time_sec": float(np.max(solve_times)),
            "min_solve_time_sec": float(np.min(solve_times)),
        }
    }
    with open(audit_output_path, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)

    # Also update docs/optimization_benchmark_results.json from the single source of truth
    results_json_path = root_dir / "docs" / "optimization_benchmark_results.json"
    reconciled_benchmark_summary = []
    reconciled_cost_breakdown = []
    for item in audit_summary:
        pol_m = item["mean"]
        reconciled_benchmark_summary.append({
            "Policy": item["policy"],
            "Fill Rate (%)": round(pol_m["fill_rate"], 2),
            "Service Level (%)": round(pol_m["service_level"], 2),
            "Total Cost ($)": round(pol_m["total_cost"], 2),
            "Mean Total Cost ($)": round(pol_m["total_cost"], 2),
            "Stockout Units": int(round(pol_m["stockout_units"])),
            "Average Inventory (Units)": round(pol_m["avg_inventory"], 2),
            "Holding Cost ($)": round(pol_m["holding_cost"], 2),
            "Stockout Cost ($)": round(pol_m["stockout_cost"], 2),
            "Procurement Cost ($)": round(pol_m["procurement_cost"], 2),
            "Fixed Setup ($)": round(pol_m["fixed_cost"], 2),
            "Transport ($)": round(pol_m["transport_cost"], 2),
            "Facility Operating ($)": round(pol_m["operating_cost"], 2),
            "Net Profit ($)": round(pol_m["net_profit"], 2),
            "Mean Profit ($)": round(pol_m["net_profit"], 2),
            "Inventory Turnover": round(pol_m["inventory_turnover"], 2),
        })
        reconciled_cost_breakdown.append({
            "Policy": item["policy"],
            "Holding Cost ($)": round(pol_m["holding_cost"], 2),
            "Stockout Cost ($)": round(pol_m["stockout_cost"], 2),
            "Procurement Cost ($)": round(pol_m["procurement_cost"], 2),
            "Fixed Setup ($)": round(pol_m["fixed_cost"], 2),
            "Transport ($)": round(pol_m["transport_cost"], 2),
            "Facility Operating ($)": round(pol_m["operating_cost"], 2),
            "Total Cost ($)": round(pol_m["total_cost"], 2),
            "Net Profit ($)": round(pol_m["net_profit"], 2),
        })

    bench_data = {
        "configuration": {
            "warehouses": config.n_warehouses,
            "products": config.n_products,
            "suppliers": config.n_suppliers,
            "horizon_days": config.episode_days,
            "seeds": seeds,
        },
        "benchmark_summary": reconciled_benchmark_summary,
        "detailed_cost_breakdown": reconciled_cost_breakdown,
        "per_seed_results": audit_summary,
        "solver_diagnostics": audit_payload["solver_diagnostics"],
    }
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(bench_data, f, indent=2)

    print(f"\nSaved granular audit payload to: {audit_output_path}", flush=True)
    print(f"Synchronized single source of truth benchmark results to: {results_json_path}", flush=True)


if __name__ == "__main__":
    run_audit()
