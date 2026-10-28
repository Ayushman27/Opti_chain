"""
OptiChain AI - Phase 2 Baseline Benchmark Runner
Executes multi-seed evaluation of classical inventory policies against the Digital Twin simulator.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from optichain.config import SimulationConfig, get_default_config
from optichain.simulator.environment import DigitalTwinSimulator
from optichain.simulator.evaluator import PolicyEvaluator
from optichain.baselines.s_S_policy import MinMaxPolicy
from optichain.baselines.rop_policy import ROPPolicy
from optichain.baselines.periodic_review import PeriodicReviewPolicy


def main():
    print("=" * 80)
    print(" OptiChain AI - Phase 2 Digital Twin Simulation & Heuristic Policy Benchmark ")
    print("=" * 80)

    # Phase 2 Baseline Setting: 1 Warehouse, 5 Products, 3 Suppliers, 90 Days
    config = get_default_config()

    print(f"\nSimulation Environment:")
    print(f"  Warehouses : {config.n_warehouses} (Distribution Center WH_01)")
    print(f"  Products   : {config.n_products} (SKUs: PROD_001 to PROD_005)")
    print(f"  Suppliers  : {config.n_suppliers} (Tier-1 Suppliers: SUP_01 to SUP_03)")
    print(f"  Horizon    : {config.episode_days} days\n")

    policies = [
        ROPPolicy(name="Reorder Point (ROP) Order-Up-To Target", service_factor_z=1.65),
        MinMaxPolicy(name="Min-Max (s, S) Policy", service_factor_z=1.65),
        PeriodicReviewPolicy(name="Periodic Review (R, S) 7-day", review_period_days=7, service_factor_z=1.65),
    ]

    evaluator = PolicyEvaluator(config)
    seeds = [42, 101, 777]
    print(f"Evaluating policies across {len(seeds)} matched random seeds {seeds} under active disruptions...")
    benchmark_df = evaluator.benchmark(policies, seeds=seeds)

    print("\n--- PHASE 2 HEURISTIC BENCHMARK RESULTS ---")
    print(benchmark_df.to_string(index=False))
    print("\nBenchmark completed successfully!\n")


if __name__ == "__main__":
    main()
