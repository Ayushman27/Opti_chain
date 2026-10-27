"""
OptiChain AI - Baseline Benchmark Runner
Executes multi-seed evaluation of classical inventory policies against the Digital Twin simulator.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from optichain.core.entities import SimulationConfig
from optichain.simulator.environment import DigitalTwinSimulator
from optichain.simulator.evaluator import PolicyEvaluator
from optichain.baselines.s_S_policy import MinMaxPolicy
from optichain.baselines.rop_policy import ROPPolicy
from optichain.baselines.periodic_review import PeriodicReviewPolicy


def main():
    print("=" * 70)
    print(" OptiChain AI - Digital Twin Simulation & Baseline Policy Benchmark ")
    print("=" * 70)

    # 3 Warehouses, 50 Products, 10 Suppliers, 90 Days Episode
    config = SimulationConfig(
        n_warehouses=3,
        n_products=50,
        n_suppliers=10,
        episode_days=90,
        random_seed=42,
    )

    print(f"\nConfiguration:")
    print(f"  Warehouses : {config.n_warehouses}")
    print(f"  Products   : {config.n_products}")
    print(f"  Suppliers  : {config.n_suppliers}")
    print(f"  Horizon    : {config.episode_days} days\n")

    policies = [
        MinMaxPolicy(name="Min-Max (s, S) Policy", service_factor_z=1.65),
        ROPPolicy(name="Reorder Point (ROP/EOQ)", service_factor_z=1.65),
        PeriodicReviewPolicy(name="Periodic Review (R, S) 7-day", review_period_days=7),
    ]

    evaluator = PolicyEvaluator(config)
    print("Evaluating policies across 3 seeds [42, 101, 777] under active disruptions...")
    benchmark_df = evaluator.benchmark(policies, seeds=[42, 101, 777])

    print("\n--- BENCHMARK RESULTS ---")
    print(benchmark_df.to_string(index=False))
    print("\nBenchmark completed successfully!\n")


if __name__ == "__main__":
    main()
