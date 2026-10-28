#!/usr/bin/env python3
"""
OptiChain AI - Experiment Tracking CLI
Command-line utility for inspecting, listing, and comparing reproducible experiment records.

Usage:
    python scripts/track_experiment.py list [--limit 20]
    python scripts/track_experiment.py show <run_id>
    python scripts/track_experiment.py compare <run_id_1> <run_id_2> ...
    python scripts/track_experiment.py sample
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optichain.mlops.experiment_tracker import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(description="OptiChain AI Experiment Tracking CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List
    list_parser = subparsers.add_parser("list", help="List recent experiment runs")
    list_parser.add_argument("--limit", type=int, default=20, help="Max runs to display")
    list_parser.add_argument("--dir", type=str, default="artifacts/experiments", help="Registry directory")

    # Show
    show_parser = subparsers.add_parser("show", help="Display full details of a specific run")
    show_parser.add_argument("run_id", type=str, help="Unique run ID")
    show_parser.add_argument("--dir", type=str, default="artifacts/experiments", help="Registry directory")

    # Compare
    compare_parser = subparsers.add_parser("compare", help="Compare multiple runs side-by-side")
    compare_parser.add_argument("run_ids", nargs="+", help="Run IDs to compare")
    compare_parser.add_argument("--dir", type=str, default="artifacts/experiments", help="Registry directory")

    # Sample run
    subparsers.add_parser("sample", help="Record a verified sample experiment run")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    tracker = ExperimentTracker(output_dir=getattr(args, "dir", "artifacts/experiments"))

    if args.command == "list":
        runs = tracker.list_experiments(limit=args.limit)
        if not runs:
            print("No experiment records found in registry.")
            return
        print(f"\n{'Run ID':<30} {'Timestamp (UTC)':<24} {'Type':<22} {'Policy/Model':<18} {'Cost ($)':<12} {'Commit'}")
        print("-" * 115)
        for r in runs:
            cost_str = f"{r.get('total_cost_mean'):.2f}" if r.get('total_cost_mean') is not None else "-"
            print(f"{r.get('run_id'):<30} {r.get('timestamp_utc')[:19]:<24} {r.get('experiment_type',''):<22} {r.get('policy_or_model',''):<18} {cost_str:<12} {r.get('git_commit','')[:7]}")

    elif args.command == "show":
        exp = tracker.get_experiment(args.run_id)
        if not exp:
            print(f"Error: Run '{args.run_id}' not found.")
            sys.exit(1)
        print(json.dumps(exp, indent=2, default=str))

    elif args.command == "compare":
        df = tracker.compare_experiments(args.run_ids)
        if df.empty:
            print("No matching runs found to compare.")
            return
        print("\n=== Experiment Comparison ===")
        print(df.to_string(index=False))

    elif args.command == "sample":
        print("Recording sample policy benchmark experiment...")
        run_id = tracker.record_multi_seed_simulation(
            policy_name="OptiChain_MILP_Ridge",
            seeds=[42, 101, 2024],
            per_seed_results=[
                {"total_cost": 435210.0, "service_level": 88.89, "fill_rate": 98.42, "stockout_units": 142.0, "holding_cost": 42100.0, "stockout_cost": 31200.0, "net_profit": 92300.0},
                {"total_cost": 421450.0, "service_level": 87.78, "fill_rate": 97.91, "stockout_units": 156.0, "holding_cost": 41500.0, "stockout_cost": 33100.0, "net_profit": 96100.0},
                {"total_cost": 428942.0, "service_level": 92.22, "fill_rate": 98.75, "stockout_units": 98.0,  "holding_cost": 43200.0, "stockout_cost": 21500.0, "net_profit": 105974.0},
            ],
            simulation_config={"n_warehouses": 1, "n_products": 5, "n_suppliers": 3, "episode_days": 90},
            run_notes="Verification sample run for MLOps registry tracking",
        )
        print(f"Sample experiment recorded successfully! Run ID: {run_id}")


if __name__ == "__main__":
    main()
