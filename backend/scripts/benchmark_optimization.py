"""
OptiChain AI - Phase 4 Optimization Benchmark Runner
Executes rigorous multi-seed comparative benchmark comparing:
1. Reorder Point (ROP) with Order-Up-To Target
2. Min-Max (s, S) Policy
3. Periodic Review (R, S) 7-day
4. Forecast-Driven MILP Optimizer

Uses identical demand scenarios, stochastic disruption events, starting inventory,
and cost definitions across matched random seeds [42, 101, 777].
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from scripts.audit_optimization import run_audit


if __name__ == "__main__":
    run_audit()

