"""
Benchmarks route
GET /api/v1/benchmarks
"""

from fastapi import APIRouter, HTTPException
from optichain.api.schemas import (
    BenchmarksResponse,
    PolicyBenchmarkItem,
    PolicyMetricStats,
)
from optichain.api.dependencies import load_optimization_benchmark

router = APIRouter(tags=["Benchmarks"])


@router.get("/benchmarks", response_model=BenchmarksResponse)
def get_benchmarks() -> BenchmarksResponse:
    """
    Returns complete multi-seed benchmark results comparing Forecast-Driven MILP against
    Periodic Review, Min-Max, and ROP baseline policies.
    Includes cost breakdowns, per-seed dispersion, and solver diagnostics.
    """
    try:
        raw_data = load_optimization_benchmark()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Benchmark artifact unavailable: {str(e)}"
        )

    config = raw_data.get("configuration", {})
    summary_list = raw_data.get("benchmark_summary", [])
    cost_breakdown = raw_data.get("detailed_cost_breakdown", [])
    per_seed_results = raw_data.get("per_seed_results", [])
    solver_diagnostics = raw_data.get("solver_diagnostics")

    # Build per-seed stats lookup
    stats_lookup = {}
    for entry in per_seed_results:
        pol_name = entry.get("policy", "")
        means = entry.get("mean", {})
        stds = entry.get("std", {})
        stats_lookup[pol_name] = {
            k: PolicyMetricStats(mean=float(v), std=float(stds.get(k, 0.0)) if stds else None)
            for k, v in means.items()
        }

    policies = []
    for item in summary_list:
        pol_name = item.get("Policy", "")
        pol_stats = stats_lookup.get(pol_name)

        pol_item = PolicyBenchmarkItem(
            policy=pol_name,
            fill_rate_percent=float(item.get("Fill Rate (%)", 0.0)),
            service_level_percent=float(item.get("Service Level (%)", 0.0)),
            total_cost_usd=float(item.get("Total Cost ($)", 0.0)),
            stockout_units=float(item.get("Stockout Units", 0.0)),
            average_inventory_units=float(item.get("Average Inventory (Units)", 0.0)),
            net_profit_usd=float(item.get("Net Profit ($)", 0.0)),
            inventory_turnover=float(item.get("Inventory Turnover", 0.0)),
            holding_cost_usd=float(item.get("Holding Cost ($)", 0.0)),
            stockout_cost_usd=float(item.get("Stockout Cost ($)", 0.0)),
            procurement_cost_usd=float(item.get("Procurement Cost ($)", 0.0)),
            fixed_setup_usd=float(item.get("Fixed Setup ($)", 0.0)),
            transport_usd=float(item.get("Transport ($)", 0.0)),
            facility_operating_usd=float(item.get("Facility Operating ($)", 0.0)),
            stats=pol_stats,
        )
        policies.append(pol_item)

    seeds = config.get("seeds", [42, 101, 777])
    seeds_str = ", ".join(str(s) for s in seeds)
    provenance = (
        f"Evaluated across {len(seeds)} matched seeds ({seeds_str}) on a 90-day horizon "
        f"with {config.get('products', 5)} SKUs and {config.get('suppliers', 3)} suppliers."
    )

    return BenchmarksResponse(
        configuration=config,
        policies=policies,
        detailed_cost_breakdown=cost_breakdown,
        per_seed_results=per_seed_results,
        solver_diagnostics=solver_diagnostics,
        provenance=provenance,
    )
