"""
Dashboard summary route
GET /api/v1/dashboard/summary
"""

from fastapi import APIRouter, HTTPException
from optichain.api.schemas import DashboardSummaryResponse
from optichain.api.dependencies import load_optimization_benchmark

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/summary", response_model=DashboardSummaryResponse)
def get_dashboard_summary() -> DashboardSummaryResponse:
    """
    Returns executive KPI summary from the validated multi-seed optimization benchmark artifact.
    Honest data provenance from docs/optimization_benchmark_results.json.
    """
    try:
        data = load_optimization_benchmark()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Benchmark artifact unavailable: {str(e)}"
        )

    summary_list = data.get("benchmark_summary", [])
    if not summary_list:
        raise HTTPException(
            status_code=500,
            detail="Benchmark artifact missing 'benchmark_summary' key."
        )

    policy_map = {item.get("Policy", ""): item for item in summary_list}
    policy_names = list(policy_map.keys())

    # Find Forecast-Driven MILP and ROP baseline
    milp_key = next((k for k in policy_names if "MILP" in k), None)
    rop_key = next((k for k in policy_names if "ROP" in k), None)

    if not milp_key or not rop_key:
        raise HTTPException(
            status_code=500,
            detail="Benchmark artifact missing required MILP or ROP policy entries."
        )

    milp_data = policy_map[milp_key]
    rop_data = policy_map[rop_key]

    milp_cost = float(milp_data["Total Cost ($)"])
    rop_cost = float(rop_data["Total Cost ($)"])
    milp_profit = float(milp_data["Net Profit ($)"])
    rop_profit = float(rop_data["Net Profit ($)"])

    cost_savings_pct = round(((rop_cost - milp_cost) / rop_cost) * 100.0, 2)
    profit_uplift_pct = round(((milp_profit - rop_profit) / rop_profit) * 100.0, 2)

    seeds = data.get("configuration", {}).get("seeds", [42, 101, 777])
    seeds_str = ", ".join(str(s) for s in seeds)
    provenance = f"Validated multi-seed benchmark artifact docs/optimization_benchmark_results.json (seeds: {seeds_str})"

    return DashboardSummaryResponse(
        data_source=provenance,
        policies=policy_names,
        baseline_policy=rop_key,
        optimal_policy=milp_key,
        total_cost=milp_cost,
        net_profit=milp_profit,
        fill_rate_percent=float(milp_data["Fill Rate (%)"]),
        service_level_percent=float(milp_data["Service Level (%)"]),
        stockout_units=float(milp_data["Stockout Units"]),
        average_inventory_units=float(milp_data["Average Inventory (Units)"]),
        inventory_turnover=float(milp_data["Inventory Turnover"]),
        cost_savings_vs_rop_percent=cost_savings_pct,
        net_profit_uplift_percent=profit_uplift_pct,
        last_updated="2026-10-28",
    )
