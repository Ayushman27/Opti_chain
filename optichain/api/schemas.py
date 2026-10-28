"""
OptiChain AI - API Pydantic Schemas
Strict schemas for request validation, domain types, metrics, and responses.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# -------------------------------------------------------------------------
# Health Schemas
# -------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall service status: healthy, degraded, or unhealthy")
    version: str = Field(..., description="Application version")
    service: str = Field(default="OptiChain AI Backend", description="Service identifier")
    cbc_solver_available: bool = Field(..., description="Whether the CBC MILP solver binary is available")
    benchmark_artifact_present: bool = Field(..., description="Whether precomputed benchmark results exist on disk")
    timestamp: str = Field(..., description="Current server ISO 8601 timestamp")


# -------------------------------------------------------------------------
# Dashboard Summary Schemas
# -------------------------------------------------------------------------

class DashboardSummaryResponse(BaseModel):
    data_source: str = Field(..., description="Provenance of metrics, e.g. benchmark artifact vs fresh run")
    policies: List[str] = Field(..., description="Names of all evaluated policies")
    baseline_policy: str = Field(..., description="Reference baseline policy name (e.g. ROP Order-Up-To)")
    optimal_policy: str = Field(..., description="Best performing policy name (e.g. Forecast-Driven MILP)")
    
    # KPIs for the top-performing policy (MILP)
    total_cost: float = Field(..., description="Mean total operating cost in USD")
    net_profit: float = Field(..., description="Mean net profit in USD")
    fill_rate_percent: float = Field(..., description="Customer fill rate percentage (units fulfilled / demand)")
    service_level_percent: float = Field(..., description="Cycle service level percentage (order cycles without stockouts)")
    stockout_units: float = Field(..., description="Mean total unmet stockout units")
    average_inventory_units: float = Field(..., description="Mean inventory units on hand per step")
    inventory_turnover: float = Field(..., description="Annualized inventory turnover ratio")
    
    # Relative improvements
    cost_savings_vs_rop_percent: float = Field(..., description="Cost savings vs ROP policy in percent")
    net_profit_uplift_percent: float = Field(..., description="Profit uplift vs ROP policy in percent")
    last_updated: Optional[str] = Field(None, description="Artifact timestamp if available")


# -------------------------------------------------------------------------
# Benchmarks Schemas
# -------------------------------------------------------------------------

class PolicyMetricStats(BaseModel):
    mean: float = Field(..., description="Sample mean")
    std: Optional[float] = Field(None, description="Sample standard deviation across seeds (ddof=1)")


class PolicyBenchmarkItem(BaseModel):
    policy: str = Field(..., description="Policy display name")
    fill_rate_percent: float = Field(..., description="Fill rate percentage")
    service_level_percent: float = Field(..., description="Cycle service level percentage")
    total_cost_usd: float = Field(..., description="Total supply chain cost in USD")
    stockout_units: float = Field(..., description="Mean stockout units")
    average_inventory_units: float = Field(..., description="Average inventory units on hand")
    net_profit_usd: float = Field(..., description="Net profit in USD")
    inventory_turnover: float = Field(..., description="Inventory turnover ratio")
    
    # Cost components
    holding_cost_usd: float = Field(..., description="Holding cost in USD")
    stockout_cost_usd: float = Field(..., description="Stockout penalty cost in USD")
    procurement_cost_usd: float = Field(..., description="Unit procurement cost in USD")
    fixed_setup_usd: float = Field(..., description="Fixed order setup cost in USD")
    transport_usd: float = Field(..., description="Transportation freight cost in USD")
    facility_operating_usd: float = Field(..., description="Facility overhead cost in USD")
    
    # Statistical dispersion where available
    stats: Optional[Dict[str, PolicyMetricStats]] = Field(None, description="Per-metric mean and sample std")


class BenchmarksResponse(BaseModel):
    configuration: Dict[str, Any] = Field(..., description="Benchmark simulation network dimensions and seeds")
    policies: List[PolicyBenchmarkItem] = Field(..., description="Summary metrics for each policy")
    detailed_cost_breakdown: List[Dict[str, Any]] = Field(..., description="Cost breakdown tables")
    per_seed_results: List[Dict[str, Any]] = Field(..., description="Raw seed-by-seed results for seeds 42, 101, 777")
    solver_diagnostics: Optional[Dict[str, Any]] = Field(None, description="CBC solver solve time, optimality distributions")
    provenance: str = Field(..., description="Data origin and seed documentation")
    units: Dict[str, str] = Field(
        default={
            "costs": "USD ($)",
            "percentages": "Percent (%) - absolute points",
            "inventory": "Physical units",
            "turnover": "Ratio (Cost of Goods Sold / Average Inventory)",
        },
        description="Explicit units definition",
    )


# -------------------------------------------------------------------------
# Forecasting Schemas
# -------------------------------------------------------------------------

class ForecastQuery(BaseModel):
    product_id: Optional[str] = Field(None, description="Optional target product ID (e.g. PROD_001)")
    model: Optional[str] = Field("Ridge", description="Forecasting model: Ridge, LightGBM, Seasonal_Naive_7d, Moving_Average_7d")
    horizon: Optional[int] = Field(14, ge=1, le=30, description="Forecast horizon in days (1-30)")


class ForecastMetrics(BaseModel):
    wape_percent: float = Field(..., description="Weighted Absolute Percentage Error (%)")
    mae: float = Field(..., description="Mean Absolute Error (units)")
    rmse: float = Field(..., description="Root Mean Squared Error (units)")
    bias_percent: float = Field(..., description="Forecast Bias percentage (%)")


class ForecastResponse(BaseModel):
    product_id: str = Field(..., description="Target product identifier")
    model_name: str = Field(..., description="Active model algorithm")
    horizon_days: int = Field(..., description="Forecast horizon length in days")
    train_days: int = Field(..., description="Number of historical days used for model fit")
    
    # Time series points
    historical_steps: List[int] = Field(..., description="Day indices of historical actual demand")
    historical_actual: List[float] = Field(..., description="Historical actual demand observations")
    
    evaluation_steps: List[int] = Field(..., description="Day indices of out-of-sample evaluation period")
    evaluation_actual: List[float] = Field(..., description="Actual demand during test evaluation")
    evaluation_predictions: List[float] = Field(..., description="Model predictions during test evaluation")
    
    future_steps: List[int] = Field(..., description="Day indices for future horizon steps")
    future_predictions: List[float] = Field(..., description="Forward forecast predictions")
    
    metrics: ForecastMetrics = Field(..., description="Model evaluation performance metrics for active SKU and selected horizon")
    benchmark_metrics: Optional[ForecastMetrics] = Field(None, description="Validated benchmark metrics across all products and full 27-day test period")
    evaluation_scope: str = Field(..., description="Scope of the SKU-specific evaluation window")
    benchmark_scope: str = Field(
        default="Full 27-day test window (Days 63-89) across all 5 SKUs (135 observations)",
        description="Scope of the benchmark metrics from docs/forecast_benchmark_results.json",
    )
    available_models: List[str] = Field(..., description="List of supported models")
    available_products: List[str] = Field(..., description="List of available products")
    leakage_audit_status: str = Field(
        default="Strict zero-leakage temporal split enforced. Future evaluation demand withheld from training features.",
        description="Confirmation of anti-leakage compliance",
    )



# -------------------------------------------------------------------------
# Inventory Schemas
# -------------------------------------------------------------------------

class ProductInventoryItem(BaseModel):
    product_id: str = Field(..., description="Product ID")
    product_name: str = Field(..., description="Product Name")
    category: str = Field(..., description="Product Category")
    unit_cost: float = Field(..., description="Procurement cost per unit ($)")
    unit_price: float = Field(..., description="Selling price per unit ($)")
    on_hand_units: float = Field(..., description="Current physical stock on hand")
    pipeline_units: float = Field(..., description="Inventory currently ordered in transit")
    recent_demand_rate: float = Field(..., description="Average daily demand over recent steps")
    reorder_point: float = Field(..., description="Calculated Reorder Point (ROP)")
    safety_stock: float = Field(..., description="Calculated safety stock buffer")
    fulfilled_units: float = Field(..., description="Total cumulative demand fulfilled")
    unmet_units: float = Field(..., description="Total cumulative demand lost to stockout")
    stock_status: str = Field(..., description="Status: NORMAL, LOW_STOCK, CRITICAL_STOCKOUT, or OVERSTOCKED")
    recommendation: Optional[str] = Field(None, description="Operational replenishment recommendation")


class InventoryResponse(BaseModel):
    as_of_step: int = Field(..., description="Simulation step / day corresponding to snapshot")
    total_on_hand_units: float = Field(..., description="Sum of on-hand inventory across all products")
    total_pipeline_units: float = Field(..., description="Sum of inbound pipeline orders")
    total_fulfilled_demand: float = Field(..., description="Total fulfilled units to date")
    total_unmet_demand: float = Field(..., description="Total unmet units to date")
    overall_fill_rate_percent: float = Field(..., description="Current overall fill rate (%)")
    products: List[ProductInventoryItem] = Field(..., description="Product-level inventory states")
    data_source: str = Field(..., description="Digital twin simulator state provenance")


# -------------------------------------------------------------------------
# Optimization Schemas
# -------------------------------------------------------------------------

class OptimizationRequest(BaseModel):
    horizon: int = Field(default=7, ge=1, le=30, description="Planning horizon in days (1-30)")
    safety_buffer_factor: float = Field(default=0.2, ge=0.0, le=2.0, description="Safety buffer factor above forecasted demand")
    time_limit_sec: int = Field(default=5, ge=1, le=60, description="PuLP/CBC solver timeout in seconds (1-60)")
    forecast_method: str = Field(
        default="ridge",
        description="Forecast method: 'heuristic_lag7', 'seasonal_naive', 'ridge', 'lgbm'",
    )
    product_ids: Optional[List[str]] = Field(None, description="Optional subset of product IDs to plan for")

    @field_validator("forecast_method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        allowed = {"heuristic_lag7", "seasonal_naive", "ridge", "lgbm"}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid forecast_method '{v}'. Allowed methods: {sorted(allowed)}")
        return v.lower()


class RecommendedOrder(BaseModel):
    step: int = Field(..., description="Planning horizon step when order is placed")
    product_id: str = Field(..., description="Ordered product ID")
    warehouse_id: str = Field(..., description="Target warehouse destination")
    supplier_id: str = Field(..., description="Selected supplier ID")
    quantity: float = Field(..., description="Recommended order quantity in units")
    unit_cost_usd: float = Field(..., description="Unit procurement price ($)")
    estimated_cost_usd: float = Field(..., description="Estimated line cost ($)")
    lead_time_days: int = Field(..., description="Expected supplier lead time (days)")


class OptimizationResponse(BaseModel):
    solver_status: str = Field(..., description="PuLP solver status string (e.g. 'Optimal', 'IntegerFeasible', 'Infeasible')")
    termination_reason: str = Field(..., description="Classification: ProvenOptimal, FeasibleIncumbentUnproven, Infeasible, or Error")
    is_proven_optimal: bool = Field(..., description="True ONLY if CBC mathematically proved optimality (status=Optimal)")
    is_feasible: bool = Field(..., description="True if a feasible solution was found")
    solve_time_sec: float = Field(..., description="Elapsed solver execution time in seconds")
    planning_horizon_days: int = Field(..., description="Horizon length used in optimization")
    forecast_method_used: str = Field(..., description="Demand forecasting method feeding the MILP")
    
    recommended_orders: List[RecommendedOrder] = Field(..., description="List of suggested replenishment orders")
    total_recommended_units: float = Field(..., description="Total units recommended across all orders")
    total_estimated_cost_usd: float = Field(..., description="Total estimated procurement cost ($)")
    
    decision_review_notice: str = Field(
        default="READ-ONLY DECISION REVIEW: This is an analytical decision recommendation generated by the PuLP/CBC MILP optimizer. No live purchase orders have been submitted or committed to suppliers.",
        description="Explicit non-execution disclaimer",
    )


# -------------------------------------------------------------------------
# Simulation Schemas
# -------------------------------------------------------------------------

class SimulationRunRequest(BaseModel):
    policy: str = Field(default="min_max", description="Policy name: 'min_max', 'rop', 'periodic', 'forecast', 'milp'")
    episode_days: int = Field(default=90, ge=5, le=180, description="Simulation duration in days (5-180)")
    random_seed: int = Field(default=42, ge=0, le=1000000, description="RNG seed for reproducibility")
    n_warehouses: int = Field(default=1, ge=1, le=5, description="Number of warehouses")
    n_products: int = Field(default=5, ge=1, le=50, description="Number of products")
    n_suppliers: int = Field(default=3, ge=1, le=20, description="Number of suppliers")

    @field_validator("policy")
    @classmethod
    def validate_policy(cls, v: str) -> str:
        allowed = {"min_max", "rop", "periodic", "forecast", "milp"}
        if v.lower() not in allowed:
            raise ValueError(f"Unknown policy '{v}'. Allowed: {sorted(allowed)}")
        return v.lower()


class SimulationRunResponse(BaseModel):
    run_id: str = Field(..., description="Unique simulation execution identifier")
    policy_name: str = Field(..., description="Policy display name")
    episode_days: int = Field(..., description="Simulated days")
    seed: int = Field(..., description="Random seed used")
    kpis: Dict[str, Any] = Field(..., description="Final KPI metrics (fill rate, cost, profit, etc.)")
    cost_breakdown: Dict[str, float] = Field(..., description="Detailed breakdown of all cost components")
    step_history_sample: List[Dict[str, Any]] = Field(..., description="Sample of step transitions (first, mid, last steps)")
    total_steps: int = Field(..., description="Total completed step count")
    disruptions_count: int = Field(..., description="Number of disruption events simulated")
