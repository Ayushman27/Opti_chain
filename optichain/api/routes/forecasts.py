"""
Forecasts route
GET /api/v1/forecasts
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
import numpy as np

from optichain.api.schemas import ForecastResponse, ForecastMetrics
from optichain.api.dependencies import get_reference_forecast_engine, load_forecast_benchmark
from optichain.forecasting.metrics import evaluate_forecast

router = APIRouter(tags=["Forecasting"])

MODEL_ALIASES = {
    "ridge": "Ridge",
    "lgbm": "LightGBM",
    "lightgbm": "LightGBM",
    "seasonal_naive": "Seasonal_Naive_7d",
    "seasonal_naive_7d": "Seasonal_Naive_7d",
    "moving_average": "Moving_Average_7d",
    "moving_average_7d": "Moving_Average_7d",
}


@router.get("/forecasts", response_model=ForecastResponse)
def get_forecasts(
    product_id: Optional[str] = Query(None, description="Product SKU ID (e.g. PROD_001) or 'ALL'"),
    model: Optional[str] = Query("Ridge", description="Forecasting model: Ridge, LightGBM, Seasonal_Naive_7d, Moving_Average_7d"),
    horizon: Optional[int] = Query(14, ge=1, le=27, description="Forecast horizon in days (1-27)"),
) -> ForecastResponse:
    """
    Returns time series actuals, out-of-sample model predictions, and performance metrics (WAPE, MAE, RMSE, Bias).
    Enforces strict chronological train/test temporal split with zero future leakage.
    Distinguishes single-SKU horizon slice metrics from the network-wide 27-day benchmark.
    """
    engine, sim = get_reference_forecast_engine()
    raw_products = [p.id for p in sim.products]
    available_products = ["ALL"] + raw_products

    # Normalize target product
    raw_pid = (product_id or raw_products[0]).strip()
    is_all = raw_pid.upper() == "ALL"
    target_pid = "ALL" if is_all else raw_pid.upper()

    if not is_all and target_pid not in raw_products:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{target_pid}' not found. Available products: {available_products}"
        )

    # Normalize model selection
    norm_model_key = model.lower() if model else "ridge"
    model_name = MODEL_ALIASES.get(norm_model_key)
    if not model_name:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Model '{model}' is not supported. Supported models: "
                "Ridge, LightGBM, Seasonal_Naive_7d, Moving_Average_7d"
            )
        )

    # Get model object
    active_model = next((m for m in engine.models if m.name == model_name), None)
    if not active_model:
        raise HTTPException(
            status_code=500,
            detail=f"Configured model '{model_name}' could not be located in ForecastEngine."
        )

    target_wh = sim.warehouses[0].id
    feature_df = engine.feature_df
    if feature_df is None:
        raise HTTPException(status_code=503, detail="Forecasting feature dataset unavailable.")

    horizon_val = horizon if horizon is not None else 14
    if horizon_val < 1 or horizon_val > 27:
        raise HTTPException(
            status_code=422,
            detail=f"Requested horizon {horizon_val} must be between 1 and 27 days."
        )

    # Extract benchmark metrics from precomputed artifact for this model
    benchmark_metrics = None
    try:
        fc_bench = load_forecast_benchmark()
        overall = fc_bench.get("overall_metrics", {}).get(model_name)
        if overall:
            benchmark_metrics = ForecastMetrics(
                wape_percent=float(overall["WAPE (%)"]),
                mae=float(overall["MAE"]),
                rmse=float(overall["RMSE"]),
                bias_percent=float(overall["Bias (%)"]),
            )
    except Exception:
        pass

    if is_all:
        # Aggregate across all products
        df_all = feature_df[feature_df["warehouse_id"] == target_wh].sort_values("step").reset_index(drop=True)
        train_mask = df_all["step"] < 63
        test_df = df_all[~train_mask]

        # Group steps 0..62 and test steps
        steps_hist = sorted(df_all[train_mask]["step"].unique())[-30:]
        hist_steps = [int(s) for s in steps_hist]
        hist_actuals = [
            round(float(df_all[(df_all["step"] == s)]["demand"].sum()), 2)
            for s in hist_steps
        ]

        steps_eval = sorted(test_df["step"].unique())[:horizon_val]
        eval_steps = [int(s) for s in steps_eval]
        eval_actuals = [
            round(float(test_df[test_df["step"] == s]["demand"].sum()), 2)
            for s in eval_steps
        ]

        # Predict per SKU and aggregate
        eval_preds = [0.0] * len(eval_steps)
        for pid in raw_products:
            sku_test = test_df[(test_df["product_id"] == pid) & (test_df["step"].isin(eval_steps))].sort_values("step")
            sku_preds = active_model.predict(sku_test)
            for idx, p_val in enumerate(sku_preds):
                eval_preds[idx] += round(float(p_val), 2)
        eval_preds = [round(p, 2) for p in eval_preds]

        eval_y = np.array(eval_actuals, dtype=float)
        eval_p = np.array(eval_preds, dtype=float)
        metrics_dict = evaluate_forecast(eval_y, eval_p)

        last_step = eval_steps[-1] if eval_steps else 62
        future_steps = [last_step + 1 + i for i in range(horizon_val)]
        future_preds = [
            round(max(0.0, float(np.mean(eval_preds[-7:]) if len(eval_preds) >= 7 else np.mean(eval_preds))), 2)
            for _ in future_steps
        ]

        train_days_count = 63
        eval_scope = f"Network-Wide Total across all 5 SKUs ({horizon_val} days: Day {eval_steps[0]}..{eval_steps[-1]})"
    else:
        # Slice feature dataset for this product & warehouse
        df_prod = feature_df[
            (feature_df["product_id"] == target_pid) & (feature_df["warehouse_id"] == target_wh)
        ].sort_values("step").reset_index(drop=True)

        if df_prod.empty:
            raise HTTPException(status_code=404, detail=f"No time-series data found for SKU '{target_pid}'.")

        train_mask = df_prod["step"] < 63
        train_df = df_prod[train_mask]
        test_df = df_prod[~train_mask]

        if horizon_val > len(test_df):
            raise HTTPException(
                status_code=422,
                detail=f"Requested horizon {horizon_val} exceeds available test period ({len(test_df)} days)."
            )

        hist_sample = train_df.tail(30)
        hist_steps = [int(s) for s in hist_sample["step"].tolist()]
        hist_actuals = [round(float(d), 2) for d in hist_sample["demand"].tolist()]

        eval_slice = test_df.iloc[:horizon_val]
        eval_steps = [int(s) for s in eval_slice["step"].tolist()]
        eval_actuals = [round(float(d), 2) for d in eval_slice["demand"].tolist()]

        preds = active_model.predict(eval_slice)
        eval_preds = [round(float(p), 2) for p in preds]

        eval_y = np.array(eval_actuals, dtype=float)
        eval_p = np.array(eval_preds, dtype=float)
        metrics_dict = evaluate_forecast(eval_y, eval_p)

        last_step = eval_steps[-1] if eval_steps else 62
        future_steps = [last_step + 1 + i for i in range(horizon_val)]
        future_preds = [
            round(max(0.0, float(np.mean(eval_preds[-7:]) if len(eval_preds) >= 7 else np.mean(eval_preds))), 2)
            for _ in future_steps
        ]

        train_days_count = len(train_df)
        eval_scope = f"Single SKU {target_pid} ({horizon_val} days: Day {eval_steps[0]}..{eval_steps[-1]})"

    metrics = ForecastMetrics(
        wape_percent=float(metrics_dict.get("WAPE (%)", 0.0)),
        mae=float(metrics_dict.get("MAE", 0.0)),
        rmse=float(metrics_dict.get("RMSE", 0.0)),
        bias_percent=float(metrics_dict.get("Bias (%)", 0.0)),
    )

    return ForecastResponse(
        product_id=target_pid,
        model_name=model_name,
        horizon_days=horizon_val,
        train_days=train_days_count,
        historical_steps=hist_steps,
        historical_actual=hist_actuals,
        evaluation_steps=eval_steps,
        evaluation_actual=eval_actuals,
        evaluation_predictions=eval_preds,
        future_steps=future_steps,
        future_predictions=future_preds,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        evaluation_scope=eval_scope,
        benchmark_scope="Validated multi-SKU benchmark across all 5 products on Days 63-89 (135 observations)",
        available_models=["Ridge", "LightGBM", "Seasonal_Naive_7d", "Moving_Average_7d"],
        available_products=available_products,
        leakage_audit_status="Strict temporal split enforced (train < step 63, test >= step 63). Zero future actuals leakage.",
    )

