"""
OptiChain AI - Demand Forecasting ML Benchmark Runner
Executes chronological train/test and walk-forward backtests comparing:
1. Seasonal Naive (7-day lag)
2. Moving Average (7-day rolling)
3. Ridge Regression (L2 regularization)
4. LightGBM (Gradient Boosted Trees)

Evaluates:
- Overall test period
- Normal demand periods
- Disruption periods

Saves benchmark metrics and generates actual-vs-predicted plots to docs/images/demand_forecast_comparison.png.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from optichain.config import get_default_config
from optichain.data.generator import SupplyChainDataGenerator
from optichain.forecasting.models import ForecastEngine


def main():
    print("=" * 80)
    print(" OptiChain AI - Phase 3 Demand Forecasting ML Benchmark & Evaluation ")
    print("=" * 80)

    # Standard Phase 1 & 2 baseline configuration: 1 Warehouse, 5 Products, 3 Suppliers, 90 Days
    config = get_default_config()
    print(f"\nConfiguration:")
    print(f"  Warehouses : {config.n_warehouses}")
    print(f"  Products   : {config.n_products}")
    print(f"  Suppliers  : {config.n_suppliers}")
    print(f"  Horizon    : {config.episode_days} days (T=90)")
    print(f"  Random Seed: {config.random_seed}\n")

    # Generate synthetic network & demand
    generator = SupplyChainDataGenerator(config)
    products = generator.generate_products()
    warehouses = generator.generate_warehouses()
    suppliers = generator.generate_suppliers(products)
    disruptions = generator.generate_disruptions(suppliers, warehouses, products)
    demand_matrix = generator.generate_demand_matrix(products, warehouses, disruptions)

    print(f"Generated demand matrix shape: {demand_matrix.shape} (T={demand_matrix.shape[0]}, W={demand_matrix.shape[1]}, P={demand_matrix.shape[2]})")
    active_disruptions = [f"{d.name} (Days {d.start_step}..{d.start_step + d.duration_steps - 1})" for d in disruptions]
    print(f"Active Disruptions: {', '.join(active_disruptions) if active_disruptions else 'None'}\n")


    # Chronological Backtest (70% Train, 30% Test: Train=Days 0..62, Test=Days 63..89)
    train_ratio = 0.70
    split_step = int(round(config.episode_days * train_ratio))
    print(f"Chronological Split: Train Days 0..{split_step - 1} | Test Days {split_step}..{config.episode_days - 1}")

    engine = ForecastEngine(random_seed=config.random_seed)
    results = engine.train_and_backtest(
        demand_matrix,
        products,
        warehouses,
        disruptions,
        train_split_ratio=train_ratio,
    )

    # 1. Overall Test Set Metrics
    print("\n" + "=" * 60)
    print(f" 1. OVERALL TEST PERIOD METRICS (Days {split_step}..{config.episode_days - 1})")
    print("=" * 60)

    summary_df = pd.DataFrame(results["summary_table"])
    print(summary_df.to_string(index=False))

    # 2. Normal Periods Metrics
    print("\n" + "=" * 60)
    print(" 2. NORMAL DEMAND PERIODS METRICS")
    print("=" * 60)
    normal_metrics = results["normal_periods"]
    normal_rows = []
    for model_name, m in normal_metrics.items():
        normal_rows.append({
            "Model": model_name,
            "WAPE (%)": m["WAPE (%)"],
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "Bias (%)": m["Bias (%)"],
        })
    normal_df = pd.DataFrame(normal_rows).sort_values(by="WAPE (%)").reset_index(drop=True)
    print(normal_df.to_string(index=False))

    # 3. Disruption Period Stress Test (Viral Demand Surge Days 25..31)
    print("\n" + "=" * 60)
    print(" 3. DISRUPTION PERIOD STRESS TEST (Viral Surge Days 25..31)")
    print("=" * 60)
    # Train on pre-disruption history (Days 0..24), test across the viral demand spike (Days 25..35)
    surge_engine = ForecastEngine(random_seed=config.random_seed)
    surge_results = surge_engine.train_and_backtest(
        demand_matrix[:36, :, :],
        products,
        warehouses,
        disruptions,
        train_split_ratio=(25.0 / 36.0),
    )
    disrupt_metrics = surge_results.get("disruption_periods", {})
    if disrupt_metrics:
        disrupt_rows = []
        for model_name, m in disrupt_metrics.items():
            disrupt_rows.append({
                "Model": model_name,
                "WAPE (%)": m["WAPE (%)"],
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "Bias (%)": m["Bias (%)"],
            })
        disrupt_df = pd.DataFrame(disrupt_rows).sort_values(by="WAPE (%)").reset_index(drop=True)
        print(disrupt_df.to_string(index=False))
    else:
        print("Disruption metrics evaluated.")


    # 4. Rolling-Origin Walk-Forward CV Evaluation
    print("\n" + "=" * 60)
    print(" 4. ROLLING-ORIGIN WALK-FORWARD EVALUATION (Horizon=7 Days)")
    print("=" * 60)
    cv_df = engine.rolling_origin_evaluation(
        demand_matrix,
        products,
        warehouses,
        disruptions,
        initial_train_days=45,
        step_days=7,
        forecast_horizon=7,
    )
    cv_summary = (
        cv_df.groupby("Model")[["WAPE (%)", "MAE", "RMSE", "Bias (%)"]]
        .mean()
        .reset_index()
        .sort_values(by="WAPE (%)")
    )
    print("Mean Across Rolling-Origin Walk-Forward Folds:")
    print(cv_summary.round(2).to_string(index=False))

    # 5. Generate and Save Actual vs. Predicted Visualizations
    images_dir = root_dir / "docs" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    plot_path = images_dir / "demand_forecast_comparison.png"

    fig, axes = plt.subplots(len(products), 1, figsize=(12, 14), sharex=True)
    if len(products) == 1:
        axes = [axes]

    test_steps = list(range(split_step, config.episode_days))

    for p_idx, prod in enumerate(products):
        ax = axes[p_idx]
        actual_test_demand = demand_matrix[split_step:, 0, p_idx]

        # Extract predictions for this SKU
        prod_mask = (engine.feature_df["product_id"] == prod.id) & (engine.feature_df["step"] >= split_step)
        sub_df = engine.feature_df[prod_mask]

        lgbm_preds = engine.lgbm_model.predict(sub_df)
        ridge_preds = engine.ridge_model.predict(sub_df)
        naive_preds = engine.seasonal_naive_model.predict(sub_df)

        ax.plot(test_steps, actual_test_demand, label="Actual Demand", color="black", linewidth=2.2, marker="o", markersize=3.5)
        ax.plot(test_steps, lgbm_preds, label="LightGBM", color="#0284c7", linewidth=1.8, linestyle="-")
        ax.plot(test_steps, ridge_preds, label="Ridge", color="#10b981", linewidth=1.5, linestyle="--")
        ax.plot(test_steps, naive_preds, label="Seasonal Naive (7d)", color="#f59e0b", linewidth=1.2, linestyle=":")

        ax.set_title(f"Demand Forecast vs Actual: {prod.name} ({prod.id})", fontsize=11, fontweight="bold")
        ax.set_ylabel("Daily Units", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.85, fontsize=8)

    axes[-1].set_xlabel("Simulation Day (t)", fontsize=10, fontweight="bold")
    plt.suptitle(f"OptiChain AI - Phase 3 Demand Forecasting Model Comparison (Test Days {split_step}..{config.episode_days - 1})", fontsize=13, fontweight="bold")
    plt.tight_layout()

    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nSaved forecast comparison plot to: {plot_path}")

    # Also save structured JSON results
    json_path = root_dir / "docs" / "forecast_benchmark_results.json"
    benchmark_payload = {
        "configuration": {
            "warehouses": config.n_warehouses,
            "products": config.n_products,
            "horizon_days": config.episode_days,
            "train_days": split_step,
            "test_days": config.episode_days - split_step,
            "random_seed": config.random_seed,
        },
        "overall_metrics": results["overall_metrics"],
        "normal_metrics": normal_metrics,
        "disruption_metrics": disrupt_metrics,
        "walk_forward_mean": cv_summary.to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)
    print(f"Saved benchmark metrics JSON to: {json_path}")
    print("\nBenchmark completed successfully!\n")


if __name__ == "__main__":
    main()
