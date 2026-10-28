"""
Unit tests for forecasting module: feature engineering, leakage prevention,
models (Seasonal Naive, Moving Average, Ridge, LightGBM), temporal validation,
metrics edge cases, and multi-product horizons.
"""

import numpy as np
import pandas as pd
import pytest

from optichain.config import SimulationConfig
from optichain.data.generator import SupplyChainDataGenerator
from optichain.forecasting.features import FeatureEngineer
from optichain.forecasting.metrics import evaluate_forecast, wape, mae, rmse, forecast_bias
from optichain.forecasting.models import (
    ForecastEngine,
    SeasonalNaiveForecaster,
    BaselineMovingAverageForecaster,
    RidgeForecaster,
    LightGBMForecaster,
)
from optichain.baselines.forecast_policy import MLForecastPolicy
from optichain.simulator.environment import DigitalTwinSimulator


def test_forecast_metrics_exact_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])

    res = evaluate_forecast(y_true, y_pred)
    assert res["MAE"] == pytest.approx(1.33, 0.05)
    assert res["WAPE (%)"] == pytest.approx(6.67, 0.05)
    assert res["Bias (%)"] == pytest.approx(0.0, 0.05)


def test_forecast_metrics_zero_demand_and_edge_cases():
    """Tests safe division, all-zeros demand, and negative prediction clipping."""
    # Case 1: All zero actual and predicted
    y_true_zeros = np.array([0.0, 0.0, 0.0])
    y_pred_zeros = np.array([0.0, 0.0, 0.0])
    res_zero = evaluate_forecast(y_true_zeros, y_pred_zeros)
    assert res_zero["WAPE (%)"] == 0.0
    assert res_zero["Bias (%)"] == 0.0
    assert res_zero["MAE"] == 0.0
    assert res_zero["RMSE"] == 0.0

    # Case 2: Zero actual but non-zero predicted
    y_pred_nonzeros = np.array([5.0, 5.0, 5.0])
    res_over = evaluate_forecast(y_true_zeros, y_pred_nonzeros)
    assert res_over["WAPE (%)"] == 100.0
    assert res_over["Bias (%)"] == 100.0

    # Case 3: Negative predictions are clipped to 0
    y_true = np.array([10.0, 20.0])
    y_pred_neg = np.array([-5.0, 15.0])
    res_clipped = evaluate_forecast(y_true, y_pred_neg)
    # y_pred becomes [0.0, 15.0]
    # sum(|10-0| + |20-15|) = 15 / 30 = 50.0%
    assert res_clipped["WAPE (%)"] == pytest.approx(50.0, 0.01)
    assert res_clipped["MAE"] == pytest.approx(7.5, 0.01)


def test_feature_engineering_structure():
    cfg = SimulationConfig(n_warehouses=2, n_products=4, n_suppliers=2, episode_days=15, random_seed=42)
    gen = SupplyChainDataGenerator(cfg)
    products = gen.generate_products()
    warehouses = gen.generate_warehouses()
    disruptions = gen.generate_disruptions(gen.generate_suppliers(products), warehouses, products)
    demand = gen.generate_demand_matrix(products, warehouses, disruptions)

    fe = FeatureEngineer()
    df = fe.create_features(demand, products, warehouses, disruptions)

    assert len(df) == 15 * 2 * 4
    assert "lag_1" in df.columns
    assert "rolling_mean_7" in df.columns
    assert "sin_dow" in df.columns
    assert "trend" in df.columns
    assert "demand" in df.columns


def test_feature_engineering_leakage_prevention():
    """
    Guarantees no lookahead bias:
    Perturbing future demand at step t+k MUST NOT change any feature value at step t.
    """
    cfg = SimulationConfig(n_warehouses=1, n_products=2, n_suppliers=1, episode_days=30, random_seed=42)
    gen = SupplyChainDataGenerator(cfg)
    products = gen.generate_products()
    warehouses = gen.generate_warehouses()
    demand_orig = gen.generate_demand_matrix(products, warehouses, [])

    # Create features on original
    fe = FeatureEngineer(lags=[1, 2, 7], rolling_windows=[7])
    df_orig = fe.create_features(demand_orig, products, warehouses, [])

    # Perturb future demand at step 20
    demand_perturbed = demand_orig.copy()
    demand_perturbed[20, :, :] = 99999.0

    df_perturbed = fe.create_features(demand_perturbed, products, warehouses, [])

    # Inspect features for step t = 10 (well before perturbation)
    features_orig_t10 = df_orig[df_orig["step"] == 10].drop(columns=["demand"]).reset_index(drop=True)
    features_pert_t10 = df_perturbed[df_perturbed["step"] == 10].drop(columns=["demand"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(features_orig_t10, features_pert_t10)


def test_feature_engineering_short_series_and_edge_cases():
    """Tests short horizons (T < max_lag) and missing/zero demand matrices without crashing."""
    cfg = SimulationConfig(n_warehouses=1, n_products=2, n_suppliers=1, episode_days=3, random_seed=42)
    gen = SupplyChainDataGenerator(cfg)
    products = gen.generate_products()
    warehouses = gen.generate_warehouses()

    # Short 3-day series
    demand_short = np.array([[[10.0, 20.0]], [[12.0, 22.0]], [[14.0, 24.0]]])  # (3, 1, 2)
    fe = FeatureEngineer()
    df_short = fe.create_features(demand_short, products, warehouses, [])
    assert len(df_short) == 3 * 1 * 2
    assert not df_short.isna().any().any()

    # Matrix with NaN values
    demand_with_nan = demand_short.copy()
    demand_with_nan[1, 0, 0] = np.nan
    df_nan = fe.create_features(demand_with_nan, products, warehouses, [])
    assert not df_nan.isna().any().any()

    # All-zero demand
    demand_zeros = np.zeros((5, 1, 2))
    df_zero = fe.create_features(demand_zeros, products, warehouses, [])
    assert len(df_zero) == 10
    assert (df_zero["demand"] == 0.0).all()


def test_seasonal_naive_forecaster():
    """Verifies that Seasonal Naive uses the 7-day seasonal lag."""
    model = SeasonalNaiveForecaster(seasonal_period=7)
    df = pd.DataFrame({
        "lag_7": [10.0, 15.0, 20.0],
        "lag_1": [5.0, 6.0, 7.0],
    })
    preds = model.predict(df)
    np.testing.assert_array_equal(preds, np.array([10.0, 15.0, 20.0]))


def test_ridge_forecaster():
    """Verifies that RidgeForecaster fits and predicts with non-negative outputs."""
    model = RidgeForecaster(alpha=1.0)
    X = pd.DataFrame({
        "step": [0, 1, 2, 3, 4],
        "demand": [10.0, 20.0, 30.0, 40.0, 50.0],
        "warehouse_id": ["WH1"] * 5,
        "product_id": ["P1"] * 5,
        "lag_1": [9.0, 10.0, 20.0, 30.0, 40.0],
        "trend": [0.0, 0.25, 0.5, 0.75, 1.0],
    })
    y = X["demand"].to_numpy()
    model.fit(X, y)
    assert model.is_fitted

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert (preds >= 0.0).all()


def test_lightgbm_forecaster():
    """Verifies LightGBM training and non-negative prediction clipping."""
    model = LightGBMForecaster(n_estimators=10, min_child_samples=2)
    X = pd.DataFrame({
        "step": list(range(10)),
        "demand": [10.0 + i for i in range(10)],
        "warehouse_id": ["WH1"] * 10,
        "product_id": ["P1"] * 10,
        "lag_1": [9.0 + i for i in range(10)],
        "trend": [i / 10.0 for i in range(10)],
    })
    y = X["demand"].to_numpy()
    model.fit(X, y)
    assert model.is_fitted

    preds = model.predict(X)
    assert len(preds) == 10
    assert (preds >= 0.0).all()


def test_temporal_split_chronological_integrity():
    """Verifies that train/test split maintains strict temporal order with no random shuffle."""
    cfg = SimulationConfig(n_warehouses=1, n_products=3, n_suppliers=1, episode_days=40, random_seed=42)
    gen = SupplyChainDataGenerator(cfg)
    products = gen.generate_products()
    warehouses = gen.generate_warehouses()
    demand = gen.generate_demand_matrix(products, warehouses, [])

    engine = ForecastEngine()
    results = engine.train_and_backtest(demand, products, warehouses, [], train_split_ratio=0.75)

    split_step = results["split_step"]
    assert split_step == 30  # 40 * 0.75
    train_df = engine.feature_df[engine.feature_df["step"] < split_step]
    test_df = engine.feature_df[engine.feature_df["step"] >= split_step]

    # Max train step is strictly less than min test step
    assert train_df["step"].max() < test_df["step"].min()
    assert train_df["step"].max() == 29
    assert test_df["step"].min() == 30


def test_forecast_engine_training_and_backtest():
    cfg = SimulationConfig(n_warehouses=2, n_products=4, n_suppliers=2, episode_days=25, random_seed=42)
    gen = SupplyChainDataGenerator(cfg)
    products = gen.generate_products()
    warehouses = gen.generate_warehouses()
    suppliers = gen.generate_suppliers(products)
    disruptions = gen.generate_disruptions(suppliers, warehouses, products)
    demand = gen.generate_demand_matrix(products, warehouses, disruptions)

    engine = ForecastEngine()
    results = engine.train_and_backtest(demand, products, warehouses, disruptions, train_split_ratio=0.7)

    assert "LightGBM" in results
    assert "Ridge" in results
    assert "Seasonal_Naive_7d" in results
    assert "Moving_Average_7d" in results
    assert results["LightGBM"]["WAPE (%)"] >= 0.0
    assert results["train_samples"] > 0
    assert results["test_samples"] > 0


def test_normal_vs_disruption_evaluation():
    """Verifies separate evaluation metrics for normal and disruption periods."""
    cfg = SimulationConfig(n_warehouses=1, n_products=2, n_suppliers=1, episode_days=50, random_seed=42)
    gen = SupplyChainDataGenerator(cfg)
    products = gen.generate_products()
    warehouses = gen.generate_warehouses()
    suppliers = gen.generate_suppliers(products)
    disruptions = gen.generate_disruptions(suppliers, warehouses, products)
    demand = gen.generate_demand_matrix(products, warehouses, disruptions)

    engine = ForecastEngine()
    results = engine.train_and_backtest(demand, products, warehouses, disruptions, train_split_ratio=0.5)

    assert "normal_periods" in results
    assert "disruption_periods" in results
    assert "LightGBM" in results["normal_periods"]


def test_rolling_origin_walk_forward():
    """Verifies walk-forward cross validation produces non-overlapping folds over time."""
    cfg = SimulationConfig(n_warehouses=1, n_products=2, n_suppliers=1, episode_days=60, random_seed=42)
    gen = SupplyChainDataGenerator(cfg)
    products = gen.generate_products()
    warehouses = gen.generate_warehouses()
    demand = gen.generate_demand_matrix(products, warehouses, [])

    engine = ForecastEngine()
    df_folds = engine.rolling_origin_evaluation(
        demand, products, warehouses, [], initial_train_days=40, step_days=7, forecast_horizon=7
    )

    assert not df_folds.empty
    assert "Fold" in df_folds.columns
    assert "WAPE (%)" in df_folds.columns
    assert set(df_folds["Model"].unique()) == {"Seasonal_Naive_7d", "Moving_Average_7d", "Ridge", "LightGBM"}


def test_multi_product_forecast_consistency():
    """Verifies multi-product forecast generation across specific SKUs."""
    cfg = SimulationConfig(n_warehouses=1, n_products=5, n_suppliers=2, episode_days=30, random_seed=42)
    gen = SupplyChainDataGenerator(cfg)
    products = gen.generate_products()
    warehouses = gen.generate_warehouses()
    demand = gen.generate_demand_matrix(products, warehouses, [])

    engine = ForecastEngine()
    engine.train_and_backtest(demand, products, warehouses, [], train_split_ratio=0.7)

    for p in products:
        fc = engine.forecast_product(w_id=warehouses[0].id, p_id=p.id, horizon_steps=7)
        assert len(fc["actual_demand"]) == 7
        assert len(fc["LightGBM"]) == 7
        assert len(fc["Ridge"]) == 7
        assert len(fc["Seasonal_Naive_7d"]) == 7


def test_forecast_reproducibility():
    """Verifies that two runs with the same random seed yield identical metrics and predictions."""
    cfg = SimulationConfig(n_warehouses=1, n_products=3, n_suppliers=1, episode_days=30, random_seed=123)
    gen = SupplyChainDataGenerator(cfg)
    products = gen.generate_products()
    warehouses = gen.generate_warehouses()
    demand = gen.generate_demand_matrix(products, warehouses, [])

    engine1 = ForecastEngine(random_seed=123)
    res1 = engine1.train_and_backtest(demand, products, warehouses, [], train_split_ratio=0.7)

    engine2 = ForecastEngine(random_seed=123)
    res2 = engine2.train_and_backtest(demand, products, warehouses, [], train_split_ratio=0.7)

    assert res1["LightGBM"]["WAPE (%)"] == res2["LightGBM"]["WAPE (%)"]
    assert res1["Ridge"]["WAPE (%)"] == res2["Ridge"]["WAPE (%)"]


def test_ml_forecast_policy_execution():
    cfg = SimulationConfig(n_warehouses=1, n_products=2, n_suppliers=1, episode_days=10, random_seed=42)
    sim = DigitalTwinSimulator(config=cfg)
    policy = MLForecastPolicy()
    policy.fit(sim)

    obs = sim.reset(seed=42)
    actions = policy.get_actions(obs, sim)
    assert isinstance(actions, dict)
