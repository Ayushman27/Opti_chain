"""
Unit tests for forecasting module: feature engineering, LightGBM model, backtesting, and metrics.
"""

import numpy as np
import pytest
from optichain.core.entities import SimulationConfig
from optichain.data.generator import SupplyChainDataGenerator
from optichain.forecasting.features import FeatureEngineer
from optichain.forecasting.metrics import evaluate_forecast, wape, mae, rmse, forecast_bias
from optichain.forecasting.models import ForecastEngine
from optichain.baselines.forecast_policy import MLForecastPolicy
from optichain.simulator.environment import DigitalTwinSimulator


def test_forecast_metrics_exact_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])

    res = evaluate_forecast(y_true, y_pred)
    assert res["MAE"] == pytest.approx(1.33, 0.05)
    assert res["WAPE (%)"] == pytest.approx(6.67, 0.05)
    assert res["Bias (%)"] == pytest.approx(0.0, 0.05)


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
    assert "demand" in df.columns


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
    assert "Moving_Average_7d" in results
    assert results["LightGBM"]["WAPE (%)"] >= 0.0
    assert results["train_samples"] > 0
    assert results["test_samples"] > 0


def test_ml_forecast_policy_execution():
    cfg = SimulationConfig(n_warehouses=1, n_products=2, n_suppliers=1, episode_days=10, random_seed=42)
    sim = DigitalTwinSimulator(config=cfg)
    policy = MLForecastPolicy()
    policy.fit(sim)

    obs = sim.reset(seed=42)
    actions = policy.get_actions(obs, sim)
    assert isinstance(actions, dict)
