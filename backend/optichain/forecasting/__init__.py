"""
OptiChain AI - Forecasting Module
Provides time-series feature engineering, accuracy metrics (WAPE, MAE, RMSE, Bias),
baseline models (Seasonal Naive, Moving Average), ML models (Ridge, LightGBM),
and ForecastEngine for temporal backtesting and rolling-origin evaluation.
"""

from optichain.forecasting.features import FeatureEngineer
from optichain.forecasting.metrics import evaluate_forecast, wape, mae, rmse, forecast_bias
from optichain.forecasting.models import (
    BaseDemandForecaster,
    SeasonalNaiveForecaster,
    BaselineMovingAverageForecaster,
    RidgeForecaster,
    LightGBMForecaster,
    ForecastEngine,
)

__all__ = [
    "FeatureEngineer",
    "evaluate_forecast",
    "wape",
    "mae",
    "rmse",
    "forecast_bias",
    "BaseDemandForecaster",
    "SeasonalNaiveForecaster",
    "BaselineMovingAverageForecaster",
    "RidgeForecaster",
    "LightGBMForecaster",
    "ForecastEngine",
]
