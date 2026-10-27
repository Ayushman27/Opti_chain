"""
OptiChain AI - Forecasting Evaluation Metrics
Computes industry standard metrics: WAPE, MAE, RMSE, and Forecast Bias.
"""

from __future__ import annotations
from typing import Dict
import numpy as np


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (WAPE = sum(|y - y_hat|) / sum(y) * 100)"""
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage Forecast Bias = sum(y_hat - y) / sum(y) * 100"""
    denom = np.sum(y_true)
    if denom == 0:
        return 0.0
    return float(np.sum(y_pred - y_true) / denom * 100.0)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes all core forecasting accuracy metrics."""
    # Clip negative predictions to 0 for demand
    y_pred_clipped = np.clip(y_pred, 0, None)
    return {
        "WAPE (%)": round(wape(y_true, y_pred_clipped), 2),
        "MAE": round(mae(y_true, y_pred_clipped), 2),
        "RMSE": round(rmse(y_true, y_pred_clipped), 2),
        "Bias (%)": round(forecast_bias(y_true, y_pred_clipped), 2),
    }
