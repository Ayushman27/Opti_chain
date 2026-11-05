"""
OptiChain AI - Forecasting Evaluation Metrics
Computes industry standard metrics: WAPE, MAE, RMSE, and Forecast Bias.
"""

from __future__ import annotations
from typing import Dict
import numpy as np


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error (WAPE):
    WAPE = sum(|y - y_hat|) / sum(|y|) * 100%

    Safe handling:
    - If sum(|y|) == 0 and sum(|y_hat|) == 0 -> 0.0%
    - If sum(|y|) == 0 and sum(|y_hat|) > 0 -> 100.0% (predicted demand when actual was zero)
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    denom = float(np.sum(np.abs(yt)))
    numer = float(np.sum(np.abs(yt - yp)))

    if denom == 0.0:
        return 0.0 if numer == 0.0 else 100.0

    return float((numer / denom) * 100.0)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error: MAE = mean(|y - y_hat|)"""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if len(yt) == 0:
        return 0.0
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error: RMSE = sqrt(mean((y - y_hat)^2))"""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if len(yt) == 0:
        return 0.0
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Percentage Forecast Bias = sum(y_hat - y) / sum(y) * 100%
    Positive value indicates over-forecasting; negative indicates under-forecasting.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    denom = float(np.sum(yt))
    if denom == 0.0:
        numer = float(np.sum(yp - yt))
        return 0.0 if numer == 0.0 else (100.0 if numer > 0 else -100.0)

    return float((np.sum(yp - yt) / denom) * 100.0)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes all core forecasting accuracy metrics with non-negative clipping."""
    # Demand is strictly non-negative: clip negative predictions to 0
    yp_clipped = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    yt = np.asarray(y_true, dtype=float)

    return {
        "WAPE (%)": round(wape(yt, yp_clipped), 2),
        "MAE": round(mae(yt, yp_clipped), 2),
        "RMSE": round(rmse(yt, yp_clipped), 2),
        "Bias (%)": round(forecast_bias(yt, yp_clipped), 2),
    }

