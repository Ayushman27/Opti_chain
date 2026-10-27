"""
OptiChain AI - Forecasting Models & Backtesting Pipeline
Implements LightGBM gradient boosted trees, moving average baselines,
and multi-step temporal backtesting for supply chain demand.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from optichain.core.entities import Product, Warehouse, Disruption
from optichain.forecasting.features import FeatureEngineer
from optichain.forecasting.metrics import evaluate_forecast


class BaselineMovingAverageForecaster:
    """Simple 7-day rolling mean baseline forecaster."""

    def __init__(self, window: int = 7):
        self.window = window

    def predict(self, feature_df: pd.DataFrame) -> np.ndarray:
        return feature_df[f"rolling_mean_{self.window}"].to_numpy()


class LightGBMForecaster:
    """LightGBM time-series demand regression model."""

    def __init__(self, n_estimators: int = 150, learning_rate: float = 0.05, max_depth: int = 6):
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
        )
        self.feature_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.feature_cols = [c for c in X.columns if c not in ["step", "demand", "warehouse_id", "product_id"]]
        self.model.fit(X[self.feature_cols], y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X[self.feature_cols])
        return np.clip(preds, 0, None)


class ForecastEngine:
    """
    Coordinates end-to-end dataset generation, training, backtesting,
    and out-of-sample forward projections.
    """

    def __init__(self):
        self.fe = FeatureEngineer()
        self.lgbm_model = LightGBMForecaster()
        self.ma_model = BaselineMovingAverageForecaster()
        self.feature_df: Optional[pd.DataFrame] = None
        self.is_fitted: bool = False

    def train_and_backtest(
        self,
        demand_matrix: np.ndarray,
        products: List[Product],
        warehouses: List[Warehouse],
        disruptions: List[Disruption] = [],
        train_split_ratio: float = 0.70,
    ) -> Dict[str, Any]:
        """
        Creates features, splits temporally into train/test, evaluates
        Moving Average vs LightGBM, and returns comprehensive metrics.
        """
        self.feature_df = self.fe.create_features(
            demand_matrix, products, warehouses, disruptions
        )

        T = demand_matrix.shape[0]
        split_step = int(T * train_split_ratio)

        train_df = self.feature_df[self.feature_df["step"] < split_step].copy()
        test_df = self.feature_df[self.feature_df["step"] >= split_step].copy()

        y_train = train_df["demand"].to_numpy()
        y_test = test_df["demand"].to_numpy()

        # Fit LightGBM
        self.lgbm_model.fit(train_df, y_train)
        self.is_fitted = True

        # Generate predictions
        lgbm_preds = self.lgbm_model.predict(test_df)
        ma_preds = self.ma_model.predict(test_df)

        lgbm_metrics = evaluate_forecast(y_test, lgbm_preds)
        ma_metrics = evaluate_forecast(y_test, ma_preds)

        return {
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "split_step": split_step,
            "LightGBM": lgbm_metrics,
            "Moving_Average_7d": ma_metrics,
            "y_test_sample": y_test[:100].tolist(),
            "lgbm_preds_sample": [round(float(p), 1) for p in lgbm_preds[:100]],
            "ma_preds_sample": [round(float(p), 1) for p in ma_preds[:100]],
        }

    def predict_next_lead_time_demand(
        self,
        w_id: str,
        p_id: str,
        current_step: int,
        lead_time_days: int,
        fallback_mean: float,
    ) -> float:
        """
        Predicts total demand over the upcoming lead-time window [current_step, current_step + lead_time_days].
        """
        if not self.is_fitted or self.feature_df is None:
            return fallback_mean * lead_time_days

        subset = self.feature_df[
            (self.feature_df["warehouse_id"] == w_id)
            & (self.feature_df["product_id"] == p_id)
            & (self.feature_df["step"] >= current_step)
            & (self.feature_df["step"] < current_step + lead_time_days)
        ]

        if subset.empty:
            return fallback_mean * lead_time_days

        preds = self.lgbm_model.predict(subset)
        return float(np.sum(preds))
