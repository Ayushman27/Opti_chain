"""
OptiChain AI - Forecasting Models & Backtesting Pipeline
Implements LightGBM gradient boosted trees, Ridge regression, seasonal naive,
and moving average baselines with strict temporal backtesting and rolling-origin validation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from optichain.core.entities import Product, Warehouse, DisruptionEvent
from optichain.forecasting.features import FeatureEngineer
from optichain.forecasting.metrics import evaluate_forecast


class BaseDemandForecaster(ABC):
    """Abstract interface for all supply chain demand forecasting models."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass


class SeasonalNaiveForecaster(BaseDemandForecaster):
    """
    Seasonal Naive Baseline Forecaster:
    Predicts demand at step t using the observed demand from the exact same
    day of the seasonal cycle (lag 7 by default for weekly seasonality).
    """

    def __init__(self, seasonal_period: int = 7, name: str = "Seasonal_Naive_7d"):
        super().__init__(name=name)
        self.seasonal_period = seasonal_period
        self.fallback_mean: float = 10.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.fallback_mean = float(np.mean(y)) if len(y) > 0 else 10.0
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        col = f"lag_{self.seasonal_period}"
        if col in X.columns:
            preds = X[col].to_numpy(dtype=float).copy()
        elif "lag_1" in X.columns:
            preds = X["lag_1"].to_numpy(dtype=float).copy()
        else:
            preds = np.full(len(X), self.fallback_mean, dtype=float)

        return np.clip(preds, 0, None)


class BaselineMovingAverageForecaster(BaseDemandForecaster):
    """Rolling moving average baseline forecaster (7-day window)."""

    def __init__(self, window: int = 7, name: str = "Moving_Average_7d"):
        super().__init__(name=name)
        self.window = window
        self.fallback_mean: float = 10.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.fallback_mean = float(np.mean(y)) if len(y) > 0 else 10.0
        self.is_fitted = True

    def predict(self, feature_df: pd.DataFrame) -> np.ndarray:
        col = f"rolling_mean_{self.window}"
        if col in feature_df.columns:
            preds = feature_df[col].to_numpy(dtype=float).copy()
        else:
            preds = np.full(len(feature_df), self.fallback_mean, dtype=float)
        return np.clip(preds, 0, None)


class RidgeForecaster(BaseDemandForecaster):
    """
    Ridge Regression demand forecaster with feature scaling and L2 regularization.
    """

    def __init__(self, alpha: float = 1.0, random_state: int = 42, name: str = "Ridge"):
        super().__init__(name=name)
        self.alpha = alpha
        self.random_state = random_state
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=random_state)),
        ])
        self.feature_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.feature_cols = [
            c for c in X.columns
            if c not in ["step", "demand", "warehouse_id", "product_id"]
        ]
        X_mat = X[self.feature_cols].to_numpy(dtype=float)
        # Fill any unexpected NaN with 0
        X_mat = np.nan_to_num(X_mat, nan=0.0)
        self.pipeline.fit(X_mat, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(len(X), dtype=float)
        X_mat = X[self.feature_cols].to_numpy(dtype=float)
        X_mat = np.nan_to_num(X_mat, nan=0.0)
        preds = self.pipeline.predict(X_mat)
        return np.clip(preds, 0, None)


class LightGBMForecaster(BaseDemandForecaster):
    """
    LightGBM Gradient Boosted Trees time-series demand regression model.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 5,
        random_state: int = 42,
        name: str = "LightGBM",
    ):
        super().__init__(name=name)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            random_state=random_state,
            verbosity=-1,
        )
        self.feature_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.feature_cols = [
            c for c in X.columns
            if c not in ["step", "demand", "warehouse_id", "product_id"]
        ]
        self.model.fit(X[self.feature_cols], y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(len(X), dtype=float)
        preds = self.model.predict(X[self.feature_cols])
        return np.clip(preds, 0, None)


class ForecastEngine:
    """
    Coordinates end-to-end dataset generation, feature engineering,
    strict chronological temporal train/test splitting, multi-model evaluation,
    normal vs disruption slicing, rolling-origin walk-forward CV,
    and out-of-sample forward projections.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.fe = FeatureEngineer()
        self.seasonal_naive_model = SeasonalNaiveForecaster(seasonal_period=7)
        self.ma_model = BaselineMovingAverageForecaster(window=7)
        self.ridge_model = RidgeForecaster(alpha=1.0, random_state=random_seed)
        self.lgbm_model = LightGBMForecaster(random_state=random_seed)
        self.models: List[BaseDemandForecaster] = [
            self.seasonal_naive_model,
            self.ma_model,
            self.ridge_model,
            self.lgbm_model,
        ]
        self.feature_df: Optional[pd.DataFrame] = None
        self.is_fitted: bool = False
        self.last_results: Dict[str, Any] = {}

    def train_and_backtest(
        self,
        demand_matrix: np.ndarray,
        products: List[Product],
        warehouses: List[Warehouse],
        disruptions: List[DisruptionEvent] = [],
        train_split_ratio: float = 0.70,
    ) -> Dict[str, Any]:
        """
        Creates features, splits chronologically into train/test,
        trains candidate models (Seasonal Naive, Moving Average, Ridge, LightGBM),
        evaluates overall test performance as well as normal vs disruption sub-periods,
        and returns structured benchmarking metrics.
        """
        self.feature_df = self.fe.create_features(
            demand_matrix, products, warehouses, disruptions
        )

        T = demand_matrix.shape[0]
        split_step = int(round(T * train_split_ratio))

        # Chronological train/test split: strictly step < split_step vs step >= split_step
        train_df = self.feature_df[self.feature_df["step"] < split_step].copy()
        test_df = self.feature_df[self.feature_df["step"] >= split_step].copy()

        y_train = train_df["demand"].to_numpy(dtype=float)
        y_test = test_df["demand"].to_numpy(dtype=float)

        # Fit all candidate models on train set
        for model in self.models:
            model.fit(train_df, y_train)
        self.is_fitted = True

        # Generate predictions for all models
        predictions: Dict[str, np.ndarray] = {}
        overall_metrics: Dict[str, Dict[str, float]] = {}

        for model in self.models:
            preds = model.predict(test_df)
            predictions[model.name] = preds
            overall_metrics[model.name] = evaluate_forecast(y_test, preds)

        # Slice test evaluation into Normal vs Disruption periods
        is_disruption = test_df["disruption_active"].to_numpy() == 1
        is_normal = ~is_disruption

        normal_metrics: Dict[str, Dict[str, float]] = {}
        disruption_metrics: Dict[str, Dict[str, float]] = {}

        if np.any(is_normal):
            y_normal = y_test[is_normal]
            for model_name, preds in predictions.items():
                normal_metrics[model_name] = evaluate_forecast(y_normal, preds[is_normal])

        if np.any(is_disruption):
            y_disrupt = y_test[is_disruption]
            for model_name, preds in predictions.items():
                disruption_metrics[model_name] = evaluate_forecast(y_disrupt, preds[is_disruption])

        # Comparative summary DataFrame
        summary_rows = []
        for model in self.models:
            m = overall_metrics[model.name]
            summary_rows.append({
                "Model": model.name,
                "WAPE (%)": m["WAPE (%)"],
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "Bias (%)": m["Bias (%)"],
            })
        summary_df = pd.DataFrame(summary_rows).sort_values(by="WAPE (%)").reset_index(drop=True)

        lgbm_preds = predictions["LightGBM"]
        ma_preds = predictions["Moving_Average_7d"]
        ridge_preds = predictions["Ridge"]
        naive_preds = predictions["Seasonal_Naive_7d"]

        results: Dict[str, Any] = {
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "split_step": split_step,
            "horizon_days": T - split_step,
            "LightGBM": overall_metrics["LightGBM"],
            "Ridge": overall_metrics["Ridge"],
            "Moving_Average_7d": overall_metrics["Moving_Average_7d"],
            "Seasonal_Naive_7d": overall_metrics["Seasonal_Naive_7d"],
            "overall_metrics": overall_metrics,
            "normal_periods": normal_metrics,
            "disruption_periods": disruption_metrics,
            "summary_table": summary_df.to_dict(orient="records"),
            "y_test_sample": y_test[:100].tolist(),
            "lgbm_preds_sample": [round(float(p), 1) for p in lgbm_preds[:100]],
            "ma_preds_sample": [round(float(p), 1) for p in ma_preds[:100]],
            "ridge_preds_sample": [round(float(p), 1) for p in ridge_preds[:100]],
            "naive_preds_sample": [round(float(p), 1) for p in naive_preds[:100]],
        }
        self.last_results = results
        return results

    def rolling_origin_evaluation(
        self,
        demand_matrix: np.ndarray,
        products: List[Product],
        warehouses: List[Warehouse],
        disruptions: List[DisruptionEvent] = [],
        initial_train_days: int = 45,
        step_days: int = 7,
        forecast_horizon: int = 7,
    ) -> pd.DataFrame:
        """
        Executes Walk-Forward / Rolling-Origin cross-validation across time horizons.
        Ensures strict chronological boundaries with zero lookahead bias.
        """
        # Generate fresh feature table for the specific demand matrix to guarantee data alignment
        feature_df = self.fe.create_features(
            demand_matrix, products, warehouses, disruptions
        )
        self.feature_df = feature_df

        T = demand_matrix.shape[0]
        fold_results: List[Dict[str, Any]] = []

        cutoff = initial_train_days
        fold = 1

        while cutoff + forecast_horizon <= T:
            train_df = feature_df[feature_df["step"] < cutoff]
            test_df = feature_df[
                (feature_df["step"] >= cutoff)
                & (feature_df["step"] < cutoff + forecast_horizon)
            ]


            y_train = train_df["demand"].to_numpy(dtype=float)
            y_test = test_df["demand"].to_numpy(dtype=float)

            # Fit and evaluate each model on this fold
            for model in self.models:
                model.fit(train_df, y_train)
                preds = model.predict(test_df)
                m = evaluate_forecast(y_test, preds)
                fold_results.append({
                    "Fold": fold,
                    "Train_End_Day": cutoff - 1,
                    "Test_Days": f"{cutoff}..{cutoff + forecast_horizon - 1}",
                    "Model": model.name,
                    "WAPE (%)": m["WAPE (%)"],
                    "MAE": m["MAE"],
                    "RMSE": m["RMSE"],
                    "Bias (%)": m["Bias (%)"],
                })

            cutoff += step_days
            fold += 1

        df_folds = pd.DataFrame(fold_results)
        return df_folds

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
        Used for dynamic base-stock inventory replenishment decisions.
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

    def forecast_product(
        self,
        w_id: str,
        p_id: str,
        horizon_steps: int = 7,
    ) -> Dict[str, Any]:
        """
        Generates forward projections for a specific warehouse and product across all candidate models.
        """
        if not self.is_fitted or self.feature_df is None:
            raise RuntimeError("ForecastEngine must be trained before calling forecast_product().")

        subset = self.feature_df[
            (self.feature_df["warehouse_id"] == w_id)
            & (self.feature_df["product_id"] == p_id)
        ].tail(horizon_steps)

        if subset.empty:
            return {}

        return {
            "steps": subset["step"].tolist(),
            "actual_demand": subset["demand"].tolist(),
            "LightGBM": [round(float(p), 1) for p in self.lgbm_model.predict(subset)],
            "Ridge": [round(float(p), 1) for p in self.ridge_model.predict(subset)],
            "Seasonal_Naive_7d": [round(float(p), 1) for p in self.seasonal_naive_model.predict(subset)],
            "Moving_Average_7d": [round(float(p), 1) for p in self.ma_model.predict(subset)],
        }
