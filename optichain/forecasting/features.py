"""
OptiChain AI - Time-Series Feature Engineering for Demand Forecasting
Constructs rich tabular feature matrices from multi-echelon demand tensors,
including lags, rolling statistical windows, calendar signals, and disruption flags.
"""

from __future__ import annotations
import math
from typing import List, Tuple
import numpy as np
import pandas as pd

from optichain.core.entities import Product, Warehouse, Disruption


class FeatureEngineer:
    """
    Transforms multi-echelon demand tensors and entity metadata into ML feature sets.
    """

    def __init__(self, lags: List[int] = [1, 2, 3, 7, 14], rolling_windows: List[int] = [7, 14]):
        self.lags = lags
        self.rolling_windows = rolling_windows

    def create_features(
        self,
        demand_matrix: np.ndarray,  # Shape: (T, W, P)
        products: List[Product],
        warehouses: List[Warehouse],
        disruptions: List[Disruption] = [],
    ) -> pd.DataFrame:
        T, W, P = demand_matrix.shape
        records = []

        # Category mapping
        cat_map = {p.id: p.category for p in products}
        cost_map = {p.id: p.unit_cost for p in products}

        for t in range(T):
            dow = t % 7
            is_weekend = 1 if dow in [5, 6] else 0
            sin_dow = math.sin(2.0 * math.pi * dow / 7.0)
            cos_dow = math.cos(2.0 * math.pi * dow / 7.0)

            # Check if any disruption is active today
            disr_active = 1 if any(d.is_active(t) for d in disruptions) else 0

            for w in range(W):
                w_id = warehouses[w].id
                for p in range(P):
                    p_id = products[p].id
                    actual_demand = demand_matrix[t, w, p]

                    row = {
                        "step": t,
                        "warehouse_id": w_id,
                        "product_id": p_id,
                        "category": cat_map[p_id],
                        "unit_cost": cost_map[p_id],
                        "day_of_week": dow,
                        "is_weekend": is_weekend,
                        "sin_dow": sin_dow,
                        "cos_dow": cos_dow,
                        "disruption_active": disr_active,
                        "demand": actual_demand,
                    }

                    # Lag features
                    for lag in self.lags:
                        if t - lag >= 0:
                            row[f"lag_{lag}"] = float(demand_matrix[t - lag, w, p])
                        else:
                            row[f"lag_{lag}"] = float(demand_matrix[0, w, p])

                    # Rolling features (using historical demand prior to step t)
                    for win in self.rolling_windows:
                        start_idx = max(0, t - win)
                        hist = demand_matrix[start_idx:t, w, p] if t > 0 else np.array([demand_matrix[0, w, p]])
                        row[f"rolling_mean_{win}"] = float(np.mean(hist))
                        row[f"rolling_std_{win}"] = float(np.std(hist)) if len(hist) > 1 else 0.0
                        row[f"rolling_max_{win}"] = float(np.max(hist))

                    records.append(row)

        df = pd.DataFrame(records)
        # One-hot encode category
        df = pd.get_dummies(df, columns=["category"], drop_first=False)
        return df
