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

        # Clean NaN or missing values in input demand tensor
        clean_demand = np.nan_to_num(demand_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Entity metadata mappings
        cat_map = {p.id: p.category for p in products}
        cost_map = {p.id: p.unit_cost for p in products}
        price_map = {p.id: p.selling_price for p in products}

        for t in range(T):
            dow = t % 7
            is_weekend = 1 if dow in [5, 6] else 0
            sin_dow = math.sin(2.0 * math.pi * dow / 7.0)
            cos_dow = math.cos(2.0 * math.pi * dow / 7.0)
            trend = float(t) / float(max(1, T - 1))

            # Check if any disruption is active today
            disr_active = 1 if any(d.is_active(t) for d in disruptions) else 0

            for w in range(W):
                w_id = warehouses[w].id
                for p in range(P):
                    p_id = products[p].id
                    actual_demand = float(clean_demand[t, w, p])

                    row = {
                        "step": t,
                        "trend": trend,
                        "warehouse_id": w_id,
                        "product_id": p_id,
                        "category": cat_map.get(p_id, "General"),
                        "unit_cost": cost_map.get(p_id, 10.0),
                        "selling_price": price_map.get(p_id, 20.0),
                        "day_of_week": dow,
                        "is_weekend": is_weekend,
                        "sin_dow": sin_dow,
                        "cos_dow": cos_dow,
                        "disruption_active": disr_active,
                        "demand": actual_demand,
                    }

                    # Lag features: strictly historical steps (t - lag)
                    for lag in self.lags:
                        if t - lag >= 0:
                            row[f"lag_{lag}"] = float(clean_demand[t - lag, w, p])
                        else:
                            # Short series or start of horizon: use initial observation or 0.0
                            row[f"lag_{lag}"] = float(clean_demand[0, w, p]) if T > 0 else 0.0

                    # Rolling features: strictly prior to step t (t' < t) to prevent lookahead leakage
                    for win in self.rolling_windows:
                        start_idx = max(0, t - win)
                        hist = clean_demand[start_idx:t, w, p] if t > 0 else np.array([clean_demand[0, w, p]])
                        row[f"rolling_mean_{win}"] = float(np.mean(hist))
                        row[f"rolling_std_{win}"] = float(np.std(hist)) if len(hist) > 1 else 0.0
                        row[f"rolling_min_{win}"] = float(np.min(hist))
                        row[f"rolling_max_{win}"] = float(np.max(hist))

                    records.append(row)

        df = pd.DataFrame(records)
        # One-hot encode category and product identifiers
        df = pd.get_dummies(df, columns=["category"], drop_first=False)
        # Also create product identifier dummy indicators
        for p in products:
            col_name = f"is_prod_{p.id}"
            df[col_name] = (df["product_id"] == p.id).astype(int)

        return df

