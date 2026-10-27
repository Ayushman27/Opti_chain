"""
OptiChain AI - System Configuration Module
Defines strongly-typed, validated simulation configuration dataclass.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class SimulationConfig:
    """
    Simulation environment and data generation configuration parameters.
    Default baseline settings: 1 Warehouse, 5 Products, 3 Suppliers, 90-day horizon.
    """
    n_warehouses: int = 1
    n_products: int = 5
    n_suppliers: int = 3
    episode_days: int = 90
    random_seed: int = 42
    backlog_unmet_demand: bool = False
    transport_cost_per_unit: float = 1.5
    trend_slope: float = 0.12
    seasonality_amplitude: float = 0.25
    weekend_boost: float = 0.20

    def __post_init__(self):
        if self.n_warehouses < 1:
            raise ValueError(f"n_warehouses must be >= 1, got {self.n_warehouses}")
        if self.n_products < 1:
            raise ValueError(f"n_products must be >= 1, got {self.n_products}")
        if self.n_suppliers < 1:
            raise ValueError(f"n_suppliers must be >= 1, got {self.n_suppliers}")
        if self.episode_days < 1:
            raise ValueError(f"episode_days must be >= 1, got {self.episode_days}")
        if self.transport_cost_per_unit < 0.0:
            raise ValueError(
                f"transport_cost_per_unit must be >= 0.0, got {self.transport_cost_per_unit}"
            )
        if self.trend_slope < -1.0:
            raise ValueError(f"trend_slope must be >= -1.0, got {self.trend_slope}")
        if self.seasonality_amplitude < 0.0:
            raise ValueError(
                f"seasonality_amplitude must be >= 0.0, got {self.seasonality_amplitude}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)

    def model_copy(self, update: Dict[str, Any] | None = None) -> SimulationConfig:
        data = asdict(self)
        if update:
            data.update(update)
        return SimulationConfig(**data)


def get_default_config() -> SimulationConfig:
    """Returns standard Phase 1 default configuration: 1 WH, 5 Products, 3 Suppliers."""
    return SimulationConfig(
        n_warehouses=1,
        n_products=5,
        n_suppliers=3,
        episode_days=90,
        random_seed=42,
    )


def get_benchmark_config() -> SimulationConfig:
    """Returns expanded benchmark configuration: 3 WH, 50 Products, 10 Suppliers."""
    return SimulationConfig(
        n_warehouses=3,
        n_products=50,
        n_suppliers=10,
        episode_days=90,
        random_seed=42,
    )
