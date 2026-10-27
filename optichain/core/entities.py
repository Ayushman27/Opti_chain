"""
OptiChain AI - Core Data Entities & Domain Models
Provides typed, validated dataclasses for products, warehouses, suppliers,
orders, and disruption events.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

# Re-export SimulationConfig from config module
from optichain.config import SimulationConfig


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    IN_TRANSIT = "IN_TRANSIT"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"


class DisruptionType(str, Enum):
    DEMAND_SPIKE = "DEMAND_SPIKE"
    SUPPLIER_DELAY = "SUPPLIER_DELAY"
    SUPPLIER_HALT = "SUPPLIER_HALT"
    PORT_CONGESTION = "PORT_CONGESTION"
    WAREHOUSE_CAPACITY_DROP = "WAREHOUSE_CAPACITY_DROP"


@dataclass
class Product:
    """
    Physical stock-keeping unit (SKU) definition with commercial, holding, and penalty costs.
    """
    id: str
    name: str
    unit_cost: float
    selling_price: float
    holding_cost_rate: float
    stockout_penalty: float
    category: str = "General"
    weight_kg: float = 1.0
    volume_m3: float = 0.01

    def __post_init__(self):
        if self.unit_cost < 0.0:
            raise ValueError(f"unit_cost must be >= 0.0, got {self.unit_cost}")
        if self.selling_price < 0.0:
            raise ValueError(f"selling_price must be >= 0.0, got {self.selling_price}")
        if self.holding_cost_rate < 0.0:
            raise ValueError(f"holding_cost_rate must be >= 0.0, got {self.holding_cost_rate}")
        if self.stockout_penalty < 0.0:
            raise ValueError(f"stockout_penalty must be >= 0.0, got {self.stockout_penalty}")
        if self.weight_kg <= 0.0:
            raise ValueError(f"weight_kg must be > 0.0, got {self.weight_kg}")
        if self.volume_m3 <= 0.0:
            raise ValueError(f"volume_m3 must be > 0.0, got {self.volume_m3}")

    @property
    def unit_margin(self) -> float:
        return self.selling_price - self.unit_cost

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Warehouse:
    """
    Regional distribution center or fulfillment warehouse node.
    """
    id: str
    name: str
    storage_capacity: float
    location: str = "Primary Region"
    daily_fixed_cost: float = 500.0

    def __post_init__(self):
        if self.storage_capacity <= 0.0:
            raise ValueError(f"storage_capacity must be > 0.0, got {self.storage_capacity}")
        if self.daily_fixed_cost < 0.0:
            raise ValueError(f"daily_fixed_cost must be >= 0.0, got {self.daily_fixed_cost}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Supplier:
    """
    Tier-1 supplier with catalog pricing, minimum order quantities (MOQ), and stochastic lead times.
    """
    id: str
    name: str
    base_lead_time_days: int
    location: str = "Supplier Hub"
    lead_time_std: float = 1.0
    reliability: float = 0.95
    fixed_order_cost: float = 100.0
    catalog: Dict[str, float] = field(default_factory=dict)
    min_order_quantity: Dict[str, int] = field(default_factory=dict)
    daily_capacity: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.base_lead_time_days < 1:
            raise ValueError(f"base_lead_time_days must be >= 1, got {self.base_lead_time_days}")
        if self.lead_time_std < 0.0:
            raise ValueError(f"lead_time_std must be >= 0.0, got {self.lead_time_std}")
        if not (0.0 <= self.reliability <= 1.0):
            raise ValueError(f"reliability must be in [0.0, 1.0], got {self.reliability}")
        if self.fixed_order_cost < 0.0:
            raise ValueError(f"fixed_order_cost must be >= 0.0, got {self.fixed_order_cost}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Order:
    """
    Purchase replenishment order placed with a supplier.
    """
    id: str
    step_created: int
    supplier_id: str
    warehouse_id: str
    product_id: str
    quantity: int
    unit_cost: float
    expected_arrival_step: int
    fixed_order_cost: float = 0.0
    transport_cost: float = 0.0
    actual_arrival_step: Optional[int] = None
    status: OrderStatus = OrderStatus.IN_TRANSIT

    def __post_init__(self):
        if self.quantity < 1:
            raise ValueError(f"quantity must be >= 1, got {self.quantity}")
        if self.unit_cost < 0.0:
            raise ValueError(f"unit_cost must be >= 0.0, got {self.unit_cost}")
        if self.fixed_order_cost < 0.0:
            raise ValueError(f"fixed_order_cost must be >= 0.0, got {self.fixed_order_cost}")
        if self.transport_cost < 0.0:
            raise ValueError(f"transport_cost must be >= 0.0, got {self.transport_cost}")

    @property
    def total_cost(self) -> float:
        return (self.unit_cost * self.quantity) + self.fixed_order_cost + self.transport_cost

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DisruptionEvent:
    """
    Exogenous supply chain shock or systemic event (demand spike, supplier shutdown, port delay).
    """
    id: str
    name: str
    type: DisruptionType
    start_step: int
    duration_steps: int
    impact_multiplier: float = 2.0
    target_supplier_id: Optional[str] = None
    target_warehouse_id: Optional[str] = None
    target_product_id: Optional[str] = None
    description: str = ""

    def __post_init__(self):
        if self.start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {self.start_step}")
        if self.duration_steps < 1:
            raise ValueError(f"duration_steps must be >= 1, got {self.duration_steps}")
        if self.impact_multiplier < 0.0:
            raise ValueError(f"impact_multiplier must be >= 0.0, got {self.impact_multiplier}")

    def is_active(self, current_step: int) -> bool:
        return self.start_step <= current_step < (self.start_step + self.duration_steps)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


# Backwards compatibility alias
Disruption = DisruptionEvent
