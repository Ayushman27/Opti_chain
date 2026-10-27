"""
OptiChain AI - Core Data Entities & Domain Models
Provides typed, validated domain models for products, warehouses, suppliers,
orders, and disruption events.
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


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


class Product(BaseModel):
    id: str = Field(..., description="Unique product SKU identifier, e.g. PROD_001")
    name: str = Field(..., description="Product commercial name")
    category: str = Field(default="General", description="Category classification")
    unit_cost: float = Field(..., ge=0.0, description="Base procurement cost per unit")
    selling_price: float = Field(..., ge=0.0, description="Market sales price per unit")
    holding_cost_rate: float = Field(
        ..., ge=0.0, description="Holding/carrying cost per unit per day"
    )
    stockout_penalty: float = Field(
        ..., ge=0.0, description="Penalty cost per unit of unsatisfied customer demand"
    )
    weight_kg: float = Field(default=1.0, ge=0.0, description="Unit shipping weight")
    volume_m3: float = Field(default=0.01, ge=0.0, description="Unit storage volume")

    @property
    def unit_margin(self) -> float:
        return self.selling_price - self.unit_cost


class Warehouse(BaseModel):
    id: str = Field(..., description="Warehouse code, e.g. WH_01")
    name: str = Field(..., description="Warehouse facility name")
    location: str = Field(default="Primary Region", description="Geographic location")
    storage_capacity: float = Field(
        ..., ge=0.0, description="Total storage capacity in units or m3"
    )
    daily_fixed_cost: float = Field(
        default=500.0, ge=0.0, description="Fixed daily facility operating cost"
    )


class Supplier(BaseModel):
    id: str = Field(..., description="Supplier code, e.g. SUP_01")
    name: str = Field(..., description="Supplier company name")
    location: str = Field(default="Supplier Hub", description="Geographic origin")
    base_lead_time_days: int = Field(
        ..., ge=1, description="Expected average lead time in days"
    )
    lead_time_std: float = Field(
        default=1.0, ge=0.0, description="Standard deviation of delivery lead time"
    )
    reliability: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Probability of on-time delivery"
    )
    fixed_order_cost: float = Field(
        default=100.0, ge=0.0, description="Fixed setup cost per purchase order"
    )
    catalog: Dict[str, float] = Field(
        default_factory=dict,
        description="Map of product_id -> unit purchase cost offered by this supplier",
    )
    min_order_quantity: Dict[str, int] = Field(
        default_factory=dict,
        description="Minimum order quantity (MOQ) per product_id",
    )
    daily_capacity: Dict[str, int] = Field(
        default_factory=dict,
        description="Maximum unit production capacity per product per day",
    )


class Order(BaseModel):
    id: str
    step_created: int
    supplier_id: str
    warehouse_id: str
    product_id: str
    quantity: int = Field(..., ge=1)
    unit_cost: float = Field(..., ge=0.0)
    fixed_order_cost: float = Field(default=0.0, ge=0.0)
    transport_cost: float = Field(default=0.0, ge=0.0)
    expected_arrival_step: int
    actual_arrival_step: Optional[int] = None
    status: OrderStatus = OrderStatus.IN_TRANSIT

    @property
    def total_cost(self) -> float:
        return (self.unit_cost * self.quantity) + self.fixed_order_cost + self.transport_cost


class Disruption(BaseModel):
    id: str
    name: str
    type: DisruptionType
    start_step: int = Field(..., ge=0)
    duration_steps: int = Field(..., ge=1)
    impact_multiplier: float = Field(default=2.0, ge=0.0)
    target_supplier_id: Optional[str] = None
    target_warehouse_id: Optional[str] = None
    target_product_id: Optional[str] = None
    description: str = ""

    def is_active(self, current_step: int) -> bool:
        return self.start_step <= current_step < (self.start_step + self.duration_steps)


class SimulationConfig(BaseModel):
    n_warehouses: int = Field(default=3, ge=1)
    n_products: int = Field(default=50, ge=1)
    n_suppliers: int = Field(default=10, ge=1)
    episode_days: int = Field(default=90, ge=1)
    random_seed: int = Field(default=42)
    backlog_unmet_demand: bool = Field(
        default=False,
        description="If True, unmet demand is backlogged; if False, it represents lost sales.",
    )
    transport_cost_per_unit: float = Field(
        default=1.5, ge=0.0, description="Transport cost per unit shipped"
    )
