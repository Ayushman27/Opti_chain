"""
OptiChain AI - Base Policy Interface
Defines the standard contract for inventory replenishment policies
including classical OR heuristics, ML-driven policies, and DRL agents.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from optichain.simulator.environment import DigitalTwinSimulator


class BasePolicy(ABC):
    """
    Abstract interface for inventory replenishment policies.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, sim: DigitalTwinSimulator) -> None:
        """
        Calibrate or initialize policy parameters based on the simulator network structure,
        lead times, and historical demand statistics.
        """
        pass

    @abstractmethod
    def get_actions(
        self, state: Dict[str, Any], sim: DigitalTwinSimulator
    ) -> Dict[Tuple[str, str, str], int]:
        """
        Given the current state snapshot and simulator references,
        return replenishment orders: (warehouse_id, product_id, supplier_id) -> quantity.
        """
        pass
