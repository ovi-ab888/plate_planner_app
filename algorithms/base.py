"""
Base Optimizer Class
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseOptimizer(ABC):
    """Base class for all plate ratio optimizers"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int):
        self.demand = demand
        self.capacity = capacity
        self.max_plates = max_plates
        self.name = "Base Optimizer"
        self.version = "V0"
    
    @abstractmethod
    def optimize(self) -> List[Dict[str, Any]]:
        """Main optimization method"""
        pass
