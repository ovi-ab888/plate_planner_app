"""
V1 - Plate Ratio System (Basic Version)
Simple proportional distribution algorithm
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met
import math


class V1Optimizer(BaseOptimizer):
    """V1 - Basic Plate Ratio System"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int):
        super().__init__(demand, capacity, max_plates)
        self.name = "V1 - Plate Ratio System"
        self.version = "V1"
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V1 optimization"""
        remaining = self.demand.copy()
        plates = []
        
        for i in range(self.max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            
            # Create layout using balanced method
            layout = create_valid_layout(active, self.capacity, "balanced")
            
            # Calculate sheets
            possible_sheets = []
            for tag, ups in layout.items():
                if ups > 0 and remaining.get(tag, 0) > 0:
                    possible_sheets.append(math.ceil(remaining[tag] / ups))
            
            sheets = max(1, min(possible_sheets)) if possible_sheets else 1
            
            # Update remaining
            for tag, ups in layout.items():
                remaining[tag] = max(0, remaining[tag] - (ups * sheets))
            
            plates.append({
                "name": plate_name(i + 1),
                "layout": layout,
                "sheets": sheets
            })
        
        # Ensure all demand is met
        return ensure_demand_met(plates, self.demand)
