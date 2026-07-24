"""
V2 - Common Sheet Optimizer
Greedy allocation with common sheet count
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met
import math


class V2Optimizer(BaseOptimizer):
    """V2 - Common Sheet Optimizer"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int):
        super().__init__(demand, capacity, max_plates)
        self.name = "V2 - Common Sheet Optimizer"
        self.version = "V2"
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V2 optimization"""
        remaining = self.demand.copy()
        plates = []
        
        for i in range(self.max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            
            # Create layout using greedy method
            layout = create_valid_layout(active, self.capacity, "greedy")
            
            # Calculate sheets - use minimum to avoid waste
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
