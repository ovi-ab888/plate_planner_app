"""
V15 - DP Repair Engine
Uses dynamic programming for repair and optimization
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy


class V15Optimizer(BaseOptimizer):
    """V15 - DP Repair Engine"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int):
        super().__init__(demand, capacity, max_plates)
        self.name = "V15 - DP Repair Engine"
        self.version = "V15"
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V15 optimization with DP repair"""
        # Start with V3 solution
        from algorithms.v3 import V3Optimizer
        base = V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
        
        if not base:
            return self._fallback()
        
        best = copy.deepcopy(base)
        best_waste = calculate_waste_percent(best, self.demand)
        
        # Try different repair strategies
        for _ in range(100):
            trial = copy.deepcopy(best)
            
            # Strategy 1: Fix plate layouts
            for plate in trial:
                layout = plate["layout"]
                tags = list(layout.keys())
                
                # Check if any tag has 0 UPS
                for tag in self.demand.keys():
                    if tag not in layout:
                        # Add tag with minimum UPS
                        layout[tag] = 1
                
                # Repair capacity
                while sum(layout.values()) > self.capacity:
                    # Remove from tags with excess
                    max_tag = max(layout, key=layout.get)
                    if layout[max_tag] > 1 and max_tag in self.demand:
                        layout[max_tag] -= 1
                    else:
                        break
                
                while sum(layout.values()) < self.capacity:
                    # Add to tags with highest demand
                    best_tag = max(self.demand.keys(), key=lambda t: self.demand[t] / (layout.get(t, 1) + 1))
                    layout[best_tag] = layout.get(best_tag, 0) + 1
            
            # Strategy 2: Optimize sheets
            for plate in trial:
                layout = plate["layout"]
                current_sheets = plate["sheets"]
                
                # Try to reduce sheets if possible
                for tag, ups in layout.items():
                    if ups > 0:
                        needed = math.ceil(self.demand.get(tag, 0) / ups)
                        if needed < current_sheets:
                            # Check if reducing sheets still meets demand
                            remaining = self.demand.copy()
                            for p in trial:
                                if p is plate:
                                    continue
                                for t, u in p["layout"].items():
                                    remaining[t] = max(0, remaining.get(t, 0) - u * p["sheets"])
                            
                            # Check if reduced sheets still works
                            if all(remaining.get(t, 0) <= ups * needed for t, ups in layout.items()):
                                plate["sheets"] = needed
            
            trial = ensure_demand_met(trial, self.demand)
            waste = calculate_waste_percent(trial, self.demand)
            
            if waste < best_waste:
                best = copy.deepcopy(trial)
                best_waste = waste
        
        return ensure_demand_met(best, self.demand)
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
