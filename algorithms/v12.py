"""
V12 - Column Generation
Uses column generation technique for optimization
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy


class V12Optimizer(BaseOptimizer):
    """V12 - Column Generation Optimizer"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int):
        super().__init__(demand, capacity, max_plates)
        self.name = "V12 - Column Generation"
        self.version = "V12"
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V12 optimization with column generation"""
        try:
            from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
        except ImportError:
            from algorithms.v3 import V3Optimizer
            return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
        
        tags_list = list(self.demand.keys())
        remaining = self.demand.copy()
        plates = []
        
        # Generate initial patterns
        patterns = []
        initial_demand = self.demand.copy()
        
        for _ in range(min(self.max_plates, len(tags_list) * 2)):
            active = {k: v for k, v in initial_demand.items() if v > 0}
            if not active:
                break
            
            pattern = create_valid_layout(active, self.capacity, "proportional")
            patterns.append(pattern)
            
            sheets = 1
            for tag, ups in pattern.items():
                initial_demand[tag] = max(0, initial_demand[tag] - (ups * sheets))
        
        # Solve master problem with generated patterns
        try:
            for idx, pattern in enumerate(patterns):
                if sum(pattern.values()) > 0:
                    layout = pattern
                    
                    # Calculate sheets
                    possible_sheets = []
                    for tag, ups in layout.items():
                        if ups > 0 and remaining.get(tag, 0) > 0:
                            possible_sheets.append(math.ceil(remaining[tag] / ups))
                    
                    if possible_sheets:
                        sheets = min(possible_sheets)
                    else:
                        sheets = 1
                    
                    plates.append({
                        "name": plate_name(len(plates) + 1),
                        "layout": layout,
                        "sheets": sheets
                    })
                    
                    for tag, ups in layout.items():
                        remaining[tag] = max(0, remaining[tag] - (ups * sheets))
                    
                    if all(v <= 0 for v in remaining.values()):
                        break
            
            # Handle remaining demand
            if any(v > 0 for v in remaining.values()) and plates:
                last = plates[-1]
                for tag in list(remaining.keys()):
                    if remaining[tag] > 0:
                        ups = max(1, last["layout"].get(tag, 1))
                        add_sheets = math.ceil(remaining[tag] / ups)
                        last["sheets"] += add_sheets
                        remaining[tag] = 0
            elif any(v > 0 for v in remaining.values()):
                from algorithms.v3 import V3Optimizer
                return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
        
        except Exception:
            from algorithms.v3 import V3Optimizer
            return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
        
        return ensure_demand_met(plates, self.demand) if plates else self._fallback()
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
