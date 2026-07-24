"""
V5 - AI Mutation Engine
Uses random mutations to find better layouts
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy
import random


class V5Optimizer(BaseOptimizer):
    """V5 - AI Mutation Engine"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int, iterations: int = 80):
        super().__init__(demand, capacity, max_plates)
        self.name = "V5 - AI Mutation Engine"
        self.version = "V5"
        self.iterations = iterations
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V5 optimization with mutations"""
        best_score = 999999
        best_plates = None
        
        for attempt in range(self.iterations):
            remaining = copy.deepcopy(self.demand)
            plates = []
            
            for i in range(self.max_plates):
                active = {k: v for k, v in remaining.items() if v > 0}
                if not active:
                    break
                
                layout = create_valid_layout(active, self.capacity, "greedy")
                
                options = [math.ceil(remaining[tag] / layout[tag]) for tag in layout if layout[tag] > 0]
                if not options:
                    break
                
                options = sorted(list(set(options)))
                sheets = max(1, random.choice(options))
                
                for tag, ups in layout.items():
                    remaining[tag] = max(0, remaining[tag] - (ups * sheets))
                
                plates.append({"name": plate_name(i + 1), "layout": layout, "sheets": sheets})
            
            if any(v > 0 for v in remaining.values()) and plates:
                last = plates[-1]
                for tag in remaining:
                    if remaining[tag] > 0:
                        ups = max(1, last["layout"].get(tag, 1))
                        extra = math.ceil(remaining[tag] / ups)
                        last["sheets"] += extra
                        remaining[tag] = 0
            
            waste = calculate_waste_percent(plates, self.demand)
            if waste < best_score:
                best_score = waste
                best_plates = copy.deepcopy(plates)
        
        return ensure_demand_met(best_plates, self.demand) if best_plates else self._fallback()
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
