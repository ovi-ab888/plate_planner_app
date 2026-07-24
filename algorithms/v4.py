"""
V4 - Multi-Variation Optimizer
Tests multiple variations to find best
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy


class V4Optimizer(BaseOptimizer):
    """V4 - Multi-Variation Optimizer"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int):
        super().__init__(demand, capacity, max_plates)
        self.name = "V4 - Multi-Variation Optimizer"
        self.version = "V4"
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V4 optimization with multiple variations"""
        best_score = 999999
        best_plates = None
        
        for variation in range(15):
            remaining = copy.deepcopy(self.demand)
            plates = []
            
            for i in range(self.max_plates):
                active = {k: v for k, v in remaining.items() if v > 0}
                if not active:
                    break
                
                layout = create_valid_layout(active, self.capacity, "proportional")
                
                possible = [math.ceil(remaining[tag] / layout[tag]) for tag in layout if layout[tag] > 0]
                if not possible:
                    break
                
                possible = sorted(possible)
                strategy_index = min(variation % len(possible), len(possible) - 1)
                sheets = max(1, possible[strategy_index])
                
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
                best_plates = plates
        
        return ensure_demand_met(best_plates, self.demand) if best_plates else self._fallback()
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
