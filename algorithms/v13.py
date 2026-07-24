"""
V13 - Hybrid Local Search
Combines multiple local search techniques
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy
import random


class V13Optimizer(BaseOptimizer):
    """V13 - Hybrid Local Search"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int):
        super().__init__(demand, capacity, max_plates)
        self.name = "V13 - Hybrid Local Search"
        self.version = "V13"
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V13 optimization with hybrid local search"""
        # Start with V3 solution
        from algorithms.v3 import V3Optimizer
        best = V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
        
        if not best:
            return self._fallback()
        
        best_waste = calculate_waste_percent(best, self.demand)
        
        # Multiple local search strategies
        for _ in range(50):
            # Strategy 1: Swap UPS between plates
            trial1 = copy.deepcopy(best)
            if len(trial1) >= 2:
                i, j = random.sample(range(len(trial1)), 2)
                tags_i = list(trial1[i]["layout"].keys())
                tags_j = list(trial1[j]["layout"].keys())
                
                if tags_i and tags_j:
                    a = random.choice(tags_i)
                    b = random.choice(tags_j)
                    
                    if trial1[i]["layout"].get(a, 0) > 1:
                        trial1[i]["layout"][a] -= 1
                        trial1[j]["layout"][b] = trial1[j]["layout"].get(b, 0) + 1
            
            trial1 = ensure_demand_met(trial1, self.demand)
            waste1 = calculate_waste_percent(trial1, self.demand)
            
            if waste1 < best_waste:
                best = copy.deepcopy(trial1)
                best_waste = waste1
            
            # Strategy 2: Merge plates if possible
            trial2 = copy.deepcopy(best)
            if len(trial2) >= 2:
                i, j = random.sample(range(len(trial2)), 2)
                combined = trial2[i]["layout"].copy()
                
                for tag, ups in trial2[j]["layout"].items():
                    combined[tag] = combined.get(tag, 0) + ups
                
                if sum(combined.values()) <= self.capacity:
                    trial2[i]["layout"] = combined
                    trial2[i]["sheets"] = max(trial2[i]["sheets"], trial2[j]["sheets"])
                    trial2.pop(j)
            
            trial2 = ensure_demand_met(trial2, self.demand)
            waste2 = calculate_waste_percent(trial2, self.demand)
            
            if waste2 < best_waste:
                best = copy.deepcopy(trial2)
                best_waste = waste2
        
        return ensure_demand_met(best, self.demand)
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
