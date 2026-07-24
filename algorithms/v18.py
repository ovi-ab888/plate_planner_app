"""
V18 - Global Multi-Plate Optimizer
Optimizes all plates simultaneously
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy


class V18Optimizer(BaseOptimizer):
    """V18 - Global Multi-Plate Optimizer"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int):
        super().__init__(demand, capacity, max_plates)
        self.name = "V18 - Global Multi-Plate Optimizer"
        self.version = "V18"
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V18 optimization - global optimization"""
        candidates = []
        
        # Try different algorithms
        algos = [
            self._optimize_v3,
            self._optimize_v5,
            self._optimize_v7,
            self._optimize_v11,
            self._optimize_hybrid
        ]
        
        for algo in algos:
            try:
                result = algo()
                if result:
                    waste = calculate_waste_percent(result, self.demand)
                    candidates.append((waste, result))
            except:
                pass
        
        if not candidates:
            from algorithms.v3 import V3Optimizer
            return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
        
        candidates.sort(key=lambda x: x[0])
        return ensure_demand_met(candidates[0][1], self.demand)
    
    def _optimize_v3(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
    
    def _optimize_v5(self):
        from algorithms.v5 import V5Optimizer
        return V5Optimizer(self.demand, self.capacity, self.max_plates).optimize()
    
    def _optimize_v7(self):
        from algorithms.v7 import V7Optimizer
        return V7Optimizer(self.demand, self.capacity, self.max_plates).optimize()
    
    def _optimize_v11(self):
        from algorithms.v11 import V11Optimizer
        return V11Optimizer(self.demand, self.capacity, self.max_plates).optimize()
    
    def _optimize_hybrid(self):
        """Hybrid optimization combining multiple strategies"""
        # Start with V3
        from algorithms.v3 import V3Optimizer
        plates = V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
        
        if not plates:
            return None
        
        # Try to improve
        best_waste = calculate_waste_percent(plates, self.demand)
        
        for _ in range(50):
            trial = copy.deepcopy(plates)
            
            # Randomly adjust plates
            for plate in trial:
                tags = list(plate["layout"].keys())
                if len(tags) >= 2:
                    a, b = random.sample(tags, 2)
                    if plate["layout"][a] > 1:
                        plate["layout"][a] -= 1
                        plate["layout"][b] += 1
                        
                        if sum(plate["layout"].values()) > self.capacity:
                            plate["layout"][a] += 1
                            plate["layout"][b] -= 1
            
            trial = ensure_demand_met(trial, self.demand)
            waste = calculate_waste_percent(trial, self.demand)
            
            if waste < best_waste:
                plates = copy.deepcopy(trial)
                best_waste = waste
        
        return plates
