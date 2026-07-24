"""
V7 - Simulated Annealing
Uses simulated annealing for optimization
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy
import random


class V7Optimizer(BaseOptimizer):
    """V7 - Simulated Annealing"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int, iterations: int = 100):
        super().__init__(demand, capacity, max_plates)
        self.name = "V7 - Simulated Annealing"
        self.version = "V7"
        self.iterations = iterations
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V7 optimization with simulated annealing"""
        # Start with V3 solution
        from algorithms.v3 import V3Optimizer
        base = V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
        
        if not base:
            return self._fallback()
        
        best = copy.deepcopy(base)
        best_waste = calculate_waste_percent(best, self.demand)
        current = copy.deepcopy(best)
        current_waste = best_waste
        
        temperature = 100.0
        
        for _ in range(self.iterations):
            # Create neighbor solution
            neighbor = copy.deepcopy(current)
            
            for plate in neighbor:
                tags = list(plate["layout"].keys())
                if len(tags) < 2:
                    continue
                
                a, b = random.sample(tags, 2)
                if plate["layout"][a] > 1:
                    plate["layout"][a] -= 1
                    plate["layout"][b] += 1
                    
                    if sum(plate["layout"].values()) > self.capacity:
                        plate["layout"][a] += 1
                        plate["layout"][b] -= 1
            
            neighbor = ensure_demand_met(neighbor, self.demand)
            neighbor_waste = calculate_waste_percent(neighbor, self.demand)
            
            # Accept or reject
            delta = neighbor_waste - current_waste
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current = copy.deepcopy(neighbor)
                current_waste = neighbor_waste
                
                if current_waste < best_waste:
                    best = copy.deepcopy(current)
                    best_waste = current_waste
            
            # Cool down
            temperature *= 0.99
        
        return ensure_demand_met(best, self.demand)
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
