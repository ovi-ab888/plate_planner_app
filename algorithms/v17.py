"""
V17 - AI Evolution Engine
Uses evolutionary strategies with AI-inspired operators
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy
import random


class V17Optimizer(BaseOptimizer):
    """V17 - AI Evolution Engine"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int, generations: int = 100):
        super().__init__(demand, capacity, max_plates)
        self.name = "V17 - AI Evolution Engine"
        self.version = "V17"
        self.generations = generations
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V17 optimization with AI evolution"""
        population = []
        
        # Initialize population with different algorithms
        from algorithms.v3 import V3Optimizer
        from algorithms.v5 import V5Optimizer
        from algorithms.v7 import V7Optimizer
        
        try:
            population.append(V3Optimizer(self.demand, self.capacity, self.max_plates).optimize())
            population.append(V5Optimizer(self.demand, self.capacity, self.max_plates).optimize())
            population.append(V7Optimizer(self.demand, self.capacity, self.max_plates).optimize())
        except:
            pass
        
        # Fill rest with random variations
        while len(population) < 20:
            candidate = V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
            if candidate:
                population.append(candidate)
        
        best_solution = None
        best_waste = float('inf')
        
        for generation in range(self.generations):
            scored = []
            
            for sol in population:
                if sol:
                    waste = calculate_waste_percent(sol, self.demand)
                    scored.append((waste, sol))
                    
                    if waste < best_waste:
                        best_waste = waste
                        best_solution = copy.deepcopy(sol)
            
            scored.sort(key=lambda x: x[0])
            elites = [copy.deepcopy(x[1]) for x in scored[:5]]
            new_population = elites.copy()
            
            # Generate offspring
            while len(new_population) < 20:
                parent = copy.deepcopy(random.choice(elites))
                
                # Apply mutation
                for plate in parent:
                    tags = list(plate["layout"].keys())
                    
                    if len(tags) >= 2:
                        a, b = random.sample(tags, 2)
                        if plate["layout"][a] > 1:
                            plate["layout"][a] -= 1
                            plate["layout"][b] += 1
                            
                            if sum(plate["layout"].values()) > self.capacity:
                                plate["layout"][a] += 1
                                plate["layout"][b] -= 1
                
                parent = ensure_demand_met(parent, self.demand)
                new_population.append(parent)
            
            population = new_population
        
        return ensure_demand_met(best_solution, self.demand) if best_solution else self._fallback()
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
