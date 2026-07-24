"""
V24 - Differential Evolution Optimizer
Uses Differential Evolution algorithm
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy
import random


class V24Optimizer(BaseOptimizer):
    """V24 - Differential Evolution Optimizer"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int,
                 population_size: int = 20, generations: int = 50,
                 F: float = 0.8, CR: float = 0.9):
        super().__init__(demand, capacity, max_plates)
        self.name = "V24 - Differential Evolution"
        self.version = "V24"
        self.population_size = population_size
        self.generations = generations
        self.F = F
        self.CR = CR
        self.tags = list(demand.keys())
        self.n_tags = len(self.tags)
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V24 optimization with Differential Evolution"""
        if self.n_tags == 0:
            return []
        
        def encode_plates_to_vector(plates):
            """Encode plates to a vector for DE"""
            vector = []
            for plate in plates:
                for tag in self.tags:
                    vector.append(plate["layout"].get(tag, 0) / self.capacity)
                vector.append(plate["sheets"] / 1000.0)
            
            # Pad if needed
            while len(vector) < self.max_plates * (self.n_tags + 1):
                vector.append(0)
            
            return vector[:self.max_plates * (self.n_tags + 1)]
        
        def decode_vector_to_plates(vector):
            """Decode vector back to plates"""
            plates = []
            for i in range(self.max_plates):
                start_idx = i * (self.n_tags + 1)
                layout = {}
                
                for j, tag in enumerate(self.tags):
                    ups = int(vector[start_idx + j] * self.capacity)
                    if ups > 0:
                        layout[tag] = max(1, min(ups, self.capacity))
                
                if not layout:
                    continue
                
                # Fix layout
                active = {tag: self.demand.get(tag, 0) for tag in layout.keys()}
                layout = create_valid_layout(active, self.capacity, "proportional")
                
                sheets = max(1, int(vector[start_idx + self.n_tags] * 1000))
                
                plates.append({
                    "name": plate_name(len(plates) + 1),
                    "layout": layout,
                    "sheets": sheets
                })
            
            return ensure_demand_met(plates, self.demand) if plates else None
        
        def evaluate(plates):
            """Evaluate solution quality"""
            if not plates:
                return float('inf')
            return calculate_waste_percent(plates, self.demand)
        
        # Initialize population
        population = []
        from algorithms.v3 import V3Optimizer
        
        for _ in range(self.population_size):
            base_plates = V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
            if base_plates:
                population.append(encode_plates_to_vector(base_plates))
            else:
                # Generate random solution
                remaining = self.demand.copy()
                plates = []
                for _ in range(self.max_plates):
                    active = {k: v for k, v in remaining.items() if v > 0}
                    if not active:
                        break
                    layout = create_valid_layout(active, self.capacity, "balanced")
                    sheets = max(1, min(math.ceil(remaining[t] / layout.get(t, 1)) for t in active))
                    plates.append({"layout": layout, "sheets": sheets})
                    for tag, ups in layout.items():
                        remaining[tag] = max(0, remaining[tag] - (ups * sheets))
                plates = ensure_demand_met(plates, self.demand)
                population.append(encode_plates_to_vector(plates))
        
        best_solution = None
        best_waste = float('inf')
        
        # DE main loop
        for generation in range(self.generations):
            new_population = []
            
            for i, target in enumerate(population):
                # Select three random individuals
                candidates = [idx for idx in range(self.population_size) if idx != i]
                if len(candidates) < 3:
                    a, b, c = 0, 1, 2
                else:
                    a, b, c = random.sample(candidates, 3)
                
                # Mutation
                mutant = []
                for j in range(len(target)):
                    val = population[a][j] + self.F * (population[b][j] - population[c][j])
                    mutant.append(max(0.0, min(1.0, val)))
                
                # Crossover
                trial = []
                for j in range(len(target)):
                    if random.random() < self.CR:
                        trial.append(mutant[j])
                    else:
                        trial.append(target[j])
                
                # Evaluate
                trial_plates = decode_vector_to_plates(trial)
                trial_waste = evaluate(trial_plates)
                
                target_plates = decode_vector_to_plates(target)
                target_waste = evaluate(target_plates)
                
                # Selection
                if trial_waste < target_waste:
                    new_population.append(trial)
                    if trial_waste < best_waste:
                        best_waste = trial_waste
                        best_solution = trial_plates
                else:
                    new_population.append(target)
                    if target_waste < best_waste:
                        best_waste = target_waste
                        best_solution = target_plates
            
            population = new_population
        
        return ensure_demand_met(best_solution, self.demand) if best_solution else self._fallback()
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
