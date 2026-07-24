"""
V21 - ACO Optimizer
Ant Colony Optimization for plate ratio
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy
import random


class V21Optimizer(BaseOptimizer):
    """V21 - ACO Optimizer"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int,
                 ants: int = 15, iterations: int = 30,
                 alpha: float = 1.0, beta: float = 2.0,
                 evaporation: float = 0.5):
        super().__init__(demand, capacity, max_plates)
        self.name = "V21 - ACO Optimizer"
        self.version = "V21"
        self.ants = ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.tags = list(demand.keys())
        self.pheromone = {}
        
        # Initialize pheromone
        for i in range(len(self.tags)):
            for j in range(1, self.capacity + 1):
                self.pheromone[(i, j)] = 1.0
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V21 optimization with Ant Colony"""
        if not self.tags:
            return []
        
        best_plates = None
        best_waste = float('inf')
        
        for iteration in range(self.iterations):
            iteration_best_plates = None
            iteration_best_waste = float('inf')
            
            for ant in range(self.ants):
                plates = self._construct_solution()
                if plates:
                    waste = calculate_waste_percent(plates, self.demand)
                    
                    if waste < iteration_best_waste:
                        iteration_best_waste = waste
                        iteration_best_plates = copy.deepcopy(plates)
                    
                    if waste < best_waste:
                        best_waste = waste
                        best_plates = copy.deepcopy(plates)
            
            if iteration_best_plates:
                self._update_pheromone(iteration_best_plates, iteration_best_waste)
            
            if best_waste == 0:
                break
        
        return ensure_demand_met(best_plates, self.demand) if best_plates else self._fallback()
    
    def _construct_solution(self):
        """Construct a solution using ACO"""
        remaining = self.demand.copy()
        plates = []
        
        for plate_idx in range(self.max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            
            # Start with proportional layout
            layout = create_valid_layout(active, self.capacity, "proportional")
            
            # Use ACO to refine layout
            remaining_cap = self.capacity - sum(layout.values())
            
            if remaining_cap > 0 and active:
                active_tags = list(active.keys())
                while remaining_cap > 0 and active_tags:
                    for tag in active_tags:
                        if remaining_cap <= 0:
                            break
                        tag_idx = self.tags.index(tag)
                        
                        # Use pheromone to decide extra UPS
                        ups_options = list(range(1, min(remaining_cap, active[tag]) + 1))
                        if ups_options:
                            probabilities = []
                            for ups in ups_options:
                                tau = self.pheromone.get((tag_idx, ups), 1.0) ** self.alpha
                                eta = (1.0 / (ups + 1)) ** self.beta
                                probabilities.append(tau * eta)
                            
                            if probabilities and sum(probabilities) > 0:
                                total_prob = sum(probabilities)
                                probs = [p / total_prob for p in probabilities]
                                chosen_ups = random.choices(ups_options, weights=probs)[0]
                                layout[tag] = layout.get(tag, 0) + chosen_ups
                                remaining_cap -= chosen_ups
            
            # Ensure exact capacity
            while sum(layout.values()) > self.capacity:
                max_tag = max(layout, key=layout.get)
                if layout[max_tag] > 1:
                    layout[max_tag] -= 1
                else:
                    break
            
            while sum(layout.values()) < self.capacity:
                max_tag = max(active, key=active.get)
                layout[max_tag] = layout.get(max_tag, 0) + 1
            
            # Calculate sheets
            sheets_list = []
            for tag, ups in layout.items():
                if ups > 0 and remaining.get(tag, 0) > 0:
                    sheets_list.append(math.ceil(remaining[tag] / ups))
            
            sheets = max(1, min(sheets_list)) if sheets_list else 1
            
            # Update remaining
            for tag, ups in layout.items():
                remaining[tag] = max(0, remaining[tag] - (ups * sheets))
            
            plates.append({
                "name": plate_name(len(plates) + 1),
                "layout": layout,
                "sheets": sheets
            })
            
            if all(v <= 0 for v in remaining.values()):
                break
        
        return ensure_demand_met(plates, self.demand) if plates else None
    
    def _update_pheromone(self, plates, waste):
        """Update pheromone based on solution quality"""
        # Evaporation
        for key in list(self.pheromone.keys()):
            self.pheromone[key] *= (1 - self.evaporation)
        
        # Deposit pheromone
        deposit = 10.0 / (waste + 1) if waste > 0 else 100.0
        
        for plate in plates:
            for tag, ups in plate["layout"].items():
                if tag in self.tags:
                    tag_idx = self.tags.index(tag)
                    self.pheromone[(tag_idx, ups)] = self.pheromone.get((tag_idx, ups), 1.0) + deposit
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
