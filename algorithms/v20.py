"""
V20 - PSO Optimizer
Particle Swarm Optimization
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy
import random


class V20Optimizer(BaseOptimizer):
    """V20 - PSO Optimizer"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int,
                 particles: int = 20, iterations: int = 50):
        super().__init__(demand, capacity, max_plates)
        self.name = "V20 - PSO Optimizer"
        self.version = "V20"
        self.particles = particles
        self.iterations = iterations
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V20 optimization with PSO"""
        best_global_plates = None
        best_global_waste = float('inf')
        
        class Particle:
            def __init__(self):
                self.plates = []
                remaining = self.demand.copy()
                
                for _ in range(self.max_plates):
                    active = {k: v for k, v in remaining.items() if v > 0}
                    if not active:
                        break
                    
                    layout = create_valid_layout(active, self.capacity, "proportional")
                    
                    # Add randomness
                    if len(layout) >= 2:
                        tags_list = list(layout.keys())
                        a, b = random.sample(tags_list, 2)
                        if layout.get(a, 0) > 1:
                            layout[a] = layout.get(a, 1) - 1
                            layout[b] = layout.get(b, 0) + 1
                    
                    sheets = max(1, min(math.ceil(remaining[tag] / layout.get(tag, 1)) for tag in active))
                    self.plates.append({"layout": layout, "sheets": sheets})
                    
                    for tag, ups in layout.items():
                        remaining[tag] = max(0, remaining[tag] - (ups * sheets))
                
                if any(v > 0 for v in remaining.values()) and self.plates:
                    last = self.plates[-1]
                    for tag in remaining:
                        if remaining[tag] > 0:
                            ups = max(1, last["layout"].get(tag, 1))
                            last["sheets"] += math.ceil(remaining[tag] / ups)
                            remaining[tag] = 0
                
                self.plates = ensure_demand_met(self.plates, self.demand)
                self.best_plates = copy.deepcopy(self.plates)
                self.best_waste = calculate_waste_percent(self.plates, self.demand)
            
            def update_fitness(self):
                waste = calculate_waste_percent(self.plates, self.demand)
                if waste < self.best_waste:
                    self.best_waste = waste
                    self.best_plates = copy.deepcopy(self.plates)
                return waste
        
        # Initialize swarm
        swarm = []
        for _ in range(self.particles):
            p = Particle()
            swarm.append(p)
            
            # Update global best
            if p.best_waste < best_global_waste:
                best_global_waste = p.best_waste
                best_global_plates = copy.deepcopy(p.best_plates)
        
        # PSO iterations
        for iteration in range(self.iterations):
            for particle in swarm:
                # Update particle fitness
                waste = particle.update_fitness()
                
                # Update global best
                if waste < best_global_waste:
                    best_global_waste = waste
                    best_global_plates = copy.deepcopy(particle.best_plates)
                
                # Mutation - move towards global best
                if random.random() < 0.3 and particle.plates:
                    plate_idx = random.randint(0, len(particle.plates) - 1)
                    layout = particle.plates[plate_idx]["layout"]
                    
                    if len(layout) >= 2 and best_global_plates:
                        # Copy some characteristics from global best
                        if plate_idx < len(best_global_plates):
                            global_layout = best_global_plates[plate_idx]["layout"]
                            
                            for tag in layout.keys():
                                if tag in global_layout:
                                    # Move towards global best
                                    if layout[tag] < global_layout[tag]:
                                        layout[tag] += 1
                                    elif layout[tag] > global_layout[tag]:
                                        layout[tag] -= 1
                    
                    # Ensure capacity
                    while sum(layout.values()) > self.capacity:
                        max_tag = max(layout, key=layout.get)
                        if layout[max_tag] > 1:
                            layout[max_tag] -= 1
                        else:
                            break
                    
                    while sum(layout.values()) < self.capacity:
                        max_tag = max(layout, key=lambda t: layout.get(t, 0))
                        layout[max_tag] = layout.get(max_tag, 0) + 1
            
            # Re-evaluate all particles
            for particle in swarm:
                particle.plates = ensure_demand_met(particle.plates, self.demand)
                
                # Apply some random mutation
                if random.random() < 0.1 and particle.plates:
                    plate_idx = random.randint(0, len(particle.plates) - 1)
                    layout = particle.plates[plate_idx]["layout"]
                    tags = list(layout.keys())
                    
                    if len(tags) >= 2:
                        a, b = random.sample(tags, 2)
                        if layout[a] > 1:
                            layout[a] -= 1
                            layout[b] += 1
        
        return ensure_demand_met(best_global_plates, self.demand) if best_global_plates else self._fallback()
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
