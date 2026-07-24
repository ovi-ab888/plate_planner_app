"""
V25 - Pareto Optimizer
Multi-objective optimization with Pareto front
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy
import random


class Individual:
    """Individual for Pareto optimization"""
    def __init__(self, plates, demand):
        self.plates = plates
        self.demand = demand
        self.waste = calculate_waste_percent(plates, demand)
        self.total_plates = len(plates)
        self.total_sheets = sum(p["sheets"] for p in plates)
    
    def dominates(self, other):
        """Check if this individual dominates another"""
        better_in_one = False
        for attr in ['waste', 'total_plates', 'total_sheets']:
            if getattr(self, attr) < getattr(other, attr):
                better_in_one = True
            elif getattr(self, attr) > getattr(other, attr):
                return False
        return better_in_one


class V25Optimizer(BaseOptimizer):
    """V25 - Pareto Optimizer"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int,
                 population_size: int = 30, generations: int = 50):
        super().__init__(demand, capacity, max_plates)
        self.name = "V25 - Pareto Optimizer"
        self.version = "V25"
        self.population_size = population_size
        self.generations = generations
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V25 optimization with Pareto front"""
        
        def create_individual():
            """Create random individual"""
            remaining = self.demand.copy()
            plates = []
            
            for _ in range(self.max_plates):
                active = {k: v for k, v in remaining.items() if v > 0}
                if not active:
                    break
                
                layout = create_valid_layout(active, self.capacity, "greedy")
                sheets = max(1, min(math.ceil(remaining[tag] / layout.get(tag, 1)) for tag in active))
                plates.append({"layout": layout, "sheets": sheets})
                
                for tag, ups in layout.items():
                    remaining[tag] = max(0, remaining[tag] - (ups * sheets))
            
            if any(v > 0 for v in remaining.values()) and plates:
                last = plates[-1]
                for tag in remaining:
                    if remaining[tag] > 0:
                        ups = max(1, last["layout"].get(tag, 1))
                        last["sheets"] += math.ceil(remaining[tag] / ups)
                        remaining[tag] = 0
            
            return Individual(ensure_demand_met(plates, self.demand), self.demand)
        
        def mutate(ind):
            """Mutate an individual"""
            new_plates = copy.deepcopy(ind.plates)
            if new_plates:
                plate_idx = random.randint(0, len(new_plates) - 1)
                layout = new_plates[plate_idx]["layout"]
                if len(layout) >= 2:
                    tags_list = list(layout.keys())
                    a, b = random.sample(tags_list, 2)
                    if layout[a] > 1:
                        layout[a] -= 1
                        layout[b] += 1
            return Individual(ensure_demand_met(new_plates, self.demand), self.demand)
        
        def crossover(ind1, ind2):
            """Crossover between two individuals"""
            point = random.randint(1, min(len(ind1.plates), len(ind2.plates)) - 1)
            child_plates = ind1.plates[:point] + ind2.plates[point:]
            
            remaining = self.demand.copy()
            fixed_plates = []
            for plate in child_plates:
                active = {k: v for k, v in remaining.items() if v > 0}
                if not active:
                    break
                
                new_layout = plate["layout"].copy()
                sheets = plate["sheets"]
                fixed_plates.append({"layout": new_layout, "sheets": sheets})
                
                for tag, ups in new_layout.items():
                    remaining[tag] = max(0, remaining[tag] - (ups * sheets))
            
            if any(v > 0 for v in remaining.values()) and fixed_plates:
                last = fixed_plates[-1]
                for tag in remaining:
                    if remaining[tag] > 0:
                        ups = max(1, last["layout"].get(tag, 1))
                        last["sheets"] += math.ceil(remaining[tag] / ups)
                        remaining[tag] = 0
            
            return Individual(ensure_demand_met(fixed_plates, self.demand), self.demand)
        
        def non_dominated_sort(population):
            """Perform non-dominated sorting"""
            fronts = []
            remaining = set(range(len(population)))
            
            while remaining:
                front = []
                for i in list(remaining):
                    dominated = False
                    for j in list(remaining):
                        if i != j and population[j].dominates(population[i]):
                            dominated = True
                            break
                    if not dominated:
                        front.append(i)
                
                for i in front:
                    remaining.remove(i)
                fronts.append([population[i] for i in front])
            
            return fronts
        
        # Initialize population
        population = [create_individual() for _ in range(self.population_size)]
        
        # Evolution loop
        for generation in range(self.generations):
            offspring = []
            
            # Create offspring
            while len(offspring) < self.population_size:
                parents = random.sample(population, 2)
                child = crossover(parents[0], parents[1])
                if random.random() < 0.1:
                    child = mutate(child)
                offspring.append(child)
            
            # Combine and sort
            combined = population + offspring
            fronts = non_dominated_sort(combined)
            
            # Build new population
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= self.population_size:
                    new_population.extend(front)
                else:
                    # Sort by waste and add best
                    front.sort(key=lambda x: x.waste)
                    new_population.extend(front[:self.population_size - len(new_population)])
                    break
            
            population = new_population
        
        # Get best individual from first front
        fronts = non_dominated_sort(population)
        best = min(fronts[0], key=lambda x: x.waste)
        
        return best.plates if best.plates else self._fallback()
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
