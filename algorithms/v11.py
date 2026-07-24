"""
V11 - Genetic Algorithm
Evolutionary optimization using genetic operators
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy
import random


class V11Optimizer(BaseOptimizer):
    """V11 - Genetic Algorithm"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int,
                 population_size: int = 30, generations: int = 50,
                 mutation_rate: float = 0.1, elite_size: int = 5):
        super().__init__(demand, capacity, max_plates)
        self.name = "V11 - Genetic Algorithm"
        self.version = "V11"
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V11 genetic algorithm optimization"""
        
        def create_individual():
            remaining = self.demand.copy()
            plates = []
            
            for p in range(self.max_plates):
                active = {k: v for k, v in remaining.items() if v > 0}
                if not active:
                    break
                
                layout = create_valid_layout(active, self.capacity, "greedy")
                sheets = max(1, min(math.ceil(remaining[tag] / layout.get(tag, 1)) for tag in active))
                
                for tag, ups in layout.items():
                    remaining[tag] = max(0, remaining[tag] - (ups * sheets))
                
                plates.append({"layout": layout, "sheets": sheets})
            
            if any(v > 0 for v in remaining.values()) and plates:
                last = plates[-1]
                for tag in remaining:
                    if remaining[tag] > 0:
                        ups = max(1, last["layout"].get(tag, 1))
                        last["sheets"] += math.ceil(remaining[tag] / ups)
                        remaining[tag] = 0
            
            return plates
        
        def fitness(plates):
            return 100 - calculate_waste_percent(plates, self.demand)
        
        def crossover(parent1, parent2):
            point = random.randint(1, min(len(parent1), len(parent2)) - 1)
            child = parent1[:point] + parent2[point:]
            
            remaining = self.demand.copy()
            new_plates = []
            
            for p in child:
                active = {k: v for k, v in remaining.items() if v > 0}
                if not active:
                    break
                
                layout = p.get("layout", {})
                if sum(layout.values()) != self.capacity:
                    layout = create_valid_layout(active, self.capacity, "greedy")
                
                sheets = p.get("sheets", 1)
                new_plates.append({"layout": layout, "sheets": sheets})
                
                for tag, ups in layout.items():
                    remaining[tag] = max(0, remaining[tag] - (ups * sheets))
            
            if any(v > 0 for v in remaining.values()) and new_plates:
                last = new_plates[-1]
                for tag in remaining:
                    if remaining[tag] > 0:
                        ups = max(1, last["layout"].get(tag, 1))
                        last["sheets"] += math.ceil(remaining[tag] / ups)
                        remaining[tag] = 0
            
            return new_plates
        
        def mutate(plates):
            if random.random() > self.mutation_rate:
                return plates
            
            mutated = copy.deepcopy(plates)
            if mutated:
                plate_idx = random.randint(0, len(mutated) - 1)
                layout = mutated[plate_idx]["layout"]
                
                if len(layout) >= 2:
                    tags = list(layout.keys())
                    a, b = random.sample(tags, 2)
                    if layout[a] > 1:
                        layout[a] -= 1
                        layout[b] += 1
            
            return mutated
        
        # Initialize population
        population = [create_individual() for _ in range(self.population_size)]
        
        # Evolution loop
        for generation in range(self.generations):
            fitness_scores = [fitness(ind) for ind in population]
            
            # Select elites
            elite_indices = sorted(range(len(fitness_scores)), 
                                  key=lambda i: fitness_scores[i], reverse=True)[:self.elite_size]
            new_population = [population[i] for i in elite_indices]
            
            # Create offspring
            while len(new_population) < self.population_size:
                tournament = random.sample(list(zip(population, fitness_scores)), 5)
                parent1 = max(tournament, key=lambda x: x[1])[0]
                
                tournament = random.sample(list(zip(population, fitness_scores)), 5)
                parent2 = max(tournament, key=lambda x: x[1])[0]
                
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # Get best individual
        best_idx = max(range(len(population)), key=lambda i: fitness(population[i]))
        return ensure_demand_met(population[best_idx], self.demand)
