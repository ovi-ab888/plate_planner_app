"""
V26 - NN Predictor Optimizer
Uses neural network inspired prediction
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy
import random


class SimplePredictor:
    """Simple pattern predictor inspired by neural networks"""
    
    def __init__(self):
        self.patterns = {}
    
    def learn_from_plate(self, layout, waste):
        """Learn from a plate configuration"""
        key = tuple(sorted(layout.items()))
        if key not in self.patterns:
            self.patterns[key] = []
        self.patterns[key].append(waste)
    
    def predict_layout(self, active, capacity):
        """Predict best layout for given active items"""
        best_pattern = None
        best_score = float('inf')
        
        # Find similar patterns
        for pattern, wastes in self.patterns.items():
            avg_waste = sum(wastes) / len(wastes)
            pattern_dict = dict(pattern)
            
            # Check if pattern covers same items
            if set(pattern_dict.keys()) == set(active.keys()):
                # Calculate similarity
                total_active = sum(active.values())
                similarity = sum(abs(pattern_dict.get(t, 0) - (active.get(t, 0) / total_active * capacity)) 
                               for t in active.keys())
                score = avg_waste + similarity * 0.1
                
                if score < best_score:
                    best_score = score
                    best_pattern = pattern_dict
        
        if best_pattern:
            return best_pattern
        
        # Fallback to balanced layout
        return create_valid_layout(active, capacity, "balanced")


class V26Optimizer(BaseOptimizer):
    """V26 - NN Predictor Optimizer"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int):
        super().__init__(demand, capacity, max_plates)
        self.name = "V26 - NN Predictor"
        self.version = "V26"
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V26 optimization with NN predictor"""
        
        predictor = SimplePredictor()
        population_size = 20
        generations = 40
        
        def create_individual_with_ml():
            """Create individual using ML predictor"""
            remaining = self.demand.copy()
            plates = []
            
            for _ in range(self.max_plates):
                active = {k: v for k, v in remaining.items() if v > 0}
                if not active:
                    break
                
                layout = predictor.predict_layout(active, self.capacity)
                
                # Calculate sheets
                sheets_list = []
                for tag, ups in layout.items():
                    if ups > 0 and remaining.get(tag, 0) > 0:
                        sheets_list.append(math.ceil(remaining[tag] / ups))
                
                sheets = max(1, min(sheets_list)) if sheets_list else 1
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
            
            return ensure_demand_met(plates, self.demand)
        
        def mutate_with_ml(plates):
            """Mutate using ML insights"""
            new_plates = copy.deepcopy(plates)
            if new_plates:
                plate_idx = random.randint(0, len(new_plates) - 1)
                layout = new_plates[plate_idx]["layout"]
                
                if len(layout) >= 2:
                    tags_list = list(layout.keys())
                    a, b = random.sample(tags_list, 2)
                    if layout[a] > 1:
                        layout[a] -= 1
                        layout[b] += 1
            return ensure_demand_met(new_plates, self.demand)
        
        def crossover_plates(p1, p2):
            """Crossover with ML enhancement"""
            point = random.randint(1, min(len(p1), len(p2)) - 1)
            child = p1[:point] + p2[point:]
            
            remaining = self.demand.copy()
            fixed = []
            for plate in child:
                active = {k: v for k, v in remaining.items() if v > 0}
                if not active:
                    break
                
                layout = plate["layout"].copy()
                sheets = plate["sheets"]
                fixed.append({"layout": layout, "sheets": sheets})
                
                for tag, ups in layout.items():
                    remaining[tag] = max(0, remaining[tag] - (ups * sheets))
            
            if any(v > 0 for v in remaining.values()) and fixed:
                last = fixed[-1]
                for tag in remaining:
                    if remaining[tag] > 0:
                        ups = max(1, last["layout"].get(tag, 1))
                        last["sheets"] += math.ceil(remaining[tag] / ups)
                        remaining[tag] = 0
            
            return ensure_demand_met(fixed, self.demand)
        
        # Initialize population
        population = []
        for _ in range(population_size):
            ind = create_individual_with_ml()
            population.append(ind)
        
        # Evolution with learning
        for generation in range(generations):
            # Evaluate and learn
            scored = [(calculate_waste_percent(ind, self.demand), ind) for ind in population]
            scored.sort(key=lambda x: x[0])
            
            # Learn from best individuals
            for i in range(min(3, len(scored))):
                waste, best_ind = scored[i]
                for plate in best_ind:
                    predictor.learn_from_plate(plate["layout"], waste)
            
            # Select elites
            elites = [copy.deepcopy(scored[i][1]) for i in range(min(3, len(scored)))]
            new_population = elites.copy()
            
            # Create offspring
            while len(new_population) < population_size:
                p1 = copy.deepcopy(random.choice(elites))
                p2 = copy.deepcopy(random.choice(elites))
                child = crossover_plates(p1, p2)
                
                if random.random() < 0.3:
                    child = mutate_with_ml(child)
                
                new_population.append(child)
            
            population = new_population
        
        # Get best solution
        best_idx = min(range(len(population)), 
                      key=lambda i: calculate_waste_percent(population[i], self.demand))
        return ensure_demand_met(population[best_idx], self.demand) if population else self._fallback()
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
