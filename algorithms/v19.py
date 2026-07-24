"""
V19 - CP-SAT Optimizer
Uses Google OR-Tools CP-SAT solver
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import plate_name, ensure_demand_met
import math


class V19Optimizer(BaseOptimizer):
    """V19 - CP-SAT Optimizer using OR-Tools"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int, time_limit: int = 5):
        super().__init__(demand, capacity, max_plates)
        self.name = "V19 - CP-SAT Optimizer"
        self.version = "V19"
        self.time_limit = time_limit
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V19 optimization using CP-SAT"""
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            from algorithms.v3 import V3Optimizer
            return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
        
        tags = list(self.demand.keys())
        n_tags = len(tags)
        
        if n_tags == 0:
            return []
        
        model = cp_model.CpModel()
        max_possible_plates = self.max_plates
        
        # Variables: UPS for each plate and tag
        ups = {}
        for i in range(max_possible_plates):
            for idx, tag in enumerate(tags):
                max_ups = min(self.capacity, self.demand.get(tag, 0))
                ups[(i, idx)] = model.NewIntVar(0, max_ups, f'ups_{i}_{tag}')
        
        # Variables: Sheets for each plate
        sheets = {}
        for i in range(max_possible_plates):
            sheets[i] = model.NewIntVar(0, sum(self.demand.values()), f'sheets_{i}')
        
        # Variables: Plate usage
        plate_used = {}
        for i in range(max_possible_plates):
            plate_used[i] = model.NewBoolVar(f'used_{i}')
        
        # Constraints
        for i in range(max_possible_plates):
            total_ups = sum(ups[(i, idx)] for idx in range(n_tags))
            model.Add(total_ups <= self.capacity)
            model.Add(total_ups == 0).OnlyEnforceIf(plate_used[i].Not())
            model.Add(total_ups > 0).OnlyEnforceIf(plate_used[i])
        
        # Demand constraints
        for idx, tag in enumerate(tags):
            total_produced = sum(ups[(i, idx)] * sheets[i] for i in range(max_possible_plates))
            model.Add(total_produced >= self.demand[tag])
        
        # Sheet constraints
        for i in range(max_possible_plates):
            model.Add(sheets[i] >= 1).OnlyEnforceIf(plate_used[i])
            model.Add(sheets[i] == 0).OnlyEnforceIf(plate_used[i].Not())
        
        # Objective: Minimize total sheets
        total_sheets = sum(sheets[i] for i in range(max_possible_plates))
        model.Minimize(total_sheets)
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            plates = []
            for i in range(max_possible_plates):
                if solver.Value(plate_used[i]) and solver.Value(sheets[i]) > 0:
                    layout = {}
                    for idx, tag in enumerate(tags):
                        ups_val = solver.Value(ups[(i, idx)])
                        if ups_val > 0:
                            layout[tag] = ups_val
                    
                    if layout:
                        plates.append({
                            "name": plate_name(len(plates) + 1),
                            "layout": layout,
                            "sheets": int(solver.Value(sheets[i]))
                        })
            
            return ensure_demand_met(plates, self.demand) if plates else self._fallback()
        
        return self._fallback()
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer
