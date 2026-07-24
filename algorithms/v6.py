"""
V6 - Integer Solver
Uses PuLP for integer programming optimization
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import plate_name, ensure_demand_met
import math


class V6Optimizer(BaseOptimizer):
    """V6 - Integer Solver using PuLP"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int):
        super().__init__(demand, capacity, max_plates)
        self.name = "V6 - Integer Solver"
        self.version = "V6"
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V6 optimization using PuLP"""
        try:
            from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
        except ImportError:
            from algorithms.v3 import V3Optimizer
            return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
        
        remaining = self.demand.copy()
        plates = []
        
        for plate_num in range(self.max_plates):
            active_tags = [t for t in self.demand.keys() if remaining[t] > 0]
            if not active_tags:
                break
            
            try:
                model = LpProblem(f"Plate_{plate_num}", LpMinimize)
                ups = {t: LpVariable(f"UPS_{t}", lowBound=0, cat="Integer") for t in active_tags}
                sheets = LpVariable("Sheets", lowBound=1, cat="Integer")
                
                # Minimize excess production
                model += lpSum(ups[t] * sheets - remaining[t] for t in active_tags)
                model += lpSum(ups[t] for t in active_tags) == self.capacity
                
                for t in active_tags:
                    model += ups[t] * sheets >= remaining[t]
                
                model.solve()
                
                if model.status == 1:
                    layout = {t: int(value(ups[t])) for t in active_tags}
                    sheet_count = int(value(sheets))
                    
                    # Ensure no zero UPS
                    for tag in active_tags:
                        if layout.get(tag, 0) == 0:
                            layout[tag] = 1
                    
                    # Fix capacity if needed
                    while sum(layout.values()) > self.capacity:
                        max_tag = max(layout, key=layout.get)
                        if layout[max_tag] > 1:
                            layout[max_tag] -= 1
                        else:
                            break
                    
                    plates.append({
                        "name": plate_name(plate_num + 1),
                        "layout": layout,
                        "sheets": sheet_count
                    })
                    
                    for t in active_tags:
                        remaining[t] = max(0, remaining[t] - layout[t] * sheet_count)
                else:
                    from algorithms.v3 import V3Optimizer
                    return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
            
            except Exception:
                from algorithms.v3 import V3Optimizer
                return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
        
        return ensure_demand_met(plates, self.demand) if plates else self._fallback()
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
