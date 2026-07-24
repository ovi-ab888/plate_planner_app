"""
V23 - Branch & Bound Optimizer
Exact optimization using Branch and Bound
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy


class V23Optimizer(BaseOptimizer):
    """V23 - Branch & Bound Optimizer"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int):
        super().__init__(demand, capacity, max_plates)
        self.name = "V23 - Branch & Bound Optimizer"
        self.version = "V23"
        self.best_solution = None
        self.best_waste = float('inf')
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V23 optimization with Branch & Bound"""
        # Start with a good initial solution
        from algorithms.v3 import V3Optimizer
        initial = V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
        
        if initial:
            self.best_solution = initial
            self.best_waste = calculate_waste_percent(initial, self.demand)
        
        # Branch and bound search
        self._branch_and_bound([], self.demand.copy(), 0)
        
        return self.best_solution if self.best_solution else self._fallback()
    
    def _branch_and_bound(self, current_plates: List[Dict], remaining: Dict[str, int], depth: int):
        """Recursive branch and bound search"""
        # Check if we've exceeded max plates
        if depth >= self.max_plates:
            return
        
        # Check if all demand is met
        if all(v <= 0 for v in remaining.values()):
            # Complete solution found
            plates = ensure_demand_met(current_plates, self.demand)
            waste = calculate_waste_percent(plates, self.demand)
            
            if waste < self.best_waste:
                self.best_waste = waste
                self.best_solution = copy.deepcopy(plates)
            return
        
        # Lower bound check
        if current_plates:
            waste = calculate_waste_percent(current_plates, self.demand)
            if waste >= self.best_waste:
                return
        
        active = {k: v for k, v in remaining.items() if v > 0}
        if not active:
            return
        
        # Generate possible layouts
        layouts = self._generate_layouts(active, self.capacity)
        
        # Try layouts in order of potential
        for layout in layouts[:10]:  # Limit branching
            # Calculate sheets for this layout
            sheets_list = []
            for tag, ups in layout.items():
                if ups > 0 and remaining.get(tag, 0) > 0:
                    sheets_list.append(math.ceil(remaining[tag] / ups))
            
            if not sheets_list:
                continue
            
            sheets = max(1, min(sheets_list))
            
            # Update remaining
            new_remaining = remaining.copy()
            for tag, ups in layout.items():
                new_remaining[tag] = max(0, new_remaining.get(tag, 0) - (ups * sheets))
            
            # Create new plate
            new_plate = {
                "name": plate_name(len(current_plates) + 1),
                "layout": layout,
                "sheets": sheets
            }
            
            # Recurse
            self._branch_and_bound(
                current_plates + [new_plate],
                new_remaining,
                depth + 1
            )
    
    def _generate_layouts(self, active: Dict[str, int], capacity: int) -> List[Dict[str, int]]:
        """Generate possible layouts for given active items"""
        layouts = []
        tags = list(active.keys())
        n = len(tags)
        
        if n == 0:
            return []
        
        if n == 1:
            return [{tags[0]: capacity}]
        
        # Generate variations
        base = create_valid_layout(active, capacity, "balanced")
        layouts.append(base)
        
        # Generate variations by redistributing UPS
        for _ in range(20):
            layout = base.copy()
            
            if len(layout) >= 2:
                a, b = tags[:2]
                if layout.get(a, 0) > 1:
                    layout[a] = layout.get(a, 0) - 1
                    layout[b] = layout.get(b, 0) + 1
            
            # Fix capacity
            while sum(layout.values()) > capacity:
                max_tag = max(layout, key=layout.get)
                if layout[max_tag] > 1:
                    layout[max_tag] -= 1
                else:
                    break
            
            while sum(layout.values()) < capacity:
                max_tag = max(active, key=lambda t: active[t] / (layout.get(t, 1) + 1))
                layout[max_tag] = layout.get(max_tag, 0) + 1
            
            if layout not in layouts:
                layouts.append(layout)
        
        return layouts
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
