"""
V10 - Greedy with Lookahead Optimizer
Greedy allocation with lookahead optimization
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy


class V10Optimizer(BaseOptimizer):
    """V10 - Greedy with Lookahead"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int):
        super().__init__(demand, capacity, max_plates)
        self.name = "V10 - Greedy with Lookahead"
        self.version = "V10"
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V10 optimization with lookahead"""
        best_plates = None
        best_waste = float('inf')
        
        # Try different lookahead depths
        for lookahead in range(1, 4):
            remaining = self.demand.copy()
            plates = []
            
            for i in range(self.max_plates):
                active = {k: v for k, v in remaining.items() if v > 0}
                if not active:
                    break
                
                # Use lookahead to choose best layout
                layout = self._lookahead_layout(active, self.capacity, lookahead)
                
                # Calculate sheets
                possible_sheets = []
                for tag, ups in layout.items():
                    if ups > 0 and remaining.get(tag, 0) > 0:
                        possible_sheets.append(math.ceil(remaining[tag] / ups))
                
                sheets = max(1, min(possible_sheets)) if possible_sheets else 1
                
                # Update remaining
                for tag, ups in layout.items():
                    remaining[tag] = max(0, remaining[tag] - (ups * sheets))
                
                plates.append({
                    "name": plate_name(i + 1),
                    "layout": layout,
                    "sheets": sheets
                })
            
            # Ensure all demand is met
            plates = ensure_demand_met(plates, self.demand)
            waste = calculate_waste_percent(plates, self.demand)
            
            if waste < best_waste:
                best_waste = waste
                best_plates = copy.deepcopy(plates)
        
        return best_plates if best_plates else self._fallback()
    
    def _lookahead_layout(self, active: Dict[str, int], capacity: int, lookahead: int) -> Dict[str, int]:
        """Choose layout with lookahead"""
        tags = list(active.keys())
        
        if len(tags) <= 1:
            return create_valid_layout(active, capacity, "balanced")
        
        # Generate candidate layouts
        candidates = []
        
        # Start with balanced
        base_layout = create_valid_layout(active, capacity, "balanced")
        candidates.append(base_layout)
        
        # Generate variations
        for _ in range(10 * lookahead):
            layout = base_layout.copy()
            
            # Mutate
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
            
            candidates.append(layout)
        
        # Score each candidate with lookahead
        best_layout = base_layout
        best_score = float('inf')
        
        for layout in candidates:
            # Simulate lookahead
            score = self._simulate_lookahead(layout, active, capacity, lookahead)
            
            if score < best_score:
                best_score = score
                best_layout = layout
        
        return best_layout
    
    def _simulate_lookahead(self, layout: Dict[str, int], active: Dict[str, int], 
                           capacity: int, depth: int) -> float:
        """Simulate lookahead to score a layout"""
        if depth == 0:
            # Calculate waste for current layout
            remaining = active.copy()
            for tag, ups in layout.items():
                if ups > 0:
                    sheets = math.ceil(remaining.get(tag, 0) / ups)
                    remaining[tag] = max(0, remaining.get(tag, 0) - ups * sheets)
            
            return sum(remaining.values())
        
        # Simulate one step ahead
        remaining = active.copy()
        for tag, ups in layout.items():
            if ups > 0:
                sheets = math.ceil(remaining.get(tag, 0) / ups)
                remaining[tag] = max(0, remaining.get(tag, 0) - ups * sheets)
        
        # Check if all done
        if all(v == 0 for v in remaining.values()):
            return 0
        
        # Create next layout and recurse
        next_active = {k: v for k, v in remaining.items() if v > 0}
        if not next_active:
            return sum(remaining.values())
        
        next_layout = create_valid_layout(next_active, capacity, "balanced")
        return self._simulate_lookahead(next_layout, remaining, capacity, depth - 1)
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
