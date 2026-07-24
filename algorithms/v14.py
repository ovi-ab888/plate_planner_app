"""
V14 - Adaptive Threshold Optimizer
Uses adaptive thresholds for optimization
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy


class V14Optimizer(BaseOptimizer):
    """V14 - Adaptive Threshold Optimizer"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int):
        super().__init__(demand, capacity, max_plates)
        self.name = "V14 - Adaptive Threshold Optimizer"
        self.version = "V14"
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V14 optimization with adaptive thresholds"""
        best_plates = None
        best_waste = float('inf')
        
        # Try different threshold levels
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            remaining = self.demand.copy()
            plates = []
            
            for i in range(self.max_plates):
                active = {k: v for k, v in remaining.items() if v > 0}
                if not active:
                    break
                
                # Create layout with threshold
                layout = self._threshold_layout(active, self.capacity, threshold)
                
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
    
    def _threshold_layout(self, active: Dict[str, int], capacity: int, threshold: float) -> Dict[str, int]:
        """Create layout with adaptive threshold"""
        total_qty = sum(active.values())
        layout = {}
        
        # First pass: allocate based on threshold
        remaining_cap = capacity
        for tag, qty in active.items():
            ratio = qty / total_qty
            if ratio >= threshold:
                ups = max(1, int(ratio * capacity))
                layout[tag] = ups
                remaining_cap -= ups
        
        # Second pass: distribute remaining capacity
        if remaining_cap > 0:
            remaining_tags = [t for t in active.keys() if t not in layout or layout.get(t, 0) == 0]
            if remaining_tags:
                # Give 1 to each remaining tag
                for tag in remaining_tags:
                    if remaining_cap > 0:
                        layout[tag] = 1
                        remaining_cap -= 1
                
                # Distribute extra capacity
                while remaining_cap > 0:
                    best_tag = max(active.keys(), key=lambda t: active[t] / (layout.get(t, 1) + 1))
                    layout[best_tag] = layout.get(best_tag, 0) + 1
                    remaining_cap -= 1
            else:
                # All tags already have allocation, distribute extra
                while remaining_cap > 0:
                    best_tag = max(active.keys(), key=lambda t: active[t] / (layout.get(t, 1) + 1))
                    layout[best_tag] = layout.get(best_tag, 0) + 1
                    remaining_cap -= 1
        
        # Ensure all tags have at least 1 if possible
        for tag in active.keys():
            if layout.get(tag, 0) == 0 and len(active) <= capacity:
                layout[tag] = 1
        
        # Fix capacity if needed
        while sum(layout.values()) > capacity:
            max_tag = max(layout, key=layout.get)
            if layout[max_tag] > 1:
                layout[max_tag] -= 1
            else:
                break
        
        return layout
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
