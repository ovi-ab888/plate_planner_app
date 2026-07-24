"""
V9 - Smart Clustering Optimizer
High-Efficiency Multi-Phase Dynamic Chunking Engine for extreme variance
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy


class V9Optimizer(BaseOptimizer):
    """V9 - Smart Clustering Optimizer"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int):
        super().__init__(demand, capacity, max_plates)
        self.name = "V9 - Smart Clustering Optimizer"
        self.version = "V9"
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V9 - High-Efficiency Multi-Phase Dynamic Chunking Engine"""
        if not self.demand:
            return []
        
        remaining = copy.deepcopy(self.demand)
        plates = []
        
        for p_idx in range(self.max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            
            # Sort by quantity (highest first)
            sorted_active = sorted(active.items(), key=lambda x: x[1], reverse=True)
            total_active_qty = sum(active.values())
            
            # Create layout using proportional allocation
            layout = {}
            for tag, qty in sorted_active:
                share = qty / total_active_qty
                allocated_ups = int(round(share * self.capacity))
                if qty > 0 and allocated_ups < 1:
                    allocated_ups = 0
                layout[tag] = allocated_ups
            
            # Fix capacity - remove excess
            while sum(layout.values()) > self.capacity:
                max_tag = max(layout, key=lambda k: (layout[k], remaining[k]))
                if layout[max_tag] > 0:
                    layout[max_tag] -= 1
                else:
                    break
            
            # Fix capacity - add if needed
            while sum(layout.values()) < self.capacity:
                max_pressure_tag = max(active.keys(), key=lambda k: remaining[k] / (layout.get(k, 0) + 1))
                layout[max_pressure_tag] = layout.get(max_pressure_tag, 0) + 1
            
            # Calculate sheets
            valid_sheets = []
            for tag, ups in layout.items():
                if ups > 0:
                    valid_sheets.append(math.ceil(remaining[tag] / ups))
            
            if not valid_sheets:
                break
            
            valid_sheets.sort()
            
            # Use 25th percentile for sheet count (reduces waste)
            target_idx = min(int(len(valid_sheets) * 0.25), len(valid_sheets) - 1)
            sheets = max(1, valid_sheets[target_idx])
            
            # Update remaining
            for tag, ups in layout.items():
                if ups > 0:
                    remaining[tag] = max(0, remaining[tag] - (ups * sheets))
            
            plates.append({
                "name": plate_name(p_idx + 1),
                "layout": layout,
                "sheets": sheets
            })
        
        # Ensure all demand is met
        plates = ensure_demand_met(plates, self.demand)
        
        # Validate and fix plate capacity for all plates
        for plate in plates:
            layout = plate["layout"]
            total_ups = sum(layout.values())
            
            if total_ups != self.capacity:
                # Fix the layout to match capacity
                while sum(layout.values()) > self.capacity:
                    max_tag = max(layout, key=layout.get)
                    if layout[max_tag] > 1:
                        layout[max_tag] -= 1
                    else:
                        break
                
                while sum(layout.values()) < self.capacity:
                    best_tag = max(self.demand.keys(), key=lambda t: self.demand.get(t, 0) / (layout.get(t, 1) + 1))
                    layout[best_tag] = layout.get(best_tag, 0) + 1
                
                plate["layout"] = layout
        
        return plates
