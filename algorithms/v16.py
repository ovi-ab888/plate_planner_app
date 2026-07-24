"""
V16 - Plate Merge Optimizer
Optimizes by merging compatible plates
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy


class V16Optimizer(BaseOptimizer):
    """V16 - Plate Merge Optimizer"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int):
        super().__init__(demand, capacity, max_plates)
        self.name = "V16 - Plate Merge Optimizer"
        self.version = "V16"
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V16 optimization with plate merging"""
        # Start with V3 solution
        from algorithms.v3 import V3Optimizer
        plates = V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
        
        if not plates:
            return self._fallback()
        
        # Try to merge plates
        merged = []
        skip = set()
        
        for i in range(len(plates)):
            if i in skip:
                continue
            
            current = copy.deepcopy(plates[i])
            
            # Try to merge with other plates
            for j in range(i + 1, len(plates)):
                if j in skip:
                    continue
                
                candidate = copy.deepcopy(plates[j])
                combined = current["layout"].copy()
                
                # Add UPS from candidate
                for tag, ups in candidate["layout"].items():
                    combined[tag] = combined.get(tag, 0) + ups
                
                # Check if combined layout fits in one plate
                if sum(combined.values()) <= self.capacity:
                    current["layout"] = combined
                    current["sheets"] = max(current["sheets"], candidate["sheets"])
                    skip.add(j)
            
            merged.append(current)
        
        # If merging reduced plates, add back if needed
        if len(merged) < len(plates):
            # Check if demand is still met
            temp_plates = ensure_demand_met(merged, self.demand)
            if len(temp_plates) <= self.max_plates:
                return ensure_demand_met(temp_plates, self.demand)
        
        # Try alternative merging strategy
        merged2 = []
        skip = set()
        
        for i in range(len(plates)):
            if i in skip:
                continue
            
            current = copy.deepcopy(plates[i])
            
            for j in range(i + 1, len(plates)):
                if j in skip:
                    continue
                
                candidate = copy.deepcopy(plates[j])
                
                # Check if plates can be combined
                combined_ups = sum(current["layout"].values()) + sum(candidate["layout"].values())
                
                if combined_ups <= self.capacity:
                    combined_layout = current["layout"].copy()
                    for tag, ups in candidate["layout"].items():
                        combined_layout[tag] = combined_layout.get(tag, 0) + ups
                    
                    current["layout"] = combined_layout
                    current["sheets"] = max(current["sheets"], candidate["sheets"])
                    skip.add(j)
            
            merged2.append(current)
        
        # Choose better solution
        waste1 = calculate_waste_percent(merged, self.demand)
        waste2 = calculate_waste_percent(merged2, self.demand)
        
        if waste1 <= waste2:
            return ensure_demand_met(merged, self.demand)
        else:
            return ensure_demand_met(merged2, self.demand)
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
