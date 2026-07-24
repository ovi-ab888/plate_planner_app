"""
V8 - MCTS Tree Search
Monte Carlo Tree Search optimization
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy
import random


class MCTSNodeV8:
    """Node for MCTS tree search"""
    def __init__(self, layout: dict, remaining: dict, capacity: int, parent=None):
        self.layout = layout
        self.remaining = remaining.copy()
        self.capacity = capacity
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0
    
    def best_child(self, c_param: float = 1.4):
        if not self.children:
            return None
        choices = []
        for child in self.children:
            if child.visits == 0:
                ucb = float('inf')
            else:
                ucb = (child.score / child.visits) + c_param * math.sqrt(
                    2 * math.log(self.visits) / child.visits
                )
            choices.append((ucb, child))
        return max(choices, key=lambda x: x[0])[1]


class V8Optimizer(BaseOptimizer):
    """V8 - MCTS Tree Search"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int, iterations: int = 80):
        super().__init__(demand, capacity, max_plates)
        self.name = "V8 - MCTS Tree Search"
        self.version = "V8"
        self.iterations = iterations
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V8 optimization with MCTS"""
        remaining = self.demand.copy()
        plates = []
        
        for plate_num in range(self.max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            
            # Initial layout
            root_layout = create_valid_layout(active, self.capacity, "balanced")
            sheets = max(1, min(math.ceil(active[t] / root_layout[t]) for t in root_layout if root_layout[t] > 0))
            
            root = MCTSNodeV8(root_layout, active, self.capacity)
            
            # MCTS iterations
            for _ in range(self.iterations):
                node = root
                
                # Selection
                while node.children:
                    next_node = node.best_child()
                    if next_node is None:
                        break
                    node = next_node
                
                # Expansion
                current_layout = node.layout.copy()
                tags = list(current_layout.keys())
                
                if len(tags) >= 2:
                    a, b = random.sample(tags, 2)
                    if current_layout.get(a, 0) > 1:
                        current_layout[a] -= 1
                        current_layout[b] = current_layout.get(b, 0) + 1
                
                # Ensure exact capacity
                final_layout = self._adjust_to_exact_capacity(current_layout, self.capacity, active)
                
                # Simulation
                waste = sum(max(0, ups * sheets - active.get(tag, 0)) for tag, ups in final_layout.items())
                score = -waste
                
                # Backpropagation
                new_node = MCTSNodeV8(final_layout, active, self.capacity, node)
                node.children.append(new_node)
                
                current_node = node
                while current_node:
                    current_node.visits += 1
                    current_node.score += score
                    current_node = current_node.parent
            
            # Select best child
            if root.children:
                best_child = max(root.children, key=lambda c: (c.score / c.visits) if c.visits > 0 else 0)
                best_layout = best_child.layout
            else:
                best_layout = root_layout
            
            plates.append({
                "name": plate_name(plate_num + 1),
                "layout": best_layout,
                "sheets": sheets
            })
            
            for tag, ups in best_layout.items():
                remaining[tag] = max(0, remaining[tag] - ups * sheets)
        
        return ensure_demand_met(plates, self.demand)
    
    def _adjust_to_exact_capacity(self, layout: dict, capacity: int, remaining: dict) -> dict:
        """Adjust layout to exact capacity"""
        if not layout or not remaining:
            return layout
        
        new_layout = layout.copy()
        
        # Add until capacity reached
        while sum(new_layout.values()) < capacity:
            if not new_layout:
                break
            best_tag = max(new_layout.keys(), key=lambda t: remaining.get(t, 0) / new_layout.get(t, 1))
            new_layout[best_tag] = new_layout.get(best_tag, 0) + 1
        
        # Remove until capacity reached
        while sum(new_layout.values()) > capacity:
            candidates = [t for t in new_layout if new_layout[t] > 1]
            if not candidates:
                break
            smallest_tag = min(candidates, key=lambda t: new_layout[t])
            new_layout[smallest_tag] -= 1
        
        return new_layout
