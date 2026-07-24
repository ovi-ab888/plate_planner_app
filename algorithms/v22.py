"""
V22 - Q-Learning Optimizer
Reinforcement Learning based optimization
"""

from typing import Dict, List, Any
from algorithms.base import BaseOptimizer
from algorithms.v1_helpers import create_valid_layout, plate_name, ensure_demand_met, calculate_waste_percent
import math
import copy
import random


class QLearningPlateOptimizer:
    """Q-Learning optimizer for plate ratio"""
    
    def __init__(self, demand, capacity, max_plates, learning_rate=0.1, discount=0.9, epsilon=0.1):
        self.demand = demand
        self.capacity = capacity
        self.max_plates = max_plates
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.q_table = {}
        self.tags = list(demand.keys())
    
    def get_state_key(self, remaining, current_layout):
        """Create state key for Q-table"""
        remaining_tuple = tuple(remaining.get(t, 0) for t in self.tags)
        layout_tuple = tuple(current_layout.get(t, 0) for t in self.tags)
        return (remaining_tuple, layout_tuple)
    
    def get_action(self, state, possible_actions):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        
        q_values = [self.q_table.get((state, action), 0) for action in possible_actions]
        max_q = max(q_values) if q_values else 0
        best_actions = [a for a, q in zip(possible_actions, q_values) if q == max_q]
        return random.choice(best_actions) if best_actions else random.choice(possible_actions)
    
    def get_possible_actions(self, layout):
        """Generate possible actions from current layout"""
        actions = []
        tags_list = list(layout.keys())
        if len(tags_list) >= 2:
            for i in range(len(tags_list)):
                for j in range(len(tags_list)):
                    if i != j and layout[tags_list[i]] > 1:
                        actions.append(('mutate', tags_list[i], tags_list[j]))
        return actions
    
    def apply_action(self, layout, action):
        """Apply action to layout"""
        new_layout = layout.copy()
        if action[0] == 'mutate':
            _, a, b = action
            new_layout[a] -= 1
            new_layout[b] = new_layout.get(b, 0) + 1
        return new_layout
    
    def optimize(self, episodes=30):
        """Main Q-learning optimization loop"""
        best_plates = None
        best_waste = float('inf')
        
        for episode in range(episodes):
            remaining = self.demand.copy()
            plates = []
            
            for plate_num in range(self.max_plates):
                active = {k: v for k, v in remaining.items() if v > 0}
                if not active:
                    break
                
                # Start with balanced layout
                layout = create_valid_layout(active, self.capacity, "balanced")
                state = self.get_state_key(remaining, layout)
                
                # Get possible actions
                possible_actions = self.get_possible_actions(layout)
                
                if possible_actions and len(plates) < self.max_plates - 1:
                    action = self.get_action(state, possible_actions)
                    new_layout = self.apply_action(layout, action)
                    
                    # Ensure exact capacity
                    while sum(new_layout.values()) > self.capacity:
                        max_tag = max(new_layout, key=new_layout.get)
                        if new_layout[max_tag] > 1:
                            new_layout[max_tag] -= 1
                        else:
                            break
                    
                    while sum(new_layout.values()) < self.capacity:
                        max_tag = max(active, key=active.get)
                        new_layout[max_tag] = new_layout.get(max_tag, 0) + 1
                    
                    # Calculate reward
                    new_sheets = max(1, min(math.ceil(remaining[t] / new_layout.get(t, 1)) for t in active))
                    waste = sum(max(0, new_layout.get(t, 0) * new_sheets - remaining.get(t, 0)) for t in active)
                    reward = -waste
                    
                    # Update Q-value
                    next_state = self.get_state_key(remaining, new_layout)
                    old_q = self.q_table.get((state, action), 0)
                    next_actions = self.get_possible_actions(new_layout)
                    next_max_q = max([self.q_table.get((next_state, a), 0) for a in next_actions]) if next_actions else 0
                    new_q = old_q + self.lr * (reward + self.discount * next_max_q - old_q)
                    self.q_table[(state, action)] = new_q
                    
                    layout = new_layout
                
                # Calculate sheets
                sheets_list = []
                for tag, ups in layout.items():
                    if ups > 0 and remaining.get(tag, 0) > 0:
                        sheets_list.append(math.ceil(remaining[tag] / ups))
                
                sheets = max(1, min(sheets_list)) if sheets_list else 1
                
                # Update remaining
                for tag, ups in layout.items():
                    remaining[tag] = max(0, remaining[tag] - (ups * sheets))
                
                plates.append({
                    "name": plate_name(len(plates) + 1),
                    "layout": layout,
                    "sheets": sheets
                })
            
            plates = ensure_demand_met(plates, self.demand)
            waste = calculate_waste_percent(plates, self.demand)
            
            if waste < best_waste:
                best_waste = waste
                best_plates = copy.deepcopy(plates)
            
            # Decay epsilon
            self.epsilon *= 0.99
        
        return ensure_demand_met(best_plates, self.demand) if best_plates else None


class V22Optimizer(BaseOptimizer):
    """V22 - Q-Learning Optimizer"""
    
    def __init__(self, demand: Dict[str, int], capacity: int, max_plates: int, episodes: int = 30):
        super().__init__(demand, capacity, max_plates)
        self.name = "V22 - Q-Learning Optimizer"
        self.version = "V22"
        self.episodes = episodes
    
    def optimize(self) -> List[Dict[str, Any]]:
        """Execute V22 optimization with Q-Learning"""
        optimizer = QLearningPlateOptimizer(
            self.demand, 
            self.capacity, 
            self.max_plates,
            learning_rate=0.1,
            discount=0.9,
            epsilon=0.1
        )
        result = optimizer.optimize(self.episodes)
        return result if result else self._fallback()
    
    def _fallback(self):
        from algorithms.v3 import V3Optimizer
        return V3Optimizer(self.demand, self.capacity, self.max_plates).optimize()
