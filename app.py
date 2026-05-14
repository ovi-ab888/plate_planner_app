# app_final.py — 10-in-1 PLATE RATIO COMPARATOR
# V3 to V10 Complete | Compare All Algorithms | Pick Best
# Design by Ovi

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
from io import BytesIO
from collections import Counter
from math import ceil, floor
import string
import copy
import random
import math
from datetime import datetime

# Try to import PuLP for V8
try:
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

st.set_page_config(
    page_title="10-in-1 Plate Ratio Comparator | Ovi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================================================
#  PASSWORD CHECK SYSTEM
# ================================================================
def check_password():
    expected = None
    try:
        expected = st.secrets.get("app_password", None)
    except Exception:
        expected = None
    if expected is None:
        expected = os.environ.get("PEPCO_APP_PASSWORD")
    if expected is None:
        st.error("App password not configured.")
        return False

    def _password_entered():
        if st.session_state.get("password") == expected:
            st.session_state["password_correct"] = True
            try:
                del st.session_state["password"]
            except Exception:
                pass
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", None) is True:
        return True

    st.markdown("""
    <style>
        .stApp { background: black !important; }
        .main > div { background: transparent !important; padding: 0 !important; }
        .block-container { padding: 0rem !important; max-width: 52% !important; }
        .stTextInput input {
            background: rgba(255,255,255,0.1) !important;
            border: 2px solid #333 !important;
            border-radius: 10px !important;
            color: white !important;
            text-align: center !important;
        }
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 1rem 0rem 1rem;
            text-align: center;
        }
        .main-header h1 { color: white; font-size: 2.5rem; }
        .designer-name { color: #ffd700; }
        .password-container {
            max-width: 450px;
            margin: 60px auto 0 auto;
            padding: 2.5rem;
            background: rgba(0, 0, 0, 0.85);
            border-radius: 20px;
            text-align: center;
        }
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>📊 10-in-1 Plate Ratio Comparator</h1>
        <p>V3 • V4 • V5 • V6 • V7 • V8 • V9 • V10 | Compare All | Pick Best</p>
        <p class="designer-name">✨ Design by Ovi ✨</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="height: 40px;"></div><div class="password-container"><h2>🔐 Access Code</h2><p>Enter your access code to continue</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.text_input("Password", type="password", key="password", on_change=_password_entered, label_visibility="collapsed")
    
    if st.session_state.get("password_correct") is False:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.error("❌ Incorrect password. Contact Mr. Ovi.")
    return False

if not check_password():
    st.stop()

# ================================================================
# CSS FOR MAIN APP
# ================================================================
st.markdown("""
<style>
    .stApp { background: black !important; }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { color: white; font-size: 2.5rem; }
    .card {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #333;
    }
    .card-title {
        font-size: 1.3rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        border-bottom: 2px solid #667eea;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .best-algo {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
        border: 2px solid gold;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        background: #1a1a1a;
        border-radius: 15px;
        margin-top: 2rem;
    }
    .tag-display {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #667eea;
        color: #667eea;
        font-weight: bold;
        text-align: center;
    }
    .warning {
        background: #332700;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        color: #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# HELPER FUNCTIONS
# ================================================================
def plate_name(n):
    n -= 1
    chars = string.ascii_uppercase
    out = ""
    while True:
        out = chars[n % 26] + out
        n = n // 26 - 1
        if n < 0:
            break
    return out

def calculate_waste_percent(plates, demand):
    total_produced = 0
    total_demand = sum(demand.values())
    for tag in demand:
        produced_qty = 0
        for p in plates:
            ups = p["layout"].get(tag, 0)
            produced_qty += ups * p["sheets"]
        total_produced += produced_qty
    if total_produced == 0:
        return 100
    waste = total_produced - total_demand
    return round((waste / total_produced) * 100, 2)

def build_full_summary(plates, demand, original_qty):
    rows = []
    sl = 1
    for tag in demand.keys():
        row = {
            "SL": sl,
            "Tag": tag,
            "Original QTY": original_qty[tag],
            "Produced (+Add-on)": demand[tag]
        }
        for p in plates:
            ups = p["layout"].get(tag, 0)
            row[f"Plate {p['name']}"] = ups
        total_produced = 0
        for p in plates:
            ups = p["layout"].get(tag, 0)
            total_produced += ups * p["sheets"]
        excess = total_produced - demand[tag]
        excess_percent = round((excess / demand[tag]) * 100, 2) if demand[tag] else 0
        row["Total Produced QTY"] = total_produced
        row["Excess"] = excess
        row["Excess %"] = f"{excess_percent}%"
        rows.append(row)
        sl += 1
    
    df = pd.DataFrame(rows)
    
    total_row = {
        "SL": "📊",
        "Tag": "TOTAL",
        "Original QTY": df["Original QTY"].sum(),
        "Produced (+Add-on)": df["Produced (+Add-on)"].sum(),
    }
    for p in plates:
        total_row[f"Plate {p['name']}"] = df[f"Plate {p['name']}"].sum()
    total_row["Total Produced QTY"] = df["Total Produced QTY"].sum()
    total_row["Excess"] = df["Excess"].sum()
    total_row["Excess %"] = f"{round((total_row['Excess'] / total_row['Produced (+Add-on)']) * 100, 2) if total_row['Produced (+Add-on)'] > 0 else 0}%"
    
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    return df

# ================================================================
# ALGORITHM V3 - ORIGINAL PLATE RATIO SYSTEM
# ================================================================
def smart_layout_v3(demand, cap):
    total = sum(demand.values())
    if total == 0:
        return {}
    floor_vals = {}
    remainders = {}
    for k, v in demand.items():
        ratio = (v / total) * cap
        floor_vals[k] = floor(ratio)
        remainders[k] = ratio - floor_vals[k]
    layout = dict(floor_vals)
    for k in layout:
        if layout[k] == 0:
            layout[k] = 1
    while sum(layout.values()) > cap:
        biggest = max(layout, key=layout.get)
        if layout[biggest] > 1:
            layout[biggest] -= 1
        else:
            break
    remaining_cap = cap - sum(layout.values())
    while remaining_cap > 0:
        best = max(remainders, key=remainders.get)
        layout[best] += 1
        remainders[best] = 0
        remaining_cap -= 1
    return layout

def v3_optimizer(demand, cap, max_plates):
    remaining = demand.copy()
    plates = []
    for i in range(max_plates):
        if not any(v > 0 for v in remaining.values()):
            break
        layout = smart_layout_v3(remaining, cap)
        if not layout:
            break
        possible = [ceil(remaining[k] / v) for k, v in layout.items() if v > 0]
        sheets = max(1, min(possible))
        for k, v in layout.items():
            remaining[k] = max(0, remaining[k] - (v * sheets))
        plates.append({"name": plate_name(len(plates) + 1), "layout": layout, "sheets": sheets})
    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for k in remaining:
            if remaining[k] > 0:
                per_sheet = max(1, last["layout"].get(k, 1))
                add_sheets = ceil(remaining[k] / per_sheet)
                last["sheets"] += add_sheets
                remaining[k] = 0
    return plates

# ================================================================
# ALGORITHM V4 - COMMON SHEET OPTIMIZER
# ================================================================
def v4_optimizer(demand, capacity, max_plates):
    total_qty = sum(demand.values())
    target_sheets = ceil(total_qty / capacity)
    remaining = demand.copy()
    plates = []
    for p in range(max_plates):
        active = {k: v for k, v in remaining.items() if v > 0}
        if not active:
            break
        ideal = {tag: qty / target_sheets for tag, qty in active.items()}
        layout = {k: max(1, round(v)) for k, v in ideal.items()}
        while sum(layout.values()) > capacity:
            biggest = max(layout, key=layout.get)
            if layout[biggest] > 1:
                layout[biggest] -= 1
            else:
                break
        while sum(layout.values()) < capacity:
            biggest = max(active, key=active.get)
            layout[biggest] += 1
        possible_sheets = [ceil(remaining[tag] / layout[tag]) for tag in layout if layout[tag] > 0]
        sheets = max(1, min(possible_sheets))
        for tag, ups in layout.items():
            remaining[tag] = max(0, remaining[tag] - (ups * sheets))
        plates.append({"name": plate_name(len(plates) + 1), "layout": layout, "sheets": sheets})
    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for tag in remaining:
            if remaining[tag] > 0:
                ups = max(1, last["layout"].get(tag, 1))
                add_sheets = ceil(remaining[tag] / ups)
                last["sheets"] += add_sheets
                remaining[tag] = 0
    return plates

# ================================================================
# ALGORITHM V5 - SMART DECIMAL BALANCING
# ================================================================
def build_balanced_layout_v5(remaining, capacity):
    active = {k: v for k, v in remaining.items() if v > 0}
    if not active:
        return {}
    total_qty = sum(active.values())
    layout = {}
    decimals = {}
    for tag, qty in active.items():
        ideal = (qty / total_qty) * capacity
        base = int(ideal)
        if base < 1:
            base = 1
        layout[tag] = base
        decimals[tag] = ideal - int(ideal)
    while sum(layout.values()) > capacity:
        biggest = max(layout, key=layout.get)
        if layout[biggest] > 1:
            layout[biggest] -= 1
        else:
            break
    while sum(layout.values()) < capacity:
        best = max(decimals, key=decimals.get)
        layout[best] += 1
        decimals[best] = 0
    return layout

def v5_optimizer(demand, capacity, max_plates):
    remaining = demand.copy()
    plates = []
    for i in range(max_plates):
        active = {k: v for k, v in remaining.items() if v > 0}
        if not active:
            break
        layout = build_balanced_layout_v5(active, capacity)
        candidate_sheets = [ceil(remaining[tag] / layout[tag]) for tag in layout if layout[tag] > 0]
        sheets = max(1, min(candidate_sheets))
        for tag, ups in layout.items():
            remaining[tag] = max(0, remaining[tag] - (ups * sheets))
        plates.append({"name": plate_name(len(plates) + 1), "layout": layout, "sheets": sheets})
    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for tag in remaining:
            if remaining[tag] > 0:
                ups = max(1, last["layout"].get(tag, 1))
                extra_sheets = ceil(remaining[tag] / ups)
                last["sheets"] += extra_sheets
                remaining[tag] = 0
    return plates

# ================================================================
# ALGORITHM V6 - MULTI-VARIATION OPTIMIZER
# ================================================================
def proportional_layout_v6(remaining, capacity):
    active = {k: v for k, v in remaining.items() if v > 0}
    if not active:
        return {}
    total_qty = sum(active.values())
    layout = {}
    decimal_map = {}
    for tag, qty in active.items():
        ideal = (qty / total_qty) * capacity
        base = int(ideal)
        if base < 1:
            base = 1
        layout[tag] = base
        decimal_map[tag] = ideal - int(ideal)
    while sum(layout.values()) > capacity:
        biggest = max(layout, key=layout.get)
        if layout[biggest] > 1:
            layout[biggest] -= 1
        else:
            break
    while sum(layout.values()) < capacity:
        best = max(decimal_map, key=decimal_map.get)
        layout[best] += 1
        decimal_map[best] = 0
    return layout

def v6_optimizer(demand, capacity, max_plates):
    best_score = 999999
    best_plates = None
    for variation in range(15):
        remaining = copy.deepcopy(demand)
        plates = []
        for p in range(max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            layout = proportional_layout_v6(active, capacity)
            possible = [ceil(remaining[tag] / layout[tag]) for tag in layout if layout[tag] > 0]
            if not possible:
                break
            possible = sorted(possible)
            strategy_index = min(variation % len(possible), len(possible) - 1)
            sheets = max(1, possible[strategy_index])
            for tag, ups in layout.items():
                remaining[tag] = max(0, remaining[tag] - (ups * sheets))
            plates.append({"name": plate_name(len(plates) + 1), "layout": layout, "sheets": sheets})
        if any(v > 0 for v in remaining.values()) and plates:
            last = plates[-1]
            for tag in remaining:
                if remaining[tag] > 0:
                    ups = max(1, last["layout"].get(tag, 1))
                    add_sheets = ceil(remaining[tag] / ups)
                    last["sheets"] += add_sheets
                    remaining[tag] = 0
        waste_percent = calculate_waste_percent(plates, demand)
        if waste_percent < best_score:
            best_score = waste_percent
            best_plates = plates
    return best_plates

# ================================================================
# ALGORITHM V7 - AI MUTATION ENGINE
# ================================================================
def generate_layout_v7(active, capacity):
    total_qty = sum(active.values())
    layout = {}
    decimal_map = {}
    for tag, qty in active.items():
        ideal = (qty / total_qty) * capacity
        base = floor(ideal)
        if base < 1:
            base = 1
        layout[tag] = base
        decimal_map[tag] = ideal - floor(ideal)
    random_tags = list(active.keys())
    random.shuffle(random_tags)
    while sum(layout.values()) > capacity:
        biggest = max(layout, key=layout.get)
        if layout[biggest] > 1:
            layout[biggest] -= 1
        else:
            break
    while sum(layout.values()) < capacity:
        best = max(decimal_map, key=decimal_map.get)
        layout[best] += 1
        decimal_map[best] = 0
    if len(layout) >= 2:
        for _ in range(2):
            a = random.choice(random_tags)
            b = random.choice(random_tags)
            if a != b and layout[a] > 1:
                layout[a] -= 1
                layout[b] += 1
                if sum(layout.values()) > capacity:
                    layout[b] -= 1
                    layout[a] += 1
    return layout

def v7_optimizer(demand, capacity, max_plates, iterations=150):
    best_score = 999999
    best_plates = None
    for attempt in range(iterations):
        remaining = copy.deepcopy(demand)
        plates = []
        for p in range(max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            layout = generate_layout_v7(active, capacity)
            options = [ceil(remaining[tag] / layout[tag]) for tag in layout if layout[tag] > 0]
            if not options:
                break
            options = sorted(list(set(options)))
            sheets = max(1, random.choice(options))
            for tag, ups in layout.items():
                remaining[tag] = max(0, remaining[tag] - (ups * sheets))
            plates.append({"name": plate_name(len(plates) + 1), "layout": layout, "sheets": sheets})
        if any(v > 0 for v in remaining.values()) and plates:
            last = plates[-1]
            for tag in remaining:
                if remaining[tag] > 0:
                    ups = max(1, last["layout"].get(tag, 1))
                    extra = ceil(remaining[tag] / ups)
                    last["sheets"] += extra
                    remaining[tag] = 0
        waste_percent = calculate_waste_percent(plates, demand)
        if waste_percent < best_score:
            best_score = waste_percent
            best_plates = copy.deepcopy(plates)
    return best_plates

# ================================================================
# ALGORITHM V8 - INTEGER LINEAR SOLVER (PuLP)
# ================================================================
def v8_optimizer(demand, capacity, max_plates):
    if not PULP_AVAILABLE:
        return None
    
    remaining = demand.copy()
    plates = []
    
    for plate_num in range(max_plates):
        active_tags = [t for t in demand.keys() if remaining[t] > 0]
        if not active_tags:
            break
        
        try:
            model = LpProblem(f"Plate_{plate_num}", LpMinimize)
            
            ups = {t: LpVariable(f"UPS_{t}", lowBound=1, cat="Integer") for t in active_tags}
            sheets = LpVariable("Sheets", lowBound=1, cat="Integer")
            
            # Objective: minimize excess
            excess_vars = []
            for t in active_tags:
                excess = ups[t] * sheets - remaining[t]
                excess_vars.append(excess)
            
            model += lpSum(excess_vars)
            
            # Constraints
            model += lpSum(ups[t] for t in active_tags) == capacity
            
            for t in active_tags:
                model += ups[t] * sheets >= remaining[t]
            
            model.solve()
            
            if model.status == 1:  # Optimal found
                layout = {t: int(value(ups[t])) for t in active_tags}
                sheet_count = int(value(sheets))
                
                plates.append({
                    "name": plate_name(plate_num + 1),
                    "layout": layout,
                    "sheets": sheet_count
                })
                
                for t in active_tags:
                    remaining[t] -= layout[t] * sheet_count
                    remaining[t] = max(0, remaining[t])
            else:
                # Fallback to V5
                return v5_optimizer(demand, capacity, max_plates)
                
        except Exception:
            return v5_optimizer(demand, capacity, max_plates)
    
    return plates if plates else v5_optimizer(demand, capacity, max_plates)

# ================================================================
# ALGORITHM V9 - SIMULATED ANNEALING
# ================================================================
def v9_optimizer(demand, capacity, max_plates, iterations=300):
    
    def calculate_waste(layout, sheets, remaining):
        waste = 0
        for tag, ups in layout.items():
            produced = ups * sheets
            waste += max(0, produced - remaining.get(tag, 0))
        return waste
    
    def mutate_layout(layout, capacity):
        new_layout = layout.copy()
        tags = list(new_layout.keys())
        if len(tags) >= 2:
            a, b = random.sample(tags, 2)
            if new_layout[a] > 1:
                new_layout[a] -= 1
                new_layout[b] += 1
        return new_layout
    
    def initial_layout(active, capacity):
        total = sum(active.values())
        layout = {}
        for tag, qty in active.items():
            ups = max(1, int((qty / total) * capacity))
            layout[tag] = ups
        while sum(layout.values()) > capacity:
            max_tag = max(layout, key=layout.get)
            if layout[max_tag] > 1:
                layout[max_tag] -= 1
            else:
                break
        return layout
    
    remaining = demand.copy()
    plates = []
    
    for plate_num in range(max_plates):
        active = {k: v for k, v in remaining.items() if v > 0}
        if not active:
            break
        
        current = initial_layout(active, capacity)
        sheets = max(1, min(ceil(active[t] / current[t]) for t in current))
        current_score = calculate_waste(current, sheets, active)
        
        best = current.copy()
        best_score = current_score
        temperature = 100.0
        
        for i in range(iterations):
            candidate = mutate_layout(current, capacity)
            candidate_score = calculate_waste(candidate, sheets, active)
            
            delta = candidate_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current = candidate
                current_score = candidate_score
                
                if current_score < best_score:
                    best = current.copy()
                    best_score = current_score
            
            temperature *= 0.995
        
        plates.append({
            "name": plate_name(plate_num + 1),
            "layout": best,
            "sheets": sheets
        })
        
        for tag, ups in best.items():
            remaining[tag] -= ups * sheets
            remaining[tag] = max(0, remaining[tag])
    
    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for tag in remaining:
            if remaining[tag] > 0:
                ups = max(1, last["layout"].get(tag, 1))
                add_sheets = ceil(remaining[tag] / ups)
                last["sheets"] += add_sheets
                remaining[tag] = 0
    
    return plates

# ================================================================
# ALGORITHM V10 - MCTS TREE SEARCH
# ================================================================
class MCTSNode:
    def __init__(self, layout, remaining, capacity, parent=None):
        self.layout = layout
        self.remaining = remaining.copy()
        self.capacity = capacity
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0
    
    def get_possible_moves(self):
        moves = []
        tags = list(self.layout.keys())
        for i, a in enumerate(tags):
            for b in tags[i+1:]:
                if self.layout[a] > 1:
                    moves.append((a, b))
                if self.layout[b] > 1:
                    moves.append((b, a))
        return moves
    
    def best_child(self, c_param=1.4):
        choices = []
        for child in self.children:
            if child.visits == 0:
                ucb = float('inf')
            else:
                ucb = (child.score / child.visits) + c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            choices.append((ucb, child))
        return max(choices, key=lambda x: x[0])[1]

def v10_optimizer(demand, capacity, max_plates, iterations=150):
    
    def initial_layout(active, capacity):
        total = sum(active.values())
        layout = {}
        for tag, qty in active.items():
            ups = max(1, int((qty / total) * capacity))
            layout[tag] = ups
        while sum(layout.values()) > capacity:
            max_tag = max(layout, key=layout.get)
            if layout[max_tag] > 1:
                layout[max_tag] -= 1
            else:
                break
        return layout
    
    remaining = demand.copy()
    plates = []
    
    for plate_num in range(max_plates):
        active = {k: v for k, v in remaining.items() if v > 0}
        if not active:
            break
        
        root_layout = initial_layout(active, capacity)
        sheets = max(1, min(ceil(active[t] / root_layout[t]) for t in root_layout))
        
        root = MCTSNode(root_layout, active, capacity)
        
        for _ in range(iterations):
            node = root
            
            # Selection
            while node.children and len(node.children) >= len(node.get_possible_moves()):
                node = node.best_child()
            
            # Expansion
            if node.children:
                possible_moves = node.get_possible_moves()
                existing_moves = [(c.layout, c.remaining) for c in node.children]
                for move in possible_moves:
                    new_layout = node.layout.copy()
                    a, b = move
                    new_layout[a] -= 1
                    new_layout[b] += 1
                    if (new_layout, node.remaining) not in existing_moves:
                        child = MCTSNode(new_layout, node.remaining, capacity, node)
                        node.children.append(child)
                        node = child
                        break
            
            # Simulation (rollout)
            waste = 0
            for tag, ups in node.layout.items():
                produced = ups * sheets
                waste += max(0, produced - node.remaining.get(tag, 0))
            score = -waste  # Negative because we want to maximize
            
            # Backpropagation
            while node:
                node.visits += 1
                node.score += score
                node = node.parent
        
        # Select best child
        if root.children:
            best_child = max(root.children, key=lambda c: c.score / c.visits if c.visits > 0 else 0)
            best_layout = best_child.layout
        else:
            best_layout = root_layout
        
        plates.append({
            "name": plate_name(plate_num + 1),
            "layout": best_layout,
            "sheets": sheets
        })
        
        for tag, ups in best_layout.items():
            remaining[tag] -= ups * sheets
            remaining[tag] = max(0, remaining[tag])
    
    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for tag in remaining:
            if remaining[tag] > 0:
                ups = max(1, last["layout"].get(tag, 1))
                add_sheets = ceil(remaining[tag] / ups)
                last["sheets"] += add_sheets
                remaining[tag] = 0
    
    return plates

# ================================================================
# UI - MAIN APP
# ================================================================
st.markdown("""
<div class="main-header">
    <h1>🔬 10-in-1 Plate Ratio Comparator</h1>
    <p>V3 • V4 • V5 • V6 • V7 • V8 • V9 • V10 — Compare All | Pick the Best</p>
</div>
""", unsafe_allow_html=True)

# Configuration Panel
st.markdown('<div class="card"><div class="card-title">⚙️ Production Configuration</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    n = st.number_input("🏷️ Number of Items", 1, 500, 1)
with col2:
    cap = st.number_input("📀 Plate Capacity", 1, 200, 10)
with col3:
    maxp = st.number_input("🎨 Max Plates", 1, 30, 3)
with col4:
    addon = st.number_input("📈 Add-on %", 0.0, 50.0, 0.0, step=0.5)
st.markdown('</div>', unsafe_allow_html=True)

# Tag Quantity Section
st.markdown('<div class="card"><div class="card-title">📦 Item Quantity Details</div>', unsafe_allow_html=True)

tags = []
qty = []

for i in range(n):
    col1, col2 = st.columns([1, 2])
    with col1:
        item_name = f"Item {i+1}"
        st.markdown(f"<div class='tag-display'>{item_name}</div>", unsafe_allow_html=True)
    with col2:
        q = st.number_input(f"Quantity for {item_name}", 0, 100000, step=10, key=f"qty_{i}", label_visibility="collapsed")
    tags.append(item_name)
    qty.append(q)

st.markdown('</div>', unsafe_allow_html=True)

# Data Preparation
original_qty = {t: int(q) for t, q in zip(tags, qty) if q > 0}
demand = {t: ceil(int(q) * (1 + addon / 100)) for t, q in zip(tags, qty) if q > 0}

# Show PuLP warning if needed
if not PULP_AVAILABLE:
    st.markdown('<div class="warning">⚠️ PuLP library not installed. V8 (Integer Solver) will use V5 fallback. Run: pip install pulp</div>', unsafe_allow_html=True)

# Generate Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_clicked = st.button("🚀 COMPARE ALL 10 ALGORITHMS", use_container_width=True)

if generate_clicked:
    if not demand:
        st.error("⚠️ Please enter at least one item with quantity greater than 0")
        st.stop()
    
    with st.spinner("🔄 Running 10 algorithms simultaneously..."):
        results = {}
        
        # V3
        plates_v3 = v3_optimizer(demand, cap, maxp)
        results["V3 - Plate Ratio System"] = plates_v3
        
        # V4
        plates_v4 = v4_optimizer(demand, cap, maxp)
        results["V4 - Common Sheet Optimizer"] = plates_v4
        
        # V5
        plates_v5 = v5_optimizer(demand, cap, maxp)
        results["V5 - Smart Decimal Balancing"] = plates_v5
        
        # V6
        plates_v6 = v6_optimizer(demand, cap, maxp)
        results["V6 - Multi-Variation Optimizer"] = plates_v6
        
        # V7
        plates_v7 = v7_optimizer(demand, cap, maxp, iterations=100)
        results["V7 - AI Mutation Engine"] = plates_v7
        
        # V8
        plates_v8 = v8_optimizer(demand, cap, maxp)
        results["V8 - Integer Solver"] = plates_v8 if plates_v8 else v5_optimizer(demand, cap, maxp)
        
        # V9
        plates_v9 = v9_optimizer(demand, cap, maxp, iterations=200)
        results["V9 - Simulated Annealing"] = plates_v9
        
        # V10
        plates_v10 = v10_optimizer(demand, cap, maxp, iterations=100)
        results["V10 - MCTS Tree Search"] = plates_v10
        
        # Calculate waste percentages
        comparison_data = []
        for algo_name, plates in results.items():
            waste = calculate_waste_percent(plates, demand)
            total_plates = len(plates)
            total_sheets = sum(p["sheets"] for p in plates)
            comparison_data.append({
                "Algorithm": algo_name,
                "Waste %": waste,
                "Total Plates": total_plates,
                "Total Sheets": total_sheets
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("Waste %")
        
        best_algo = comparison_df.iloc[0]["Algorithm"]
        best_waste = comparison_df.iloc[0]["Waste %"]
        
        # Store in session state
        for algo_name, plates in results.items():
            st.session_state[f'plates_{algo_name.replace(" ", "_")}'] = plates
        st.session_state['demand'] = demand
        st.session_state['original_qty'] = original_qty
        st.session_state['comparison_df'] = comparison_df
        st.session_state['best_algo'] = best_algo
        st.session_state['best_waste'] = best_waste
        st.session_state['results'] = results
    
    # Show Best Algorithm
    st.markdown(f"""
    <div class="best-algo" style="margin-bottom: 2rem;">
        <div class="metric-value">🏆 BEST ALGORITHM: {best_algo}</div>
        <div class="metric-label">Waste Percentage: {best_waste}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show Comparison Table
    st.markdown("## 📊 Algorithm Comparison (Sorted by Waste %)")
    
    def highlight_best(row):
        if row["Algorithm"] == best_algo:
            return ['background-color: #2e7d32; color: white'] * len(row)
        return [''] * len(row)
    
    styled_df = comparison_df.style.apply(highlight_best, axis=1).format({"Waste %": "{:.2f}%"})
    st.dataframe(styled_df, use_container_width=True)
    
    # Selection for Export
    st.markdown("---")
    st.markdown("## 📥 Select Plan to Export")
    
    selected_algo = st.radio(
        "Choose which algorithm's detailed report to download:",
        options=comparison_df["Algorithm"].tolist(),
        index=0,
        horizontal=True
    )
    
    selected_plates = results[selected_algo]
    algo_name_clean = selected_algo.replace(" ", "_").replace("-", "_")
    
    full_df = build_full_summary(selected_plates, demand, original_qty)
    
    st.markdown(f"### 📋 Preview: {selected_algo}")
    st.dataframe(full_df, use_container_width=True)
    
    st.markdown("### 🧾 Plate Configuration Details")
    plate_rows = []
    for idx, p in enumerate(selected_plates, 1):
        plate_rows.append({
            "SL": idx,
            "Plate ID": p["name"],
            "Sheets Required": p["sheets"],
            "Total UPS": sum(p["layout"].values()),
            "Layout": ", ".join([f"{k}:{v}" for k, v in p["layout"].items()])
        })
    plate_details_df = pd.DataFrame(plate_rows)
    st.dataframe(plate_details_df, use_container_width=True)
    
    # Export Excel
    st.markdown("### 📥 Download Report")
    bio_excel = BytesIO()
    with pd.ExcelWriter(bio_excel, engine="openpyxl") as writer:
        comparison_df.to_excel(writer, sheet_name="Algorithm_Comparison", index=False)
        full_df.to_excel(writer, sheet_name=f"{algo_name_clean}_Summary", index=False)
        plate_details_df.to_excel(writer, sheet_name="Plate_Details", index=False)
    bio_excel.seek(0)
    
    st.download_button(
        "📊 Download Excel Report (Full Comparison + Selected Plan)",
        data=bio_excel,
        file_name=f"plate_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🔬 10-in-1 Plate Ratio Comparator — V3 • V4 • V5 • V6 • V7 • V8 • V9 • V10</p>
    <p style="color: #667eea;">✨ Design & Developed by <strong>Ovi</strong> ✨</p>
    <p style="font-size:0.8rem;">V8 requires PuLP: pip install pulp | All 10 algorithms run simultaneously</p>
</div>
""", unsafe_allow_html=True)
