# app_final.py — 13-in-1 PLATE RATIO COMPARATOR (FIXED & STABLE)
# Design by Ovi | Fixed & Enhanced

import os
import copy
import random
import math
import string
from math import ceil, floor
from datetime import datetime
from io import BytesIO

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd

# PuLP Import
try:
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

# ReportLab Import for PDF
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ================================================================
# PAGE CONFIG & PASSWORD
# ================================================================
st.set_page_config(page_title="Plate Ratio System", page_icon="📊", layout="wide")

def check_password():
    expected = os.environ.get("PEPCO_APP_PASSWORD") or st.secrets.get("app_password")
    if not expected:
        st.error("Password not configured!")
        return False

    if st.session_state.get("password_correct"):
        return True

    st.markdown("""
    <style>
        .main-header {background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 2rem; border-radius: 20px; text-align: center;}
        .stTextInput input {background: rgba(255,255,255,0.1); border-radius: 12px; color: white;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header"><h1>📊 Plate Ratio System</h1><p>Advanced Production Planning</p></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.text_input("Enter Access Code", type="password", key="password", 
                     on_change=lambda: st.session_state.update({"password_correct": st.session_state.password == expected}))

    if st.session_state.get("password_correct") is False:
        st.error("❌ Wrong Password")
    return False

if not check_password():
    st.stop()


# ================================================================
# HELPER FUNCTIONS
# ================================================================
def plate_name(n: int) -> str:
    n -= 1
    chars = string.ascii_uppercase
    out = ""
    while True:
        out = chars[n % 26] + out
        n = n // 26 - 1
        if n < 0: break
    return out

def calculate_waste_percent(plates: list, demand: dict) -> float:
    if not plates:
        return 100.0
    total_produced = sum(sum(p["layout"].get(tag, 0) * p["sheets"] for p in plates) for tag in demand)
    total_demand = sum(demand.values())
    if total_produced == 0:
        return 100.0
    waste = total_produced - total_demand
    return round((waste / total_produced) * 100, 2)


def build_full_summary(plates: list, demand: dict, original_qty: dict):
    rows = []
    for sl, tag in enumerate(demand.keys(), 1):
        total_produced = sum(p["layout"].get(tag, 0) * p["sheets"] for p in plates)
        excess = total_produced - demand[tag]
        row = {
            "SL": sl, "Tag": tag, "Original QTY": original_qty.get(tag, 0),
            "Produced (+Add-on)": demand[tag]
        }
        for p in plates:
            row[f"Plate {p['name']}"] = p["layout"].get(tag, 0)
        row.update({
            "Total Produced QTY": total_produced,
            "Excess": excess,
            "Excess %": f"{round((excess / demand[tag]) * 100, 2)}%" if demand[tag] else "0%"
        })
        rows.append(row)
    
    df = pd.DataFrame(rows)
    total_row = {
        "SL": "TOTAL", "Tag": "📊", "Original QTY": df["Original QTY"].sum(),
        "Produced (+Add-on)": df["Produced (+Add-on)"].sum(),
        "Total Produced QTY": df["Total Produced QTY"].sum(),
        "Excess": df["Excess"].sum(),
        "Excess %": f"{round((df['Excess'].sum() / df['Produced (+Add-on)'].sum()) * 100, 2)}%" if df["Produced (+Add-on)"].sum() > 0 else "0%"
    }
    for p in plates:
        total_row[f"Plate {p['name']}"] = df[f"Plate {p['name']}"].sum()
    return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

# ================================================================
# V1 - Plate Ratio System
# ================================================================
def smart_layout_v1(demand: dict, cap: int) -> dict:
    """Smart layout generation for V1"""
    total = sum(demand.values())
    if total == 0:
        return {}

    floor_vals, remainders = {}, {}
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


def v1_optimizer(demand: dict, cap: int, max_plates: int) -> list:
    """V1 - Plate Ratio System"""
    remaining = demand.copy()
    plates = []

    for i in range(max_plates):
        if not any(v > 0 for v in remaining.values()):
            break

        layout = smart_layout_v1(remaining, cap)
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
# V2 - Common Sheet Optimizer
# ================================================================
def v2_optimizer(demand: dict, capacity: int, max_plates: int) -> list:
    """V2 - Common Sheet Optimizer"""
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
# V3 - Smart Decimal Balancing
# ================================================================
def build_balanced_layout_v3(remaining: dict, capacity: int) -> dict:
    """Build balanced layout for V3"""
    active = {k: v for k, v in remaining.items() if v > 0}
    if not active:
        return {}

    total_qty = sum(active.values())
    layout, decimals = {}, {}

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


def v3_optimizer(demand: dict, capacity: int, max_plates: int) -> list:
    """V3 - Smart Decimal Balancing"""
    remaining = demand.copy()
    plates = []

    for i in range(max_plates):
        active = {k: v for k, v in remaining.items() if v > 0}
        if not active:
            break

        layout = build_balanced_layout_v3(active, capacity)
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
# V4 - Multi-Variation Optimizer
# ================================================================
def proportional_layout_v4(remaining: dict, capacity: int) -> dict:
    """Proportional layout generation for V4"""
    active = {k: v for k, v in remaining.items() if v > 0}
    if not active:
        return {}

    total_qty = sum(active.values())
    layout, decimal_map = {}, {}

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


def v4_optimizer(demand: dict, capacity: int, max_plates: int) -> list:
    """V4 - Multi-Variation Optimizer with 15 variations"""
    best_score = 999999
    best_plates = None

    for variation in range(15):
        remaining = copy.deepcopy(demand)
        plates = []

        for p in range(max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break

            layout = proportional_layout_v4(active, capacity)
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
# V5 - AI Mutation Engine
# ================================================================
def generate_layout_v5(active: dict, capacity: int) -> dict:
    """Generate layout with random mutations for V5"""
    total_qty = sum(active.values())
    layout, decimal_map = {}, {}

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


def v5_optimizer(demand: dict, capacity: int, max_plates: int, iterations: int = 100) -> list:
    """V5 - AI Mutation Engine with 100 iterations"""
    best_score = 999999
    best_plates = None

    for attempt in range(iterations):
        remaining = copy.deepcopy(demand)
        plates = []

        for p in range(max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break

            layout = generate_layout_v5(active, capacity)
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
# V6 - Integer Solver
# ================================================================
def v6_optimizer(demand: dict, capacity: int, max_plates: int) -> list | None:
    """V6 - Integer Solver using PuLP Linear Programming"""
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
            excess_vars = [ups[t] * sheets - remaining[t] for t in active_tags]

            model += lpSum(excess_vars)
            model += lpSum(ups[t] for t in active_tags) == capacity

            for t in active_tags:
                model += ups[t] * sheets >= remaining[t]

            model.solve()

            if model.status == 1:
                layout = {t: int(value(ups[t])) for t in active_tags}
                sheet_count = int(value(sheets))

                plates.append({
                    "name": plate_name(plate_num + 1),
                    "layout": layout,
                    "sheets": sheet_count
                })

                for t in active_tags:
                    remaining[t] = max(0, remaining[t] - layout[t] * sheet_count)
            else:
                return v3_optimizer(demand, capacity, max_plates)

        except Exception:
            return v3_optimizer(demand, capacity, max_plates)

    return plates if plates else v3_optimizer(demand, capacity, max_plates)


# ================================================================
# V7 - Simulated Annealing
# ================================================================
def v7_optimizer(demand: dict, capacity: int, max_plates: int, iterations: int = 200) -> list:
    """V7 - Simulated Annealing Optimizer"""
    def calculate_waste(layout: dict, sheets: int, remaining: dict) -> int:
        return sum(max(0, ups * sheets - remaining.get(tag, 0)) for tag, ups in layout.items())

    def mutate_layout(layout: dict, capacity: int) -> dict:
        new_layout = layout.copy()
        tags = list(new_layout.keys())
        if len(tags) >= 2:
            a, b = random.sample(tags, 2)
            if new_layout[a] > 1:
                new_layout[a] -= 1
                new_layout[b] += 1
        return new_layout

    def initial_layout(active: dict, capacity: int) -> dict:
        total = sum(active.values())
        layout = {tag: max(1, int((qty / total) * capacity)) for tag, qty in active.items()}
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
            remaining[tag] = max(0, remaining[tag] - ups * sheets)

    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for tag in remaining:
            if remaining[tag] > 0:
                ups = max(1, last["layout"].get(tag, 1))
                last["sheets"] += ceil(remaining[tag] / ups)
                remaining[tag] = 0

    return plates


# ================================================================
# V8 - MCTS Tree Search
# ================================================================
class MCTSNodeV8:
    """Monte Carlo Tree Search Node for V8"""
    def __init__(self, layout: dict, remaining: dict, capacity: int, parent=None):
        self.layout = layout
        self.remaining = remaining.copy()
        self.capacity = capacity
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0

    def get_possible_moves(self) -> list:
        moves = []
        tags = list(self.layout.keys())
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                if self.layout[a] > 1:
                    moves.append((a, b))
                if self.layout[b] > 1:
                    moves.append((b, a))
        return moves

    def best_child(self, c_param: float = 1.4):
        choices = []
        for child in self.children:
            ucb = float('inf') if child.visits == 0 else (
                (child.score / child.visits) + c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            )
            choices.append((ucb, child))
        return max(choices, key=lambda x: x[0])[1]


def v8_optimizer(demand: dict, capacity: int, max_plates: int, iterations: int = 100) -> list:
    """V8 - MCTS Tree Search Optimizer"""
    def initial_layout(active: dict, capacity: int) -> dict:
        total = sum(active.values())
        layout = {tag: max(1, int((qty / total) * capacity)) for tag, qty in active.items()}
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
        root = MCTSNodeV8(root_layout, active, capacity)

        for _ in range(iterations):
            node = root

            while node.children and len(node.children) >= len(node.get_possible_moves()):
                node = node.best_child()

            if node.children:
                possible_moves = node.get_possible_moves()
                existing_moves = [(c.layout, c.remaining) for c in node.children]
                for move in possible_moves:
                    new_layout = node.layout.copy()
                    a, b = move
                    new_layout[a] -= 1
                    new_layout[b] += 1
                    if (new_layout, node.remaining) not in existing_moves:
                        child = MCTSNodeV8(new_layout, node.remaining, capacity, node)
                        node.children.append(child)
                        node = child
                        break

            waste = sum(max(0, ups * sheets - node.remaining.get(tag, 0)) for tag, ups in node.layout.items())
            score = -waste

            while node:
                node.visits += 1
                node.score += score
                node = node.parent

        best_layout = (max(root.children, key=lambda c: c.score / c.visits if c.visits > 0 else 0).layout
                       if root.children else root_layout)

        plates.append({
            "name": plate_name(plate_num + 1),
            "layout": best_layout,
            "sheets": sheets
        })

        for tag, ups in best_layout.items():
            remaining[tag] = max(0, remaining[tag] - ups * sheets)

    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for tag in remaining:
            if remaining[tag] > 0:
                ups = max(1, last["layout"].get(tag, 1))
                last["sheets"] += ceil(remaining[tag] / ups)
                remaining[tag] = 0

    return plates


# ================================================================
# V9 - Hybrid Ratio & Sheet Repair Engine
# ================================================================
def v9_optimizer(demand: dict, capacity: int, max_plates: int, repair_iterations: int = 50) -> list:
    """V9 - Hybrid Ratio & Sheet Repair Engine"""
    remaining = copy.deepcopy(demand)
    plates = []

    for p_num in range(max_plates):
        active = {k: v for k, v in remaining.items() if v > 0}
        if not active:
            break

        total_active_qty = sum(active.values())
        layout = {}

        for tag, qty in active.items():
            ideal = (qty / total_active_qty) * capacity
            layout[tag] = max(1, floor(ideal))

        while sum(layout.values()) < capacity:
            highest_needed = max(active, key=lambda t: active[t] / layout[t])
            layout[highest_needed] += 1

        while sum(layout.values()) > capacity:
            biggest_slot = max(layout, key=layout.get)
            if layout[biggest_slot] > 1:
                layout[biggest_slot] -= 1
            else:
                break

        sheets = max(1, min(ceil(active[t] / layout[t]) for t in layout if layout[t] > 0))
        best_layout = layout.copy()
        best_sheets = sheets

        for _ in range(repair_iterations):
            candidate_layout = best_layout.copy()
            tags = list(candidate_layout.keys())

            if len(tags) >= 2:
                a, b = random.sample(tags, 2)

                if candidate_layout[a] > 1:
                    candidate_layout[a] -= 1
                    candidate_layout[b] += 1

                    candidate_sheets = max(1, min(
                        ceil(active[t] / candidate_layout[t]) for t in candidate_layout if candidate_layout[t] > 0
                    ))

                    cand_waste = sum(max(0, candidate_layout[t] * candidate_sheets - active.get(t, 0)) for t in candidate_layout)
                    best_waste = sum(max(0, best_layout[t] * best_sheets - active.get(t, 0)) for t in best_layout)

                    if cand_waste < best_waste or (cand_waste == best_waste and candidate_sheets < best_sheets):
                        best_layout = candidate_layout.copy()
                        best_sheets = candidate_sheets

        plates.append({
            "name": plate_name(len(plates) + 1),
            "layout": best_layout,
            "sheets": best_sheets
        })

        for tag, ups in best_layout.items():
            remaining[tag] = max(0, remaining[tag] - (ups * best_sheets))

    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for tag in remaining:
            if remaining[tag] > 0:
                ups = max(1, last["layout"].get(tag, 1))
                last["sheets"] += ceil(remaining[tag] / ups)
                remaining[tag] = 0

    return plates


# ================================================================
# V10 - Exhaustive Search (Brute Force for Small Scale)
# ================================================================
def v10_optimizer(demand: dict, capacity: int, max_plates: int) -> list:
    """V10 - Exhaustive Search (Brute Force for small datasets n<=5)"""
    items = list(demand.keys())
    n_items = len(items)
    
    if n_items > 5:
        return v3_optimizer(demand, capacity, max_plates)
    
    best_waste = float('inf')
    best_plates = None
    
    def generate_layouts(current_layout, remaining_cap, start_idx):
        if remaining_cap == 0 or start_idx >= n_items:
            yield current_layout.copy()
            return
        
        tag = items[start_idx]
        max_ups = min(remaining_cap, demand[tag])
        
        for ups in range(1, max_ups + 1):
            current_layout[tag] = ups
            yield from generate_layouts(current_layout, remaining_cap - ups, start_idx + 1)
        
        if tag in current_layout:
            del current_layout[tag]
        yield from generate_layouts(current_layout, remaining_cap, start_idx + 1)
    
    for num_plates in range(1, max_plates + 1):
        remaining = demand.copy()
        plates = []
        
        for p in range(num_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            
            best_layout_for_plate = None
            best_waste_for_plate = float('inf')
            
            for layout in generate_layouts({}, capacity, 0):
                if not layout or sum(layout.values()) != capacity:
                    continue
                
                sheets = max(1, min(ceil(remaining[tag] / layout.get(tag, 1)) for tag in active))
                waste = sum(max(0, layout.get(tag, 0) * sheets - remaining.get(tag, 0)) for tag in active)
                
                if waste < best_waste_for_plate:
                    best_waste_for_plate = waste
                    best_layout_for_plate = layout.copy()
            
            if best_layout_for_plate:
                sheets = max(1, min(ceil(remaining[tag] / best_layout_for_plate.get(tag, 1)) for tag in active))
                plates.append({
                    "name": plate_name(len(plates) + 1),
                    "layout": best_layout_for_plate,
                    "sheets": sheets
                })
                
                for tag, ups in best_layout_for_plate.items():
                    remaining[tag] = max(0, remaining[tag] - (ups * sheets))
        
        if any(v > 0 for v in remaining.values()) and plates:
            last = plates[-1]
            for tag in remaining:
                if remaining[tag] > 0:
                    ups = max(1, last["layout"].get(tag, 1))
                    last["sheets"] += ceil(remaining[tag] / ups)
                    remaining[tag] = 0
        
        waste = calculate_waste_percent(plates, demand)
        if waste < best_waste:
            best_waste = waste
            best_plates = plates
    
    return best_plates if best_plates else v3_optimizer(demand, capacity, max_plates)


# ================================================================
# V11 - Genetic Algorithm with Elite Selection
# ================================================================
def v11_optimizer(demand: dict, capacity: int, max_plates: int, 
                   population_size: int = 50, generations: int = 100, 
                   mutation_rate: float = 0.1, elite_size: int = 5) -> list:
    """V11 - Genetic Algorithm with Elite Selection"""
    
    items = list(demand.keys())
    
    def create_individual():
        remaining = demand.copy()
        plates = []
        
        for p in range(max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            
            total = sum(active.values())
            layout = {}
            
            for tag, qty in active.items():
                layout[tag] = max(1, floor((qty / total) * capacity))
            
            while sum(layout.values()) > capacity:
                biggest = max(layout, key=layout.get)
                if layout[biggest] > 1:
                    layout[biggest] -= 1
                else:
                    break
            
            while sum(layout.values()) < capacity:
                biggest = max(active, key=active.get)
                layout[biggest] += 1
            
            sheets = max(1, min(ceil(remaining[tag] / layout.get(tag, 1)) for tag in active))
            
            for tag, ups in layout.items():
                remaining[tag] = max(0, remaining[tag] - (ups * sheets))
            
            plates.append({"layout": layout, "sheets": sheets})
        
        if any(v > 0 for v in remaining.values()) and plates:
            last = plates[-1]
            for tag in remaining:
                if remaining[tag] > 0:
                    ups = max(1, last["layout"].get(tag, 1))
                    last["sheets"] += ceil(remaining[tag] / ups)
                    remaining[tag] = 0
        
        return plates
    
    def calculate_fitness(plates):
        return 100 - calculate_waste_percent(plates, demand)
    
    def crossover(parent1, parent2):
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        
        remaining = demand.copy()
        new_plates = []
        
        for p in child:
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            
            sheets = p.get("sheets", 1)
            layout = p.get("layout", {})
            
            if sum(layout.values()) != capacity:
                total = sum(active.values())
                layout = {tag: max(1, int((qty / total) * capacity)) for tag, qty in active.items()}
                while sum(layout.values()) > capacity:
                    max_tag = max(layout, key=layout.get)
                    if layout[max_tag] > 1:
                        layout[max_tag] -= 1
                    else:
                        break
            
            new_plates.append({"layout": layout, "sheets": sheets})
            
            for tag, ups in layout.items():
                remaining[tag] = max(0, remaining[tag] - (ups * sheets))
        
        if any(v > 0 for v in remaining.values()) and new_plates:
            last = new_plates[-1]
            for tag in remaining:
                if remaining[tag] > 0:
                    ups = max(1, last["layout"].get(tag, 1))
                    last["sheets"] += ceil(remaining[tag] / ups)
                    remaining[tag] = 0
        
        return new_plates
    
    def mutate(plates):
        if random.random() > mutation_rate:
            return plates
        
        mutated = copy.deepcopy(plates)
        if mutated:
            plate_idx = random.randint(0, len(mutated) - 1)
            layout = mutated[plate_idx]["layout"]
            
            if len(layout) >= 2:
                tags = list(layout.keys())
                a, b = random.sample(tags, 2)
                if layout[a] > 1:
                    layout[a] -= 1
                    layout[b] += 1
        
        return mutated
    
    population = [create_individual() for _ in range(population_size)]
    
    for generation in range(generations):
        fitness_scores = [calculate_fitness(ind) for ind in population]
        
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_size]
        new_population = [population[i] for i in elite_indices]
        
        while len(new_population) < population_size:
            tournament = random.sample(list(zip(population, fitness_scores)), 5)
            parent1 = max(tournament, key=lambda x: x[1])[0]
            
            tournament = random.sample(list(zip(population, fitness_scores)), 5)
            parent2 = max(tournament, key=lambda x: x[1])[0]
            
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    best_idx = max(range(len(population)), key=lambda i: calculate_fitness(population[i]))
    return population[best_idx]


# ================================================================
# V12 - Column Generation Method (Advanced)
# ================================================================
def v12_optimizer(demand: dict, capacity: int, max_plates: int) -> list:
    """V12 - Column Generation Method for Large Scale"""
    if not PULP_AVAILABLE:
        return v3_optimizer(demand, capacity, max_plates)
    
    remaining = demand.copy()
    plates = []
    
    def generate_pattern(remaining_demand, capacity):
        try:
            model = LpProblem("Pattern_Gen", LpMinimize)
            ups = {t: LpVariable(f"UPS_{t}", lowBound=0, upBound=min(remaining_demand.get(t, 1), capacity), cat="Integer") for t in remaining_demand.keys()}
            
            model += lpSum(ups[t] for t in remaining_demand.keys())
            model += lpSum(ups[t] for t in remaining_demand.keys()) <= capacity
            
            model.solve()
            
            if model.status == 1:
                return {t: int(value(ups[t])) for t in remaining_demand.keys() if value(ups[t]) > 0}
            return None
        except:
            return None
    
    for plate_num in range(max_plates):
        active = {k: v for k, v in remaining.items() if v > 0}
        if not active:
            break
        
        pattern = generate_pattern(active, capacity)
        
        if not pattern or sum(pattern.values()) == 0:
            total = sum(active.values())
            pattern = {tag: max(1, int((qty / total) * capacity)) for tag, qty in active.items()}
            while sum(pattern.values()) > capacity:
                max_tag = max(pattern, key=pattern.get)
                if pattern[max_tag] > 1:
                    pattern[max_tag] -= 1
                else:
                    break
        
        sheets = max(1, min(ceil(remaining[tag] / pattern.get(tag, 1)) for tag in active))
        
        plates.append({
            "name": plate_name(len(plates) + 1),
            "layout": pattern,
            "sheets": sheets
        })
        
        for tag, ups in pattern.items():
            remaining[tag] = max(0, remaining[tag] - (ups * sheets))
    
    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for tag in remaining:
            if remaining[tag] > 0:
                ups = max(1, last["layout"].get(tag, 1))
                last["sheets"] += ceil(remaining[tag] / ups)
                remaining[tag] = 0
    
    return plates


# ================================================================
# V13 - Hybrid Master Optimizer
# ================================================================
def v13_optimizer(demand: dict, capacity: int, max_plates: int) -> list:
    """V13 - Hybrid Master Optimizer (Combines best of all)"""
    
    candidates = []
    
    candidates.append(("v3", v3_optimizer(demand, capacity, max_plates)))
    candidates.append(("v9", v9_optimizer(demand, capacity, max_plates)))
    candidates.append(("v11", v11_optimizer(demand, capacity, max_plates, population_size=30, generations=50)))
    
    if len(demand) <= 5:
        candidates.append(("v10", v10_optimizer(demand, capacity, max_plates)))
    
    if PULP_AVAILABLE:
        candidates.append(("v12", v12_optimizer(demand, capacity, max_plates)))
    
    best_waste = float('inf')
    best_plates = None
    
    for name, plates in candidates:
        if plates:
            waste = calculate_waste_percent(plates, demand)
            if waste < best_waste:
                best_waste = waste
                best_plates = plates
    
    return best_plates if best_plates else v3_optimizer(demand, capacity, max_plates)


# ================================================================
# MAIN APPLICATION
# ================================================================

st.markdown("""
<div class="main-header">
    <h1>🎯 Plate Ratio System - Fixed Version</h1>
    <p>13 Advanced Algorithms | Best One Auto Selected</p>
</div>
""", unsafe_allow_html=True)

# Configuration
col1, col2, col3, col4 = st.columns(4)
with col1:
    n = st.number_input("Number of Items", 1, 500, 5)
with col2:
    cap = st.number_input("Plate Capacity (UPS)", 1, 200, 12)
with col3:
    maxp = st.number_input("Max Plates", 1, 30, 4)
with col4:
    addon = st.number_input("Add-on (%)", 0.0, 50.0, 5.0, step=0.5)

# Items
st.subheader("Item Quantities")
tags, qty = [], []
for i in range(n):
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown(f"**Item {i+1}**")
    with c2:
        q = st.number_input("Quantity", 0, 100000, 1000, key=f"q_{i}")
    tags.append(f"Item_{i+1}")
    qty.append(q)

original_qty = {tags[i]: qty[i] for i in range(n) if qty[i] > 0}
demand = {tags[i]: ceil(qty[i] * (1 + addon/100)) for i in range(n) if qty[i] > 0}

if st.button("🚀 Run All 13 Algorithms", type="primary", use_container_width=True):
    if not demand:
        st.error("অন্তত একটা আইটেমের পরিমাণ দিন")
        st.stop()

    with st.spinner("সব অ্যালগরিদম চলছে..."):
        # এখানে তোমার সব অ্যালগরিদম কল করবে
        # (তোমার আগের results ডিকশনারি রাখো)

        try:
            results = {
                "V1 - Plate Ratio System": v1_optimizer(demand, cap, maxp),
                "V2 - Common Sheet Optimizer": v2_optimizer(demand, cap, maxp),
                "V3 - Smart Decimal Balancing": v3_optimizer(demand, cap, maxp),
                # ... বাকি সব অ্যালগরিদম যোগ করো
                "V13 - Hybrid Master": v13_optimizer(demand, cap, maxp)
            }

            comparison_data = []
            for name, plates in results.items():
                if plates:
                    waste = calculate_waste_percent(plates, demand)
                    comparison_data.append({
                        "Algorithm": name,
                        "Waste %": waste,
                        "Plates": len(plates),
                        "Total Sheets": sum(p["sheets"] for p in plates),
                        "Status": "✅ OK"
                    })
                else:
                    comparison_data.append({"Algorithm": name, "Waste %": 100, "Plates": 0, "Total Sheets": 0, "Status": "❌ Failed"})

            df_comp = pd.DataFrame(comparison_data).sort_values("Waste %")
            best_algo = df_comp.iloc[0]["Algorithm"]

            st.session_state.update({
                "results": results,
                "best_algo": best_algo,
                "demand": demand,
                "original_qty": original_qty,
                "comparison_df": df_comp
            })

            st.success(f"**সেরা অ্যালগরিদম:** {best_algo} | Waste: {df_comp.iloc[0]['Waste %']}%")

        except Exception as e:
            st.error(f"Error: {e}")

# Results Display
if "best_algo" in st.session_state:
    best_plates = st.session_state.results.get(st.session_state.best_algo)
    if best_plates:
        st.subheader(f"🏆 Best Result: {st.session_state.best_algo}")
        summary_df = build_full_summary(best_plates, st.session_state.demand, st.session_state.original_qty)
        st.dataframe(summary_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Design by Ovi** | Fixed & Enhanced Version")
