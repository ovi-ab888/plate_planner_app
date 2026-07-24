"""
Common helper functions used across all algorithms
"""

import string
import math
from typing import Dict, List, Any
import pandas as pd


def plate_name(n: int) -> str:
    """Convert number to Excel-style column name"""
    n -= 1
    chars = string.ascii_uppercase
    out = ""
    while True:
        out = chars[n % 26] + out
        n = n // 26 - 1
        if n < 0:
            break
    return out


def create_valid_layout(active: Dict[str, int], capacity: int, method: str = "balanced") -> Dict[str, int]:
    """Create a layout that respects capacity"""
    if not active:
        return {}
    
    total_qty = sum(active.values())
    n_items = len(active)
    
    if n_items > capacity:
        layout = {}
        
        if method == "balanced":
            for tag, qty in active.items():
                ups = max(1, int((qty / total_qty) * capacity))
                layout[tag] = ups
            
            while sum(layout.values()) > capacity:
                max_tag = max(layout, key=layout.get)
                if layout[max_tag] > 1:
                    layout[max_tag] -= 1
                else:
                    min_tag = min(active, key=lambda t: active[t])
                    if layout.get(min_tag, 0) > 0:
                        layout[min_tag] = 0
                    else:
                        break
            
            while sum(layout.values()) < capacity:
                max_tag = max(active, key=lambda t: active[t] / (layout.get(t, 1) + 1))
                layout[max_tag] = layout.get(max_tag, 0) + 1
            
            return layout
        
        elif method == "greedy":
            for tag in active.keys():
                layout[tag] = 1
            
            remaining_cap = capacity - sum(layout.values())
            if remaining_cap > 0:
                sorted_items = sorted(active.items(), key=lambda x: x[1], reverse=True)
                for tag, _ in sorted_items:
                    if remaining_cap <= 0:
                        break
                    layout[tag] = layout.get(tag, 1) + 1
                    remaining_cap -= 1
            
            return layout
        
        elif method == "proportional":
            for tag, qty in active.items():
                ups = int((qty / total_qty) * capacity)
                if ups < 1:
                    ups = 1 if len(active) <= capacity else 0
                layout[tag] = ups
            
            while sum(layout.values()) > capacity:
                max_tag = max(layout, key=layout.get)
                if layout[max_tag] > 1:
                    layout[max_tag] -= 1
                else:
                    break
            
            while sum(layout.values()) < capacity:
                max_tag = max(active, key=lambda t: active[t] / (layout.get(t, 1) + 1))
                layout[max_tag] = layout.get(max_tag, 0) + 1
            
            return layout
    
    layout = {}
    
    if method == "balanced":
        for tag, qty in active.items():
            ideal = (qty / total_qty) * capacity
            base = int(ideal)
            if base < 1:
                base = 1
            layout[tag] = base
        
        while sum(layout.values()) > capacity:
            max_tag = max(layout, key=layout.get)
            if layout[max_tag] > 1:
                layout[max_tag] -= 1
            else:
                break
        
        while sum(layout.values()) < capacity:
            fractional = {}
            for tag, qty in active.items():
                ideal = (qty / total_qty) * capacity
                fractional[tag] = ideal - int(ideal)
            best = max(fractional, key=fractional.get)
            layout[best] = layout.get(best, 0) + 1
    
    else:
        for tag, qty in active.items():
            ups = max(1, int((qty / total_qty) * capacity))
            layout[tag] = ups
        
        while sum(layout.values()) > capacity:
            max_tag = max(layout, key=layout.get)
            if layout[max_tag] > 1:
                layout[max_tag] -= 1
            else:
                break
        
        while sum(layout.values()) < capacity:
            max_tag = max(active, key=lambda t: active[t] / (layout.get(t, 1) + 1))
            layout[max_tag] = layout.get(max_tag, 0) + 1
    
    return layout


def ensure_demand_met(plates: List[Dict[str, Any]], demand: Dict[str, int]) -> List[Dict[str, Any]]:
    """Ensure all demand is met"""
    if not plates:
        return plates
    
    for tag in demand.keys():
        total_produced = 0
        for plate in plates:
            total_produced += plate["layout"].get(tag, 0) * plate["sheets"]
        
        if total_produced < demand.get(tag, 0):
            shortfall = demand.get(tag, 0) - total_produced
            if plates:
                last_plate = plates[-1]
                ups = last_plate["layout"].get(tag, 1)
                additional_sheets = math.ceil(shortfall / max(1, ups))
                last_plate["sheets"] += additional_sheets
    
    for idx, plate in enumerate(plates):
        if "name" not in plate:
            plate["name"] = plate_name(idx + 1)
    
    return plates


def calculate_waste_percent(plates: List[Dict[str, Any]], demand: Dict[str, int]) -> float:
    """Calculate waste percentage"""
    total_produced = 0
    total_demand = sum(demand.values())
    
    if total_demand == 0:
        return 0.0
    
    for tag in demand:
        produced_qty = 0
        for p in plates:
            if p and "layout" in p:
                ups = p["layout"].get(tag, 0)
                produced_qty += ups * p.get("sheets", 0)
        total_produced += produced_qty
    
    if total_produced == 0:
        return 100.0
    
    waste = total_produced - total_demand
    waste_percent = (waste / total_produced) * 100
    
    if waste_percent < 0:
        waste_percent = 0.0
    
    return round(waste_percent, 2)


def build_full_summary(plates: List[Dict[str, Any]], demand: Dict[str, int], 
                       original_qty: Dict[str, int]) -> pd.DataFrame:
    """Build complete summary DataFrame"""
    rows = []
    sl = 1
    
    for tag in demand.keys():
        row = {
            "SL": sl,
            "Tag": tag,
            "Original QTY": original_qty.get(tag, 0),
            "Produced (+Add-on)": demand[tag]
        }
        
        for idx, p in enumerate(plates):
            if p and "layout" in p and "name" in p:
                ups = p["layout"].get(tag, 0)
                row[f"Plate {p['name']}"] = ups
            else:
                row[f"Plate {idx+1}"] = 0
        
        total_produced = 0
        for p in plates:
            if p and "layout" in p:
                ups = p["layout"].get(tag, 0)
                sheets = p.get("sheets", 0)
                total_produced += ups * sheets
        
        excess = total_produced - demand[tag]
        excess_percent = round((excess / demand[tag]) * 100, 2) if demand[tag] else 0
        
        row["Total Produced QTY"] = total_produced
        row["Excess"] = max(0, excess)
        row["Excess %"] = f"{max(0, excess_percent)}%"
        rows.append(row)
        sl += 1
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    total_row = {
        "SL": "📊",
        "Tag": "TOTAL",
        "Original QTY": df["Original QTY"].sum(),
        "Produced (+Add-on)": df["Produced (+Add-on)"].sum(),
    }
    
    for idx, p in enumerate(plates):
        col_name = f"Plate {p['name']}" if "name" in p else f"Plate {idx+1}"
        if col_name in df.columns:
            total_row[col_name] = df[col_name].sum()
        else:
            total_row[col_name] = 0
    
    total_row["Total Produced QTY"] = df["Total Produced QTY"].sum()
    total_excess = df["Excess"].sum()
    total_row["Excess"] = total_excess
    
    total_produced_qty = total_row["Total Produced QTY"]
    total_excess_percent = round((total_excess / total_produced_qty) * 100, 2) if total_produced_qty > 0 else 0
    total_row["Excess %"] = f"{total_excess_percent}%"
    
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    return df
