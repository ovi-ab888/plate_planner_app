# app.py — 5-IN-1 PLATE RATIO COMPARATOR (UPDATED)
# Fixed Tag List + PDF + Excel Report
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
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

st.set_page_config(
    page_title="5-in-1 Plate Ratio Comparator | Ovi",
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
        <h1>📊 5-in-1 Plate Ratio Comparator</h1>
        <p>Compare 5 Algorithms | Pick Best Waste % | Export Selected Plan</p>
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
    div[data-testid="stTextInput"] input {
        background: #1a1a1a !important;
        color: white !important;
        border: 1px solid #333 !important;
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

def calculate_waste_percent(plates, demand, original_qty):
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
    """Build dataframe exactly like the required format"""
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
    
    # Add TOTAL row
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

def generate_pdf_report(plates, demand, original_qty, algo_name, waste_percent):
    """Generate PDF report in the same format"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, alignment=TA_CENTER, textColor=colors.HexColor('#667eea'))
    subtitle_style = ParagraphStyle('CustomSubtitle', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER, textColor=colors.grey)
    header_style = ParagraphStyle('Header', parent=styles['Normal'], fontSize=12, alignment=TA_LEFT, textColor=colors.white)
    
    story = []
    
    # Title
    story.append(Paragraph("📊 Plate Ratio System - Production Report", title_style))
    story.append(Paragraph(f"Algorithm: {algo_name} | Waste: {waste_percent}% | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    story.append(Spacer(1, 20))
    
    # Build Production Summary Table
    summary_data = [["SL", "Tag", "Original QTY", "Produced (+Add-on)"]]
    for p in plates:
        summary_data[0].append(f"Plate {p['name']}")
    summary_data[0].extend(["Total Produced QTY", "Excess", "Excess %"])
    
    sl = 1
    for tag in demand.keys():
        row = [str(sl), tag, str(original_qty[tag]), str(demand[tag])]
        total_produced = 0
        for p in plates:
            ups = p["layout"].get(tag, 0)
            row.append(str(ups))
            total_produced += ups * p["sheets"]
        excess = total_produced - demand[tag]
        excess_percent = f"{round((excess / demand[tag]) * 100, 2) if demand[tag] else 0}%"
        row.extend([str(total_produced), str(excess), excess_percent])
        summary_data.append(row)
        sl += 1
    
    # Total row
    total_row = ["📊", "TOTAL", str(sum(original_qty.values())), str(sum(demand.values()))]
    for p in plates:
        total_row.append(str(sum(df[f"Plate {p['name']}"].sum() for df in [pd.DataFrame([{f"Plate {p['name']}": 0}])])))
    total_produced_sum = sum(original_qty.values())  # Placeholder
    total_excess_sum = 0
    total_row.extend([str(total_produced_sum), str(total_excess_sum), "0%"])
    
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Plate Information Table
    plate_data = [["SL", "Plate ID", "Sheets Required", "Total UPS"]]
    for idx, p in enumerate(plates, 1):
        plate_data.append([str(idx), p["name"], str(p["sheets"]), str(sum(p["layout"].values()))])
    
    plate_table = Table(plate_data)
    plate_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    story.append(plate_table)
    story.append(Spacer(1, 20))
    
    # Footer
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.grey)
    story.append(Paragraph("This Report Generated by Ovi's Plate Ratio System", footer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ================================================================
# ALGORITHMS (V3 to V7) - Same as before
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
    produced = Counter()
    for i in range(max_plates):
        if not any(v > 0 for v in remaining.values()):
            break
        layout = smart_layout_v3(remaining, cap)
        if not layout:
            break
        possible = [ceil(remaining[k] / v) for k, v in layout.items() if v > 0]
        sheets = max(1, min(possible))
        for k, v in layout.items():
            produced_qty = v * sheets
            remaining[k] = max(0, remaining[k] - produced_qty)
            produced[k] += produced_qty
        plates.append({"name": plate_name(len(plates) + 1), "layout": layout, "sheets": sheets})
    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for k in remaining:
            if remaining[k] > 0:
                per_sheet = max(1, last["layout"].get(k, 1))
                add_sheets = ceil(remaining[k] / per_sheet)
                last["sheets"] += add_sheets
                produced[k] += add_sheets * per_sheet
                remaining[k] = 0
    return plates

def v4_optimizer(demand, capacity, max_plates):
    total_qty = sum(demand.values())
    target_sheets = ceil(total_qty / capacity)
    remaining = demand.copy()
    plates = []
    for p in range(max_plates):
        active = {k: v for k, v in remaining.items() if v > 0}
        if not active:
            break
        ideal = {}
        for tag, qty in active.items():
            ideal[tag] = qty / target_sheets
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
        produced = {}
        for tag, ups in layout.items():
            qty = ups * sheets
            produced[tag] = qty
            remaining[tag] = max(0, remaining[tag] - qty)
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
        produced = {}
        for tag, ups in layout.items():
            qty = ups * sheets
            produced[tag] = qty
            remaining[tag] = max(0, remaining[tag] - qty)
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
            produced = {}
            for tag, ups in layout.items():
                qty = ups * sheets
                produced[tag] = qty
                remaining[tag] = max(0, remaining[tag] - qty)
            plates.append({"name": plate_name(len(plates) + 1), "layout": layout, "sheets": sheets})
        if any(v > 0 for v in remaining.values()) and plates:
            last = plates[-1]
            for tag in remaining:
                if remaining[tag] > 0:
                    ups = max(1, last["layout"].get(tag, 1))
                    add_sheets = ceil(remaining[tag] / ups)
                    last["sheets"] += add_sheets
                    remaining[tag] = 0
        total_produced = 0
        total_demand = sum(demand.values())
        for tag in demand:
            produced_qty = 0
            for p in plates:
                produced_qty += p["layout"].get(tag, 0) * p["sheets"]
            total_produced += produced_qty
        waste = total_produced - total_demand
        waste_percent = (waste / total_produced) * 100 if total_produced > 0 else 999
        if waste_percent < best_score:
            best_score = waste_percent
            best_plates = plates
    return best_plates

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
            produced = {}
            for tag, ups in layout.items():
                qty = ups * sheets
                produced[tag] = qty
                remaining[tag] = max(0, remaining[tag] - qty)
            plates.append({"name": plate_name(len(plates) + 1), "layout": layout, "sheets": sheets})
        if any(v > 0 for v in remaining.values()) and plates:
            last = plates[-1]
            for tag in remaining:
                if remaining[tag] > 0:
                    ups = max(1, last["layout"].get(tag, 1))
                    extra = ceil(remaining[tag] / ups)
                    last["sheets"] += extra
                    remaining[tag] = 0
        total_produced = 0
        total_demand = sum(demand.values())
        for tag in demand:
            produced_qty = 0
            for p in plates:
                produced_qty += p["layout"].get(tag, 0) * p["sheets"]
            total_produced += produced_qty
        waste = total_produced - total_demand
        waste_percent = (waste / total_produced) * 100 if total_produced > 0 else 999
        if waste_percent < best_score:
            best_score = waste_percent
            best_plates = copy.deepcopy(plates)
    return best_plates

# ================================================================
# UI - MAIN APP
# ================================================================
st.markdown("""
<div class="main-header">
    <h1>🔬 5-in-1 Plate Ratio Comparator</h1>
    <p>V3 • V4 • V5 • V6 • V7 — Compare Waste % | Pick the Best</p>
</div>
""", unsafe_allow_html=True)

# Configuration Panel
st.markdown('<div class="card"><div class="card-title">⚙️ Production Configuration</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    n = st.number_input("🏷️ Number of Items", 1, 20, 3)
with col2:
    cap = st.number_input("📀 Plate Capacity", 1, 64, 10)
with col3:
    maxp = st.number_input("🎨 Max Plates", 1, 20, 3)
with col4:
    addon = st.number_input("📈 Add-on %", 0.0, 50.0, 0.0, step=0.5)
st.markdown('</div>', unsafe_allow_html=True)

# Tag Quantity Section - FIXED LIST (Non-editable names)
st.markdown('<div class="card"><div class="card-title">📦 Item Quantity Details</div>', unsafe_allow_html=True)

# Predefined item names (Non-editable)
default_items = ["Item A", "Item B", "Item C", "Item D", "Item E", "Item F", "Item G", "Item H", "Item I", "Item J",
                 "Item K", "Item L", "Item M", "Item N", "Item O", "Item P", "Item Q", "Item R", "Item S", "Item T"]

tags = []
qty = []

for i in range(n):
    col1, col2 = st.columns(2)
    with col1:
        # Non-editable tag name - using display only
        st.markdown(f"<div style='background:#2a2a2a; padding:10px; border-radius:8px; color:#667eea; font-weight:bold;'>{default_items[i]}</div>", unsafe_allow_html=True)
        tag_name = default_items[i]  # Fixed name
    with col2:
        q = st.number_input(f"Quantity for {tag_name}", 0, 100000, step=10, key=f"qty_{i}")
    tags.append(tag_name)
    qty.append(q)

st.markdown('</div>', unsafe_allow_html=True)

# Data Preparation
original_qty = {t: int(q) for t, q in zip(tags, qty) if q > 0}
demand = {t: ceil(int(q) * (1 + addon / 100)) for t, q in zip(tags, qty) if q > 0}

# Generate Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_clicked = st.button("🚀 COMPARE ALL 5 ALGORITHMS", use_container_width=True)

if generate_clicked:
    if not demand:
        st.error("⚠️ Please enter at least one item with quantity greater than 0")
        st.stop()
    
    with st.spinner("🔄 Running 5 algorithms simultaneously..."):
        plates_v3 = v3_optimizer(demand, cap, maxp)
        plates_v4 = v4_optimizer(demand, cap, maxp)
        plates_v5 = v5_optimizer(demand, cap, maxp)
        plates_v6 = v6_optimizer(demand, cap, maxp)
        plates_v7 = v7_optimizer(demand, cap, maxp)
        
        waste_v3 = calculate_waste_percent(plates_v3, demand, original_qty)
        waste_v4 = calculate_waste_percent(plates_v4, demand, original_qty)
        waste_v5 = calculate_waste_percent(plates_v5, demand, original_qty)
        waste_v6 = calculate_waste_percent(plates_v6, demand, original_qty)
        waste_v7 = calculate_waste_percent(plates_v7, demand, original_qty)
        
        comparison_data = {
            "Algorithm": ["V3 - Plate Ratio System", "V4 - Common Sheet Optimizer", "V5 - Smart Decimal Balancing", "V6 - Multi-Variation Optimizer", "V7 - AI Mutation Engine"],
            "Waste %": [waste_v3, waste_v4, waste_v5, waste_v6, waste_v7],
            "Total Plates": [len(plates_v3), len(plates_v4), len(plates_v5), len(plates_v6), len(plates_v7)],
            "Total Sheets": [sum(p["sheets"] for p in plates_v3), sum(p["sheets"] for p in plates_v4), sum(p["sheets"] for p in plates_v5), sum(p["sheets"] for p in plates_v6), sum(p["sheets"] for p in plates_v7)]
        }
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("Waste %")
        
        best_algo = comparison_df.iloc[0]["Algorithm"]
        best_waste = comparison_df.iloc[0]["Waste %"]
        
        st.session_state['plates_v3'] = plates_v3
        st.session_state['plates_v4'] = plates_v4
        st.session_state['plates_v5'] = plates_v5
        st.session_state['plates_v6'] = plates_v6
        st.session_state['plates_v7'] = plates_v7
        st.session_state['demand'] = demand
        st.session_state['original_qty'] = original_qty
        st.session_state['comparison_df'] = comparison_df
        st.session_state['best_algo'] = best_algo
        st.session_state['best_waste'] = best_waste
    
    st.markdown(f"""
    <div class="best-algo" style="margin-bottom: 2rem;">
        <div class="metric-value">🏆 BEST ALGORITHM: {best_algo}</div>
        <div class="metric-label">Waste Percentage: {best_waste}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📊 Algorithm Comparison")
    
    def highlight_best(row):
        if row["Algorithm"] == best_algo:
            return ['background-color: #2e7d32; color: white'] * len(row)
        return [''] * len(row)
    
    styled_df = comparison_df.style.apply(highlight_best, axis=1).format({"Waste %": "{:.2f}%"})
    st.dataframe(styled_df, use_container_width=True)
    
    st.markdown("---")
    st.markdown("## 📥 Select Plan to Export")
    
    selected_algo = st.radio(
        "Choose which algorithm's detailed report to download:",
        options=comparison_df["Algorithm"].tolist(),
        index=0,
        horizontal=True
    )
    
    if selected_algo == "V3 - Plate Ratio System":
        selected_plates = plates_v3
        algo_name = "V3_Plate_Ratio_System"
    elif selected_algo == "V4 - Common Sheet Optimizer":
        selected_plates = plates_v4
        algo_name = "V4_Common_Sheet_Optimizer"
    elif selected_algo == "V5 - Smart Decimal Balancing":
        selected_plates = plates_v5
        algo_name = "V5_Smart_Decimal_Balancing"
    elif selected_algo == "V6 - Multi-Variation Optimizer":
        selected_plates = plates_v6
        algo_name = "V6_Multi_Variation_Optimizer"
    else:
        selected_plates = plates_v7
        algo_name = "V7_AI_Mutation_Engine"
    
    selected_waste = calculate_waste_percent(selected_plates, demand, original_qty)
    
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
            "Total UPS": sum(p["layout"].values())
        })
    plate_details_df = pd.DataFrame(plate_rows)
    st.dataframe(plate_details_df, use_container_width=True)
    
    # Export Options - PDF and Excel
    st.markdown("### 📥 Download Report")
    col1, col2 = st.columns(2)
    
    with col1:
        # Excel Export
        bio_excel = BytesIO()
        with pd.ExcelWriter(bio_excel, engine="openpyxl") as writer:
            full_df.to_excel(writer, sheet_name="Production Summary", index=False)
            plate_details_df.to_excel(writer, sheet_name="Plate Details", index=False)
            comparison_df.to_excel(writer, sheet_name="Algorithm Comparison", index=False)
        bio_excel.seek(0)
        
        st.download_button(
            "📊 Download Excel Report",
            data=bio_excel,
            file_name=f"{algo_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col2:
        # PDF Export
        pdf_buffer = generate_pdf_report(selected_plates, demand, original_qty, selected_algo, selected_waste)
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_buffer,
            file_name=f"{algo_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🔬 5-in-1 Plate Ratio Comparator — V3 • V4 • V5 • V6 • V7</p>
    <p style="color: #667eea;">✨ Design & Developed by <strong>Ovi</strong> ✨</p>
</div>
""", unsafe_allow_html=True)
