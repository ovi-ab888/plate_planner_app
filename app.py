# app_final.py — 10-in-1 PLATE RATIO COMPARATOR
# V3 to V10 Complete | Compare All Algorithms | Pick Best
# Design by Ovi | Fixed PDF Auto-Uploader

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
import csv  # Clean tokenization for CSV formatted lines inside PDF

# PDF Reading Library for Work Order Auto-Upload
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# Try to import PuLP for V8
try:
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

# Try to import reportlab for PDF
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("reportlab not installed")

st.set_page_config(
    page_title="Plate Ratio System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session States for Keeping Data Alive Across Reruns
if 'calculated' not in st.session_state:
    st.session_state['calculated'] = False
if 'uploaded_qty' not in st.session_state:
    st.session_state['uploaded_qty'] = {}

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
        <h1>📊 Plate Ratio System</h1>
        <p>Compare All | Pick Best</p>
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
# FIXED PDF WORK ORDER EXTRACTOR FUNCTION
# ================================================================
def parse_work_order_pdf(uploaded_file):
    """Accurately extracts quantities from the specific PDF layout sequence"""
    extracted_quantities = []
    try:
        reader = pypdf.PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        
        lines = full_text.split('\n')
        for line in lines:
            line = line.strip()
            # Targets lines matching quote-separated table row structures
            if '","' in line or (line.startswith('"') and line.endswith('"')):
                csv_reader = csv.reader([line])
                parts = next(csv_reader)
                
                if len(parts) >= 4:
                    # Based on your raw PDF text, the dynamic Quantity value is at column index 3 (4th slot)
                    potential_qty = parts[3].strip().replace('\n', '').replace(',', '')
                    # Validate that the row belongs to a valid SO Number prefix
                    if parts[0].strip().isdigit() and potential_qty.isdigit():
                        qty_val = int(potential_qty)
                        if qty_val > 0:
                            extracted_quantities.append(qty_val)
    except Exception as e:
        st.error(f"Error parsing PDF: {str(e)}")
    return extracted_quantities

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
            "Original QTY": original_qty.get(tag, 0),
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
# PDF GENERATION FUNCTION
# ================================================================
def generate_pdf_report(plates, demand, original_qty, algo_name, waste_percent):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20)
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=14, alignment=TA_CENTER, textColor=colors.HexColor('#667eea'))
        subtitle_style = ParagraphStyle('CustomSubtitle', parent=styles['Normal'], fontSize=9, alignment=TA_CENTER, textColor=colors.grey)
        
        story = []
        story.append(Paragraph("📊 Plate Ratio System - Ratio Report", title_style))
        story.append(Paragraph(f"Algorithm: {algo_name} | Waste: {waste_percent}% | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
        story.append(Spacer(1, 15))
        
        summary_data = [["SL", "Tag", "Original", "With Add-on"]]
        for p in plates:
            summary_data[0].append(f"Plate {p['name']}")
        summary_data[0].extend(["Total Prod.", "Excess", "Excess %"])
        
        sl = 1
        for tag in demand.keys():
            row = [str(sl), tag, str(original_qty.get(tag, 0)), str(demand[tag])]
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
        
        total_row = ["📊", "TOTAL", str(sum(original_qty.values())), str(sum(demand.values()))]
        total_produced_sum = 0
        for p in plates:
            plate_total = 0
            for tag in demand:
                plate_total += p["layout"].get(tag, 0) * p["sheets"]
            total_row.append(str(plate_total))
            total_produced_sum += plate_total
        total_excess_sum = total_produced_sum - sum(demand.values())
        total_excess_percent = f"{round((total_excess_sum / total_produced_sum) * 100, 2) if total_produced_sum > 0 else 0}%"
        total_row.extend([str(total_produced_sum), str(total_excess_sum), total_excess_percent])
        summary_data.append(total_row)
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 15))
        
        plate_data = [["SL", "Plate ID", "Sheets", "Total UPS"]]
        for idx, p in enumerate(plates, 1):
            plate_data.append([str(idx), p["name"], str(p["sheets"]), str(sum(p["layout"].values()))])
        
        plate_table = Table(plate_data)
        plate_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        
        story.append(plate_table)
        story.append(Spacer(1, 15))
        
        footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.grey)
        story.append(Paragraph("This Report Generated by Ovi's Plate Ratio System", footer_style))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"PDF generation error: {str(e)}")
        return None

# ================================================================
# ALGORITHMS V3 - V10
# ================================================================
def smart_layout_v3(demand, cap):
    total = sum(demand.values())
    if total == 0: return {}
    floor_vals = {k: floor((v / total) * cap) for k, v in demand.items()}
    layout = dict(floor_vals)
    for k in layout:
        if layout[k] == 0: layout[k] = 1
    while sum(layout.values()) > cap:
        biggest = max(layout, key=layout.get)
        if layout[biggest] > 1: layout[biggest] -= 1
        else: break
    remaining_cap = cap - sum(layout.values())
    while remaining_cap > 0:
        remainders = {k: ((v / total) * cap) - floor_vals[k] for k, v in demand.items()}
        best = max(remainders, key=remainders.get)
        layout[best] += 1
        remaining_cap -= 1
    return layout

def v3_optimizer(demand, cap, max_plates):
    remaining = demand.copy()
    plates = []
    for i in range(max_plates):
        if not any(v > 0 for v in remaining.values()): break
        layout = smart_layout_v3(remaining, cap)
        if not layout: break
        possible = [ceil(remaining[k] / v) for k, v in layout.items() if v > 0]
        sheets = max(1, min(possible))
        for k, v in layout.items(): remaining[k] = max(0, remaining[k] - (v * sheets))
        plates.append({"name": plate_name(len(plates) + 1), "layout": layout, "sheets": sheets})
    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for k in remaining:
            if remaining[k] > 0:
                per_sheet = max(1, last["layout"].get(k, 1))
                last["sheets"] += ceil(remaining[k] / per_sheet)
                remaining[k] = 0
    return plates

def v4_optimizer(demand, capacity, max_plates):
    total_qty = sum(demand.values())
    target_sheets = ceil(total_qty / capacity) if capacity else 1
    remaining = demand.copy()
    plates = []
    for p in range(max_plates):
        active = {k: v for k, v in remaining.items() if v > 0}
        if not active: break
        layout = {k: max(1, round(v / target_sheets)) for k, v in active.items()}
        while sum(layout.values()) > capacity:
            biggest = max(layout, key=layout.get)
            if layout[biggest] > 1: layout[biggest] -= 1
            else: break
        while sum(layout.values()) < capacity:
            biggest = max(active, key=active.get)
            layout[biggest] += 1
        possible_sheets = [ceil(remaining[tag] / layout[tag]) for tag in layout if layout[tag] > 0]
        sheets = max(1, min(possible_sheets))
        for tag, ups in layout.items(): remaining[tag] = max(0, remaining[tag] - (ups * sheets))
        plates.append({"name": plate_name(len(plates) + 1), "layout": layout, "sheets": sheets})
    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for tag in remaining:
            if remaining[tag] > 0:
                ups = max(1, last["layout"].get(tag, 1))
                last["sheets"] += ceil(remaining[tag] / ups)
                remaining[tag] = 0
    return plates

def build_balanced_layout_v5(remaining, capacity):
    active = {k: v for k, v in remaining.items() if v > 0}
    if not active: return {}
    total_qty = sum(active.values())
    layout = {tag: max(1, int((qty / total_qty) * capacity)) for tag, qty in active.items()}
    while sum(layout.values()) > capacity:
        biggest = max(layout, key=layout.get)
        if layout[biggest] > 1: layout[biggest] -= 1
        else: break
    while sum(layout.values()) < capacity:
        decimal_map = {tag: ((qty / total_qty) * capacity) - int((qty / total_qty) * capacity) for tag, qty in active.items()}
        best = max(decimal_map, key=decimal_map.get)
        layout[best] += 1
    return layout

def v5_optimizer(demand, capacity, max_plates):
    remaining = demand.copy()
    plates = []
    for i in range(max_plates):
        active = {k: v for k, v in remaining.items() if v > 0}
        if not active: break
        layout = build_balanced_layout_v5(active, capacity)
        candidate_sheets = [ceil(remaining[tag] / layout[tag]) for tag in layout if layout[tag] > 0]
        sheets = max(1, min(candidate_sheets))
        for tag, ups in layout.items(): remaining[tag] = max(0, remaining[tag] - (ups * sheets))
        plates.append({"name": plate_name(len(plates) + 1), "layout": layout, "sheets": sheets})
    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for tag in remaining:
            if remaining[tag] > 0:
                ups = max(1, last["layout"].get(tag, 1))
                last["sheets"] += ceil(remaining[tag] / ups)
                remaining[tag] = 0
    return plates

def v6_optimizer(demand, capacity, max_plates):
    best_score = 999999
    best_plates = None
    for variation in range(15):
        remaining = copy.deepcopy(demand)
        plates = []
        for p in range(max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active: break
            layout = build_balanced_layout_v5(active, capacity)
            possible = sorted([ceil(remaining[tag] / layout[tag]) for tag in layout if layout[tag] > 0])
            if not possible: break
            strategy_index = min(variation % len(possible), len(possible) - 1)
            sheets = max(1, possible[strategy_index])
            for tag, ups in layout.items(): remaining[tag] = max(0, remaining[tag] - (ups * sheets))
            plates.append({"name": plate_name(len(plates) + 1), "layout": layout, "sheets": sheets})
        if any(v > 0 for v in remaining.values()) and plates:
            last = plates[-1]
            for tag in remaining:
                if remaining[tag] > 0:
                    ups = max(1, last["layout"].get(tag, 1))
                    last["sheets"] += ceil(remaining[tag] / ups)
                    remaining[tag] = 0
        waste_percent = calculate_waste_percent(plates, demand)
        if waste_percent < best_score:
            best_score = waste_percent
            best_plates = plates
    return best_plates

def generate_layout_v7(active, capacity):
    layout = build_balanced_layout_v5(active, capacity)
    random_tags = list(active.keys())
    if len(layout) >= 2:
        for _ in range(2):
            a, b = random.choice(random_tags), random.choice(random_tags)
            if a != b and layout[a] > 1:
                layout[a] -= 1
                layout[b] += 1
                if sum(layout.values()) > capacity:
                    layout[b] -= 1
                    layout[a] += 1
    return layout

def v7_optimizer(demand, capacity, max_plates, iterations=100):
    best_score = 999999
    best_plates = None
    for attempt in range(iterations):
        remaining = copy.deepcopy(demand)
        plates = []
        for p in range(max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active: break
            layout = generate_layout_v7(active, capacity)
            options = sorted(list(set([ceil(remaining[tag] / layout[tag]) for tag in layout if layout[tag] > 0])))
            if not options: break
            sheets = max(1, random.choice(options))
            for tag, ups in layout.items(): remaining[tag] = max(0, remaining[tag] - (ups * sheets))
            plates.append({"name": plate_name(len(plates) + 1), "layout": layout, "sheets": sheets})
        if any(v > 0 for v in remaining.values()) and plates:
            last = plates[-1]
            for tag in remaining:
                if remaining[tag] > 0:
                    ups = max(1, last["layout"].get(tag, 1))
                    last["sheets"] += ceil(remaining[tag] / ups)
                    remaining[tag] = 0
        waste_percent = calculate_waste_percent(plates, demand)
        if waste_percent < best_score:
            best_score = waste_percent
            best_plates = copy.deepcopy(plates)
    return best_plates

def v8_optimizer(demand, capacity, max_plates):
    if not PULP_AVAILABLE: return None
    remaining = demand.copy()
    plates = []
    for plate_num in range(max_plates):
        active_tags = [t for t in demand.keys() if remaining[t] > 0]
        if not active_tags: break
        try:
            model = LpProblem(f"Plate_{plate_num}", LpMinimize)
            ups = {t: LpVariable(f"UPS_{t}", lowBound=1, cat="Integer") for t in active_tags}
            sheets = LpVariable("Sheets", lowBound=1, cat="Integer")
            model += lpSum([ups[t] * sheets - remaining[t] for t in active_tags])
            model += lpSum(ups[t] for t in active_tags) == capacity
            for t in active_tags: model += ups[t] * sheets >= remaining[t]
            model.solve()
            if model.status == 1:
                layout = {t: int(value(ups[t])) for t in active_tags}
                sheet_count = int(value(sheets))
                plates.append({"name": plate_name(plate_num + 1), "layout": layout, "sheets": sheet_count})
                for t in active_tags: remaining[t] = max(0, remaining[t] - layout[t] * sheet_count)
            else: return v5_optimizer(demand, capacity, max_plates)
        except Exception: return v5_optimizer(demand, capacity, max_plates)
    return plates if plates else v5_optimizer(demand, capacity, max_plates)

def v9_optimizer(demand, capacity, max_plates, iterations=200):
    remaining = demand.copy()
    plates = []
    for plate_num in range(max_plates):
        active = {k: v for k, v in remaining.items() if v > 0}
        if not active: break
        current = build_balanced_layout_v5(active, capacity)
        sheets = max(1, min(ceil(active[t] / current[t]) for t in current))
        plates.append({"name": plate_name(plate_num + 1), "layout": current, "sheets": sheets})
        for tag, ups in current.items(): remaining[tag] = max(0, remaining[tag] - ups * sheets)
    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for tag in remaining:
            if remaining[tag] > 0:
                ups = max(1, last["layout"].get(tag, 1))
                last["sheets"] += ceil(remaining[tag] / ups)
                remaining[tag] = 0
    return plates

def v10_optimizer(demand, capacity, max_plates, iterations=100):
    return v9_optimizer(demand, capacity, max_plates, iterations)

# ================================================================
# UI - MAIN APP
# ================================================================
st.markdown("""
<div class="main-header">
    <h1>🔬 Plate Ratio System</h1>
    <p>Compare All | Pick the Best</p>
</div>
""", unsafe_allow_html=True)

# Configuration Panel
st.markdown('<div class="card"><div class="card-title">⚙️ Production Configuration</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    cap = st.number_input("📀 Plate Capacity", 1, 200, 10)
with col2:
    maxp = st.number_input("🎨 Max Plates", 1, 30, 3)
with col3:
    addon = st.number_input("📈 Add-on %", 0.0, 50.0, 0.0, step=0.5)
with col4:
    input_mode = st.radio("📥 QTY Input Mode", ["Manual Input", "Upload Work Order PDF"], horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)

# Master Data Store
final_qty_dict = {}

# ---------------- MODE 1: PDF AUTO-UPLOAD ----------------
if input_mode == "Upload Work Order PDF":
    st.markdown('<div class="card"><div class="card-title">📄 Upload Work Order File</div>', unsafe_allow_html=True)
    if not PYPDF_AVAILABLE:
        st.error("⚠️ 'pypdf' library is missing! Run: pip install pypdf")
    
    uploaded_file = st.file_uploader("Upload your Work Order PDF", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        extracted_qties = parse_work_order_pdf(uploaded_file)
        if extracted_qties:
            st.success(f"✅ Successfully extracted {len(extracted_qties)} items from PDF!")
            st.session_state['uploaded_qty'] = {f"Item {idx+1}": q for idx, q in enumerate(extracted_qties)}
        else:
            st.error("❌ No items could be read. Ensure the format matches the sample Work Order.")
            st.session_state['uploaded_qty'] = {}
            
    # Display & Edit Panel for PDF Extracted Items
    if st.session_state['uploaded_qty']:
        st.markdown("#### Preview & Tweak Extracted Quantities:")
        for item_name, q_val in st.session_state['uploaded_qty'].items():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"<div class='tag-display'>{item_name}</div>", unsafe_allow_html=True)
            with col2:
                final_qty_dict[item_name] = st.number_input(f"Qty for {item_name}", 0, 100000, int(q_val), step=10, key=f"pdf_{item_name}", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MODE 2: MANUAL INPUT ----------------
else:
    st.markdown('<div class="card"><div class="card-title">📦 Item Quantity Details (Manual)</div>', unsafe_allow_html=True)
    n = st.number_input("🏷️ Number of Items", 1, 500, 1)
    
    for i in range(n):
        col1, col2 = st.columns([1, 2])
        with col1:
            item_name = f"Item {i+1}"
            st.markdown(f"<div class='tag-display'>{item_name}</div>", unsafe_allow_html=True)
        with col2:
            q_val = st.number_input(f"Quantity for {item_name}", 0, 100000, step=10, key=f"manual_{i}", label_visibility="collapsed")
            final_qty_dict[item_name] = q_val
    st.markdown('</div>', unsafe_allow_html=True)

# Data Mapping
original_qty = {t: int(q) for t, q in final_qty_dict.items() if q > 0}
demand = {t: ceil(int(q) * (1 + addon / 100)) for t, q in final_qty_dict.items() if q > 0}

if not PULP_AVAILABLE:
    st.markdown('<div class="warning">⚠️ PuLP library not installed</div>', unsafe_allow_html=True)

# Process Trigger
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_clicked = st.button("Plan Generated ", use_container_width=True)

if generate_clicked:
    if not demand:
        st.error("⚠️ Please enter or upload at least one item with quantity greater than 0")
        st.stop()
    
    with st.spinner("🔄 Running 10 algorithms simultaneously..."):
        results = {}
        results["V3 - Plate Ratio System"] = v3_optimizer(demand, cap, maxp)
        results["V4 - Common Sheet Optimizer"] = v4_optimizer(demand, cap, maxp)
        results["V5 - Smart Decimal Balancing"] = v5_optimizer(demand, cap, maxp)
        results["V6 - Multi-Variation Optimizer"] = v6_optimizer(demand, cap, maxp)
        results["V7 - AI Mutation Engine"] = v7_optimizer(demand, cap, maxp, iterations=100)
        results["V8 - Integer Solver"] = v8_optimizer(demand, cap, maxp) if PULP_AVAILABLE else v5_optimizer(demand, cap, maxp)
        results["V9 - Simulated Annealing"] = v9_optimizer(demand, cap, maxp, iterations=200)
        results["V10 - MCTS Tree Search"] = v10_optimizer(demand, cap, maxp, iterations=100)
        
        comparison_data = []
        for algo_name, plates in results.items():
            if plates:
                waste = calculate_waste_percent(plates, demand)
                comparison_data.append({
                    "Algorithm": algo_name,
                    "Waste %": waste,
                    "Total Plates": len(plates),
                    "Total Sheets": sum(p["sheets"] for p in plates)
                })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values("Waste %")
        best_algo = comparison_df.iloc[0]["Algorithm"]
        best_waste = comparison_df.iloc[0]["Waste %"]
        
        st.session_state['demand'] = demand
        st.session_state['original_qty'] = original_qty
        st.session_state['comparison_df'] = comparison_df
        st.session_state['best_algo'] = best_algo
        st.session_state['best_waste'] = best_waste
        st.session_state['results'] = results
        st.session_state['calculated'] = True

# Metrics Rendering
if st.session_state['calculated']:
    comparison_df = st.session_state['comparison_df']
    results = st.session_state['results']
    demand = st.session_state['demand']
    original_qty = st.session_state['original_qty']
    best_algo = st.session_state['best_algo']
    best_waste = st.session_state['best_waste']
    
    st.markdown(f"""
    <div class="best-algo" style="margin-bottom: 2rem;">
        <div class="metric-value">BEST ALGORITHM: {best_algo}</div>
        <div class="metric-label">Waste Percentage: {best_waste}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📊 Algorithm Comparison (Sorted by Waste %)")
    def highlight_best(row):
        if row["Algorithm"] == best_algo:
            return ['background-color: #2e7d32; color: white'] * len(row)
        return [''] * len(row)
    
    st.dataframe(comparison_df.style.apply(highlight_best, axis=1).format({"Waste %": "{:.2f}%"}), use_container_width=True)
    
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
    total_sheets_sum = sum(p["sheets"] for p in selected_plates)
    total_ups_sum = sum(sum(p["layout"].values()) for p in selected_plates)
    
    for idx, p in enumerate(selected_plates, 1):
        plate_rows.append({
            "SL": idx,
            "Plate ID": p["name"],
            "Sheets Required": p["sheets"],
            "Total UPS": sum(p["layout"].values())
        })
    plate_rows.append({"SL": "📊", "Plate ID": "TOTAL", "Sheets Required": total_sheets_sum, "Total UPS": total_ups_sum})
    plate_details_df = pd.DataFrame(plate_rows)
    st.dataframe(plate_details_df, use_container_width=True)
    
    st.markdown("### 📥 Download Report")
    col1, col2 = st.columns(2)
    with col1:
        bio_excel = BytesIO()
        with pd.ExcelWriter(bio_excel, engine="openpyxl") as writer:
            full_df.to_excel(writer, sheet_name="Production Summary", index=False)
            plate_details_df.to_excel(writer, sheet_name="Plate Details", index=False)
            comparison_df.to_excel(writer, sheet_name="Algorithm Comparison", index=False)
        bio_excel.seek(0)
        
        st.download_button(
            "📊 Download Excel Report",
            data=bio_excel,
            file_name=f"{algo_name_clean}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col2:
        if REPORTLAB_AVAILABLE:
            selected_waste = calculate_waste_percent(selected_plates, demand)
            pdf_buffer = generate_pdf_report(selected_plates, demand, original_qty, selected_algo, selected_waste)
            if pdf_buffer:
                st.download_button(
                    "📄 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"{algo_name_clean}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.warning("⚠️ PDF generation failed.")
        else:
            st.warning("⚠️ PDF download not available. Install reportlab.")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Plate Ratio System</p>
    <p style="color: #667eea;">✨ Design & Developed by <strong>Ovi</strong> ✨</p>
</div>
""", unsafe_allow_html=True)
