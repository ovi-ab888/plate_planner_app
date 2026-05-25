# app_final.py — 26-in-1 PLATE RATIO COMPARATOR (WITH ADVANCED ALGORITHMS)
# V1 to V26 Complete | Compare All Algorithms | Pick Best
# Design by Ovi

import os
import copy
import random
import math
import string
from collections import Counter
from math import ceil, floor
from datetime import datetime
from io import BytesIO

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd

# ================================================================
# LIBRARY IMPORTS & CHECKS
# ================================================================

# Try to import PuLP for Integer Solver
try:
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, LpInteger
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

# Try to import OR-Tools for V19
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

# ================================================================
# STREAMLIT PAGE CONFIGURATION
# ================================================================
st.set_page_config(
    page_title="Plate Ratio System - Complete Edition",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def check_password():
    expected = None
    try:
        expected = st.secrets.get("app_password", None)
    except Exception:
        pass
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
            st.session_state["wrong_password"] = True

    if st.session_state.get("password_correct", None) is True:
        return True

    # ================== FULL STYLING ==================
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        * { font-family: 'Inter', sans-serif; }
        
        .stApp {
            background: linear-gradient(-45deg, #0f0c29, #1a1a3e, #24243e, #1a1a3e);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .main > div { background: transparent !important; padding: 0 !important; }
        .block-container { padding: 0rem !important; max-width: 90% !important; }
        
        /* Main Header */
        .main-header {
            background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.15) 100%);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 30px;
            margin: 1rem 1rem 0rem 1rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .main-header h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0;
        }
        
        .password-container {
            max-width: 460px;
            margin: 40px auto 8px auto;
            padding: 2.8rem 2rem 1.8rem 2rem;
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(20px);
            border-radius: 32px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.3);
        }
        
        .lock-icon {
            font-size: 3rem;
            margin: 1rem 0;
            animation: bounce 2s infinite;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        .stTextInput { margin-top: -10px !important; }
        .stTextInput input {
            background: rgba(255,255,255,0.08) !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            border-radius: 30px !important;
            color: white !important;
            text-align: center !important;
            font-size: 1.1rem !important;
            padding: 0.9rem 1.5rem !important;
            letter-spacing: 3px;
        }
        
        #MainMenu, header, footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    # ================== HEADER ==================
    st.markdown("""
    <div class="main-header">
        <h1>Plate Ratio System</h1>
        <p>Intelligent Production Planning & Ratio Optimization</p>
        <p style="font-size: 0.85rem; opacity: 0.8;">AI-Powered • Fast • Accurate</p>
        <p class="designer-name">✨ Design by Ovi ✨</p>
    </div>
    """, unsafe_allow_html=True)

    # ================== PASSWORD CARD ==================
    st.markdown("""
    <div class="password-container">
        <h2>Welcome Back</h2>
        <div class="lock-icon">🔐</div>
        <p>Enter your secure access code to continue</p>
    </div>
    """, unsafe_allow_html=True)

    # Password Input (বক্সের ঠিক নিচে)
    col1, col2, col3 = st.columns([1.4, 1.1, 1.4])
    with col2:
        st.text_input(
            label="",
            type="password",
            key="password",
            on_change=_password_entered,
            label_visibility="collapsed",
            placeholder="••••••••"
        )

    # Error Message
    if st.session_state.get("password_correct") is False:
        st.error("❌ Incorrect password. Please contact Mr. Ovi.")

    return False

# ================== APP START ==================
if not check_password():
    st.stop()

# এখান থেকে আপনার মেইন অ্যাপ কোড শুরু হবে...
st.success("✅ সফলভাবে লগইন হয়েছে!")
st.write("আপনার মেইন অ্যাপ্লিকেশন এখানে শুরু করুন...")

# ================================================================
# MODERN CSS FOR MAIN APP
# ================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
    }
    
    /* Modern Header */
    .main-header {
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding: 2rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        border-radius: 0;
    }
    
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.7);
        margin-top: 0.5rem;
    }
    
    /* Modern Cards */
    .card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        border-color: rgba(102,126,234,0.5);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        border-bottom: 2px solid #667eea;
        display: inline-block;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
    }
    
    /* Modern Metrics */
    .metric-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.2) 100%);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1rem;
        color: white;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.7);
        margin-top: 0.5rem;
    }
    
    /* Best Algorithm Banner */
    .best-algo {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        border-radius: 20px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        border: none;
        box-shadow: 0 10px 30px rgba(0,176,155,0.3);
        margin-bottom: 2rem;
    }
    
    .best-algo .metric-value {
        -webkit-text-fill-color: white;
        font-size: 1.5rem;
    }
    
    /* Modern Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 12px;
        width: 100%;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102,126,234,0.4);
    }
    
    /* Modern Inputs */
    .stNumberInput input, .stTextInput input {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 5px !important;
        color: white !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102,126,234,0.2) !important;
        background: rgba(255,255,255,0.12) !important;
    }
    
    /* Modern Dataframe */
    .stDataFrame {
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 0.5rem;
    }
    
    .stDataFrame table {
        border-radius: 12px;
    }
    
    /* Tag Display */
    .tag-display {
        background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.2) 100%);
        padding: 10px;
        border-radius: 12px;
        border: 1px solid rgba(102,126,234,0.3);
        color: #667eea;
        font-weight: 600;
        text-align: center;
        font-size: 0.9rem;
    }
    
    /* Warning & Info */
    .warning {
        background: rgba(255,193,7,0.1);
        padding: 12px;
        border-radius: 12px;
        border-left: 4px solid #ffc107;
        color: #ffc107;
        margin: 1rem 0;
    }
    
    .info {
        background: rgba(23,162,184,0.1);
        padding: 12px;
        border-radius: 12px;
        border-left: 4px solid #17a2b8;
        color: #17a2b8;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: rgba(255,255,255,0.03);
        border-radius: 20px;
        margin-top: 3rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    
    .footer p {
        color: rgba(255,255,255,0.5);
        font-size: 0.85rem;
    }
    
    /* Radio Buttons */
    .stRadio > div {
        gap: 1rem;
    }
    
    .stRadio label {
        background: rgba(255,255,255,0.05);
        padding: 0.5rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    .stRadio label:hover {
        background: rgba(102,126,234,0.2);
        border-color: #667eea;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        color: white;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# HELPER FUNCTIONS
# ================================================================
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


def calculate_waste_percent(plates: list, demand: dict) -> float:
    """Calculate waste percentage from plates and demand"""
    total_produced = 0
    total_demand = sum(demand.values())
    
    if total_demand == 0:
        return 0.0

    for tag in demand:
        produced_qty = 0
        for p in plates:
            if p and "layout" in p:  # Check if plate exists
                ups = p["layout"].get(tag, 0)
                produced_qty += ups * p.get("sheets", 0)
        total_produced += produced_qty

    if total_produced == 0:
        return 100.0

    # Waste calculation: (produced - demand) / produced * 100
    waste = total_produced - total_demand
    waste_percent = (waste / total_produced) * 100
    
    # Ensure waste is not negative (if produced < demand, it's underproduction)
    if waste_percent < 0:
        waste_percent = 0.0
    
    return round(waste_percent, 2)


def build_full_summary(plates: list, demand: dict, original_qty: dict) -> pd.DataFrame:
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
        row["Excess"] = max(0, excess)  # Can't have negative excess
        row["Excess %"] = f"{max(0, excess_percent)}%"
        rows.append(row)
        sl += 1

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Add total row
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


def generate_pdf_report(plates: list, demand: dict, original_qty: dict,
                        algo_name: str, waste_percent: float) -> BytesIO | None:
    """Generate PDF report"""
    if not REPORTLAB_AVAILABLE:
        return None

    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=landscape(A4),
            rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20
        )
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Heading1'],
            fontSize=14, alignment=TA_CENTER, textColor=colors.HexColor('#667eea')
        )
        subtitle_style = ParagraphStyle(
            'CustomSubtitle', parent=styles['Normal'],
            fontSize=9, alignment=TA_CENTER, textColor=colors.grey
        )

        story = []
        story.append(Paragraph("📊 Plate Ratio System - Ratio Report", title_style))
        story.append(Paragraph(
            f"Algorithm: {algo_name} | Waste: {waste_percent}% | "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            subtitle_style
        ))
        story.append(Spacer(1, 15))

        # Summary table
        summary_data = [["SL", "Tag", "Original", "With Add-on"]]
        for p in plates:
            summary_data[0].append(f"Plate {p['name']}")
        summary_data[0].extend(["Total Prod.", "Excess", "Excess %"])

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

        total_row = ["📊", "TOTAL", str(sum(original_qty.values())), str(sum(demand.values()))]
        total_produced_sum = 0

        for p in plates:
            plate_total = 0
            for tag in demand:
                plate_total += p["layout"].get(tag, 0) * p["sheets"]
            total_row.append(str(plate_total))
            total_produced_sum += plate_total

        total_excess_sum = total_produced_sum - sum(demand.values())
        total_excess_percent = (
            f"{round((total_excess_sum / total_produced_sum) * 100, 2) if total_produced_sum > 0 else 0}%"
        )
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

        # Plate details table
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

        footer_style = ParagraphStyle(
            'Footer', parent=styles['Normal'],
            fontSize=8, alignment=TA_CENTER, textColor=colors.grey
        )
        story.append(Paragraph("This Report Generated by Ovi's Plate Ratio System", footer_style))

        doc.build(story)
        buffer.seek(0)
        return buffer

    except Exception as e:
        return None


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
# V7 - Simulated Annealing (Fixed - Exact Capacity)
# ================================================================
def v7_optimizer(demand: dict, capacity: int, max_plates: int, iterations: int = 200) -> list:
    """V7 - Simulated Annealing Optimizer with Exact Capacity"""
    
    def calculate_waste(layout: dict, sheets: int, remaining: dict) -> int:
        return sum(max(0, ups * sheets - remaining.get(tag, 0)) for tag, ups in layout.items())

    def adjust_to_exact_capacity(layout: dict, capacity: int) -> dict:
        """Ensure total UPS exactly equals capacity"""
        current_sum = sum(layout.values())
        
        if current_sum == capacity:
            return layout
        
        new_layout = layout.copy()
        
        # If sum is less than capacity → add to highest demand tag
        while sum(new_layout.values()) < capacity:
            if not new_layout:
                break
            # Add to the tag that needs it most
            best_tag = max(new_layout.keys(), key=lambda t: remaining.get(t, 0) / new_layout[t])
            new_layout[best_tag] += 1
        
        # If sum is more than capacity → remove from smallest
        while sum(new_layout.values()) > capacity:
            if not new_layout:
                break
            smallest_tag = min(new_layout.keys(), key=lambda t: new_layout[t])
            if new_layout[smallest_tag] > 1:
                new_layout[smallest_tag] -= 1
            else:
                # If can't reduce, try another tag
                for tag in sorted(new_layout.keys(), key=lambda t: new_layout[t], reverse=True):
                    if new_layout[tag] > 1:
                        new_layout[tag] -= 1
                        break
                else:
                    break  # Can't reduce further
        
        return new_layout

    def mutate_layout(layout: dict, capacity: int) -> dict:
        """Mutate while maintaining exact capacity"""
        new_layout = layout.copy()
        tags = list(new_layout.keys())
        
        if len(tags) >= 2:
            a, b = random.sample(tags, 2)
            if new_layout[a] > 1:
                new_layout[a] -= 1
                new_layout[b] += 1
        
        # Ensure exact capacity after mutation
        return adjust_to_exact_capacity(new_layout, capacity)

    def initial_layout(active: dict, capacity: int) -> dict:
        """Create initial layout with exact capacity"""
        if not active:
            return {}
        
        total = sum(active.values())
        layout = {}
        
        for tag, qty in active.items():
            ideal = (qty / total) * capacity
            layout[tag] = max(1, int(ideal))
        
        # Adjust to exact capacity
        return adjust_to_exact_capacity(layout, capacity)

    # ===================== Main Logic =====================
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

    # Final remaining adjustment
    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for tag in remaining:
            if remaining[tag] > 0:
                ups = max(1, last["layout"].get(tag, 1))
                last["sheets"] += ceil(remaining[tag] / ups)
                remaining[tag] = 0

    return plates


# ================================================================
# V8 - MCTS Tree Search (Fixed - Exact Capacity + Best Child)
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

    def best_child(self, c_param: float = 1.4):
        """Select best child using UCB1 formula"""
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


def v8_optimizer(demand: dict, capacity: int, max_plates: int, iterations: int = 100) -> list:
    """V8 - MCTS Tree Search Optimizer with Exact Capacity"""
    
    def adjust_to_exact_capacity(layout: dict, capacity: int, remaining: dict) -> dict:
        """Ensure total UPS exactly equals capacity"""
        if not layout or not remaining:
            return layout
        
        new_layout = layout.copy()
        
        # Add if less than capacity
        while sum(new_layout.values()) < capacity:
            if not new_layout:
                break
            best_tag = max(new_layout.keys(), key=lambda t: remaining.get(t, 0) / new_layout.get(t, 1))
            new_layout[best_tag] = new_layout.get(best_tag, 0) + 1
        
        # Reduce if more than capacity
        while sum(new_layout.values()) > capacity:
            candidates = [t for t in new_layout if new_layout[t] > 1]
            if not candidates:
                break
            smallest_tag = min(candidates, key=lambda t: new_layout[t])
            new_layout[smallest_tag] -= 1
        
        return new_layout

    def initial_layout(active: dict, capacity: int) -> dict:
        """Create initial balanced layout"""
        if not active:
            return {}
        
        total = sum(active.values())
        layout = {tag: max(1, int((qty / total) * capacity)) for tag, qty in active.items()}
        
        return adjust_to_exact_capacity(layout, capacity, active)

    # ===================== Main Logic =====================
    remaining = demand.copy()
    plates = []

    for plate_num in range(max_plates):
        active = {k: v for k, v in remaining.items() if v > 0}
        if not active:
            break

        root_layout = initial_layout(active, capacity)
        sheets = max(1, min(ceil(active[t] / root_layout[t]) for t in root_layout if root_layout[t] > 0))
        
        root = MCTSNodeV8(root_layout, active, capacity)

        for _ in range(iterations):
            node = root

            # Selection
            while node.children:
                next_node = node.best_child()
                if next_node is None:
                    break
                node = next_node

            # Expansion + Simulation
            current_layout = node.layout.copy()
            
            # Mutation
            tags = list(current_layout.keys())
            if len(tags) >= 2:
                a, b = random.sample(tags, 2)
                if current_layout.get(a, 0) > 1:
                    current_layout[a] -= 1
                    current_layout[b] = current_layout.get(b, 0) + 1

            # Ensure exact capacity
            final_layout = adjust_to_exact_capacity(current_layout, capacity, active)

            # Calculate score
            waste = sum(max(0, ups * sheets - active.get(tag, 0)) 
                       for tag, ups in final_layout.items())
            score = -waste

            # Create new node and backpropagate
            new_node = MCTSNodeV8(final_layout, active, capacity, node)
            node.children.append(new_node)
            
            # Backpropagation
            current_node = node
            while current_node:
                current_node.visits += 1
                current_node.score += score
                current_node = current_node.parent

        # Select best layout
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

    # Final cleanup for remaining items
    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for tag in list(remaining.keys()):
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
# V12 - COLUMN GENERATION METHOD (Corrected)
# ================================================================
def v12_optimizer(demand: dict, capacity: int, max_plates: int) -> list:
    """V12 - Column Generation Method for Large Scale (Corrected)"""
    if not PULP_AVAILABLE:
        return v3_optimizer(demand, capacity, max_plates)
    
    remaining = demand.copy()
    plates = []
    
    # Collect all unique tags
    tags_list = list(demand.keys())
    
    def solve_master_problem(patterns):
        """Solve master problem to get dual values"""
        try:
            # Master problem: minimize waste/plates
            master = LpProblem("Master_Problem", LpMinimize)
            
            # Variables for each pattern
            pattern_vars = []
            for idx in range(len(patterns)):
                var = LpVariable(f"x_{idx}", lowBound=0, cat="Integer")
                pattern_vars.append(var)
            
            # Objective: minimize total sheets used
            master += lpSum(pattern_vars)
            
            # Constraints: meet demand for each tag
            for tag in tags_list:
                master += lpSum(patterns[idx].get(tag, 0) * pattern_vars[idx] 
                               for idx in range(len(patterns))) >= demand[tag]
            
            master.solve()
            
            if master.status == 1:
                # Get dual values for each tag
                duals = {}
                for tag in tags_list:
                    constraint = [c for c in master.constraints.values() 
                                 if tag in str(c)][0] if master.constraints else None
                    duals[tag] = constraint.pi if constraint else 0
                
                return value(master.objective), duals
            return None, None
        except Exception as e:
            return None, None
    
    def generate_pattern_with_dual(duals, capacity):
        """Subproblem: generate pattern with positive reduced cost"""
        try:
            sub = LpProblem("Subproblem", LpMaximize)
            
            # Variables for each tag
            ups = {tag: LpVariable(f"ups_{tag}", lowBound=0, upBound=capacity, cat="Integer") 
                   for tag in tags_list}
            
            # Objective: maximize reduced cost
            sub += lpSum(duals.get(tag, 0) * ups[tag] for tag in tags_list) - 1
            
            # Capacity constraint
            sub += lpSum(ups[tag] for tag in tags_list) <= capacity
            
            sub.solve()
            
            if sub.status == 1:
                pattern = {tag: int(value(ups[tag])) for tag in tags_list if value(ups[tag]) > 0}
                reduced_cost = value(sub.objective)
                return pattern, reduced_cost
            return None, -1
        except Exception:
            return None, -1
    
    # Initial patterns (simple proportional layouts)
    patterns = []
    
    # Generate initial feasible patterns
    initial_demand = demand.copy()
    for _ in range(min(max_plates, len(tags_list) * 2)):
        active = {k: v for k, v in initial_demand.items() if v > 0}
        if not active:
            break
        
        total = sum(active.values())
        pattern = {}
        for tag, qty in active.items():
            pattern[tag] = max(1, int((qty / total) * capacity))
        
        # Adjust to capacity
        while sum(pattern.values()) > capacity:
            max_tag = max(pattern, key=pattern.get)
            if pattern[max_tag] > 1:
                pattern[max_tag] -= 1
            else:
                break
        
        while sum(pattern.values()) < capacity:
            max_tag = max(active, key=active.get)
            pattern[max_tag] = pattern.get(max_tag, 0) + 1
        
        patterns.append(pattern)
        
        # Update remaining for next initial pattern
        sheets = 1
        for tag, ups in pattern.items():
            initial_demand[tag] = max(0, initial_demand[tag] - (ups * sheets))
    
    # Column Generation Loop
    iteration = 0
    max_iterations = 50
    
    while iteration < max_iterations:
        # Solve master problem with current patterns
        obj_value, duals = solve_master_problem(patterns)
        
        if duals is None:
            break
        
        # Generate new pattern
        new_pattern, reduced_cost = generate_pattern_with_dual(duals, capacity)
        
        # Stop if no improving pattern found
        if new_pattern is None or reduced_cost <= 0.001:
            break
        
        # Add new pattern
        patterns.append(new_pattern)
        iteration += 1
    
    # Build final solution from patterns
    remaining = demand.copy()
    plate_counter = 0
    
    # Sort patterns by efficiency
    pattern_efficiency = []
    for idx, pattern in enumerate(patterns):
        if sum(pattern.values()) > 0:
            total_ups = sum(pattern.values())
            pattern_efficiency.append((idx, total_ups))
    
    pattern_efficiency.sort(key=lambda x: x[1], reverse=True)
    
    for idx, _ in pattern_efficiency:
        if plate_counter >= max_plates:
            break
        
        pattern = patterns[idx]
        
        # Skip if pattern doesn't help
        active_tags = [t for t in pattern if remaining.get(t, 0) > 0]
        if not active_tags:
            continue
        
        # Calculate maximum sheets possible with this pattern
        possible_sheets = []
        for tag in active_tags:
            if pattern.get(tag, 0) > 0:
                possible_sheets.append(ceil(remaining[tag] / pattern[tag]))
        
        if not possible_sheets:
            continue
        
        sheets = min(possible_sheets)
        
        # Don't use too many plates
        if plate_counter + 1 > max_plates:
            sheets = 1
        
        plates.append({
            "name": plate_name(plate_counter + 1),
            "layout": pattern,
            "sheets": sheets
        })
        
        for tag, ups in pattern.items():
            remaining[tag] = max(0, remaining[tag] - (ups * sheets))
        
        plate_counter += 1
        
        # Check if all demand met
        if all(v <= 0 for v in remaining.values()):
            break
    
    # Handle remaining demand
    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for tag in list(remaining.keys()):
            if remaining[tag] > 0:
                ups = max(1, last["layout"].get(tag, 1))
                add_sheets = ceil(remaining[tag] / ups)
                last["sheets"] += add_sheets
                remaining[tag] = 0
    elif any(v > 0 for v in remaining.values()):
        # Fallback to V3 if no plates generated
        return v3_optimizer(demand, capacity, max_plates)
    
    return plates if plates else v3_optimizer(demand, capacity, max_plates)

# ================================================================
# INSTALL REQUIREMENT
# pip install ortools
# ================================================================

# ================================================================
# IMPORT
# ================================================================
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except:
    ORTOOLS_AVAILABLE = False

# ================================================================
# V13 - HYBRID MASTER OPTIMIZER (যোগ করুন)
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
# V14 - BASE OPTIMIZER (যোগ করুন)
# ================================================================
def v14_optimizer(demand: dict, capacity: int, max_plates: int) -> list:
    """V14 - Base Optimizer using V13 as fallback"""
    return v13_optimizer(demand, capacity, max_plates)





# ================================================================
# V15 - DYNAMIC PROGRAMMING REPAIR ENGINE
# ================================================================
def v15_optimizer(demand: dict, capacity: int, max_plates: int):

    base = v14_optimizer(demand, capacity, max_plates)

    if not base:
        return v13_optimizer(demand, capacity, max_plates)

    best = copy.deepcopy(base)
    best_waste = calculate_waste_percent(best, demand)

    for _ in range(300):

        trial = copy.deepcopy(best)

        for plate in trial:

            tags = list(plate["layout"].keys())

            if len(tags) < 2:
                continue

            a, b = random.sample(tags, 2)

            if plate["layout"][a] > 1:

                plate["layout"][a] -= 1
                plate["layout"][b] += 1

                if sum(plate["layout"].values()) > capacity:
                    plate["layout"][a] += 1
                    plate["layout"][b] -= 1

        waste = calculate_waste_percent(trial, demand)

        if waste < best_waste:
            best = copy.deepcopy(trial)
            best_waste = waste

    return best


# ================================================================
# V16 - PLATE MERGE OPTIMIZER
# ================================================================
def v16_optimizer(demand: dict, capacity: int, max_plates: int):

    plates = v15_optimizer(demand, capacity, max_plates)

    if not plates:
        return v13_optimizer(demand, capacity, max_plates)

    merged = []

    skip = set()

    for i in range(len(plates)):

        if i in skip:
            continue

        current = copy.deepcopy(plates[i])

        for j in range(i + 1, len(plates)):

            if j in skip:
                continue

            candidate = copy.deepcopy(plates[j])

            combined = current["layout"].copy()

            possible = True

            for tag, ups in candidate["layout"].items():

                combined[tag] = combined.get(tag, 0) + ups

            if sum(combined.values()) <= capacity:

                current["layout"] = combined
                current["sheets"] = max(
                    current["sheets"],
                    candidate["sheets"]
                )

                skip.add(j)

        merged.append(current)

    return merged


# ================================================================
# V17 - AI EVOLUTION ENGINE
# ================================================================
def v17_optimizer(
    demand: dict,
    capacity: int,
    max_plates: int,
    generations: int = 200
):

    population = []

    # Initial Population
    for _ in range(20):

        candidate = random.choice([
            v3_optimizer,
            v5_optimizer,
            v9_optimizer,
            v11_optimizer,
            v13_optimizer,
            v15_optimizer,
            v16_optimizer
        ])(demand, capacity, max_plates)

        population.append(candidate)

    best_solution = None
    best_waste = 999999

    for generation in range(generations):

        scored = []

        for sol in population:

            waste = calculate_waste_percent(sol, demand)

            scored.append((waste, sol))

            if waste < best_waste:
                best_waste = waste
                best_solution = copy.deepcopy(sol)

        scored.sort(key=lambda x: x[0])

        elites = [copy.deepcopy(x[1]) for x in scored[:5]]

        new_population = elites.copy()

        while len(new_population) < 20:

            parent = copy.deepcopy(random.choice(elites))

            for plate in parent:

                tags = list(plate["layout"].keys())

                if len(tags) >= 2:

                    a, b = random.sample(tags, 2)

                    if plate["layout"][a] > 1:

                        plate["layout"][a] -= 1
                        plate["layout"][b] += 1

                        if sum(plate["layout"].values()) > capacity:

                            plate["layout"][a] += 1
                            plate["layout"][b] -= 1

            new_population.append(parent)

        population = new_population

    return best_solution


# ================================================================
# V18 - GLOBAL MULTI-PLATE OPTIMIZER
# ================================================================
def v18_optimizer(demand: dict, capacity: int, max_plates: int):

    candidates = []

    algos = [
        v14_optimizer,
        v15_optimizer,
        v16_optimizer,
        v17_optimizer,
        v13_optimizer,
        v11_optimizer,
        v9_optimizer
    ]

    for algo in algos:

        try:
            result = algo(demand, capacity, max_plates)

            if result:

                waste = calculate_waste_percent(result, demand)

                candidates.append((waste, result))

        except:
            pass

    if not candidates:
        return v13_optimizer(demand, capacity, max_plates)

    candidates.sort(key=lambda x: x[0])

    return candidates[0][1]


# ================================================================
# V19 - CONSTRAINT PROGRAMMING (CP-SAT) OPTIMIZER
# ================================================================
def v19_optimizer(demand: dict, capacity: int, max_plates: int, time_limit_seconds: int = 5) -> list:
    """V19 - OR-Tools CP-SAT based optimization for plate layout"""
    if not ORTOOLS_AVAILABLE:
        return v18_optimizer(demand, capacity, max_plates)
    
    tags = list(demand.keys())
    n_tags = len(tags)
    
    if n_tags == 0:
        return []
    
    model = cp_model.CpModel()
    
    # Variables
    max_possible_plates = max_plates
    
    ups = {}
    for i in range(max_possible_plates):
        for idx, tag in enumerate(tags):
            max_ups = min(capacity, demand.get(tag, 0))
            ups[(i, idx)] = model.NewIntVar(0, max_ups, f'ups_{i}_{tag}')
    
    sheets = {}
    for i in range(max_possible_plates):
        sheets[i] = model.NewIntVar(0, sum(demand.values()), f'sheets_{i}')
    
    plate_used = {}
    for i in range(max_possible_plates):
        plate_used[i] = model.NewBoolVar(f'used_{i}')
    
    # Capacity constraint
    for i in range(max_possible_plates):
        total_ups = sum(ups[(i, idx)] for idx in range(n_tags))
        model.Add(total_ups <= capacity)
        
        # Link plate_used with UPS
        model.Add(total_ups == 0).OnlyEnforceIf(plate_used[i].Not())
        model.Add(total_ups > 0).OnlyEnforceIf(plate_used[i])
    
    # Demand constraints
    for idx, tag in enumerate(tags):
        total_produced = sum(ups[(i, idx)] * sheets[i] for i in range(max_possible_plates))
        model.Add(total_produced >= demand[tag])
    
    # Sheets constraint
    for i in range(max_possible_plates):
        model.Add(sheets[i] >= 1).OnlyEnforceIf(plate_used[i])
        model.Add(sheets[i] == 0).OnlyEnforceIf(plate_used[i].Not())
    
    # Objective: minimize total sheets
    total_sheets = sum(sheets[i] for i in range(max_possible_plates))
    model.Minimize(total_sheets)
    
    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        plates = []
        for i in range(max_possible_plates):
            if solver.Value(plate_used[i]) and solver.Value(sheets[i]) > 0:
                layout = {}
                for idx, tag in enumerate(tags):
                    ups_val = solver.Value(ups[(i, idx)])
                    if ups_val > 0:
                        layout[tag] = ups_val
                if layout:
                    plates.append({
                        "name": plate_name(len(plates) + 1),
                        "layout": layout,
                        "sheets": int(solver.Value(sheets[i]))
                    })
        return plates if plates else v18_optimizer(demand, capacity, max_plates)
    
    return v18_optimizer(demand, capacity, max_plates)


# ================================================================
# V20 - PARTICLE SWARM OPTIMIZATION (PSO)
# ================================================================
def v20_optimizer(demand: dict, capacity: int, max_plates: int, 
                   particles: int = 30, iterations: int = 100) -> list:
    """V20 - PSO based layout optimization"""
    
    tags = list(demand.keys())
    n_tags = len(tags)
    best_global_plates = None
    best_global_waste = float('inf')
    
    class Particle:
        def __init__(self):
            self.position = []
            self.velocity = []
            self.best_position = []
            self.best_waste = float('inf')
            
            # Initialize random plates
            self.plates = []
            remaining = demand.copy()
            for _ in range(max_plates):
                active = {k: v for k, v in remaining.items() if v > 0}
                if not active:
                    break
                
                total = sum(active.values())
                layout = {}
                for tag, qty in active.items():
                    layout[tag] = max(1, int((qty / total) * capacity) + random.randint(-1, 1))
                
                # Adjust to capacity
                while sum(layout.values()) > capacity:
                    max_tag = max(layout, key=layout.get)
                    if layout[max_tag] > 1:
                        layout[max_tag] -= 1
                    else:
                        break
                
                while sum(layout.values()) < capacity:
                    max_tag = max(active, key=active.get)
                    layout[max_tag] = layout.get(max_tag, 0) + 1
                
                sheets = max(1, min(ceil(remaining[tag] / layout.get(tag, 1)) for tag in active))
                
                self.plates.append({"layout": layout, "sheets": sheets})
                
                for tag, ups in layout.items():
                    remaining[tag] = max(0, remaining[tag] - (ups * sheets))
            
            if any(v > 0 for v in remaining.values()) and self.plates:
                last = self.plates[-1]
                for tag in remaining:
                    if remaining[tag] > 0:
                        ups = max(1, last["layout"].get(tag, 1))
                        last["sheets"] += ceil(remaining[tag] / ups)
                        remaining[tag] = 0
            
            self.update_fitness()
        
        def update_fitness(self):
            waste = calculate_waste_percent(self.plates, demand)
            if waste < self.best_waste:
                self.best_waste = waste
                self.best_position = copy.deepcopy(self.plates)
            return waste
    
    # Initialize particles
    swarm = [Particle() for _ in range(particles)]
    
    # PSO main loop
    w = 0.7  # inertia weight
    c1 = 1.5  # cognitive weight
    c2 = 1.5  # social weight
    
    for iteration in range(iterations):
        for particle in swarm:
            waste = particle.update_fitness()
            
            if waste < best_global_waste:
                best_global_waste = waste
                best_global_plates = copy.deepcopy(particle.plates)
            
            # Update velocity and position (simplified mutation)
            if random.random() < 0.3:
                # Mutate a random plate
                if particle.plates:
                    plate_idx = random.randint(0, len(particle.plates) - 1)
                    layout = particle.plates[plate_idx]["layout"]
                    if len(layout) >= 2:
                        tags_list = list(layout.keys())
                        a, b = random.sample(tags_list, 2)
                        if layout[a] > 1:
                            layout[a] -= 1
                            layout[b] += 1
        
        # Adaptive parameters
        w = 0.7 * (1 - iteration / iterations)
    
    return best_global_plates if best_global_plates else v18_optimizer(demand, capacity, max_plates)


# ================================================================
# V21 - ANT COLONY OPTIMIZATION (ACO)
# ================================================================
def v21_optimizer(demand: dict, capacity: int, max_plates: int,
                   ants: int = 20, iterations: int = 50,
                   alpha: float = 1.0, beta: float = 2.0,
                   evaporation: float = 0.5) -> list:
    """V21 - Ant Colony Optimization for plate layout"""
    
    tags = list(demand.keys())
    n_tags = len(tags)
    
    # Pheromone matrix
    pheromone = {}
    for i in range(n_tags):
        for j in range(1, capacity + 1):
            pheromone[(i, j)] = 1.0
    
    best_plates = None
    best_waste = float('inf')
    
    def construct_solution():
        remaining = demand.copy()
        plates = []
        
        for _ in range(max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            
            layout = {}
            remaining_cap = capacity
            
            # Ant decision making
            for tag in active.keys():
                if remaining_cap <= 0:
                    break
                
                tag_idx = tags.index(tag)
                probabilities = []
                possible_ups = list(range(1, min(remaining_cap, active[tag]) + 1))
                
                for ups in possible_ups:
                    prob = (pheromone.get((tag_idx, ups), 1.0) ** alpha) * ((1.0 / ups) ** beta)
                    probabilities.append(prob)
                
                if probabilities:
                    total_prob = sum(probabilities)
                    if total_prob > 0:
                        probs = [p / total_prob for p in probabilities]
                        chosen_ups = random.choices(possible_ups, weights=probs)[0]
                    else:
                        chosen_ups = min(possible_ups)
                    
                    layout[tag] = chosen_ups
                    remaining_cap -= chosen_ups
            
            # Fill remaining capacity
            while remaining_cap > 0:
                max_tag = max(active, key=active.get)
                layout[max_tag] = layout.get(max_tag, 0) + 1
                remaining_cap -= 1
            
            sheets = max(1, min(ceil(remaining[tag] / layout.get(tag, 1)) for tag in active))
            
            plates.append({"layout": layout, "sheets": sheets})
            
            for tag, ups in layout.items():
                remaining[tag] = max(0, remaining[tag] - (ups * sheets))
        
        return plates
    
    def update_pheromone(plates, waste):
        # Evaporate
        for key in pheromone:
            pheromone[key] *= (1 - evaporation)
        
        # Deposit
        deposit = 1.0 / (waste + 1)
        for plate in plates:
            for tag, ups in plate["layout"].items():
                tag_idx = tags.index(tag)
                pheromone[(tag_idx, ups)] += deposit
    
    # Main ACO loop
    for iteration in range(iterations):
        iteration_best_plates = None
        iteration_best_waste = float('inf')
        
        for ant in range(ants):
            plates = construct_solution()
            waste = calculate_waste_percent(plates, demand)
            
            if waste < iteration_best_waste:
                iteration_best_waste = waste
                iteration_best_plates = plates
            
            if waste < best_waste:
                best_waste = waste
                best_plates = copy.deepcopy(plates)
        
        if iteration_best_plates:
            update_pheromone(iteration_best_plates, iteration_best_waste)
    
    return best_plates if best_plates else v18_optimizer(demand, capacity, max_plates)


# ================================================================
# V22 - Q-LEARNING OPTIMIZER
# ================================================================
class QLearningPlateOptimizer:
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
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        
        q_values = [self.q_table.get((state, action), 0) for action in possible_actions]
        max_q = max(q_values) if q_values else 0
        best_actions = [a for a, q in zip(possible_actions, q_values) if q == max_q]
        return random.choice(best_actions) if best_actions else random.choice(possible_actions)
    
    def optimize(self, episodes=50):
        best_plates = None
        best_waste = float('inf')
        
        for episode in range(episodes):
            remaining = self.demand.copy()
            plates = []
            
            for plate_num in range(self.max_plates):
                active = {k: v for k, v in remaining.items() if v > 0}
                if not active:
                    break
                
                # Build initial layout
                total = sum(active.values())
                layout = {}
                for tag, qty in active.items():
                    layout[tag] = max(1, int((qty / total) * self.capacity))
                
                # Adjust to capacity
                while sum(layout.values()) > self.capacity:
                    max_tag = max(layout, key=layout.get)
                    if layout[max_tag] > 1:
                        layout[max_tag] -= 1
                    else:
                        break
                
                while sum(layout.values()) < self.capacity:
                    max_tag = max(active, key=active.get)
                    layout[max_tag] = layout.get(max_tag, 0) + 1
                
                # Q-learning improvement
                state = self.get_state_key(remaining, layout)
                possible_actions = self.get_possible_actions(layout)
                
                if possible_actions:
                    action = self.get_action(state, possible_actions)
                    new_layout = self.apply_action(layout, action)
                    
                    # Calculate reward
                    sheets = max(1, min(ceil(remaining[t] / new_layout.get(t, 1)) for t in active))
                    waste = sum(max(0, new_layout.get(t, 0) * sheets - remaining.get(t, 0)) for t in active)
                    reward = -waste
                    
                    next_state = self.get_state_key(remaining, new_layout)
                    old_q = self.q_table.get((state, action), 0)
                    next_max_q = max([self.q_table.get((next_state, a), 0) for a in self.get_possible_actions(new_layout)]) if self.get_possible_actions(new_layout) else 0
                    new_q = old_q + self.lr * (reward + self.discount * next_max_q - old_q)
                    self.q_table[(state, action)] = new_q
                    
                    layout = new_layout
                
                sheets = max(1, min(ceil(remaining[t] / layout.get(t, 1)) for t in active))
                
                plates.append({"name": plate_name(len(plates) + 1), "layout": layout, "sheets": sheets})
                
                for tag, ups in layout.items():
                    remaining[tag] = max(0, remaining[tag] - (ups * sheets))
            
            waste = calculate_waste_percent(plates, self.demand)
            if waste < best_waste:
                best_waste = waste
                best_plates = copy.deepcopy(plates)
            
            # Decay epsilon
            self.epsilon *= 0.99
        
        return best_plates
    
    def get_possible_actions(self, layout):
        """Generate possible mutation actions"""
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


def v22_optimizer(demand: dict, capacity: int, max_plates: int, episodes: int = 50) -> list:
    """V22 - Q-Learning based plate optimizer"""
    optimizer = QLearningPlateOptimizer(demand, capacity, max_plates)
    result = optimizer.optimize(episodes)
    return result if result else v18_optimizer(demand, capacity, max_plates)


# ================================================================
# V23 - BRANCH AND BOUND OPTIMIZER
# ================================================================
def v23_optimizer(demand: dict, capacity: int, max_plates: int) -> list:
    """V23 - Exact Branch and Bound algorithm for small datasets"""
    
    tags = list(demand.keys())
    n_tags = len(tags)
    
    if n_tags > 8:  # Too large for B&B
        return v18_optimizer(demand, capacity, max_plates)
    
    best_plates = None
    best_waste = float('inf')
    
    def backtrack(remaining, current_plates, plate_num):
        nonlocal best_plates, best_waste
        
        # Prune if already worse than best
        current_waste = calculate_waste_percent(current_plates, demand) if current_plates else 0
        if current_waste >= best_waste:
            return
        
        # Check if all demand met
        if all(v <= 0 for v in remaining.values()):
            if current_waste < best_waste:
                best_waste = current_waste
                best_plates = copy.deepcopy(current_plates)
            return
        
        # Max plates limit
        if plate_num >= max_plates:
            # Try to add remaining to last plate
            if current_plates:
                last_plate = current_plates[-1]
                for tag in remaining:
                    if remaining[tag] > 0:
                        ups = max(1, last_plate["layout"].get(tag, 1))
                        last_plate["sheets"] += ceil(remaining[tag] / ups)
                        remaining[tag] = 0
                
                final_waste = calculate_waste_percent(current_plates, demand)
                if final_waste < best_waste:
                    best_waste = final_waste
                    best_plates = copy.deepcopy(current_plates)
            return
        
        active = {k: v for k, v in remaining.items() if v > 0}
        if not active:
            return
        
        # Generate possible layouts
        def generate_layouts(current_layout, remaining_cap, start_idx):
            if remaining_cap == 0 or start_idx >= len(active_tags):
                yield current_layout.copy()
                return
            
            tag = active_tags[start_idx]
            max_ups = min(remaining_cap, active[tag])
            
            for ups in range(1, max_ups + 1):
                current_layout[tag] = ups
                yield from generate_layouts(current_layout, remaining_cap - ups, start_idx + 1)
            
            if tag in current_layout:
                del current_layout[tag]
            yield from generate_layouts(current_layout, remaining_cap, start_idx + 1)
        
        active_tags = list(active.keys())
        
        for layout in generate_layouts({}, capacity, 0):
            if not layout or sum(layout.values()) != capacity:
                continue
            
            sheets = max(1, min(ceil(remaining[t] / layout.get(t, 1)) for t in active_tags if layout.get(t, 0) > 0))
            
            new_remaining = remaining.copy()
            for tag, ups in layout.items():
                new_remaining[tag] = max(0, new_remaining[tag] - (ups * sheets))
            
            new_plates = current_plates.copy()
            new_plates.append({
                "name": plate_name(plate_num + 1),
                "layout": layout,
                "sheets": sheets
            })
            
            backtrack(new_remaining, new_plates, plate_num + 1)
    
    backtrack(demand.copy(), [], 0)
    
    return best_plates if best_plates else v18_optimizer(demand, capacity, max_plates)


# ================================================================
# V24 - DIFFERENTIAL EVOLUTION OPTIMIZER
# ================================================================
def v24_optimizer(demand: dict, capacity: int, max_plates: int,
                   population_size: int = 30, generations: int = 100,
                   F: float = 0.8, CR: float = 0.9) -> list:
    """V24 - Differential Evolution for continuous optimization"""
    
    tags = list(demand.keys())
    n_tags = len(tags)
    
    def encode_plates_to_vector(plates):
        """Convert plates to continuous vector"""
        vector = []
        for plate in plates:
            for tag in tags:
                vector.append(plate["layout"].get(tag, 0) / capacity)
            vector.append(plate["sheets"] / 1000.0)  # normalize
        # Pad to fixed length
        while len(vector) < max_plates * (n_tags + 1):
            vector.append(0)
        return vector[:max_plates * (n_tags + 1)]
    
    def decode_vector_to_plates(vector):
        """Convert vector back to plates"""
        plates = []
        for i in range(max_plates):
            start_idx = i * (n_tags + 1)
            layout = {}
            for j, tag in enumerate(tags):
                ups = int(vector[start_idx + j] * capacity)
                if ups > 0:
                    layout[tag] = max(1, min(ups, capacity))
            
            if not layout:
                continue
            
            # Adjust to capacity
            total_ups = sum(layout.values())
            if total_ups > capacity:
                # Scale down
                scale = capacity / total_ups
                for tag in layout:
                    layout[tag] = max(1, int(layout[tag] * scale))
            
            while sum(layout.values()) > capacity:
                max_tag = max(layout, key=layout.get)
                if layout[max_tag] > 1:
                    layout[max_tag] -= 1
                else:
                    break
            
            while sum(layout.values()) < capacity:
                if not layout:
                    break
                max_tag = max(layout, key=layout.get)
                layout[max_tag] += 1
            
            sheets = max(1, int(vector[start_idx + n_tags] * 1000))
            
            plates.append({"layout": layout, "sheets": sheets})
        
        return plates
    
    def evaluate(plates):
        return calculate_waste_percent(plates, demand)
    
    # Initialize population
    population = []
    for _ in range(population_size):
        # Create random plates
        remaining = demand.copy()
        plates = []
        for _ in range(max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            
            layout = {}
            for tag in active:
                layout[tag] = random.randint(1, min(capacity, active[tag]))
            
            # Adjust to capacity
            while sum(layout.values()) > capacity:
                max_tag = max(layout, key=layout.get)
                if layout[max_tag] > 1:
                    layout[max_tag] -= 1
                else:
                    break
            
            while sum(layout.values()) < capacity:
                max_tag = max(active, key=active.get)
                layout[max_tag] = layout.get(max_tag, 0) + 1
            
            sheets = random.randint(1, max(1, ceil(max(active.values()) / capacity)))
            
            plates.append({"layout": layout, "sheets": sheets})
            
            for tag, ups in layout.items():
                remaining[tag] = max(0, remaining[tag] - (ups * sheets))
        
        if any(v > 0 for v in remaining.values()) and plates:
            last = plates[-1]
            for tag in remaining:
                if remaining[tag] > 0:
                    ups = max(1, last["layout"].get(tag, 1))
                    last["sheets"] += ceil(remaining[tag] / ups)
                    remaining[tag] = 0
        
        population.append(encode_plates_to_vector(plates))
    
    # Main DE loop
    for generation in range(generations):
        new_population = []
        
        for i, target in enumerate(population):
            # Select three distinct random vectors
            candidates = [idx for idx in range(population_size) if idx != i]
            a, b, c = random.sample(candidates, 3)
            
            # Mutation
            mutant = [population[a][j] + F * (population[b][j] - population[c][j]) for j in range(len(target))]
            
            # Crossover
            trial = []
            for j in range(len(target)):
                if random.random() < CR:
                    trial.append(mutant[j])
                else:
                    trial.append(target[j])
            
            # Evaluate
            trial_plates = decode_vector_to_plates(trial)
            trial_waste = evaluate(trial_plates)
            
            target_plates = decode_vector_to_plates(target)
            target_waste = evaluate(target_plates)
            
            if trial_waste < target_waste:
                new_population.append(trial)
            else:
                new_population.append(target)
        
        population = new_population
    
    # Get best solution
    best_idx = min(range(population_size), key=lambda i: evaluate(decode_vector_to_plates(population[i])))
    best_plates = decode_vector_to_plates(population[best_idx])
    
    return best_plates if best_plates else v18_optimizer(demand, capacity, max_plates)

# ================================================================
# V25 - MULTI-OBJECTIVE PARETO OPTIMIZER
# ================================================================
def v25_optimizer(demand: dict, capacity: int, max_plates: int, 
                   population_size: int = 50, generations: int = 100) -> list:
    """V25 - Multi-Objective Pareto front based optimization"""
    
    class Individual:
        def __init__(self, plates):
            self.plates = plates
            self.waste = calculate_waste_percent(plates, demand)
            self.total_plates = len(plates)
            self.total_sheets = sum(p["sheets"] for p in plates)
        
        def dominates(self, other):
            """Check if this individual dominates another"""
            better_in_one = False
            for attr in ['waste', 'total_plates', 'total_sheets']:
                if getattr(self, attr) < getattr(other, attr):
                    better_in_one = True
                elif getattr(self, attr) > getattr(other, attr):
                    return False
            return better_in_one
    
    def create_individual():
        remaining = demand.copy()
        plates = []
        
        for _ in range(max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            
            total = sum(active.values())
            layout = {}
            for tag, qty in active.items():
                layout[tag] = max(1, int((qty / total) * capacity))
            
            while sum(layout.values()) > capacity:
                max_tag = max(layout, key=layout.get)
                if layout[max_tag] > 1:
                    layout[max_tag] -= 1
                else:
                    break
            
            while sum(layout.values()) < capacity:
                max_tag = max(active, key=active.get)
                layout[max_tag] = layout.get(max_tag, 0) + 1
            
            sheets = max(1, min(ceil(remaining[tag] / layout.get(tag, 1)) for tag in active))
            
            plates.append({"layout": layout, "sheets": sheets})
            
            for tag, ups in layout.items():
                remaining[tag] = max(0, remaining[tag] - (ups * sheets))
        
        if any(v > 0 for v in remaining.values()) and plates:
            last = plates[-1]
            for tag in remaining:
                if remaining[tag] > 0:
                    ups = max(1, last["layout"].get(tag, 1))
                    last["sheets"] += ceil(remaining[tag] / ups)
                    remaining[tag] = 0
        
        return Individual(plates)
    
    def mutate(individual):
        new_plates = copy.deepcopy(individual.plates)
        if new_plates:
            plate_idx = random.randint(0, len(new_plates) - 1)
            layout = new_plates[plate_idx]["layout"]
            if len(layout) >= 2:
                tags_list = list(layout.keys())
                a, b = random.sample(tags_list, 2)
                if layout[a] > 1:
                    layout[a] -= 1
                    layout[b] += 1
        return Individual(new_plates)
    
    def crossover(ind1, ind2):
        point = random.randint(1, min(len(ind1.plates), len(ind2.plates)) - 1)
        child_plates = ind1.plates[:point] + ind2.plates[point:]
        
        # Fix the child to ensure demand is met
        remaining = demand.copy()
        fixed_plates = []
        for plate in child_plates:
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            
            new_layout = plate["layout"].copy()
            sheets = plate["sheets"]
            
            fixed_plates.append({"layout": new_layout, "sheets": sheets})
            
            for tag, ups in new_layout.items():
                remaining[tag] = max(0, remaining[tag] - (ups * sheets))
        
        if any(v > 0 for v in remaining.values()) and fixed_plates:
            last = fixed_plates[-1]
            for tag in remaining:
                if remaining[tag] > 0:
                    ups = max(1, last["layout"].get(tag, 1))
                    last["sheets"] += ceil(remaining[tag] / ups)
                    remaining[tag] = 0
        
        return Individual(fixed_plates)
    
    def non_dominated_sort(population):
        fronts = []
        remaining = set(range(len(population)))
        
        while remaining:
            front = []
            for i in list(remaining):
                dominated = False
                for j in list(remaining):
                    if i != j and population[j].dominates(population[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            
            for i in front:
                remaining.remove(i)
            fronts.append([population[i] for i in front])
        
        return fronts
    
    # Initialize population
    population = [create_individual() for _ in range(population_size)]
    
    for generation in range(generations):
        # Create offspring
        offspring = []
        while len(offspring) < population_size:
            parents = random.sample(population, 2)
            child = crossover(parents[0], parents[1])
            if random.random() < 0.1:
                child = mutate(child)
            offspring.append(child)
        
        # Combine and sort
        combined = population + offspring
        fronts = non_dominated_sort(combined)
        
        # Select next generation
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= population_size:
                new_population.extend(front)
            else:
                # Sort by crowding distance (simplified - sort by waste)
                front.sort(key=lambda x: x.waste)
                new_population.extend(front[:population_size - len(new_population)])
                break
        
        population = new_population
    
    # Return the best individual (lowest waste)
    best = min(population, key=lambda x: x.waste)
    return best.plates if best.plates else v18_optimizer(demand, capacity, max_plates)

# ================================================================
# V26 - NEURAL NETWORK PREDICTOR + OPTIMIZER
# ================================================================
def v26_optimizer(demand: dict, capacity: int, max_plates: int,
                   use_ml: bool = True) -> list:
    """V26 - NN-based prediction + Genetic optimization"""
    
    # Simple ML model (simulated neural network)
    # In production, you would train actual NN. Here we use heuristics + pattern learning
    
    tags = list(demand.keys())
    n_tags = len(tags)
    
    class SimplePredictor:
        def __init__(self):
            self.patterns = {}  # Store successful patterns
            self.weights = {}   # Feature weights
        
        def learn_from_plate(self, layout, waste):
            """Learn from successful plate layouts"""
            key = tuple(sorted(layout.items()))
            if key not in self.patterns:
                self.patterns[key] = []
            self.patterns[key].append(waste)
        
        def predict_layout(self, active, capacity):
            """Predict best layout based on learned patterns"""
            active_items = sorted(active.items())
            
            # Find similar pattern
            best_pattern = None
            best_score = float('inf')
            
            for pattern, wastes in self.patterns.items():
                avg_waste = sum(wastes) / len(wastes)
                # Pattern similarity check
                pattern_dict = dict(pattern)
                if set(pattern_dict.keys()) == set(active.keys()):
                    similarity = sum(abs(pattern_dict.get(t, 0) - (active.get(t, 0) / sum(active.values()) * capacity)) 
                                   for t in active.keys())
                    score = avg_waste + similarity * 0.1
                    if score < best_score:
                        best_score = score
                        best_pattern = pattern_dict
            
            if best_pattern:
                return best_pattern
            
            # Fallback to proportional layout
            total = sum(active.values())
            layout = {}
            for tag, qty in active.items():
                layout[tag] = max(1, int((qty / total) * capacity))
            
            # Adjust to capacity
            while sum(layout.values()) > capacity:
                max_tag = max(layout, key=layout.get)
                if layout[max_tag] > 1:
                    layout[max_tag] -= 1
                else:
                    break
            
            while sum(layout.values()) < capacity:
                max_tag = max(active, key=active.get)
                layout[max_tag] = layout.get(max_tag, 0) + 1
            
            return layout
    
    predictor = SimplePredictor()
    
    # Genetic algorithm with ML guidance
    population_size = 30
    generations = 80
    
    def create_individual_with_ml():
        remaining = demand.copy()
        plates = []
        
        for _ in range(max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            
            # Use ML predictor
            layout = predictor.predict_layout(active, capacity)
            
            sheets = max(1, min(ceil(remaining[tag] / layout.get(tag, 1)) for tag in active))
            
            plates.append({"layout": layout, "sheets": sheets})
            
            for tag, ups in layout.items():
                remaining[tag] = max(0, remaining[tag] - (ups * sheets))
        
        if any(v > 0 for v in remaining.values()) and plates:
            last = plates[-1]
            for tag in remaining:
                if remaining[tag] > 0:
                    ups = max(1, last["layout"].get(tag, 1))
                    last["sheets"] += ceil(remaining[tag] / ups)
                    remaining[tag] = 0
        
        return plates
    
    def mutate_with_ml(plates):
        new_plates = copy.deepcopy(plates)
        if new_plates:
            plate_idx = random.randint(0, len(new_plates) - 1)
            layout = new_plates[plate_idx]["layout"]
            if len(layout) >= 2:
                tags_list = list(layout.keys())
                a, b = random.sample(tags_list, 2)
                if layout[a] > 1:
                    layout[a] -= 1
                    layout[b] += 1
        
        return new_plates
    
    def crossover_plates(p1, p2):
        point = random.randint(1, min(len(p1), len(p2)) - 1)
        child = p1[:point] + p2[point:]
        
        # Fix child
        remaining = demand.copy()
        fixed = []
        for plate in child:
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            
            layout = plate["layout"].copy()
            sheets = plate["sheets"]
            
            fixed.append({"layout": layout, "sheets": sheets})
            
            for tag, ups in layout.items():
                remaining[tag] = max(0, remaining[tag] - (ups * sheets))
        
        if any(v > 0 for v in remaining.values()) and fixed:
            last = fixed[-1]
            for tag in remaining:
                if remaining[tag] > 0:
                    ups = max(1, last["layout"].get(tag, 1))
                    last["sheets"] += ceil(remaining[tag] / ups)
                    remaining[tag] = 0
        
        return fixed
    
    # Initial population
    population = []
    for _ in range(population_size):
        ind = create_individual_with_ml()
        population.append(ind)
    
    # Evolution
    for generation in range(generations):
        scored = [(calculate_waste_percent(ind, demand), ind) for ind in population]
        scored.sort(key=lambda x: x[0])
        
        # Update ML model with best individuals
        for i in range(min(5, len(scored))):
            waste, best_ind = scored[i]
            for plate in best_ind:
                predictor.learn_from_plate(plate["layout"], waste)
        
        # Selection and breeding
        elites = [copy.deepcopy(scored[i][1]) for i in range(min(5, len(scored)))]
        new_population = elites.copy()
        
        while len(new_population) < population_size:
            p1 = copy.deepcopy(random.choice(elites))
            p2 = copy.deepcopy(random.choice(elites))
            child = crossover_plates(p1, p2)
            if random.random() < 0.3:
                child = mutate_with_ml(child)
            new_population.append(child)
        
        population = new_population
    
    # Return best
    best_idx = min(range(len(population)), key=lambda i: calculate_waste_percent(population[i], demand))
    return population[best_idx] if population else v18_optimizer(demand, capacity, max_plates)




# ================================================================
# MAIN UI
# ================================================================
st.markdown("""
<div class="main-header">
    <h1>Plate Ratio Intelligence System</h1>
    <p>Intelligent Production Planning & Ratio Optimization</p>
    <p style="font-size: 0.85rem; opacity: 0.6;">AI-Powered • Fast • Accurate</p>
</div>
""", unsafe_allow_html=True)

# ================== CONFIGURATION ==================
st.markdown('<div class="card"><div class="card-title" style="text-align: center; display: block; width: 100%;">⚙️ Production Configuration</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    n = st.number_input("🏷️ Number of Items", 1, 500, 1)
with col2:
    cap = st.number_input("📀 Plate Capacity (UPS)", 1, 200, 10)
with col3:
    maxp = st.number_input("🎨 Max Plates", 1, 50, 3)
with col4:
    addon = st.number_input("📈 Add-on (%)", 0.0, 50.0, 0.0, step=0.5)

st.markdown('</div>', unsafe_allow_html=True)

# ================== ITEM QUANTITY ==================
st.markdown('<div class="card"><div class="card-title" style="text-align: center; display: block; width: 100%;">📦 Item Quantity Details</div>', unsafe_allow_html=True)

tags = []
qty = []
for i in range(n):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"<div class='tag-display'>Item {i+1}</div>", unsafe_allow_html=True)
    with col2:
        q = st.number_input(f"Quantity", min_value=0, value=0, step=100, key=f"qty_{i}", label_visibility="collapsed")
    tags.append(f"Item {i+1}")
    qty.append(q)

st.markdown('</div>', unsafe_allow_html=True)

# Data Preparation
original_qty = {t: int(q) for t, q in zip(tags, qty) if q > 0}
demand = {t: ceil(int(q) * (1 + addon / 100)) for t, q in zip(tags, qty) if q > 0}

if not PULP_AVAILABLE:
    st.markdown('<div class="warning">⚠️ PuLP library not installed. Some advanced features disabled.</div>', unsafe_allow_html=True)

# ================== GENERATE BUTTON ==================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_clicked = st.button("Generate Plans ", use_container_width=True, type="primary")

# ================== AFTER GENERATE ==================
if generate_clicked:
    if not demand:
        st.error("⚠️ Please enter at least one item with quantity greater than 0")
        st.stop()

    with st.spinner("🔄 Running all algorithms... Please wait..."):
        # ================== ALGORITHMS DEFINITION ==================
        algo_functions = {
            "V1 - Plate Ratio System": lambda: v1_optimizer(demand, cap, maxp),
            "V2 - Common Sheet Optimizer": lambda: v2_optimizer(demand, cap, maxp),
            "V3 - Smart Decimal Balancing": lambda: v3_optimizer(demand, cap, maxp),
            "V4 - Multi-Variation Optimizer": lambda: v4_optimizer(demand, cap, maxp),
            "V5 - AI Mutation Engine": lambda: v5_optimizer(demand, cap, maxp, iterations=80),
            "V6 - Integer Solver": lambda: v6_optimizer(demand, cap, maxp) if PULP_AVAILABLE else v3_optimizer(demand, cap, maxp),
            "V7 - Simulated Annealing": lambda: v7_optimizer(demand, cap, maxp, iterations=150),
            "V8 - MCTS Tree Search": lambda: v8_optimizer(demand, cap, maxp, iterations=80),
            "V9 - Hybrid Ratio & Sheet Repair": lambda: v9_optimizer(demand, cap, maxp),
            "V10 - Exhaustive Search": lambda: v10_optimizer(demand, cap, maxp),
            "V11 - Genetic Algorithm": lambda: v11_optimizer(demand, cap, maxp, population_size=30, generations=50),
            "V12 - Column Generation": lambda: v12_optimizer(demand, cap, maxp) if PULP_AVAILABLE else v3_optimizer(demand, cap, maxp),
            "V13 - Hybrid Master": lambda: v13_optimizer(demand, cap, maxp),
            "V15 - DP Repair Engine": lambda: v15_optimizer(demand, cap, maxp),
            "V16 - Plate Merge Optimizer": lambda: v16_optimizer(demand, cap, maxp),
            "V17 - AI Evolution Engine": lambda: v17_optimizer(demand, cap, maxp),
            "V18 - Global Multi-Plate Optimizer": lambda: v18_optimizer(demand, cap, maxp),
        }
        
        # OR-Tools থাকলে V19 যোগ করুন
        if ORTOOLS_AVAILABLE:
            algo_functions["V19 - CP-SAT Optimizer"] = lambda: v19_optimizer(demand, cap, maxp)
        
        # V20-V26 যোগ করুন (যদি ফাংশন ডিফাইন থাকে)
        try:
            algo_functions["V20 - PSO Optimizer"] = lambda: v20_optimizer(demand, cap, maxp)
            algo_functions["V21 - ACO Optimizer"] = lambda: v21_optimizer(demand, cap, maxp)
            algo_functions["V22 - Q-Learning Optimizer"] = lambda: v22_optimizer(demand, cap, maxp)
            algo_functions["V23 - Branch & Bound"] = lambda: v23_optimizer(demand, cap, maxp)
            algo_functions["V24 - Differential Evolution"] = lambda: v24_optimizer(demand, cap, maxp)
            algo_functions["V25 - Pareto Optimizer"] = lambda: v25_optimizer(demand, cap, maxp)
            algo_functions["V26 - NN Predictor"] = lambda: v26_optimizer(demand, cap, maxp)
        except NameError:
            pass  # V20-V26 ডিফাইন না থাকলে এগিয়ে যান
        
        # Problematic algorithms for single plate
        problematic_for_single_plate = {
            "V11 - Genetic Algorithm", "V16 - Plate Merge Optimizer", 
            "V17 - AI Evolution Engine", "V19 - CP-SAT Optimizer",
            "V20 - PSO Optimizer", "V21 - ACO Optimizer", "V22 - Q-Learning Optimizer",
            "V23 - Branch & Bound", "V24 - Differential Evolution",
            "V25 - Pareto Optimizer", "V26 - NN Predictor"
        }
        
        # ========== MAIN LOOP ==========
        results = {}
        
        for algo_name, func in algo_functions.items():
            try:
                if maxp == 1 and algo_name in problematic_for_single_plate:
                    results[algo_name] = v3_optimizer(demand, cap, maxp)
                else:
                    results[algo_name] = func()
            except Exception as e:
                results[algo_name] = v3_optimizer(demand, cap, maxp)
        
        # Ensure all have results
        for algo_name in list(results.keys()):
            if not results.get(algo_name):
                results[algo_name] = v3_optimizer(demand, cap, maxp)
        
        # Filter out invalid results
        valid_results = {}
        for algo_name, plates in results.items():
            if plates:
                waste = calculate_waste_percent(plates, demand)
                if 0 <= waste <= 100:
                    valid_results[algo_name] = plates
                else:
                    valid_results[algo_name] = v3_optimizer(demand, cap, maxp)
            else:
                valid_results[algo_name] = v3_optimizer(demand, cap, maxp)
        
        results = valid_results
        
        # Update comparison
        comparison_data = []
        for algo_name, plates in results.items():
            waste = calculate_waste_percent(plates, demand)
            comparison_data.append({
                "Algorithm": algo_name,
                "Waste %": waste,
                "Total Plates": len(plates),
                "Total Sheets": sum(p.get("sheets", 0) for p in plates),
                "Status": "✅ Success" if plates else "❌ Failed"
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values("Waste %")
        best_algo = comparison_df.iloc[0]["Algorithm"]
        best_waste = comparison_df.iloc[0]["Waste %"]
        
        # Store in session state
        st.session_state['results'] = results
        st.session_state['comparison_df'] = comparison_df
        st.session_state['best_algo'] = best_algo
        st.session_state['best_waste'] = best_waste
        st.session_state['demand'] = demand
        st.session_state['original_qty'] = original_qty
        
        # ====================== UI OUTPUT ======================
        st.markdown(f"""
        <div class="best-algo">
            <div class="metric-value">🏆 BEST ALGORITHM: {best_algo}</div>
            <div class="metric-label">Waste Percentage: {best_waste}%</div>
            <div class="metric-label">✨ Total Algorithms Tested: {len(results)} ✨</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Best Algorithm Report
        st.markdown("## 📋 Best Algorithm Report")
        best_plates = results[best_algo]
        
        if best_plates:
            try:
                st.markdown("### 📊 Production Summary")
                full_df = build_full_summary(best_plates, demand, original_qty)
                if not full_df.empty:
                    st.dataframe(full_df, use_container_width=True, height=380)
                
                st.markdown("### 🧾 Plate Configuration Details")
                plate_rows = []
                total_sheets_sum = 0
                total_ups_sum = 0
                
                for idx, p in enumerate(best_plates, 1):
                    if p and "layout" in p:
                        total_ups = sum(p["layout"].values())
                        plate_name_str = p.get("name", f"Plate {idx}")
                        plate_rows.append({
                            "SL": idx,
                            "Plate ID": plate_name_str,
                            "Sheets Required": p.get("sheets", 0),
                            "Total UPS": total_ups,
                        })
                        total_sheets_sum += p.get("sheets", 0)
                        total_ups_sum += total_ups
                
                plate_rows.append({
                    "SL": "📊",
                    "Plate ID": "TOTAL",
                    "Sheets Required": total_sheets_sum,
                    "Total UPS": total_ups_sum,
                })
                
                plate_details_df = pd.DataFrame(plate_rows)
                st.dataframe(plate_details_df, use_container_width=True)
                
                # Download buttons
                st.markdown("### 📥 Download Best Report")
                col1, col2 = st.columns(2)
                
                with col1:
                    bio_excel = BytesIO()
                    with pd.ExcelWriter(bio_excel, engine="openpyxl") as writer:
                        full_df.to_excel(writer, sheet_name="Summary", index=False)
                        plate_details_df.to_excel(writer, sheet_name="Plate Details", index=False)
                        comparison_df.to_excel(writer, sheet_name="Comparison", index=False)
                    bio_excel.seek(0)
                    st.download_button(
                        "📊 Download Excel", 
                        bio_excel,
                        f"BEST_{best_algo.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        use_container_width=True
                    )
                
                with col2:
                    if REPORTLAB_AVAILABLE:
                        pdf_buffer = generate_pdf_report(best_plates, demand, original_qty, best_algo, best_waste)
                        if pdf_buffer:
                            st.download_button(
                                "📄 Download PDF", 
                                pdf_buffer,
                                f"BEST_{best_algo.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf", 
                                use_container_width=True
                            )
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                st.info("Showing comparison table instead...")
        
        # Algorithm Comparison
        st.markdown("---")
        st.markdown("## 📊 Algorithm Comparison (Sorted by Waste %)")
        
        styled_df = comparison_df.style.apply(
            lambda row: ['background-color: #2e7d32; color: white'] * len(row) 
            if row["Algorithm"] == best_algo else [''] * len(row), axis=1
        ).format({"Waste %": "{:.2f}%"})
        
        st.dataframe(styled_df, use_container_width=True, height=460)

# ====================== VIEW ANY ALGORITHM REPORT ======================
st.markdown("---")
st.markdown("## 🔍 View others Report")

if 'results' in st.session_state and st.session_state['results']:
    algo_list = list(st.session_state['results'].keys())
    
    default_index = 0
    if st.session_state.get('best_algo') in algo_list:
        default_index = algo_list.index(st.session_state['best_algo'])

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_algo = st.selectbox(
            "👇 যে অ্যালগরিদমের রিপোর্ট দেখতে চান:",
            options=algo_list,
            index=default_index,
            key="independent_algo_selector"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        view_button = st.button("📋 Report দেখুন", use_container_width=True, type="primary")

    if view_button:
        selected_plates = st.session_state['results'].get(selected_algo)
        
        if selected_plates:
            st.markdown(f"### 📊 Production Summary — **{selected_algo}**")
            
            full_df = build_full_summary(selected_plates, 
                                       st.session_state['demand'], 
                                       st.session_state['original_qty'])
            st.dataframe(full_df, use_container_width=True, height=400)

            st.markdown("### 🧾 Plate Configuration Details")
            plate_rows = []
            total_sheets = 0
            total_ups = 0
            for idx, p in enumerate(selected_plates, 1):
                ups_sum = sum(p["layout"].values())
                plate_rows.append({
                    "SL": idx,
                    "Plate ID": p["name"],
                    "Sheets Required": p["sheets"],
                    "Total UPS": ups_sum,
                })
                total_sheets += p["sheets"]
                total_ups += ups_sum

            plate_rows.append({
                "SL": "📊",
                "Plate ID": "TOTAL",
                "Sheets Required": total_sheets,
                "Total UPS": total_ups,
            })

            plate_df = pd.DataFrame(plate_rows)
            st.dataframe(plate_df, use_container_width=True)

            waste = calculate_waste_percent(selected_plates, st.session_state['demand'])
            st.success(f"**Waste: {waste}%** | Plates: {len(selected_plates)} | Total Sheets: {total_sheets}")

        else:
            st.error(f"❌ {selected_algo} এর রিপোর্ট পাওয়া যায়নি।")

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; margin-top: 3rem; border-top: 2px solid rgba(102,126,234,0.3); background: rgba(255,255,255,0.02); border-radius: 20px;">
    <p style="color: rgba(255,255,255,0.6); font-size: 0.85rem; margin: 0;">
        © 2025 Plate Ratio System | Version 26
    </p>
    <p style="color: rgba(255,255,255,0.5); font-size: 0.8rem; margin: 8px 0;">
        Enterprise Production Optimization Framework
    </p>
    <p style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 0.85rem; font-weight: 600; margin: 10px 0 0 0;">
        ✨ Developed by Ovi | All Rights Reserved ✨
    </p>
</div>
""", unsafe_allow_html=True)
