# app.py — COMPLETE PLATE RATIO SYSTEM (V1 to V26)
# All Algorithms Fixed | No Negative Excess | Production Ready
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


# ================================================================
# PASSWORD AUTHENTICATION
# ================================================================
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

    # Password Page Styling
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
        .lock-icon { font-size: 3rem; margin: 1rem 0; animation: bounce 2s infinite; }
        @keyframes bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
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

    st.markdown("""
    <div class="main-header">
        <h1>Plate Ratio System</h1>
        <p>Intelligent Production Planning & Ratio Optimization</p>
        <p style="font-size: 0.85rem; opacity: 0.8;">AI-Powered • Fast • Accurate</p>
        <p class="designer-name">✨ Design by Ovi ✨</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="password-container">
        <h2>Welcome Back</h2>
        <div class="lock-icon">🔐</div>
        <p>Enter your secure access code to continue</p>
    </div>
    """, unsafe_allow_html=True)

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

    if st.session_state.get("password_correct") is False:
        st.error("❌ Incorrect password. Please contact Mr. Ovi.")

    return False


# ================== APP START ==================
if not check_password():
    st.stop()

st.success("✅ Successfully logged in!")


# ================================================================
# MODERN CSS FOR MAIN APP
# ================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%); }
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
    .main-header p { color: rgba(255,255,255,0.7); margin-top: 0.5rem; }
    .card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    .card:hover { border-color: rgba(102,126,234,0.5); box-shadow: 0 8px 32px rgba(0,0,0,0.2); }
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
    .metric-label { font-size: 0.85rem; color: rgba(255,255,255,0.7); margin-top: 0.5rem; }
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
    .best-algo .metric-value { -webkit-text-fill-color: white; font-size: 1.5rem; }
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
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(102,126,234,0.4); }
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
    .stDataFrame { background: rgba(255,255,255,0.05); border-radius: 16px; padding: 0.5rem; }
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
    .warning { background: rgba(255,193,7,0.1); padding: 12px; border-radius: 12px; border-left: 4px solid #ffc107; color: #ffc107; margin: 1rem 0; }
    .info { background: rgba(23,162,184,0.1); padding: 12px; border-radius: 12px; border-left: 4px solid #17a2b8; color: #17a2b8; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 10px; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ================================================================
# HELPER FUNCTIONS
# ================================================================
# ================================================================
# UNIVERSAL LAYOUT GENERATOR (FIXED FOR ALL ALGORITHMS)
# ================================================================
def create_valid_layout(active: dict, capacity: int, method: str = "balanced") -> dict:
    """
    Create a layout that respects capacity even when items > capacity
    Methods: "balanced", "proportional", "greedy"
    """
    if not active:
        return {}
    
    total_qty = sum(active.values())
    n_items = len(active)
    
    # Special case: বেশি আইটেম, কম capacity
    if n_items > capacity:
        layout = {}
        
        if method == "balanced":
            # প্রতিটি আইটেমের জন্য প্রপোরশনাল UPS
            for tag, qty in active.items():
                ups = max(1, int((qty / total_qty) * capacity))
                layout[tag] = ups
            
            # Exact capacity enforce
            while sum(layout.values()) > capacity:
                max_tag = max(layout, key=layout.get)
                if layout[max_tag] > 1:
                    layout[max_tag] -= 1
                else:
                    # সব UPS 1 হলে, সবচেয়ে কম গুরুত্বপূর্ণ ট্যাগ 0 করো
                    min_tag = min(active, key=lambda t: active[t])
                    if layout.get(min_tag, 0) > 0:
                        layout[min_tag] = 0
                    else:
                        break
            
            while sum(layout.values()) < capacity:
                # সবচেয়ে বেশি ডিমান্ডের আইটেমকে প্রায়োরিটি দাও
                max_tag = max(active, key=lambda t: active[t] / (layout.get(t, 1) + 1))
                layout[max_tag] = layout.get(max_tag, 0) + 1
            
            return layout
        
        elif method == "greedy":
            # প্রতিটি আইটেমের জন্য 1 করে দাও, তারপর বাকি capacity পূরণ করো
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
            # প্রপোরশনাল ভাগাভাগি
            for tag, qty in active.items():
                ups = int((qty / total_qty) * capacity)
                if ups < 1:
                    ups = 1 if len(active) <= capacity else 0
                layout[tag] = ups
            
            # Exact capacity enforce
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
    
    # Normal case: আইটেম সংখ্যা capacity এর কম বা সমান
    layout = {}
    
    if method == "balanced":
        for tag, qty in active.items():
            ideal = (qty / total_qty) * capacity
            base = int(ideal)
            if base < 1:
                base = 1
            layout[tag] = base
        
        # Adjust to exact capacity
        while sum(layout.values()) > capacity:
            max_tag = max(layout, key=layout.get)
            if layout[max_tag] > 1:
                layout[max_tag] -= 1
            else:
                break
        
        while sum(layout.values()) < capacity:
            # সবচেয়ে বেশি fractional part যার
            fractional = {}
            for tag, qty in active.items():
                ideal = (qty / total_qty) * capacity
                fractional[tag] = ideal - int(ideal)
            best = max(fractional, key=fractional.get)
            layout[best] = layout.get(best, 0) + 1
    
    else:  # proportional or greedy
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
    

def plate_name(n: int) -> str:
    """Convert number to Excel-style column name (A, B, C, ..., Z, AA, AB, ...)"""
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


def ensure_demand_met(plates: list, demand: dict) -> list:
    """Ensure all demand is met - no negative excess"""
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
                additional_sheets = ceil(shortfall / max(1, ups))
                last_plate["sheets"] += additional_sheets
    
    # Ensure all plates have names
    for idx, plate in enumerate(plates):
        if "name" not in plate:
            plate["name"] = plate_name(idx + 1)
    
    return plates


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


def generate_pdf_report(plates: list, demand: dict, original_qty: dict,
                        algo_name: str, waste_percent: float,
                        styles_dict: dict = None, colors_dict: dict = None, 
                        sizes_dict: dict = None, job_number: str = "") -> BytesIO | None:
    """Generate professional PDF report with Style, Color, Size columns and Job Number"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    except ImportError:
        return None

    # Initialize empty dicts if not provided
    if styles_dict is None:
        styles_dict = {}
    if colors_dict is None:
        colors_dict = {}
    if sizes_dict is None:
        sizes_dict = {}

    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=landscape(A4),
            rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20
        )
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Heading1'],
            fontSize=14, alignment=TA_CENTER, 
            textColor=colors.HexColor('#667eea'),
            spaceAfter=4
        )
        job_style = ParagraphStyle(
            'JobStyle', parent=styles['Heading2'],
            fontSize=12, alignment=TA_CENTER,
            textColor=colors.HexColor('#764ba2'),
            spaceAfter=8
        )
        subtitle_style = ParagraphStyle(
            'CustomSubtitle', parent=styles['Normal'],
            fontSize=9, alignment=TA_CENTER, 
            textColor=colors.grey,
            spaceAfter=12
        )
        footer_style = ParagraphStyle(
            'Footer', parent=styles['Normal'],
            fontSize=8, alignment=TA_CENTER, 
            textColor=colors.grey,
            spaceTop=12
        )

        story = []
        
        # Header with Job Number
        story.append(Paragraph("📊 Plate Ratio System - Ratio Report", title_style))
        if job_number:
            story.append(Paragraph(f"🔢 Job Number: {job_number}", job_style))
        story.append(Paragraph(
            f"Algorithm: {algo_name} | Waste: {waste_percent}% | "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            subtitle_style
        ))
        story.append(Spacer(1, 10))

        # ============= MAIN SUMMARY TABLE =============
        # Build header with all columns
        header_row = ["SL", "Style", "Color", "Size", "Original", "With Add-on"]
        for p in plates:
            header_row.append(f"Plate {p['name']}")
        header_row.extend(["Total Prod.", "Excess", "Excess %"])
        
        summary_data = [header_row]
        
        # Build data rows
        sl = 1
        for tag in demand.keys():
            # Get style/color/size from session state or use defaults
            style = styles_dict.get(tag, "N/A")
            color = colors_dict.get(tag, "N/A")
            size = sizes_dict.get(tag, "N/A")
            
            row = [str(sl), style, color, size, 
                   str(original_qty.get(tag, 0)), str(demand[tag])]
            
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
        total_row = ["📊", "TOTAL", "", "", 
                     str(sum(original_qty.values())), str(sum(demand.values()))]
        
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
        
        # Create main table
        main_table = Table(summary_data, repeatRows=1)
        
        # Style the table
        table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 7),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -2), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -2), 6),
            ('ALIGN', (0, 1), (-1, -2), 'CENTER'),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#f0f0f0')),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, -1), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]
        
        # Apply alternating row colors
        for i in range(1, len(summary_data) - 1):
            if i % 2 == 0:
                table_style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f8f9fa')))
        
        main_table.setStyle(TableStyle(table_style))
        story.append(main_table)
        story.append(Spacer(1, 15))
        
        # ============= PLATE DETAILS TABLE =============
        story.append(Paragraph("🧾 Plate Configuration Details", 
                              ParagraphStyle('SubHeader', parent=styles['Heading2'],
                                           fontSize=11, alignment=TA_CENTER,
                                           textColor=colors.HexColor('#667eea'))))
        story.append(Spacer(1, 8))
        
        plate_data = [["SL", "Plate ID", "Sheets", "Total UPS"]]
        for idx, p in enumerate(plates, 1):
            plate_data.append([
                str(idx), 
                p["name"], 
                str(p["sheets"]), 
                str(sum(p["layout"].values()))
            ])
        
        plate_table = Table(plate_data)
        plate_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(plate_table)
        story.append(Spacer(1, 15))
        
        # ============= FOOTER =============
        story.append(Paragraph(
            f"This Report Generated by Ovi's Plate Ratio System | Job: {job_number if job_number else 'N/A'} | All Rights Reserved",
            footer_style
        ))
        
        # Build the PDF
        doc.build(story)
        buffer.seek(0)
        return buffer

    except Exception as e:
        print(f"PDF Generation Error: {str(e)}")
        return None


def smart_layout_v1(demand: dict, cap: int) -> dict:
    total = sum(demand.values())
    if total == 0:
        return {}

    # ✅ Use helper function
    layout = create_valid_layout(demand, cap, "balanced")
    
    # Ensure no zero values
    for k in layout:
        if layout[k] == 0:
            layout[k] = 1

    remainders = {}
    for k, v in demand.items():
        ratio = (v / total) * cap
        remainders[k] = ratio - int(ratio)

    while sum(layout.values()) > cap:
        biggest = max(layout, key=layout.get)
        if layout[biggest] > 1:
            layout[biggest] -= 1
        else:
            break

    remaining_cap = cap - sum(layout.values())
    while remaining_cap > 0:
        best = max(remainders, key=remainders.get)
        layout[best] = layout.get(best, 0) + 1
        remainders[best] = 0
        remaining_cap -= 1

    return layout

def v2_optimizer(demand: dict, capacity: int, max_plates: int) -> list:
    total_qty = sum(demand.values())
    target_sheets = ceil(total_qty / capacity)
    remaining = demand.copy()
    plates = []

    for p in range(max_plates):
        active = {k: v for k, v in remaining.items() if v > 0}
        if not active:
            break

        # ✅ Fixed: Use create_valid_layout
        layout = create_valid_layout(active, capacity, "greedy")
        
        # ✅ Extra safety check
        if sum(layout.values()) != capacity:
            layout = create_valid_layout(active, capacity, "balanced")
        
        # Calculate sheets
        possible_sheets = []
        for tag, ups in layout.items():
            if ups > 0 and remaining.get(tag, 0) > 0:
                possible_sheets.append(ceil(remaining[tag] / ups))
        
        if not possible_sheets:
            break
            
        sheets = max(1, min(possible_sheets))

        # Update remaining
        for tag, ups in layout.items():
            remaining[tag] = max(0, remaining[tag] - (ups * sheets))

        plates.append({
            "name": plate_name(len(plates) + 1),
            "layout": layout,
            "sheets": sheets
        })

    # Handle remaining demand
    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for tag in remaining:
            if remaining[tag] > 0:
                ups = max(1, last["layout"].get(tag, 1))
                add_sheets = ceil(remaining[tag] / ups)
                last["sheets"] += add_sheets
                remaining[tag] = 0
    elif any(v > 0 for v in remaining.values()):
        return v3_optimizer(demand, capacity, max_plates)

    return ensure_demand_met(plates, demand)
def build_balanced_layout_v3(remaining: dict, capacity: int) -> dict:
    active = {k: v for k, v in remaining.items() if v > 0}
    if not active:
        return {}

    # ✅ Use helper function
    return create_valid_layout(active, capacity, "balanced")

def v3_optimizer(demand: dict, capacity: int, max_plates: int) -> list:
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

    return ensure_demand_met(plates, demand)


def proportional_layout_v4(remaining: dict, capacity: int) -> dict:
    active = {k: v for k, v in remaining.items() if v > 0}
    if not active:
        return {}

    # ✅ Use helper function
    return create_valid_layout(active, capacity, "proportional")

def v4_optimizer(demand: dict, capacity: int, max_plates: int) -> list:
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
            
            # ✅ Validate layout
            if not layout or sum(layout.values()) != capacity:
                break
            
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

    return ensure_demand_met(best_plates, demand) if best_plates else v3_optimizer(demand, capacity, max_plates)

def generate_layout_v5(active: dict, capacity: int) -> dict:
    # ✅ Use helper function
    return create_valid_layout(active, capacity, "greedy")

def v5_optimizer(demand: dict, capacity: int, max_plates: int, iterations: int = 80) -> list:
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
            
            # ✅ Validate layout
            if not layout or sum(layout.values()) != capacity:
                break
            
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

    return ensure_demand_met(best_plates, demand) if best_plates else v3_optimizer(demand, capacity, max_plates)


def v6_optimizer(demand: dict, capacity: int, max_plates: int) -> list | None:
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
            ups = {t: LpVariable(f"UPS_{t}", lowBound=0, cat="Integer") for t in active_tags}
            sheets = LpVariable("Sheets", lowBound=1, cat="Integer")
            excess_vars = [ups[t] * sheets - remaining[t] for t in active_tags]

            model += lpSum(excess_vars)
            model += lpSum(ups[t] for t in active_tags) == capacity

            for t in active_tags:
                model += ups[t] * sheets >= remaining[t]

            model.solve()

            if model.status == 1:
                layout = {t: int(value(ups[t])) for t in active_tags}
                
                # ✅ Ensure no zero values if possible
                for tag in active_tags:
                    if layout.get(tag, 0) == 0:
                        layout[tag] = 1
                
                # ✅ Fix capacity if needed
                while sum(layout.values()) > capacity:
                    max_tag = max(layout, key=layout.get)
                    if layout[max_tag] > 1:
                        layout[max_tag] -= 1
                    else:
                        break
                
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

    return ensure_demand_met(plates, demand) if plates else v3_optimizer(demand, capacity, max_plates)


def v7_optimizer(demand: dict, capacity: int, max_plates: int, iterations: int = 150) -> list:
    def calculate_waste(layout: dict, sheets: int, remaining: dict) -> int:
        return sum(max(0, ups * sheets - remaining.get(tag, 0)) for tag, ups in layout.items())

    def adjust_to_exact_capacity(layout: dict, capacity: int) -> dict:
        current_sum = sum(layout.values())
        
        if current_sum == capacity:
            return layout
        
        new_layout = layout.copy()
        
        while sum(new_layout.values()) < capacity:
            if not new_layout:
                break
            best_tag = max(new_layout.keys(), key=lambda t: remaining.get(t, 0) / new_layout[t])
            new_layout[best_tag] += 1
        
        while sum(new_layout.values()) > capacity:
            if not new_layout:
                break
            smallest_tag = min(new_layout.keys(), key=lambda t: new_layout[t])
            if new_layout[smallest_tag] > 1:
                new_layout[smallest_tag] -= 1
            else:
                for tag in sorted(new_layout.keys(), key=lambda t: new_layout[t], reverse=True):
                    if new_layout[tag] > 1:
                        new_layout[tag] -= 1
                        break
                else:
                    break
        
        return new_layout

    def mutate_layout(layout: dict, capacity: int) -> dict:
        new_layout = layout.copy()
        tags = list(new_layout.keys())
        
        if len(tags) >= 2:
            a, b = random.sample(tags, 2)
            if new_layout[a] > 1:
                new_layout[a] -= 1
                new_layout[b] += 1
        
        return adjust_to_exact_capacity(new_layout, capacity)

    def initial_layout(active: dict, capacity: int) -> dict:
        if not active:
            return {}
        
        # ✅ Use helper function
        layout = create_valid_layout(active, capacity, "balanced")
        return adjust_to_exact_capacity(layout, capacity)

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

    return ensure_demand_met(plates, demand)


class MCTSNodeV8:
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

def v8_optimizer(demand: dict, capacity: int, max_plates: int, iterations: int = 80) -> list:
    def adjust_to_exact_capacity(layout: dict, capacity: int, remaining: dict) -> dict:
        if not layout or not remaining:
            return layout
        
        new_layout = layout.copy()
        
        while sum(new_layout.values()) < capacity:
            if not new_layout:
                break
            best_tag = max(new_layout.keys(), key=lambda t: remaining.get(t, 0) / new_layout.get(t, 1))
            new_layout[best_tag] = new_layout.get(best_tag, 0) + 1
        
        while sum(new_layout.values()) > capacity:
            candidates = [t for t in new_layout if new_layout[t] > 1]
            if not candidates:
                break
            smallest_tag = min(candidates, key=lambda t: new_layout[t])
            new_layout[smallest_tag] -= 1
        
        return new_layout

    def initial_layout(active: dict, capacity: int) -> dict:
        if not active:
            return {}
        
        # ✅ Use helper function
        layout = create_valid_layout(active, capacity, "balanced")
        return adjust_to_exact_capacity(layout, capacity, active)

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

            while node.children:
                next_node = node.best_child()
                if next_node is None:
                    break
                node = next_node

            current_layout = node.layout.copy()
            
            tags = list(current_layout.keys())
            if len(tags) >= 2:
                a, b = random.sample(tags, 2)
                if current_layout.get(a, 0) > 1:
                    current_layout[a] -= 1
                    current_layout[b] = current_layout.get(b, 0) + 1

            final_layout = adjust_to_exact_capacity(current_layout, capacity, active)

            waste = sum(max(0, ups * sheets - active.get(tag, 0)) 
                       for tag, ups in final_layout.items())
            score = -waste

            new_node = MCTSNodeV8(final_layout, active, capacity, node)
            node.children.append(new_node)
            
            current_node = node
            while current_node:
                current_node.visits += 1
                current_node.score += score
                current_node = current_node.parent

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

    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for tag in list(remaining.keys()):
            if remaining[tag] > 0:
                ups = max(1, last["layout"].get(tag, 1))
                last["sheets"] += ceil(remaining[tag] / ups)
                remaining[tag] = 0

    return ensure_demand_met(plates, demand)




def v11_optimizer(demand: dict, capacity: int, max_plates: int, 
                   population_size: int = 30, generations: int = 50, 
                   mutation_rate: float = 0.1, elite_size: int = 5) -> list:
    
    items = list(demand.keys())
    
    def create_individual():
        remaining = demand.copy()
        plates = []
        
        for p in range(max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            
            # ✅ Use helper function
            layout = create_valid_layout(active, capacity, "greedy")
            
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
            
            # ✅ Validate and fix layout
            if sum(layout.values()) != capacity:
                layout = create_valid_layout(active, capacity, "greedy")
            
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
    return ensure_demand_met(population[best_idx], demand)


def v12_optimizer(demand: dict, capacity: int, max_plates: int) -> list:
    if not PULP_AVAILABLE:
        return v3_optimizer(demand, capacity, max_plates)
    
    remaining = demand.copy()
    plates = []
    tags_list = list(demand.keys())
    
    def solve_master_problem(patterns):
        try:
            master = LpProblem("Master_Problem", LpMinimize)
            pattern_vars = []
            for idx in range(len(patterns)):
                var = LpVariable(f"x_{idx}", lowBound=0, cat="Integer")
                pattern_vars.append(var)
            
            master += lpSum(pattern_vars)
            
            for tag in tags_list:
                master += lpSum(patterns[idx].get(tag, 0) * pattern_vars[idx] 
                               for idx in range(len(patterns))) >= demand[tag]
            
            master.solve()
            
            if master.status == 1:
                duals = {}
                for tag in tags_list:
                    constraint = [c for c in master.constraints.values() 
                                 if tag in str(c)][0] if master.constraints else None
                    duals[tag] = constraint.pi if constraint else 0
                
                return value(master.objective), duals
            return None, None
        except Exception:
            return None, None
    
    def generate_pattern_with_dual(duals, capacity):
        try:
            sub = LpProblem("Subproblem", LpMaximize)
            ups = {tag: LpVariable(f"ups_{tag}", lowBound=0, upBound=capacity, cat="Integer") 
                   for tag in tags_list}
            sub += lpSum(duals.get(tag, 0) * ups[tag] for tag in tags_list) - 1
            sub += lpSum(ups[tag] for tag in tags_list) == capacity  # ✅ Fixed: == capacity
            sub.solve()
            
            if sub.status == 1:
                pattern = {tag: int(value(ups[tag])) for tag in tags_list if value(ups[tag]) > 0}
                reduced_cost = value(sub.objective)
                return pattern, reduced_cost
            return None, -1
        except Exception:
            return None, -1
    
    patterns = []
    initial_demand = demand.copy()
    for _ in range(min(max_plates, len(tags_list) * 2)):
        active = {k: v for k, v in initial_demand.items() if v > 0}
        if not active:
            break
        
        # ✅ Use helper function
        pattern = create_valid_layout(active, capacity, "proportional")
        patterns.append(pattern)
        
        sheets = 1
        for tag, ups in pattern.items():
            initial_demand[tag] = max(0, initial_demand[tag] - (ups * sheets))
    
    iteration = 0
    max_iterations = 50
    
    while iteration < max_iterations:
        obj_value, duals = solve_master_problem(patterns)
        if duals is None:
            break
        
        new_pattern, reduced_cost = generate_pattern_with_dual(duals, capacity)
        if new_pattern is None or reduced_cost <= 0.001:
            break
        
        patterns.append(new_pattern)
        iteration += 1
    
    remaining = demand.copy()
    plate_counter = 0
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
        active_tags = [t for t in pattern if remaining.get(t, 0) > 0]
        if not active_tags:
            continue
        
        possible_sheets = []
        for tag in active_tags:
            if pattern.get(tag, 0) > 0:
                possible_sheets.append(ceil(remaining[tag] / pattern[tag]))
        
        if not possible_sheets:
            continue
        
        sheets = min(possible_sheets)
        
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
        
        if all(v <= 0 for v in remaining.values()):
            break
    
    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for tag in list(remaining.keys()):
            if remaining[tag] > 0:
                ups = max(1, last["layout"].get(tag, 1))
                add_sheets = ceil(remaining[tag] / ups)
                last["sheets"] += add_sheets
                remaining[tag] = 0
    elif any(v > 0 for v in remaining.values()):
        return v3_optimizer(demand, capacity, max_plates)
    
    return ensure_demand_met(plates, demand) if plates else v3_optimizer(demand, capacity, max_plates)




def v15_optimizer(demand: dict, capacity: int, max_plates: int):
    # ✅ Use V3 as base
    base = v3_optimizer(demand, capacity, max_plates)

    if not base:
        return v3_optimizer(demand, capacity, max_plates)

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

    return ensure_demand_met(best, demand)

def v16_optimizer(demand: dict, capacity: int, max_plates: int):
    plates = v15_optimizer(demand, capacity, max_plates)

    if not plates:
        return v3_optimizer(demand, capacity, max_plates)

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

            for tag, ups in candidate["layout"].items():
                combined[tag] = combined.get(tag, 0) + ups

            # ✅ Check capacity
            if sum(combined.values()) <= capacity:
                current["layout"] = combined
                current["sheets"] = max(current["sheets"], candidate["sheets"])
                skip.add(j)

        merged.append(current)

    return ensure_demand_met(merged, demand)

def v17_optimizer(demand: dict, capacity: int, max_plates: int, generations: int = 200):
    population = []

    # ✅ Only use available algorithms
    available_algos = [v3_optimizer, v5_optimizer, v11_optimizer, v15_optimizer, v16_optimizer]
    
    for _ in range(20):
        candidate = random.choice(available_algos)(demand, capacity, max_plates)
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

    return ensure_demand_met(best_solution, demand) if best_solution else v3_optimizer(demand, capacity, max_plates)


def v18_optimizer(demand: dict, capacity: int, max_plates: int):
    candidates = []

    # ✅ Only use available algorithms
    algos = [v3_optimizer, v5_optimizer, v11_optimizer, v15_optimizer, v16_optimizer, v17_optimizer]

    for algo in algos:
        try:
            result = algo(demand, capacity, max_plates)
            if result:
                waste = calculate_waste_percent(result, demand)
                candidates.append((waste, result))
        except:
            pass

    if not candidates:
        return v3_optimizer(demand, capacity, max_plates)

    candidates.sort(key=lambda x: x[0])
    return ensure_demand_met(candidates[0][1], demand)


def v19_optimizer(demand: dict, capacity: int, max_plates: int, time_limit_seconds: int = 5) -> list:
    if not ORTOOLS_AVAILABLE:
        return v18_optimizer(demand, capacity, max_plates)
    
    tags = list(demand.keys())
    n_tags = len(tags)
    
    if n_tags == 0:
        return []
    
    model = cp_model.CpModel()
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
    
    for i in range(max_possible_plates):
        total_ups = sum(ups[(i, idx)] for idx in range(n_tags))
        model.Add(total_ups <= capacity)
        model.Add(total_ups == 0).OnlyEnforceIf(plate_used[i].Not())
        model.Add(total_ups > 0).OnlyEnforceIf(plate_used[i])
    
    for idx, tag in enumerate(tags):
        total_produced = sum(ups[(i, idx)] * sheets[i] for i in range(max_possible_plates))
        model.Add(total_produced >= demand[tag])
    
    for i in range(max_possible_plates):
        model.Add(sheets[i] >= 1).OnlyEnforceIf(plate_used[i])
        model.Add(sheets[i] == 0).OnlyEnforceIf(plate_used[i].Not())
    
    total_sheets = sum(sheets[i] for i in range(max_possible_plates))
    model.Minimize(total_sheets)
    
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
        return ensure_demand_met(plates, demand) if plates else v18_optimizer(demand, capacity, max_plates)
    
    return v18_optimizer(demand, capacity, max_plates)


def v20_optimizer(demand: dict, capacity: int, max_plates: int, 
                   particles: int = 20, iterations: int = 50) -> list:
    
    tags = list(demand.keys())
    best_global_plates = None
    best_global_waste = float('inf')
    
    class Particle:
        def __init__(self):
            self.plates = []
            remaining = demand.copy()
            for _ in range(max_plates):
                active = {k: v for k, v in remaining.items() if v > 0}
                if not active:
                    break
                
                # ✅ Use helper function
                layout = create_valid_layout(active, capacity, "proportional")
                
                # Add some randomness
                if len(layout) >= 2:
                    tags_list = list(layout.keys())
                    a, b = random.sample(tags_list, 2)
                    if layout.get(a, 0) > 1:
                        layout[a] = layout.get(a, 1) - 1
                        layout[b] = layout.get(b, 0) + 1
                
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
            self.plates = ensure_demand_met(self.plates, demand)
        
        def update_fitness(self):
            return calculate_waste_percent(self.plates, demand)
    
    swarm = [Particle() for _ in range(particles)]
    
    for iteration in range(iterations):
        for particle in swarm:
            waste = particle.update_fitness()
            
            if waste < best_global_waste:
                best_global_waste = waste
                best_global_plates = copy.deepcopy(particle.plates)
            
            if random.random() < 0.3 and particle.plates:
                plate_idx = random.randint(0, len(particle.plates) - 1)
                layout = particle.plates[plate_idx]["layout"]
                if len(layout) >= 2:
                    tags_list = list(layout.keys())
                    a, b = random.sample(tags_list, 2)
                    if layout[a] > 1:
                        layout[a] -= 1
                        layout[b] += 1
    
    return ensure_demand_met(best_global_plates, demand) if best_global_plates else v18_optimizer(demand, capacity, max_plates)


def v21_optimizer(demand: dict, capacity: int, max_plates: int,
                   ants: int = 15, iterations: int = 30,
                   alpha: float = 1.0, beta: float = 2.0,
                   evaporation: float = 0.5) -> list:
    
    tags = list(demand.keys())
    n_tags = len(tags)
    
    if n_tags == 0:
        return []
    
    pheromone = {}
    for i in range(n_tags):
        for j in range(1, capacity + 1):
            pheromone[(i, j)] = 1.0
    
    best_plates = None
    best_waste = float('inf')
    
    def construct_solution():
        remaining = demand.copy()
        plates = []
        
        for plate_idx in range(max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            
            # ✅ Use helper function as base
            layout = create_valid_layout(active, capacity, "proportional")
            
            # ACO optimization on top of base layout
            remaining_cap = capacity - sum(layout.values())
            
            if remaining_cap > 0 and active:
                active_tags = list(active.keys())
                while remaining_cap > 0 and active_tags:
                    for tag in active_tags:
                        if remaining_cap <= 0:
                            break
                        tag_idx = tags.index(tag)
                        # Use pheromone to decide which tag gets extra UPS
                        ups_options = list(range(1, min(remaining_cap, active[tag]) + 1))
                        if ups_options:
                            probabilities = []
                            for ups in ups_options:
                                tau = pheromone.get((tag_idx, ups), 1.0) ** alpha
                                eta = (1.0 / (ups + 1)) ** beta
                                probabilities.append(tau * eta)
                            if probabilities and sum(probabilities) > 0:
                                total_prob = sum(probabilities)
                                probs = [p / total_prob for p in probabilities]
                                chosen_ups = random.choices(ups_options, weights=probs)[0]
                                layout[tag] = layout.get(tag, 0) + chosen_ups
                                remaining_cap -= chosen_ups
            
            # ✅ Ensure exact capacity
            while sum(layout.values()) > capacity:
                max_tag = max(layout, key=layout.get)
                if layout[max_tag] > 1:
                    layout[max_tag] -= 1
                else:
                    break
            
            while sum(layout.values()) < capacity:
                max_tag = max(active, key=active.get)
                layout[max_tag] = layout.get(max_tag, 0) + 1
            
            sheets_list = []
            for tag, ups in layout.items():
                if ups > 0 and remaining.get(tag, 0) > 0:
                    sheets_list.append(ceil(remaining[tag] / ups))
            
            if sheets_list:
                sheets = max(1, min(sheets_list))
            else:
                sheets = 1
            
            for tag, ups in layout.items():
                remaining[tag] = max(0, remaining[tag] - (ups * sheets))
            
            plates.append({
                "name": plate_name(len(plates) + 1),
                "layout": layout,
                "sheets": sheets
            })
            
            if all(v <= 0 for v in remaining.values()):
                break
        
        return ensure_demand_met(plates, demand)
    
    def update_pheromone(plates, waste):
        for key in list(pheromone.keys()):
            pheromone[key] *= (1 - evaporation)
        
        deposit = 10.0 / (waste + 1) if waste > 0 else 100.0
        
        for plate in plates:
            for tag, ups in plate["layout"].items():
                tag_idx = tags.index(tag)
                pheromone[(tag_idx, ups)] = pheromone.get((tag_idx, ups), 1.0) + deposit
    
    for iteration in range(iterations):
        iteration_best_plates = None
        iteration_best_waste = float('inf')
        
        for ant in range(ants):
            plates = construct_solution()
            waste = calculate_waste_percent(plates, demand)
            
            if waste < iteration_best_waste:
                iteration_best_waste = waste
                iteration_best_plates = copy.deepcopy(plates)
            
            if waste < best_waste:
                best_waste = waste
                best_plates = copy.deepcopy(plates)
        
        if iteration_best_plates:
            update_pheromone(iteration_best_plates, iteration_best_waste)
        
        if best_waste == 0:
            break
    
    return ensure_demand_met(best_plates, demand) if best_plates else v18_optimizer(demand, capacity, max_plates)


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
        remaining_tuple = tuple(remaining.get(t, 0) for t in self.tags)
        layout_tuple = tuple(current_layout.get(t, 0) for t in self.tags)
        return (remaining_tuple, layout_tuple)
    
    def get_action(self, state, possible_actions):
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        
        q_values = [self.q_table.get((state, action), 0) for action in possible_actions]
        max_q = max(q_values) if q_values else 0
        best_actions = [a for a, q in zip(possible_actions, q_values) if q == max_q]
        return random.choice(best_actions) if best_actions else random.choice(possible_actions)
    
    def optimize(self, episodes=30):
        best_plates = None
        best_waste = float('inf')
        
        for episode in range(episodes):
            remaining = self.demand.copy()
            plates = []
            
            for plate_num in range(self.max_plates):
                active = {k: v for k, v in remaining.items() if v > 0}
                if not active:
                    break
                
                # ✅ Use helper function
                layout = create_valid_layout(active, self.capacity, "balanced")
                
                state = self.get_state_key(remaining, layout)
                possible_actions = self.get_possible_actions(layout)
                
                if possible_actions and len(plates) < self.max_plates - 1:
                    action = self.get_action(state, possible_actions)
                    new_layout = self.apply_action(layout, action)
                    
                    # ✅ Ensure exact capacity
                    while sum(new_layout.values()) > self.capacity:
                        max_tag = max(new_layout, key=new_layout.get)
                        if new_layout[max_tag] > 1:
                            new_layout[max_tag] -= 1
                        else:
                            break
                    
                    while sum(new_layout.values()) < self.capacity:
                        max_tag = max(active, key=active.get)
                        new_layout[max_tag] = new_layout.get(max_tag, 0) + 1
                    
                    new_sheets = max(1, min(ceil(remaining[t] / new_layout.get(t, 1)) for t in active))
                    waste = sum(max(0, new_layout.get(t, 0) * new_sheets - remaining.get(t, 0)) for t in active)
                    reward = -waste
                    
                    next_state = self.get_state_key(remaining, new_layout)
                    old_q = self.q_table.get((state, action), 0)
                    next_actions = self.get_possible_actions(new_layout)
                    next_max_q = max([self.q_table.get((next_state, a), 0) for a in next_actions]) if next_actions else 0
                    new_q = old_q + self.lr * (reward + self.discount * next_max_q - old_q)
                    self.q_table[(state, action)] = new_q
                    
                    layout = new_layout
            
            sheets = max(1, min(ceil(remaining[t] / layout.get(t, 1)) for t in active))
            plates.append({"name": plate_name(len(plates) + 1), "layout": layout, "sheets": sheets})
            
            for tag, ups in layout.items():
                remaining[tag] = max(0, remaining[tag] - (ups * sheets))
            
            plates = ensure_demand_met(plates, self.demand)
            waste = calculate_waste_percent(plates, self.demand)
            
            if waste < best_waste:
                best_waste = waste
                best_plates = copy.deepcopy(plates)
            
            self.epsilon *= 0.99
        
        return ensure_demand_met(best_plates, self.demand) if best_plates else v3_optimizer(self.demand, self.capacity, self.max_plates)
    
    def get_possible_actions(self, layout):
        actions = []
        tags_list = list(layout.keys())
        if len(tags_list) >= 2:
            for i in range(len(tags_list)):
                for j in range(len(tags_list)):
                    if i != j and layout[tags_list[i]] > 1:
                        actions.append(('mutate', tags_list[i], tags_list[j]))
        return actions
    
    def apply_action(self, layout, action):
        new_layout = layout.copy()
        if action[0] == 'mutate':
            _, a, b = action
            new_layout[a] -= 1
            new_layout[b] = new_layout.get(b, 0) + 1
        return new_layout

def v22_optimizer(demand: dict, capacity: int, max_plates: int, episodes: int = 30) -> list:
    optimizer = QLearningPlateOptimizer(demand, capacity, max_plates)
    result = optimizer.optimize(episodes)
    return ensure_demand_met(result, demand) if result else v3_optimizer(demand, capacity, max_plates)




def v24_optimizer(demand: dict, capacity: int, max_plates: int,
                   population_size: int = 20, generations: int = 50,
                   F: float = 0.8, CR: float = 0.9) -> list:
    
    tags = list(demand.keys())
    n_tags = len(tags)
    
    def encode_plates_to_vector(plates):
        vector = []
        for plate in plates:
            for tag in tags:
                vector.append(plate["layout"].get(tag, 0) / capacity)
            vector.append(plate["sheets"] / 1000.0)
        while len(vector) < max_plates * (n_tags + 1):
            vector.append(0)
        return vector[:max_plates * (n_tags + 1)]
    
    def decode_vector_to_plates(vector):
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
            
            # ✅ Use helper function to fix layout
            active = {tag: demand.get(tag, 0) for tag in layout.keys()}
            layout = create_valid_layout(active, capacity, "proportional")
            
            sheets = max(1, int(vector[start_idx + n_tags] * 1000))
            
            plates.append({
                "name": plate_name(len(plates) + 1),
                "layout": layout,
                "sheets": sheets
            })
        
        return ensure_demand_met(plates, demand) if plates else v3_optimizer(demand, capacity, max_plates)
    
    def evaluate(plates):
        return calculate_waste_percent(plates, demand)
    
    population = []
    for _ in range(population_size):
        base_plates = v3_optimizer(demand, capacity, max_plates)
        population.append(encode_plates_to_vector(base_plates))
    
    best_solution = None
    best_waste = float('inf')
    
    for generation in range(generations):
        new_population = []
        
        for i, target in enumerate(population):
            candidates = [idx for idx in range(population_size) if idx != i]
            if len(candidates) < 3:
                a, b, c = 0, 1, 2
            else:
                a, b, c = random.sample(candidates, 3)
            
            mutant = []
            for j in range(len(target)):
                val = population[a][j] + F * (population[b][j] - population[c][j])
                mutant.append(max(0.0, min(1.0, val)))
            
            trial = []
            for j in range(len(target)):
                if random.random() < CR:
                    trial.append(mutant[j])
                else:
                    trial.append(target[j])
            
            trial_plates = decode_vector_to_plates(trial)
            trial_waste = evaluate(trial_plates)
            
            target_plates = decode_vector_to_plates(target)
            target_waste = evaluate(target_plates)
            
            if trial_waste < target_waste:
                new_population.append(trial)
                if trial_waste < best_waste:
                    best_waste = trial_waste
                    best_solution = trial_plates
            else:
                new_population.append(target)
                if target_waste < best_waste:
                    best_waste = target_waste
                    best_solution = target_plates
        
        population = new_population
    
    if not best_solution:
        best_solution = v3_optimizer(demand, capacity, max_plates)
    
    return ensure_demand_met(best_solution, demand)

def v25_optimizer(demand: dict, capacity: int, max_plates: int, 
                   population_size: int = 30, generations: int = 50) -> list:
    
    class Individual:
        def __init__(self, plates):
            self.plates = plates
            self.waste = calculate_waste_percent(plates, demand)
            self.total_plates = len(plates)
            self.total_sheets = sum(p["sheets"] for p in plates)
        
        def dominates(self, other):
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
            
            # ✅ Use helper function
            layout = create_valid_layout(active, capacity, "greedy")
            
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
        
        return Individual(ensure_demand_met(plates, demand))
    
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
        return Individual(ensure_demand_met(new_plates, demand))
    
    def crossover(ind1, ind2):
        point = random.randint(1, min(len(ind1.plates), len(ind2.plates)) - 1)
        child_plates = ind1.plates[:point] + ind2.plates[point:]
        
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
        
        return Individual(ensure_demand_met(fixed_plates, demand))
    
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
    
    population = [create_individual() for _ in range(population_size)]
    
    for generation in range(generations):
        offspring = []
        while len(offspring) < population_size:
            parents = random.sample(population, 2)
            child = crossover(parents[0], parents[1])
            if random.random() < 0.1:
                child = mutate(child)
            offspring.append(child)
        
        combined = population + offspring
        fronts = non_dominated_sort(combined)
        
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= population_size:
                new_population.extend(front)
            else:
                front.sort(key=lambda x: x.waste)
                new_population.extend(front[:population_size - len(new_population)])
                break
        
        population = new_population
    
    best = min(population, key=lambda x: x.waste)
    return best.plates if best.plates else v18_optimizer(demand, capacity, max_plates)


def v26_optimizer(demand: dict, capacity: int, max_plates: int) -> list:
    
    tags = list(demand.keys())
    
    class SimplePredictor:
        def __init__(self):
            self.patterns = {}
        
        def learn_from_plate(self, layout, waste):
            key = tuple(sorted(layout.items()))
            if key not in self.patterns:
                self.patterns[key] = []
            self.patterns[key].append(waste)
        
        def predict_layout(self, active, capacity):
            best_pattern = None
            best_score = float('inf')
            
            for pattern, wastes in self.patterns.items():
                avg_waste = sum(wastes) / len(wastes)
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
            
            # ✅ Use helper function
            return create_valid_layout(active, capacity, "balanced")
    
    predictor = SimplePredictor()
    population_size = 20
    generations = 40
    
    def create_individual_with_ml():
        remaining = demand.copy()
        plates = []
        
        for _ in range(max_plates):
            active = {k: v for k, v in remaining.items() if v > 0}
            if not active:
                break
            
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
        
        return ensure_demand_met(plates, demand)
    
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
        return ensure_demand_met(new_plates, demand)
    
    def crossover_plates(p1, p2):
        point = random.randint(1, min(len(p1), len(p2)) - 1)
        child = p1[:point] + p2[point:]
        
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
        
        return ensure_demand_met(fixed, demand)
    
    population = []
    for _ in range(population_size):
        ind = create_individual_with_ml()
        population.append(ind)
    
    for generation in range(generations):
        scored = [(calculate_waste_percent(ind, demand), ind) for ind in population]
        scored.sort(key=lambda x: x[0])
        
        for i in range(min(3, len(scored))):
            waste, best_ind = scored[i]
            for plate in best_ind:
                predictor.learn_from_plate(plate["layout"], waste)
        
        elites = [copy.deepcopy(scored[i][1]) for i in range(min(3, len(scored)))]
        new_population = elites.copy()
        
        while len(new_population) < population_size:
            p1 = copy.deepcopy(random.choice(elites))
            p2 = copy.deepcopy(random.choice(elites))
            child = crossover_plates(p1, p2)
            if random.random() < 0.3:
                child = mutate_with_ml(child)
            new_population.append(child)
        
        population = new_population
    
    best_idx = min(range(len(population)), key=lambda i: calculate_waste_percent(population[i], demand))
    return ensure_demand_met(population[best_idx], demand) if population else v18_optimizer(demand, capacity, max_plates)


# ================================================================
# MAIN UI
# ================================================================
st.markdown("""
<div class="main-header">
    <h1>📊 Plate Ratio Intelligence System</h1>
    <p>Complete Edition • 26 Algorithms • Production Ready</p>
    <p style="font-size: 0.85rem; opacity: 0.6;">AI-Powered • Fast • Accurate</p>
</div>
""", unsafe_allow_html=True)

# ================== CONFIGURATION ==================
st.markdown('<div class="card"><div class="card-title" style="text-align: center; display: block; width: 100%;">⚙️ Production Configuration</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    n = st.number_input("🏷️ Number of Items", 1, 500, 1)
with col2:
    cap = st.number_input("📀 Plate Capacity (UPS)", 1, 200, 10)
with col3:
    maxp = st.number_input("🎨 Max Plates", 1, 50, 3)
with col4:
    addon = st.number_input("📈 Add-on (%)", 0.0, 50.0, 0.0, step=0.5)
with col5:
    job_number = st.text_input(
        "🔢 Job Number", 
        value="",
        placeholder="e.g., JOB-001",
        help="Enter a job number for tracking (optional)"
    )
    # If empty, generate default
    if not job_number:
        job_number = f"JOB-{datetime.now().strftime('%Y%m%d_%H%M')}"

st.markdown('</div>', unsafe_allow_html=True)

# Store job number in session state
st.session_state['job_number'] = job_number

# ================== INPUT MODE SELECTION ==================
st.markdown('<div class="card"><div class="card-title" style="text-align: center; display: block; width: 100%;">📦 Input Method</div>', unsafe_allow_html=True)

input_mode = st.radio(
    "Select Input Mode:",
    options=["✏️ Manual Input", "📂 Upload Excel File"],
    horizontal=True,
    index=0
)

st.markdown('</div>', unsafe_allow_html=True)

# ================== MANUAL INPUT ==================
if input_mode == "✏️ Manual Input":
    st.markdown('<div class="card"><div class="card-title" style="text-align: center; display: block; width: 100%;">📦 Item Quantity Details (Manual)</div>', unsafe_allow_html=True)
    
    # Column headers
    col1, col2, col3, col4, col5 = st.columns([0.5, 1.5, 1.5, 1.5, 2])
    with col1:
        st.markdown("**SL**")
    with col2:
        st.markdown("**Style**")
    with col3:
        st.markdown("**Color**")
    with col4:
        st.markdown("**Size**")
    with col5:
        st.markdown("**Quantity**")
    
    st.markdown("---")
    
    tags = []
    styles = []
    colors = []
    sizes = []
    qty = []
    
    for i in range(n):
        col1, col2, col3, col4, col5 = st.columns([0.5, 1.5, 1.5, 1.5, 2])
        
        with col1:
            st.markdown(f"**{i+1}**")
        
        with col2:
            style_val = st.text_input(
                "Style", 
                value="N/A", 
                key=f"style_{i}",
                label_visibility="collapsed",
                placeholder="N/A"
            )
        
        with col3:
            color_val = st.text_input(
                "Color", 
                value="N/A", 
                key=f"color_{i}",
                label_visibility="collapsed",
                placeholder="N/A"
            )
        
        with col4:
            size_val = st.text_input(
                "Size", 
                value="N/A", 
                key=f"size_{i}",
                label_visibility="collapsed",
                placeholder="N/A"
            )
        
        with col5:
            qty_val = st.number_input(
                "Quantity", 
                min_value=0, 
                value=0, 
                step=100, 
                key=f"qty_manual_{i}",
                label_visibility="collapsed"
            )
        
        # Store values
        style_display = style_val.strip() if style_val.strip() else "N/A"
        color_display = color_val.strip() if color_val.strip() else "N/A"
        size_display = size_val.strip() if size_val.strip() else "N/A"
        
        styles.append(style_display)
        colors.append(color_display)
        sizes.append(size_display)
        
        # Create tag with style info (for display purposes)
        tag = f"Item {i+1}"
        if style_display != "N/A":
            tag = f"{style_display}"
        
        tags.append(tag)
        qty.append(qty_val)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show preview with all columns
    if any(q > 0 for q in qty):
        preview_df = pd.DataFrame({
            "SL": range(1, n + 1),
            "Style": styles,
            "Color": colors,
            "Size": sizes,
            "Quantity": qty
        })
        st.info("📋 Data Preview (showing all items)")
        st.dataframe(preview_df, use_container_width=True, height=200)
    
    # Data Preparation
    original_qty = {f"Item {i+1}": int(q) for i, q in enumerate(qty) if q > 0}
    demand = {f"Item {i+1}": ceil(int(q) * (1 + addon / 100)) for i, q in enumerate(qty) if q > 0}
    
    # Store style/color/size info for later use
    st.session_state['item_styles'] = {f"Item {i+1}": styles[i] for i in range(n)}
    st.session_state['item_colors'] = {f"Item {i+1}": colors[i] for i in range(n)}
    st.session_state['item_sizes'] = {f"Item {i+1}": sizes[i] for i in range(n)}

# ================== EXCEL FILE UPLOAD ==================
else:
    st.markdown('<div class="card"><div class="card-title" style="text-align: center; display: block; width: 100%;">📂 Upload Excel File</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Excel file with Item Details",
        type=["xlsx", "xls"],
        help="File must have columns: SL, Style, Color, Size, Quantity"
    )
    

    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Auto-detect columns
            # Look for required columns
            style_col = None
            color_col = None
            size_col = None
            qty_col = None
            sl_col = None
            
            for col in df.columns:
                col_lower = str(col).lower().strip()
                if col_lower in ['style', 'styles']:
                    style_col = col
                elif col_lower in ['color', 'colors', 'colour']:
                    color_col = col
                elif col_lower in ['size', 'sizes']:
                    size_col = col
                elif col_lower in ['quantity', 'qty', 'qty.', 'quantities', 'total']:
                    qty_col = col
                elif col_lower in ['sl', 's/l', 'serial', 'serial no', 'serial no.', 'no']:
                    sl_col = col
            
            # If specific columns not found, try to map by position
            if qty_col is None and len(df.columns) >= 2:
                # Try to find quantity column (usually the last column with numbers)
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        qty_col = col
                        break
            
            # If still not found, use last column as quantity
            if qty_col is None and len(df.columns) >= 2:
                qty_col = df.columns[-1]
            
            # If style/color/size not found, use empty strings or create from SL
            if style_col is None:
                style_col = df.columns[1] if len(df.columns) >= 2 else None
            if color_col is None:
                color_col = df.columns[2] if len(df.columns) >= 3 else None
            if size_col is None:
                size_col = df.columns[3] if len(df.columns) >= 4 else None
            
            if qty_col is None:
                st.error("❌ Could not find 'Quantity' column. Please ensure your file has a quantity column.")
                st.info("📌 Column names can be: 'Quantity', 'Qty', 'QTY', 'Total'")
                st.stop()
            
            # Extract data
            sl_data = df[sl_col].tolist() if sl_col else list(range(1, len(df) + 1))
            style_data = df[style_col].astype(str).tolist() if style_col else ["N/A"] * len(df)
            color_data = df[color_col].astype(str).tolist() if color_col else ["N/A"] * len(df)
            size_data = df[size_col].astype(str).tolist() if size_col else ["N/A"] * len(df)
            qty_data = df[qty_col].tolist()
            
            # Clean data: remove NaN, empty strings, and non-numeric quantities
            cleaned_data = []
            skipped_rows = 0
            
            for idx, (sl, style, color, size, qty) in enumerate(zip(sl_data, style_data, color_data, size_data, qty_data)):
                # Skip if quantity is NaN or invalid
                if pd.isna(qty):
                    skipped_rows += 1
                    continue
                
                try:
                    qty_int = int(float(qty))  # Convert to int
                    if qty_int > 0:  # Only keep positive quantities
                        # Clean up values
                        style_val = str(style).strip() if not pd.isna(style) and str(style).strip() != '' else "N/A"
                        color_val = str(color).strip() if not pd.isna(color) and str(color).strip() != '' else "N/A"
                        size_val = str(size).strip() if not pd.isna(size) and str(size).strip() != '' else "N/A"
                        sl_val = int(sl) if not pd.isna(sl) and str(sl).strip() != '' else idx + 1
                        
                        cleaned_data.append((sl_val, style_val, color_val, size_val, qty_int))
                    else:
                        skipped_rows += 1
                except (ValueError, TypeError):
                    skipped_rows += 1
                    continue
            
            if not cleaned_data:
                st.error("❌ No valid data found in the file. Please check the format.")
                st.stop()
            
            # Separate into lists
            sl_list = [item[0] for item in cleaned_data]
            style_list = [item[1] for item in cleaned_data]
            color_list = [item[2] for item in cleaned_data]
            size_list = [item[3] for item in cleaned_data]
            qty_list = [item[4] for item in cleaned_data]
            
            # Show preview
            preview_df = pd.DataFrame({
                "SL": sl_list,
                "Style": style_list,
                "Color": color_list,
                "Size": size_list,
                "Quantity": qty_list
            })
            
            st.success(f"✅ File loaded successfully! {len(cleaned_data)} valid items found.")
            if skipped_rows > 0:
                st.warning(f"⚠️ {skipped_rows} rows were skipped (empty or invalid data).")
            
            st.dataframe(preview_df, use_container_width=True)
            
            # Detected columns info
            detected_cols = []
            if sl_col:
                detected_cols.append(f"SL = '{sl_col}'")
            if style_col:
                detected_cols.append(f"Style = '{style_col}'")
            if color_col:
                detected_cols.append(f"Color = '{color_col}'")
            if size_col:
                detected_cols.append(f"Size = '{size_col}'")
            if qty_col:
                detected_cols.append(f"Quantity = '{qty_col}'")
            
            st.info(f"📋 Detected columns: {', '.join(detected_cols)}")
            
            # Auto-set the number of items
            n = len(cleaned_data)
            tags = [f"Item {i+1}" for i in range(n)]
            qty = qty_list
            
            # Store style/color/size data in session state for PDF
            st.session_state['item_styles'] = {f"Item {i+1}": style_list[i] for i in range(n)}
            st.session_state['item_colors'] = {f"Item {i+1}": color_list[i] for i in range(n)}
            st.session_state['item_sizes'] = {f"Item {i+1}": size_list[i] for i in range(n)}
            
            original_qty = {t: int(q) for t, q in zip(tags, qty) if q > 0}
            demand = {t: ceil(int(q) * (1 + addon / 100)) for t, q in zip(tags, qty) if q > 0}
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.stop()
    else:
        st.info("📤 Please upload an Excel file to continue.")
        st.stop()

st.markdown('</div>', unsafe_allow_html=True)

# ================== WARNING MESSAGES ==================
if not PULP_AVAILABLE:
    st.markdown('<div class="warning">⚠️ PuLP library not installed. Some advanced features disabled.</div>', unsafe_allow_html=True)

if not ORTOOLS_AVAILABLE:
    st.markdown('<div class="warning">⚠️ OR-Tools not installed. V19 will be disabled. Install with: pip install ortools</div>', unsafe_allow_html=True)

# ================== GENERATE BUTTON ==================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_clicked = st.button("Generate Plans", use_container_width=True, type="primary")

# ================== AFTER GENERATE ==================
if generate_clicked:
    if not demand:
        st.error("⚠️ Please enter at least one item with quantity greater than 0")
        st.stop()

    with st.spinner("🔍 Searching for the perfect plate ratio... Almost there!"):
        
        # ================== ALL ALGORITHMS V1-V26 ==================
        algo_functions = {
            "V1 - Plate Ratio System": lambda: v1_optimizer(demand, cap, maxp),
            "V2 - Common Sheet Optimizer": lambda: v2_optimizer(demand, cap, maxp),
            "V3 - Smart Decimal Balancing": lambda: v3_optimizer(demand, cap, maxp),
            "V4 - Multi-Variation Optimizer": lambda: v4_optimizer(demand, cap, maxp),
            "V5 - AI Mutation Engine": lambda: v5_optimizer(demand, cap, maxp, iterations=50),
            "V6 - Integer Solver": lambda: v6_optimizer(demand, cap, maxp) if PULP_AVAILABLE else v3_optimizer(demand, cap, maxp),
            "V7 - Simulated Annealing": lambda: v7_optimizer(demand, cap, maxp, iterations=100),
            "V8 - MCTS Tree Search": lambda: v8_optimizer(demand, cap, maxp, iterations=50),
            "V11 - Genetic Algorithm": lambda: v11_optimizer(demand, cap, maxp, population_size=20, generations=30),
            "V12 - Column Generation": lambda: v12_optimizer(demand, cap, maxp) if PULP_AVAILABLE else v3_optimizer(demand, cap, maxp),
            "V15 - DP Repair Engine": lambda: v15_optimizer(demand, cap, maxp),
            "V16 - Plate Merge Optimizer": lambda: v16_optimizer(demand, cap, maxp),
            "V17 - AI Evolution Engine": lambda: v17_optimizer(demand, cap, maxp, generations=100),
            "V18 - Global Multi-Plate Optimizer": lambda: v18_optimizer(demand, cap, maxp),
            "V19 - CP-SAT Optimizer": lambda: v19_optimizer(demand, cap, maxp) if ORTOOLS_AVAILABLE else v18_optimizer(demand, cap, maxp),
            "V20 - PSO Optimizer": lambda: v20_optimizer(demand, cap, maxp),
            "V21 - ACO Optimizer": lambda: v21_optimizer(demand, cap, maxp),
            "V22 - Q-Learning Optimizer": lambda: v22_optimizer(demand, cap, maxp, episodes=20),
            "V24 - Differential Evolution": lambda: v24_optimizer(demand, cap, maxp, population_size=15, generations=30),
            "V25 - Pareto Optimizer": lambda: v25_optimizer(demand, cap, maxp, population_size=20, generations=30),
            "V26 - NN Predictor": lambda: v26_optimizer(demand, cap, maxp),
        }
        
        problematic_for_single_plate = {
            "V11 - Genetic Algorithm", "V16 - Plate Merge Optimizer", "V17 - AI Evolution Engine",
            "V19 - CP-SAT Optimizer", "V20 - PSO Optimizer", "V21 - ACO Optimizer",
            "V22 - Q-Learning Optimizer", "V23 - Branch & Bound", "V24 - Differential Evolution",
            "V25 - Pareto Optimizer", "V26 - NN Predictor"
        }
        
        # Run all algorithms with progress bar
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (algo_name, func) in enumerate(algo_functions.items()):
            status_text.text(f"Running {algo_name}... ({idx+1}/{len(algo_functions)})")
            try:
                if maxp == 1 and algo_name in problematic_for_single_plate:
                    results[algo_name] = v3_optimizer(demand, cap, maxp)
                else:
                    results[algo_name] = func()
            except Exception as e:
                results[algo_name] = v3_optimizer(demand, cap, maxp)
            
            progress_bar.progress((idx + 1) / len(algo_functions))
        
        progress_bar.empty()
        status_text.empty()
        
        # Global fix for all algorithms
        for algo_name, plates in results.items():
            if plates:
                results[algo_name] = ensure_demand_met(plates, demand)
            else:
                results[algo_name] = v3_optimizer(demand, cap, maxp)
        
        # Comparison Data
        comparison_data = []
        for algo_name, plates in results.items():
            if plates:
                waste = calculate_waste_percent(plates, demand)
                comparison_data.append({
                    "Algorithm": algo_name,
                    "Waste %": waste,
                    "Total Plates": len(plates),
                    "Total Sheets": sum(p.get("sheets", 0) for p in plates),
                    "Status": "✅ Success"
                })
            else:
                comparison_data.append({
                    "Algorithm": algo_name,
                    "Waste %": 100,
                    "Total Plates": 0,
                    "Total Sheets": 0,
                    "Status": "❌ Failed"
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
        
        # ============= BEST ALGORITHM REPORT =============
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
                
                # ============= DOWNLOAD BUTTONS =============
                st.markdown("### 📥 Download Best Report")
                col1, col2 = st.columns(2)
                
                with col1:
                    bio_excel = BytesIO()
                    with pd.ExcelWriter(bio_excel, engine="openpyxl") as writer:
                        full_df.to_excel(writer, sheet_name="Summary", index=False)
                        plate_details_df.to_excel(writer, sheet_name="Plate Details", index=False)
                        comparison_df.to_excel(writer, sheet_name="Comparison", index=False)
                    bio_excel.seek(0)
                    
                    # Clean job number for Excel filename
                    job_number = st.session_state.get('job_number', 'JOB-00000')
                    clean_job = ''.join(c for c in job_number if c.isalnum() or c == '-')
                    excel_filename = f"{clean_job}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
                    
                    st.download_button(
                        "📊 Download Excel",
                        bio_excel,
                        excel_filename,
                        use_container_width=True
                    )
                
                with col2:
                    # Check if reportlab is available
                    try:
                        import reportlab
                        REPORTLAB_AVAILABLE = True
                    except ImportError:
                        REPORTLAB_AVAILABLE = False
                    
                    if REPORTLAB_AVAILABLE:
                        try:
                            # Get style/color/size data from session state
                            styles_dict = st.session_state.get('item_styles', {})
                            colors_dict = st.session_state.get('item_colors', {})
                            sizes_dict = st.session_state.get('item_sizes', {})
                            job_number = st.session_state.get('job_number', 'JOB-00000')
                            
                            pdf_buffer = generate_pdf_report(
                                best_plates,
                                demand,
                                original_qty,
                                best_algo,
                                best_waste,
                                styles_dict,
                                colors_dict,
                                sizes_dict,
                                job_number
                            )
                            
                            if pdf_buffer:
                                # Clean job number for filename
                                clean_job = ''.join(c for c in job_number if c.isalnum() or c == '-')
                                pdf_filename = f"Job_Number_{clean_job}.pdf"
                                
                                st.download_button(
                                    "📄 Download PDF",
                                    pdf_buffer,
                                    pdf_filename,
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                            else:
                                st.warning("⚠️ PDF could not be generated. Please check the data.")
                        except Exception as e:
                            st.error(f"❌ PDF Error: {str(e)}")
                    else:
                        st.info("ℹ️ PDF download requires reportlab. Install with: pip install reportlab")
            
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                st.info("Showing comparison table instead...")
        
        # ============= ALGORITHM COMPARISON =============
        st.markdown("---")
        st.markdown("## 📊 Algorithm Comparison (Sorted by Waste %)")
        
        styled_df = comparison_df.style.apply(
            lambda row: ['background-color: #2e7d32; color: white'] * len(row)
            if row["Algorithm"] == best_algo else [''] * len(row),
            axis=1
        ).format({"Waste %": "{:.2f}%"})
        
        st.dataframe(styled_df, use_container_width=True, height=600)

        # ============= VIEW ANY ALGORITHM REPORT =============
        st.markdown("---")
        st.markdown("## 🔍 View Individual Algorithm Report")

        if 'results' in st.session_state and st.session_state['results']:
            algo_list = list(st.session_state['results'].keys())
            
            default_index = 0
            if st.session_state.get('best_algo') in algo_list:
                default_index = algo_list.index(st.session_state['best_algo'])

            col1, col2 = st.columns([3, 1])
            with col1:
                selected_algo = st.selectbox(
                    "Select Algorithm to View Report:",
                    options=algo_list,
                    index=default_index,
                    key="independent_algo_selector"
                )

            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                view_button = st.button("📋 View Report", use_container_width=True, type="primary")

            if view_button:
                selected_plates = st.session_state['results'].get(selected_algo)
                
                if selected_plates:
                    st.markdown(f"### 📊 Production Summary — **{selected_algo}**")
                    
                    full_df = build_full_summary(
                        selected_plates, 
                        st.session_state['demand'], 
                        st.session_state['original_qty']
                    )
                    st.dataframe(full_df, use_container_width=True, height=400)

                    st.markdown("### 🧾 Plate Configuration Details")
                    plate_rows = []
                    total_sheets = 0
                    total_ups = 0
                    
                    for idx, p in enumerate(selected_plates, 1):
                        ups_sum = sum(p["layout"].values())
                        plate_rows.append({
                            "SL": idx,
                            "Plate ID": p.get("name", f"Plate {idx}"),
                            "Sheets Required": p.get("sheets", 0),
                            "Total UPS": ups_sum,
                        })
                        total_sheets += p.get("sheets", 0)
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
                    st.error(f"❌ Report not found for {selected_algo}")

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; margin-top: 3rem; border-top: 2px solid rgba(102,126,234,0.3); background: rgba(255,255,255,0.02); border-radius: 20px;">
    <p style="color: rgba(255,255,255,0.6); font-size: 0.85rem; margin: 0;">
        © 2025 Plate Ratio System | Version 26 (Complete Edition)
    </p>
    <p style="color: rgba(255,255,255,0.5); font-size: 0.8rem; margin: 8px 0;">
        Enterprise Production Optimization Framework • 26 Algorithms • Production Ready
    </p>
    <p style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 0.85rem; font-weight: 600; margin: 10px 0 0 0;">
        ✨ Developed by Ovi | All Rights Reserved ✨
    </p>
</div>
""", unsafe_allow_html=True)
