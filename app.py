# app.py — PLATE RATIO SYSTEM | DESIGN BY OVI

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
from io import BytesIO
from collections import Counter
from math import ceil, floor
import string

st.set_page_config(
    page_title="Plate Ratio System | Design by Ovi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================================================
#  PASSWORD CHECK SYSTEM (UPDATED - HEADER + CENTERED BOX)
# ================================================================
def check_password():
    """Simple password gate using secrets or environment."""
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

    # CSS for password page - ONLY password container styling
    st.markdown("""
    <style>
        /* Black background */
        .stApp {
            background: black !important;
        }
        
        /* Remove all default streamlit containers */
        .main > div {
            background: transparent !important;
            padding: 0 !important;
        }
        
        .block-container {
            padding: 0rem !important;
            max-width: 80% !important;
        }
        
        .element-container {
            background: transparent !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        .stMarkdown {
            background: transparent !important;
        }
        
        /* Remove Streamlit text input wrapper backgrounds */
        div[data-testid="stTextInput"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        
        div[data-testid="stTextInput"] > div {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }
        
        .stTextInput {
            background: transparent !important;
            border: none !important;
        }
        
        .stTextInput > div {
            background: transparent !important;
            border: none !important;
        }
        
        /* Hide labels */
        .stTextInput label {
            display: none !important;
        }
        
        /* Input field styling */
        .stTextInput input {
            background: rgba(255,255,255,0.1) !important;
            border: 2px solid #333 !important;
            border-radius: 10px !important;
            color: white !important;
            padding: 5px !important;
            width: 100% !important;
            margin: 50 !important;
            font-size: 16px !important;
            text-align: center !important;
        }
        
        .stTextInput input:focus {
            border-color: #667eea !important;
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.2) !important;
        }
        
        /* Main header styling - Top of page */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 1rem 0rem 1rem;
            text-align: center;
        }
        
        .main-header h1 {
            color: white;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        
        .main-header p {
            color: rgba(255,255,255,0.9);
            font-size: 1.1rem;
        }
        
        .designer-name {
            color: #ffd700;
            font-size: 1rem;
            margin-top: 0.5rem;
        }
        
        /* Password container - Centered with margin top */
        .password-container {
            max-width: 450px;
            margin: 60px auto 0 auto;
            padding: 2.5rem;
            background: rgba(0, 0, 0, 0.85);
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(102,126,234,0.3);
            text-align: center;
            border: 1px solid rgba(102,126,234,0.5);
        }
        
        .password-container h2 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
            font-size: 1.8rem;
            font-weight: 700;
        }
        
        .password-container p {
            color: #aaa;
            margin-bottom: 1.5rem;
            font-size: 1rem;
        }
        
        /* Error message styling */
        .stAlert {
            background: rgba(255, 0, 0, 0.1) !important;
            border-left: 4px solid #ff4444 !important;
            color: #ff4444 !important;
            border-radius: 10px !important;
            margin-top: 1rem !important;
            max-width: 450px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        
        /* Hide Streamlit menu and footer */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Remove all padding */
        .view-container {
            padding: 0 !important;
        }
        
        html, body {
            margin: 0 !important;
            padding: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Show Header at the top
    st.markdown("""
    <div class="main-header">
        <h1>📊 Plate Ratio System</h1>
        <p>Professional UPS Ratio Optimization | Low Waste + Smart Distribution</p>
        <p class="designer-name">✨ Design by Ovi ✨</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Password form - Centered below header with some gap
    st.markdown("""
    <div style="height: 40px;"></div>
    <div class="password-container">
        <h2>🔐 Access Code</h2>
        <p>Please enter your access code to continue</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Password input (inside the container area)
    # Using empty columns to center the input
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.text_input("Enter Your Access Code", type="password", key="password", 
                      on_change=_password_entered, label_visibility="collapsed")
    
    # Error message centered
    if st.session_state.get("password_correct") is False:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.error("❌ Your password is incorrect. Please contact Mr. Ovi for assistance.")
    
    return False

# Check password before showing main app
if not check_password():
    st.stop()

# Custom CSS for main app styling
st.markdown("""
<style>
    /* Black background */
    .stApp {
        background: black !important;
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .card {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        margin-bottom: 1.5rem;
        border: 1px solid #333;
    }
    
    .card:hover {
        border-color: #667eea;
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
        display: inline-block;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #333;
        padding: 0.5rem;
        background: #1a1a1a;
        color: white;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: #1a1a1a;
        border-radius: 15px;
        margin-top: 2rem;
        border: 1px solid #333;
    }
    
    .footer p {
        color: #ccc;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .designer-credit {
        font-size: 1rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        font-size: 0.85rem;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# Plate Name Generator
# =====================================================

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

# =====================================================
# SMART BALANCED UPS - RATIO BASED
# =====================================================

def smart_layout(demand, cap):
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

# =====================================================
# AUTO PLAN
# =====================================================

def auto_plan(demand, cap, max_plates):
    remaining = demand.copy()
    plates = []
    produced = Counter()
    for i in range(max_plates):
        if not any(v > 0 for v in remaining.values()):
            break
        layout = smart_layout(remaining, cap)
        if not layout:
            break
        possible = [
            ceil(remaining[k] / v)
            for k, v in layout.items()
            if v > 0
        ]
        sheets = max(1, min(possible))
        for k, v in layout.items():
            produced_qty = v * sheets
            remaining[k] = max(0, remaining[k] - produced_qty)
            produced[k] += produced_qty
        plates.append({
            "name": plate_name(len(plates) + 1),
            "layout": layout,
            "sheets": sheets
        })
    if any(v > 0 for v in remaining.values()) and plates:
        last = plates[-1]
        for k in remaining:
            if remaining[k] > 0:
                per_sheet = max(1, last["layout"].get(k, 1))
                add_sheets = ceil(remaining[k] / per_sheet)
                last["sheets"] += add_sheets
                produced[k] += add_sheets * per_sheet
                remaining[k] = 0
    return plates, dict(produced)

# =====================================================
# UI
# =====================================================

# Header Section
st.markdown("""
<div class="main-header">
    <h1>📊 Plate Ratio System</h1>
    <p>Professional UPS Ratio Optimization | Low Waste + Smart Distribution</p>
</div>
""", unsafe_allow_html=True)

# Configuration Panel
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">⚙️ Production Configuration</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    n = st.number_input("🏷️ Tag Count", 1, 50, 1)

with col2:
    cap = st.number_input("📀 Plate Capacity", 1, 64, 10)

with col3:
    maxp = st.number_input("🎨 Max Plates", 1, 50, 3)

with col4:
    addon = st.number_input("📈 Add-on %", 0.0, 50.0, 0.0, step=0.5)

st.markdown('</div>', unsafe_allow_html=True)

# Tag Quantity Section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">📦 Tag Quantity Details</div>', unsafe_allow_html=True)

tags = []
qty = []

for i in range(n):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input(f"Tag {i+1} Name", f"Tag {i+1}", key=f"tag_{i}")
    with col2:
        q = st.number_input(f"Quantity", 0, step=10, key=f"qty_{i}")
    tags.append(name)
    qty.append(q)

st.markdown('</div>', unsafe_allow_html=True)

# Data
original_qty = {t: int(q) for t, q in zip(tags, qty) if q > 0}
demand = {t: ceil(int(q) * (1 + addon / 100)) for t, q in zip(tags, qty) if q > 0}

# Generate Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_clicked = st.button("🚀 Generate Optimized Plan", use_container_width=True)

if generate_clicked:
    if not demand:
        st.error("⚠️ Please enter at least one tag with quantity greater than 0")
        st.stop()
    
    with st.spinner("🔄 Optimizing production plan..."):
        plates, produced = auto_plan(demand, cap, maxp)
    
    rows = []
    for tag in demand:
        row = {"Tag": tag, "Original QTY": original_qty[tag], "Produced (+Add-on)": demand[tag]}
        total_produced = 0
        for p in plates:
            ups = p["layout"].get(tag, 0)
            row[f"Plate {p['name']}"] = ups
            total_produced += (ups * p["sheets"])
        excess = total_produced - demand[tag]
        excess_percent = round((excess / demand[tag]) * 100, 2) if demand[tag] else 0
        row["Total Produced QTY"] = total_produced
        row["Excess"] = excess
        row["Excess %"] = excess_percent
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    total_row = {
        "Tag": "📊 TOTAL",
        "Original QTY": df["Original QTY"].sum(),
        "Produced (+Add-on)": df["Produced (+Add-on)"].sum(),
    }
    for p in plates:
        total_row[f"Plate {p['name']}"] = df[f"Plate {p['name']}"].sum()
    total_row["Total Produced QTY"] = df["Total Produced QTY"].sum()
    total_row["Excess"] = df["Excess"].sum()
    total_row["Excess %"] = round((total_row["Excess"] / total_row["Produced (+Add-on)"]) * 100, 2) if total_row["Produced (+Add-on)"] > 0 else 0
    
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    
    total_sheets = sum(p["sheets"] for p in plates)
    total_excess = df.iloc[:-1]["Excess"].sum()
    waste_percentage = (total_excess / df.iloc[:-1]["Produced (+Add-on)"].sum() * 100) if df.iloc[:-1]["Produced (+Add-on)"].sum() > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(plates)}</div><div class="metric-label">Total Plates</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_sheets}</div><div class="metric-label">Total Sheets</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_excess:,}</div><div class="metric-label">Total Excess</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{waste_percentage:.1f}%</div><div class="metric-label">Waste Rate</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 📊 Production Summary")
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("## 🧾 Plate Configuration Details")
    plate_rows = []
    for p in plates:
        plate_rows.append({
            "Plate ID": p["name"],
            "Sheets Required": p["sheets"],
            "Total UPS": sum(p["layout"].values()),
            "Layout": ", ".join([f"{k}:{v}" for k, v in p["layout"].items()])
        })
    plate_df = pd.DataFrame(plate_rows)
    st.dataframe(plate_df, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"✅ Production plan optimized successfully! Total sheets: {total_sheets}")
    
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Production Summary", index=False)
        plate_df.to_excel(writer, sheet_name="Plate Details", index=False)
    bio.seek(0)
    
    with col2:
        st.download_button("⬇️ Download Excel Report", data=bio, file_name="plate_ratio_plan.xlsx", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>📊 Plate Ratio System — Smart UPS Ratio Optimization + Zero Waste Planning</p>
    <p class="badge">Version 3.0 | Enterprise Ready</p>
    <p class="designer-credit">✨ Design & Developed by <strong style="color:#764ba2">Ovi</strong> ✨</p>
    <p style="font-size:0.8rem; opacity:0.7;">© 2026 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
