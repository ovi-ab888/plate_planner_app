# app.py — VERSION 3 OPTIMIZER (LOW WASTE + TOTAL ROW) - PROFESSIONAL UI

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
from io import BytesIO
from collections import Counter
from math import ceil, floor
import string

st.set_page_config(
    page_title="Pre-Press Planner V3 | Professional Edition",
    page_icon="🖨️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================================================
#  PASSWORD CHECK SYSTEM
# ================================================================
def check_password():
    """Simple password gate using secrets or environment."""
    expected = None

    # Prefer streamlit secrets
    try:
        expected = st.secrets.get("app_password", None)
    except Exception:
        expected = None

    # Fallback env variable
    if expected is None:
        expected = os.environ.get("PEPCO_APP_PASSWORD")

    # If not found → error
    if expected is None:
        st.error("App password not configured. Please set 'app_password' in secrets or PEPCO_APP_PASSWORD env var.")
        return False

    # When password typed
    def _password_entered():
        if st.session_state.get("password") == expected:
            st.session_state["password_correct"] = True
            try:
                del st.session_state["password"]
            except Exception:
                pass
        else:
            st.session_state["password_correct"] = False

    # Already correct?
    if st.session_state.get("password_correct", None) is True:
        return True

    # Custom password UI - No white box
    st.markdown("""
    <style>
        /* Remove all white backgrounds */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Remove Streamlit default containers */
        .main > div {
            background: transparent !important;
        }
        
        /* Style for password container - transparent */
        .password-container {
            max-width: 450px;
            margin: 150px auto;
            padding: 2.5rem;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.3);
        }
        
        .password-container h2 {
            color: #667eea;
            margin-bottom: 0.5rem;
            font-size: 2rem;
            font-weight: 700;
        }
        
        .password-container p {
            color: #666;
            margin-bottom: 1.5rem;
            font-size: 1rem;
        }
        
        /* Style input field */
        .password-container input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .password-container input:focus {
            border-color: #667eea;
            outline: none;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.2);
        }
        
        /* Style error message */
        .stAlert {
            background: rgba(255, 0, 0, 0.1) !important;
            border-left: 4px solid #ff4444 !important;
            color: #ff4444 !important;
            border-radius: 10px !important;
        }
        
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Center everything */
        .block-container {
            padding-top: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Display password form
    with st.container():
        st.markdown('<div class="password-container">', unsafe_allow_html=True)
        st.markdown('<h2>🔐 Secure Access</h2>', unsafe_allow_html=True)
        st.markdown('<p>Please enter your access code to continue</p>', unsafe_allow_html=True)
        
        # Password input
        password = st.text_input("Enter Your Access Code", type="password", key="password", 
                                 on_change=_password_entered, label_visibility="collowed")
        
        # Wrong password message
        if st.session_state.get("password_correct") is False:
            st.error("❌ Your password is incorrect. Please contact Mr. Ovi for assistance.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return False

# Check password before showing main app
if not check_password():
    st.stop()

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
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
    
    /* Card styling - Updated border color */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 5px 20px rgba(102,126,234,0.15);
        border-color: #764ba2;
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
        display: inline-block;
    }
    
    /* Metrics styling */
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
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 0.5rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102,126,234,0.1);
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo {
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Divider styling */
    hr {
        margin: 2rem 0;
        background: linear-gradient(to right, #667eea, transparent);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        margin-top: 2rem;
    }
    
    .footer p {
        color: #2c3e50;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .designer-credit {
        font-size: 1rem;
        font-weight: 600;
        color: #667eea;
        margin-top: 0.5rem;
    }
    
    /* Badge styling */
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
# SMART BALANCED UPS
# =====================================================

def smart_layout(demand, cap):
    total = sum(demand.values())
    if total == 0:
        return {}
    raw = {}
    floor_vals = {}
    remainders = {}
    for k, v in demand.items():
        ratio = (v / total) * cap
        raw[k] = ratio
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
    <h1>🖨️ Pre-Press Planner V3</h1>
    <p>Professional Production Optimization System | Low Waste + Smart UPS Distribution</p>
</div>
""", unsafe_allow_html=True)

# Configuration Panel
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">⚙️ Production Configuration</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    n = st.number_input(
        "🏷️ Tag Count",
        min_value=1,
        max_value=50,
        value=6,
        help="Number of different tags/labels to print"
    )

with col2:
    cap = st.number_input(
        "📀 Plate Capacity",
        min_value=1,
        max_value=64,
        value=12,
        help="Maximum number of UPS per plate"
    )

with col3:
    maxp = st.number_input(
        "🎨 Max Plates",
        min_value=1,
        max_value=50,
        value=2,
        help="Maximum number of plates allowed"
    )

with col4:
    addon = st.number_input(
        "📈 Add-on %",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.5,
        help="Additional percentage for safety stock"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Tag Quantity Section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">📦 Tag Quantity Details</div>', unsafe_allow_html=True)

tags = []
qty = []

for i in range(n):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input(
            f"Tag {i+1} Name",
            f"Tag {i+1}",
            key=f"tag_{i}",
            help=f"Enter name for tag {i+1}"
        )
    with col2:
        q = st.number_input(
            f"Quantity",
            min_value=0,
            step=10,
            key=f"qty_{i}",
            help=f"Enter quantity for {name}"
        )
    tags.append(name)
    qty.append(q)

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# DATA
# =====================================================

original_qty = {
    t: int(q)
    for t, q in zip(tags, qty)
    if q > 0
}

demand = {
    t: ceil(int(q) * (1 + addon / 100))
    for t, q in zip(tags, qty)
    if q > 0
}

# Generate Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_clicked = st.button("🚀 Generate Optimized Plan", use_container_width=True)

# =====================================================
# GENERATE
# =====================================================

if generate_clicked:
    if not demand:
        st.error("⚠️ Please enter at least one tag with quantity greater than 0")
        st.stop()
    
    with st.spinner("🔄 Optimizing production plan..."):
        plates, produced = auto_plan(demand, cap, maxp)
    
    # =================================================
    # FINAL SUMMARY
    # =================================================
    
    rows = []
    for tag in demand:
        row = {
            "Tag": tag,
            "Original QTY": original_qty[tag],
            "Produced (+Add-on)": demand[tag]
        }
        total_produced = 0
        for p in plates:
            ups = p["layout"].get(tag, 0)
            row[f"Plate {p['name']}"] = ups
            total_produced += (ups * p["sheets"])
        excess = total_produced - demand[tag]
        excess_percent = (
            round((excess / demand[tag]) * 100, 2)
            if demand[tag]
            else 0
        )
        row["Total Produced QTY"] = total_produced
        row["Excess"] = excess
        row["Excess %"] = excess_percent
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # =================================================
    # TOTAL ROW
    # =================================================
    
    total_row = {
        "Tag": "📊 TOTAL",
        "Original QTY": df["Original QTY"].sum(),
        "Produced (+Add-on)": df["Produced (+Add-on)"].sum(),
    }
    for p in plates:
        col = f"Plate {p['name']}"
        total_row[col] = df[col].sum()
    total_row["Total Produced QTY"] = df["Total Produced QTY"].sum()
    total_row["Excess"] = df["Excess"].sum()
    total_row["Excess %"] = round(
        (total_row["Excess"] / total_row["Produced (+Add-on)"]) * 100, 2
    ) if total_row["Produced (+Add-on)"] > 0 else 0
    
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    
    # =================================================
    # KEY METRICS
    # =================================================
    
    total_sheets = sum(p["sheets"] for p in plates)
    total_excess = df.iloc[:-1]["Excess"].sum()
    waste_percentage = (total_excess / df.iloc[:-1]["Produced (+Add-on)"].sum() * 100) if df.iloc[:-1]["Produced (+Add-on)"].sum() > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(plates)}</div>
            <div class="metric-label">Total Plates</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_sheets}</div>
            <div class="metric-label">Total Sheets</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_excess:,}</div>
            <div class="metric-label">Total Excess</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{waste_percentage:.1f}%</div>
            <div class="metric-label">Waste Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    # =================================================
    # PRODUCTION SUMMARY TABLE
    # =================================================
    
    st.markdown("---")
    st.markdown("## 📊 Production Summary")
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    # =================================================
    # PLATE INFORMATION
    # =================================================
    
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
    st.dataframe(
        plate_df,
        use_container_width=True,
        hide_index=True
    )
    
    # =================================================
    # SUCCESS MESSAGE & EXPORT
    # =================================================
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"✅ Production plan optimized successfully! Total sheets: {total_sheets}")
    
    # =================================================
    # EXCEL EXPORT
    # =================================================
    
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Production Summary", index=False)
        plate_df.to_excel(writer, sheet_name="Plate Details", index=False)
    
    bio.seek(0)
    
    with col2:
        st.download_button(
            "⬇️ Download Excel Report",
            data=bio,
            file_name="prepress_optimized_plan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# =====================================================
# FOOTER WITH DESIGNER CREDIT
# =====================================================

st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🔥 Pre-Press Planner V3 Professional Edition — Low Waste Optimization + Smart UPS Distribution</p>
    <p class="badge">Version 3.0 | Enterprise Ready</p>
    <p class="designer-credit">✨ Design & Developed by <strong style="color:#764ba2">Md Ovi</strong> ✨</p>
    <p style="font-size:0.8rem; opacity:0.7;">© 2026 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
