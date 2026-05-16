# ================================================================
# IMPORTS
# ================================================================

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import fitz
import re
import copy
import random
import string

from io import BytesIO
from math import ceil, floor


# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="Pre-Press Planner",
    page_icon="🖨️",
    layout="wide"
)


# ================================================================
# PASSWORD CHECK SYSTEM
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
# PDF QTY UPLOAD SYSTEM
# ================================================================

def extract_qty_from_pdf(uploaded_pdf):

    import fitz
    import re
    import pandas as pd

    doc = fitz.open(
        stream=uploaded_pdf.read(),
        filetype="pdf"
    )

    full_text = ""

    for page in doc:

        full_text += page.get_text()

    # ============================================================
    # DEBUG (OPTIONAL)
    # ============================================================

    # st.text(full_text)

    # ============================================================
    # YOUR PDF STRUCTURE
    # ============================================================
    #
    # Example:
    #
    # 1-1½; YRS N/A 12 Nightwear 320.00
    #
    # PRIMARY_SIZE = 1-1½; YRS
    # QTY = 320.00
    #
    # ============================================================

    pattern = r'([0-9A-Za-z½;\\-\\s]+YRS)\\s+N/A\\s+\\d+\\s+Nightwear\\s+(\\d+\\.\\d+)'

    matches = re.findall(
        pattern,
        full_text
    )

    # ============================================================
    # BUILD DATA
    # ============================================================

    data = []

    for size, qty in matches:

        try:

            qty = int(float(qty))

            data.append({
                "Tag": size.strip(),
                "Qty": qty
            })

        except:
            pass

    # ============================================================
    # DATAFRAME
    # ============================================================

    df = pd.DataFrame(data)

    return df


# ================================================================
# INPUT METHOD SELECTOR
# ================================================================

st.sidebar.markdown("## 📥 Input Method")

input_mode = st.sidebar.radio(
    "Select Input Source",
    [
        "Manual Entry",
        "Upload PDF"
    ]
)


# ================================================================
# GLOBAL VARIABLES
# ================================================================

tags = []
qtys = []

original_qty = {}
demand = {}


# ================================================================
# MANUAL ENTRY SYSTEM
# ================================================================

if input_mode == "Manual Entry":

    st.markdown("## ✍️ Manual Qty Entry")

    col1, col2 = st.columns(2)

    n = col1.number_input(
        "Tag Count",
        1,
        100,
        5
    )

    addon = col2.number_input(
        "Add-on %",
        0.0,
        100.0,
        0.0,
        step=0.5
    )

    left, right = st.columns(2)

    for i in range(n):

        tag = left.text_input(
            f"Tag {i+1}",
            key=f"tag_{i}"
        )

        qty = right.number_input(
            f"Qty {i+1}",
            0,
            step=100,
            key=f"qty_{i}"
        )

        if tag and qty > 0:

            tags.append(tag)

            qtys.append(qty)

    original_qty = {
        t: int(q)
        for t, q in zip(tags, qtys)
    }

    demand = {
        t: ceil(
            int(q)
            *
            (1 + addon / 100)
        )
        for t, q in zip(tags, qtys)
    }


# ================================================================
# PDF UPLOAD SYSTEM
# ================================================================

if input_mode == "Upload PDF":

    st.markdown("## 📄 Upload Work Order PDF")

    addon = st.number_input(
        "Add-on %",
        0.0,
        100.0,
        0.0,
        step=0.5
    )

    uploaded_pdf = st.file_uploader(
        "Upload PDF File",
        type=["pdf"]
    )

    if uploaded_pdf:

        df_pdf = extract_qty_from_pdf(
            uploaded_pdf
        )

        # ========================================================
        # VALIDATION
        # ========================================================

        if df_pdf.empty:

            st.error(
                "❌ No Qty Found"
            )

            st.stop()

        # ========================================================
        # PREVIEW
        # ========================================================

        st.success(
            "✅ PDF Extracted Successfully"
        )

        st.dataframe(
            df_pdf,
            use_container_width=True
        )

        st.metric(
            "Total Order Qty",
            int(df_pdf["Qty"].sum())
        )

        # ========================================================
        # CONVERT TO SYSTEM
        # ========================================================

        tags = (
            df_pdf["Tag"]
            .astype(str)
            .tolist()
        )

        qtys = (
            df_pdf["Qty"]
            .astype(int)
            .tolist()
        )

        original_qty = {
            t: int(q)
            for t, q in zip(tags, qtys)
        }

        demand = {
            t: ceil(
                int(q)
                *
                (1 + addon / 100)
            )
            for t, q in zip(tags, qtys)
        }


# ================================================================
# SAMPLE TEMPLATE DOWNLOAD
# ================================================================

sample_df = pd.DataFrame({
    "Tag": ["XS", "S", "M", "L"],
    "Qty": [12000, 15000, 18000, 10000]
})

sample_buffer = BytesIO()

with pd.ExcelWriter(
    sample_buffer,
    engine="openpyxl"
) as writer:

    sample_df.to_excel(
        writer,
        index=False
    )

sample_buffer.seek(0)

st.sidebar.download_button(
    "⬇️ Download Sample Template",
    data=sample_buffer,
    file_name="sample_qty_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

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
# PDF GENERATION FUNCTION (যোগ করতে হবে)
# ================================================================

def generate_pdf_report(plates, demand, original_qty, algo_name, waste_percent):
    """Generate PDF report without Layout column"""
    try:
        # Import all reportlab components inside the function
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
        
        # Summary Table
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
        
        # Total row
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
        
        # Plate Information Table (WITHOUT LAYOUT COLUMN)
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
        
        # Footer
        footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.grey)
        story.append(Paragraph("This Report Generated by Ovi's Plate Ratio System", footer_style))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    except Exception as e:
        st.error(f"PDF generation error: {str(e)}")
        return None


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
# MAIN APP UI
# ================================================================

st.title("🖨️ Pre-Press Planner")

st.caption(
    "Industrial Label Optimization System"
)

# ================================================================
# SETTINGS
# ================================================================

col1, col2 = st.columns(2)

plate_capacity = col1.number_input(
    "Plate Capacity",
    1,
    100,
    12
)

max_plates = col2.number_input(
    "Max Plates",
    1,
    20,
    2
)


# ================================================================
# GENERATE BUTTON
# ================================================================

if st.button("🚀 Generate Plan"):

    if not demand:

        st.error(
            "❌ No Qty Data Found"
        )

        st.stop()

    # ============================================================
    # CALL YOUR ALGORITHM HERE
    # ============================================================

    st.success(
        "✅ Optimization Started"
    )

    # Example:
    #
    # result = v7_optimizer(
    #     demand,
    #     plate_capacity,
    #     max_plates
    # )

    # ============================================================
    # SHOW RESULTS
    # ============================================================

    st.write("Results Here")


# ================================================================
# EXPORT SYSTEM
# ================================================================

# YOUR EXISTING EXPORT SYSTEM HERE


# ================================================================
# FOOTER
# ================================================================

st.caption(
    "🔥 Industrial Pre-Press Optimization Software"
)
