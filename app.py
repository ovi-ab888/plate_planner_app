"""
Plate Ratio System - Complete Edition V1-V26
With Password Authentication, Modern UI & Report Download
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ================================================================
# MODULE IMPORTS
# ================================================================
from utils.auth import check_password
from utils.sidebar import render_sidebar
from utils.ui_components import render_header, render_footer
from utils.config_section import render_config
from utils.item_entry import render_manual_entry, render_excel_upload
from utils.engine import run_algorithms
from utils.report_display import render_report, render_downloads

# ================================================================
# CSS LOADER
# ================================================================
def load_css():
    """Load CSS from static/css/style.css"""
    css_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "static", "css", "style.css"
    )
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            return f"<style>{f.read()}</style>"
    return ""


# ================================================================
# STREAMLIT PAGE CONFIGURATION
# ================================================================
st.set_page_config(
    page_title="Plate Ratio System - Complete Edition",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(load_css(), unsafe_allow_html=True)

# ================================================================
# APP START - PASSWORD CHECK
# ================================================================
if not check_password():
    st.stop()

# ================================================================
# SIDEBAR
# ================================================================
input_method = render_sidebar()

# ================================================================
# MAIN HEADER
# ================================================================
render_header()

# ================================================================
# CONFIGURATION SECTION
# ================================================================
n_items, capacity, max_plates, addon_percent, job_number = render_config()

# ================================================================
# ITEM ENTRY
# ================================================================
st.markdown("""
<div style="background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border-radius: 20px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid rgba(255,255,255,0.1);">
    <h3 style="margin: 0 0 1rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; border-bottom: 2px solid #667eea; display: inline-block; padding-bottom: 0.5rem;">📦 Item Details</h3>
</div>
""", unsafe_allow_html=True)

data = []
original_qty = {}
demand = {}

if input_method == "Manual Entry":
    data = render_manual_entry(n_items)
else:
    original_qty, demand = render_excel_upload(addon_percent)
    st.markdown('</div>', unsafe_allow_html=True)

# ================================================================
# GENERATE BUTTON
# ================================================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_clicked = st.button("Generate Plans", type="primary", use_container_width=True)

# ================================================================
# RESULTS SECTION
# ================================================================
if generate_clicked:
    if not data and not demand:
        st.error("⚠️ Please enter at least one item with quantity > 0")
    else:
        # If data came from manual entry, build demand/original_qty from it
        if data:
            demand = {}
            original_qty = {}
            for item in data:
                tag = item["Style"]
                qty = item["Quantity"]
                if qty > 0:
                    demand[tag] = int(qty * (1 + addon_percent / 100))
                    original_qty[tag] = int(qty)
        # else: demand/original_qty already populated by render_excel_upload()

        if not demand:
            st.error("⚠️ No valid data found!")
        else:
            results, comparison_df, best_algo, best_waste, best_plates = run_algorithms(
                demand, capacity, max_plates
            )

            if results is None:
                st.error("❌ No algorithm succeeded!")
            else:
                render_report(
                    results, comparison_df, best_algo, best_waste,
                    best_plates, demand, original_qty, job_number
                )
                render_downloads(
                    best_plates, demand, original_qty,
                    best_algo, best_waste, job_number
                )

# ================================================================
# FOOTER
# ================================================================
render_footer()
