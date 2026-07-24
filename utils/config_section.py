"""
Configuration Section - Item count, Capacity, Max Plates, Add-on %, Job Number
"""
import streamlit as st
from datetime import datetime


def render_config():
    """Renders the 5-column configuration inputs and returns their values"""
    st.markdown("""
    <div style="background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border-radius: 20px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid rgba(255,255,255,0.1);">
        <h3 style="margin: 0 0 1rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; border-bottom: 2px solid #667eea; display: inline-block; padding-bottom: 0.5rem;">⚙️ Configuration</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        n_items = st.number_input("Number of Items", min_value=1, max_value=500, value=3)
    with col2:
        capacity = st.number_input("Plate Capacity (UPS)", min_value=1, max_value=200, value=10)
    with col3:
        max_plates = st.number_input("Max Plates", min_value=1, max_value=50, value=3)
    with col4:
        addon_percent = st.number_input("Add-on (%)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
    with col5:
        job_number = st.text_input(
            "Job Number",
            value="",
            placeholder="OFF00123456/2026",
            help="Enter a job number for tracking"
        )
        # Fila na thakle default auto-generate hobe
        if not job_number:
            job_number = f"JOB-{datetime.now().strftime('%Y%m%d_%H%M')}"

    return n_items, capacity, max_plates, addon_percent, job_number
