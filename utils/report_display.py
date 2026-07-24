"""
Report Display Module - Best Algorithm summary, plate details, Excel/PDF downloads
"""
from datetime import datetime

import pandas as pd
import streamlit as st

from utils.pdf_generator import generate_pdf_report, PDF_AVAILABLE
from utils.excel_generator import generate_excel_report, EXCEL_AVAILABLE, build_full_summary


def render_report(results, comparison_df, best_algo, best_waste, best_plates, demand, original_qty, job_number):
    """Renders best-algorithm banner, comparison table, summary table, and plate details."""
    st.markdown(f"""
    <div class="best-algo">
        <h2>BEST ALGORITHM: {best_algo}</h2>
        <p style="font-size: 1.5rem; margin: 0.5rem 0;">Waste: {best_waste}%</p>
        <p style="margin: 0;">Total Algorithms Tested: {len(results)}</p>
        <p style="margin: 0; font-size: 0.85rem; opacity: 0.8;">Job Number: {job_number}</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("All Algorithms Comparison", expanded=False):
        st.dataframe(comparison_df, use_container_width=True)

    st.markdown("---")
    st.markdown("##Best Algorithm Report")

    item_meta = st.session_state.get('item_meta', {})
    meta_columns = st.session_state.get('item_meta_columns', [])

    summary_df = build_full_summary(best_plates, demand, original_qty, item_meta, meta_columns)

    if not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True, height=350)

        st.markdown("###Plate Details")
        plate_rows = []
        total_sheets_sum = 0
        total_ups_sum = 0

        for idx, p in enumerate(best_plates, 1):
            total_ups = sum(p["layout"].values())
            plate_name_str = p.get("name", f"Plate {idx}")
            plate_rows.append({
                "SL": idx,
                "Plate ID": plate_name_str,
                "Sheets": p.get("sheets", 0),
                "Total UPS": total_ups,
            })
            total_sheets_sum += p.get("sheets", 0)
            total_ups_sum += total_ups

        plate_rows.append({
            "SL": "📊",
            "Plate ID": "**TOTAL**",
            "Sheets": total_sheets_sum,
            "Total UPS": total_ups_sum,
        })

        plate_details_df = pd.DataFrame(plate_rows)
        st.dataframe(plate_details_df, use_container_width=True)


def render_downloads(best_plates, demand, original_qty, best_algo, best_waste, job_number):
    """Renders Excel and PDF download buttons."""
    st.markdown("---")
    st.markdown("## 📥 Download Reports")

    item_meta = st.session_state.get('item_meta', {})
    meta_columns = st.session_state.get('item_meta_columns', [])

    col1, col2 = st.columns(2)

    with col1:
        excel_buffer = generate_excel_report(
            best_plates, demand, original_qty, best_algo, best_waste, job_number,
            item_meta=item_meta, meta_columns=meta_columns
        )
        if excel_buffer is not None:
            st.download_button(
                "📊 Download Excel Report",
                excel_buffer,
                f"Job Number_{job_number}_Plate Ratio.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.error("❌ Excel report could not be generated. Please check the data.")

    with col2:
        pdf_buffer = generate_pdf_report(
            best_plates, demand, original_qty, best_algo, best_waste, job_number,
            item_meta=item_meta, meta_columns=meta_columns
        )
        if pdf_buffer is not None:
            st.download_button(
                "📄 Download PDF Report",
                pdf_buffer,
                f"Job Number_{job_number}_Plate Ratio.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.info("ℹ️ PDF download requires reportlab. Install with: pip install reportlab")
