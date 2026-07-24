"""
Sidebar Module - Input Method Selection
"""
import streamlit as st
from algorithms import ALGORITHM_REGISTRY


def render_sidebar():
    """Renders sidebar and returns selected input_method"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
            <h3 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">📦 Input Method</h3>
        </div>
        """, unsafe_allow_html=True)

        input_method = st.radio(
            "Select Input Method:",
            ["Manual Entry", "Upload Excel"],
            index=1,
            label_visibility="collapsed"
        )

        st.markdown("---")

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%); border-radius: 10px; padding: 0.75rem; text-align: center; border: 1px solid rgba(102,126,234,0.2);">
            <p style="margin: 0; font-size: 0.75rem; color: rgba(255,255,255,0.5);">🧠 {len(ALGORITHM_REGISTRY)} Algorithms</p>
            <p style="margin: 0; font-size: 0.65rem; color: rgba(255,255,255,0.3);">V1 - V26 Complete</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        with st.expander("📊 About", expanded=False):
            st.markdown("""
            **26 Algorithms:**
            - V1-V10: Classical
            - V11-V18: Evolutionary
            - V19-V26: AI & ML

            **Features:**
            - Excel & PDF Reports
            - Dark/Light Mode
            - Production Ready
            """)

    return input_method
