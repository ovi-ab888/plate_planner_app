"""
UI Components - Header & Footer
"""
import streamlit as st


def render_header(title="📊 Plate Ratio Intelligence System",
                   subtitle="Complete Edition • 26 Algorithms • Production Ready"):
    """Renders the main gradient header used across the app"""
    st.markdown(f"""
    <div class="main-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
        <p style="font-size: 0.85rem; opacity: 0.8;">AI-Powered • Fast • Accurate</p>
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    """Renders the app footer"""
    st.markdown("""
    <div class="footer-border" style="text-align: center; padding: 2rem; margin-top: 3rem; border-top: 2px solid rgba(102,126,234,0.3); border-radius: 20px;">
        <p class="footer-text" style="margin: 0; font-size: 0.85rem;">
            © 2025 Plate Ratio System | Version 26 (Complete Edition)
        </p>
        <p class="footer-text" style="margin: 8px 0; font-size: 0.8rem;">
            26 Algorithms • Production Ready • AI-Powered
        </p>
        <p style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 0.85rem; font-weight: 600; margin: 10px 0 0 0;">
            ✨ Developed by Ovi | All Rights Reserved ✨
        </p>
    </div>
    """, unsafe_allow_html=True)
