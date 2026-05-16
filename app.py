# app.py — PLATE RATIO SYSTEM | DESIGN BY OVI (WITH IMPROVED PDF PARSING)

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
from io import BytesIO
from collections import Counter
from math import ceil, floor
import string
import re
import PyPDF2

st.set_page_config(
    page_title="Plate Ratio System | Design by Ovi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================================================
#  PASSWORD CHECK SYSTEM
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

    # CSS for password page
    st.markdown("""
    <style>
        .stApp { background: black !important; }
        .main > div { background: transparent !important; padding: 0 !important; }
        .block-container { padding: 0rem !important; max-width: 100% !important; }
        .stTextInput input {
            background: rgba(255,255,255,0.1) !important;
            border: 2px solid #333 !important;
            border-radius: 10px !important;
            color: white !important;
            padding: 14px !important;
            text-align: center !important;
        }
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 1rem 0rem 1rem;
            text-align: center;
        }
        .main-header h1 { color: white; font-size: 2.5rem; margin-bottom: 0.5rem; }
        .main-header p { color: rgba(255,255,255,0.9); font-size: 1.1rem; }
        .designer-name { color: #ffd700; font-size: 1rem; margin-top: 0.5rem; }
        .password-container {
            max-width: 450px;
            margin: 60px auto 0 auto;
            padding: 2.5rem;
            background: rgba(0, 0, 0, 0.85);
            border-radius: 20px;
            text-align: center;
            border: 1px solid rgba(102,126,234,0.5);
        }
        .password-container h2 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.8rem;
        }
        .password-container p { color: #aaa; margin-bottom: 1.5rem; }
        .stAlert { max-width: 450px; margin: 1rem auto !important; }
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>📊 Plate Ratio System</h1>
        <p>Professional UPS Ratio Optimization | Low Waste + Smart Distribution</p>
        <p class="designer-name">✨ Design by Ovi ✨</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="password-container">', unsafe_allow_html=True)
    st.markdown('<h2>🔐 Access Code</h2>', unsafe_allow_html=True)
    st.markdown('<p>Please enter your access code to continue</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.text_input("Enter Your Access Code", type="password", key="password", 
                      on_change=_password_entered, label_visibility="collapsed")
    
    if st.session_state.get("password_correct") is False:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.error("❌ Your password is incorrect. Please contact Mr. Ovi for assistance.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return False

if not check_password():
    st.stop()

# ================================================================
#  IMPROVED PDF QTY EXTRACTOR
# ================================================================

def extract_qty_from_pdf(pdf_file):
    """Extract quantities from PDF file - Improved version"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        all_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
        
        # Debug: Show extracted text
        st.text_area("📄 Extracted PDF Text (Debug)", all_text[:2000], height=150)
        
        quantities = {}
        
        # ============================================
        # METHOD 1: Look for patterns like "ITEM: XXX QTY: 100"
        # ============================================
        
        # Pattern for: Product Code/Name followed by quantity
        # Look for lines that have product codes and numbers
        
        lines = all_text.split('\n')
        
        # Common product code patterns (SKU, Item codes)
        sku_pattern = r'(\d{8,}|\d{4,}[A-Z0-9]+|\d+[-\s]\d+|[A-Z0-9]{6,})'
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 3:
                continue
            
            # Skip lines that are clearly not product data
            skip_keywords = ['Total', 'Value', 'USD', 'Special', 'Instruction', 'Date', 'Page', 'Order']
            if any(keyword in line for keyword in skip_keywords):
                continue
            
            # Look for numbers that might be quantities (between 1 and 1,000,000)
            numbers = re.findall(r'\b([1-9][0-9]{0,5})\b', line)
            
            # Look for product/sku patterns
            skus = re.findall(sku_pattern, line)
            
            # If line has both product code and a number
            if skus and numbers:
                for sku in skus:
                    for num in numbers:
                        qty = int(num)
                        # Skip if number is too large (over 1 million)
                        if 1 <= qty <= 1000000 and qty != 2026 and qty != 45294:
                            # Clean up SKU
                            sku_clean = sku.strip()
                            if sku_clean and len(sku_clean) > 3:
                                if sku_clean not in quantities:
                                    quantities[sku_clean] = qty
                                    break
        
        # ============================================
        # METHOD 2: Look for specific patterns
        # ============================================
        
        # Pattern: "95 MMW26 2173783000012 1" -> where last number is quantity
        pattern1 = r'([A-Z0-9\s]{10,})\s+(\d{1,5})\s*$'
        matches = re.findall(pattern1, all_text, re.MULTILINE)
        for match in matches:
            tag = match[0].strip()
            qty = int(match[1])
            if tag and 1 <= qty <= 1000000:
                if tag not in quantities:
                    quantities[tag] = qty
        
        # Pattern: Quantity before product code
        pattern2 = r'(\d{1,5})\s+([A-Z0-9\s]{10,})'
        matches = re.findall(pattern2, all_text, re.MULTILINE)
        for match in matches:
            qty = int(match[0])
            tag = match[1].strip()
            if tag and 1 <= qty <= 1000000:
                if tag not in quantities:
                    quantities[tag] = qty
        
        # ============================================
        # METHOD 3: Manual data entry option
        # ============================================
        
        if not quantities or len(quantities) == 0:
            st.warning("⚠️ Could not automatically extract quantities from PDF.")
            st.info("Please use the manual input option or ensure your PDF has product codes and quantities in a clear format.")
            
            # Show a template for expected format
            st.markdown("""
            **Expected PDF Format:**
