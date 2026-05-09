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
            max-width: 100% !important;
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
            padding: 14px !important;
            width: 100% !important;
            margin: 0 !important;
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
