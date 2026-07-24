"""
Password Authentication Module
"""
import streamlit as st
import os


def check_password():
    expected_passwords = None
    try:
        # It can be a list of passwords in secrets.toml: app_passwords = ["pass1", "pass2"]
        expected_passwords = st.secrets.get("app_passwords", None)
        if expected_passwords is None:
            # Fallback to single password secret if app_passwords isn't found
            single_pass = st.secrets.get("app_password", None)
            if single_pass:
                expected_passwords = [single_pass]
    except Exception:
        pass

    # Fallback to Environment Variable
    if expected_passwords is None:
        env_pass = os.environ.get("PEPCO_APP_PASSWORD")
        if env_pass:
            expected_passwords = [env_pass]

    if expected_passwords is None:
        st.error("App passwords not configured. Please set 'app_passwords' in secrets or PEPCO_APP_PASSWORD environment variable.")
        return False

    # Ensure it's a list or set for membership testing
    if isinstance(expected_passwords, str):
        expected_passwords = [expected_passwords]

    def _password_entered():
        entered_pass = st.session_state.get("password", "")
        if entered_pass in expected_passwords:
            st.session_state["password_correct"] = True
            try:
                del st.session_state["password"]
            except Exception:
                pass
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", None) is True:
        return True

    # Password Page Styling (flat design, no gradient/blur - consistent with style.css)
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp {
        background: #1a1a2e;
    }
    .main > div { background: transparent !important; padding: 0 !important; }
    .block-container { padding: 0rem !important; max-width: 90% !important; }
    .main-header {
        background: #23233f;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 1rem 0rem 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .main-header h1 {
        color: #8b93f8;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .password-container {
        max-width: 460px;
        margin: 40px auto 8px auto;
        padding: 2.8rem 2rem 1.8rem 2rem;
        background: #23233f;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stTextInput { margin-top: -10px !important; }
    .stTextInput input {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
        color: white !important;
        text-align: center !important;
        font-size: 1.1rem !important;
        padding: 0.9rem 1.5rem !important;
        letter-spacing: 3px;
    }
    #MainMenu, header, footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1>Plate Ratio System</h1>
        <p>Intelligent Production Planning & Ratio Optimization</p>
        <p style="font-size: 0.85rem; opacity: 0.8;">AI-Powered • Fast • Accurate</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="password-container">
        <h2 style="color: white; margin: 0;">Welcome Back</h2>
        <p style="color: rgba(255,255,255,0.7); margin-top: 1rem;">Enter your secure access code to continue</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.4, 1.1, 1.4])
    with col2:
        st.text_input(
            label="",
            type="password",
            key="password",
            on_change=_password_entered,
            label_visibility="collapsed",
            placeholder="••••••••"
        )
        if st.session_state.get("password_correct") is False:
            st.error("Your password is incorrect. Please contact the administrator.")

    return False
