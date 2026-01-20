import streamlit as st

def apply_auth_styles():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #0e1117, #1a1c24, #001f3f, #080a0f);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .main {
        background-color: #f0f2f6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        justify-content: center;
    }
    .login-container {
        padding: 2rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* Deep Navy Dark Theme */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    .stTextInput > div > div > input {
        background-color: #1a1c24;
        color: #00d4ff;
        border: 1px solid #00d4ff;
    }

    /* Glowing Button - Centered and 50% width */
    /* This targets the primary buttons on the auth page specifically to override global styles. */
    .stButton>button[kind="primary"] {
        width: 50% !important;
        margin: 0 auto !important;
        display: block !important;
        height: 3em !important;
        border-radius: 5px !important;
        background: #00d4ff !important;
        color: #fafbfc !important;
        font-weight: bold !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.4) !important;
        transition: 0.3s !important;
        border: none !important;
    }

    .stButton>button[kind="primary"]:hover {
        box-shadow: 0 0 25px rgba(0, 212, 255, 0.6) !important;
        transform: translateY(-2px) !important;
        background: #00d4ff !important;
        color: #0e1117 !important;
    }
    </style>
            
    <style>
    /* Increase size and change color of the Tab labels */
    button[data-baseweb="tab"] p {
        font-size: 24px !important;
        font-weight: bold !important;
        transition: color 0.3s ease;
    }

    /* Color for the unselected tab */
    button[data-baseweb="tab"] p {
        color: #8E9AAF; /* Soft Gray-Blue */
    }

    /* Color for the active/selected tab */
    button[aria-selected="true"] p {
        color: #00D4FF !important; /* Electric Blue */
        text-shadow: 0px 0px 5px rgba(0,123,255,0.2);
    }
    
    /* Optional: Change subheader colors inside the tabs */
    .stMarkdown h3 {
        # color: #008080;
        font-size: 28px !important;
    }
    </style>
    """, unsafe_allow_html=True)