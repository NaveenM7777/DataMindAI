import streamlit as st


def apply_theme():

    st.markdown("""
        <style>

        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }

        /* ---- App background ---- */
        .stApp {
            background: radial-gradient(circle at 20% 20%, #1a1a2e 0%, #0d0d17 45%, #05050a 100%);
            color: #e6e6f0;
        }

        /* ---- Headings ---- */
        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
            color: #ffffff !important;
            text-shadow: 0 0 12px rgba(0, 255, 200, 0.25);
        }

        h1 {
            background: linear-gradient(90deg, #00f5d4, #7b2ff7, #00c2ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-size: 200% auto;
            animation: shimmer 6s linear infinite;
        }

        @keyframes shimmer {
            0% { background-position: 0% center; }
            100% { background-position: 200% center; }
        }

        /* ---- Fade-in animation for main content ---- */
        .main .block-container {
            animation: fadeSlideIn 0.6s ease-out;
        }

        @keyframes fadeSlideIn {
            0% { opacity: 0; transform: translateY(14px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        /* ---- Cards / containers ---- */
        div[data-testid="stMetric"], .stDataFrame, .stAlert, div[data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(0, 245, 212, 0.15);
            border-radius: 14px;
            padding: 6px;
            box-shadow: 0 0 18px rgba(0, 245, 212, 0.06);
            transition: all 0.3s ease;
        }

        div[data-testid="stMetric"]:hover {
            box-shadow: 0 0 22px rgba(0, 245, 212, 0.35);
            border: 1px solid rgba(0, 245, 212, 0.5);
            transform: translateY(-2px);
        }

        /* ---- Buttons ---- */
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(135deg, #00f5d4, #7b2ff7);
            color: #05050a;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            padding: 0.55em 1.4em;
            box-shadow: 0 0 14px rgba(0, 245, 212, 0.35);
            transition: all 0.25s ease;
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            box-shadow: 0 0 26px rgba(123, 47, 247, 0.65);
            transform: translateY(-2px) scale(1.02);
            color: #ffffff;
        }

        /* ---- Inputs ---- */
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
            background-color: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid rgba(0, 245, 212, 0.25) !important;
            border-radius: 8px !important;
            color: #e6e6f0 !important;
        }

        /* ---- Chat bubbles ---- */
        div[data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 14px;
            border: 1px solid rgba(123, 47, 247, 0.2);
            box-shadow: 0 0 14px rgba(123, 47, 247, 0.08);
            animation: fadeSlideIn 0.4s ease-out;
        }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d0d17 0%, #05050a 100%);
            border-right: 1px solid rgba(0, 245, 212, 0.15);
        }

        /* ---- Progress / spinner glow ---- */
        .stSpinner > div {
            border-top-color: #00f5d4 !important;
        }

        /* ---- Scrollbar ---- */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0d0d17;
        }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(#00f5d4, #7b2ff7);
            border-radius: 10px;
        }

        </style>
    """, unsafe_allow_html=True)