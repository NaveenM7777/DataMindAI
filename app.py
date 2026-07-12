import streamlit as st
from streamlit_option_menu import option_menu
from app.grok_chat import show_grok_chat
from app.theme import apply_theme
from app.preprocessing import show_preprocessing
from app.home import show_home
from app.upload import upload_dataset
from app.preview import show_preview
from app.summary import show_summary
from app.column_info import show_column_info
from app.statistics import show_statistics
from app.visualization import show_visualization
from app.ml import show_ml
from app.feature_engineering import show_feature_engineering
from app.rag_chat import show_rag_chat
from app.agent_chat import show_agent_chat

from utils.logger import logger

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="DataMind AI",
    page_icon="🧠",
    layout="wide"
)

apply_theme()

# ==========================================================
# SIDEBAR — Branding + Navigation
# ==========================================================

with st.sidebar:

    st.markdown(
        "<h2 style='text-align:center;'>🧠 DataMind AI</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center; opacity:0.6;'>Version 1.0</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    selected = option_menu(

        menu_title=None,

        options=[
            "Preview", "Summary", "EDA", "Preprocessing",
            "Feature Engineering", "Machine Learning",
            "AI Chat", "RAG (PDF)", "AI Agent"
        ],

        icons=[
            "eye", "clipboard-data", "bar-chart-line", "brush",
            "gear", "cpu", "chat-dots", "file-earmark-text", "robot"
        ],

        menu_icon="cast",

        default_index=0,

        styles={
            "container": {"padding": "5px", "background-color": "transparent"},
            "icon": {"color": "#00f5d4", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "4px",
                "border-radius": "10px",
                "color": "#e6e6f0",
                "--hover-color": "rgba(0, 245, 212, 0.1)"
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #00f5d4, #7b2ff7)",
                "color": "#05050a",
                "font-weight": "600",
                "box-shadow": "0 0 14px rgba(0, 245, 212, 0.5)"
            },
        }
    )

# ==========================================================
# HOME
# ==========================================================

show_home()

logger.info("DataMind AI Started Successfully")

# ==========================================================
# UPLOAD DATASET
# ==========================================================

df = upload_dataset()

# ==========================================================
# ROUTE TO SELECTED SECTION
# ==========================================================

if df is not None:

    st.success("✅ Dataset Loaded Successfully")

    if selected == "Preview":
        show_preview(df)

    elif selected == "Summary":
        show_summary(df)
        show_column_info(df)
        show_statistics(df)

    elif selected == "EDA":
        show_visualization(df)

    elif selected == "Preprocessing":
        processed_df = show_preprocessing(df)
        if processed_df is not None:
            df = processed_df

    elif selected == "Feature Engineering":
        engineered_df = show_feature_engineering(df)
        if engineered_df is not None:
            df = engineered_df

    elif selected == "Machine Learning":
        show_ml(df)

    elif selected == "AI Chat":
        show_grok_chat(df)

    elif selected == "RAG (PDF)":
        show_rag_chat()

    elif selected == "AI Agent":
        show_agent_chat(df)


st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:gray; padding:20px;">
        Built by <b>Naveen Mahasamudram</b><br>
        DataMind AI © 2026
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("---")