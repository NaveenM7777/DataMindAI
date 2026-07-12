import streamlit as st


def create_tabs():
    return st.tabs([
        "📋 Preview",
        "📊 Summary",
        "📈 EDA",
        "🧹 Preprocessing",
        "⚙️ Feature Engineering",
        "🤖 ML",
        "💬 AI Chat",
        "📄 RAG",
        "🤖 Agents"
    ])