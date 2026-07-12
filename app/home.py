import streamlit as st


def show_home():

    st.title("DataMind AI")
    st.subheader("An Agentic GenAI Platform for Automated Data Science")

    st.markdown("---")

    st.info("🚀 Upload a dataset and let AI perform the complete Data Science workflow.")

    st.markdown("""
### Features

- 📂 Upload CSV / Excel
- 📋 Dataset Preview
- 📊 Dataset Summary
- 📈 Statistical Summary
- 📊 Visualizations
- 🧹 Data Preprocessing
- ⚙️ Feature Engineering
- 🤖 Machine Learning
- 💬 Gemini AI Chat
- 📄 RAG over PDFs
- 🤖 AI Agents
""")