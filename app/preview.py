import streamlit as st


def show_preview(df):
    st.markdown("---")
    st.subheader("📋 Dataset Preview")

    st.dataframe(df.head())