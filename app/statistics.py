import streamlit as st


def show_statistics(df):
    st.markdown("---")
    st.subheader("📈 Statistical Summary")

    st.dataframe(df.describe())