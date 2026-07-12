import streamlit as st


def show_summary(df):
    st.markdown("---")
    st.subheader("📊 Dataset Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    col4, col5 = st.columns(2)

    col4.metric("Duplicate Rows", df.duplicated().sum())
    col5.metric(
        "Memory Usage (KB)",
        round(df.memory_usage().sum() / 1024, 2)
    )