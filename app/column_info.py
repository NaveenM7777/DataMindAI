import streamlit as st
import pandas as pd


def show_column_info(df):

    st.markdown("---")
    st.subheader("📋 Column Information")

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(column_info)