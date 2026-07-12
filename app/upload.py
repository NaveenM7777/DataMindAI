import streamlit as st
import pandas as pd


def upload_dataset():

    uploaded_file = st.file_uploader(
        "📂 Upload CSV or Excel File",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:

        # ==========================================
        # Read CSV
        # ==========================================

        if uploaded_file.name.endswith(".csv"):

            try:
                df = pd.read_csv(uploaded_file)

            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1")

        # ==========================================
        # Read Excel
        # ==========================================

        else:

            df = pd.read_excel(uploaded_file)

        # ==========================================
        # Convert Date Columns Automatically
        # ==========================================

        for col in df.columns:

            if df[col].dtype == "object":

                try:
                    df[col] = pd.to_datetime(df[col])

                except Exception:
                    pass

        st.success("✅ Dataset Uploaded Successfully!")

        return df

    return None