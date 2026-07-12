import streamlit as st
import pandas as pd
import joblib
import io


# ==========================================================
# Download Trained Model
# ==========================================================

def download_model(model):

    buffer = io.BytesIO()

    joblib.dump(
        model,
        buffer
    )

    buffer.seek(0)

    st.download_button(

        label="💾 Download Best Model (.pkl)",

        data=buffer,

        file_name="best_model.pkl",

        mime="application/octet-stream"

    )


# ==========================================================
# Download Predictions
# ==========================================================

def download_predictions(prediction):

    prediction_df = pd.DataFrame({

        "Prediction": prediction

    })

    csv = prediction_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        label="📄 Download Predictions",

        data=csv,

        file_name="predictions.csv",

        mime="text/csv"

    )


# ==========================================================
# Download Model Comparison
# ==========================================================

def download_results(result_df):

    csv = result_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        label="📊 Download Model Comparison",

        data=csv,

        file_name="model_comparison.csv",

        mime="text/csv"

    )