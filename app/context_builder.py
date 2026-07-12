import pandas as pd
import streamlit as st


def build_dataset_context(df):
    """
    Builds a compact text summary of the current dataset state,
    pulling in whatever has been computed so far in session_state
    (preprocessing, feature engineering, ML results).
    """

    if df is None:

        return "No dataset has been uploaded yet."

    lines = []

    # ---- Basic shape ----

    lines.append(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns.")

    # ---- Columns and dtypes ----

    col_info = []

    for col in df.columns:

        dtype = str(df[col].dtype)

        n_missing = int(df[col].isnull().sum())

        n_unique = int(df[col].nunique())

        col_info.append(f"{col} ({dtype}, missing={n_missing}, unique={n_unique})")

    lines.append("Columns: " + "; ".join(col_info))

    # ---- Basic numeric stats (only if not too many columns) ----

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] > 0 and numeric_df.shape[1] <= 15:

        desc = numeric_df.describe().round(2)

        lines.append("Numeric summary:\n" + desc.to_string())

    # ---- Correlation matrix (top correlated pairs) ----

    if numeric_df.shape[1] >= 2:

        try:

            corr = numeric_df.corr().abs()

            corr_pairs = (
                corr.where(~corr.isna())
                .unstack()
                .sort_values(ascending=False)
            )

            # Remove self-correlations (always 1.0) and duplicate pairs
            corr_pairs = corr_pairs[corr_pairs < 1.0]

            seen_pairs = set()

            top_pairs_text = []

            for (col1, col2), value in corr_pairs.items():

                pair_key = tuple(sorted([col1, col2]))

                if pair_key in seen_pairs:

                    continue

                seen_pairs.add(pair_key)

                top_pairs_text.append(f"{col1} & {col2}: {value:.2f}")

                if len(top_pairs_text) >= 5:

                    break

            if top_pairs_text:

                lines.append("Top correlated feature pairs: " + "; ".join(top_pairs_text))

        except Exception:

            pass

    # ---- Feature Engineering history ----

    fe_history = st.session_state.get("fe_history", [])

    if fe_history:

        lines.append(
            "Feature engineering steps applied: " + " | ".join(fe_history)
        )

    # ---- ML results, if training has happened ----

    if "ml_result_df" in st.session_state:

        result_df = st.session_state["ml_result_df"]

        best_model = st.session_state.get("ml_best_model_name", "Unknown")

        problem_type = st.session_state.get("ml_problem_type", "Unknown")

        lines.append(f"Machine Learning has been run. Problem type: {problem_type}.")

        lines.append(f"Best model: {best_model}.")

        lines.append("Full model comparison table:\n" + result_df.to_string(index=False))

        if problem_type == "Classification":

            lines.append(
                "Metrics available: Accuracy, Precision, Recall, F1 Score, Training Time."
            )

        else:

            lines.append(
                "Metrics available: R2 Score, MAE, RMSE, Training Time."
            )

    else:

        lines.append("Machine Learning has not been run yet.")

    return "\n".join(lines)


def build_system_instruction(df):
    """
    Wraps the dataset context with instructions on how the AI
    should behave as DataMind AI's assistant.
    """

    dataset_context = build_dataset_context(df)

    system_instruction = f"""You are the AI assistant inside "DataMind AI", a data science platform.

You can answer TWO types of questions:

1. Questions about the CURRENT dataset, EDA, preprocessing, feature engineering,
   or machine learning results — for these, always use the actual dataset context
   provided below. If the context doesn't cover something (e.g. ML hasn't been run
   yet), say so clearly instead of guessing.

2. General data science, machine learning, or statistics questions — such as
   "what is Random Forest?", "explain overfitting", "what does R2 score mean?",
   "why do we scale features?", or general concepts about this project/platform
   itself. For these, answer using your own knowledge — you do NOT need the
   dataset context to answer general concept questions.

Always figure out which type of question is being asked, and answer accordingly.
If a question mixes both (e.g. "explain R2 score and what mine means"), first
explain the general concept, then apply it to the actual dataset context below.

You can also generate Python code (pandas, matplotlib, seaborn, sklearn) when asked.

Keep answers concise, practical, and easy to understand. When asked to write
code, provide clean, runnable Python code in a code block.

=== CURRENT DATASET CONTEXT ===
{dataset_context}
=== END CONTEXT ===
"""

    return system_instruction