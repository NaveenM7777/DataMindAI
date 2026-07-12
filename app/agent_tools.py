import pandas as pd
import streamlit as st

from app.context_builder import build_system_instruction
from app.grok_client import send_message as ai_chat_send_message

from app.rag_embeddings import create_query_embedding
from app.rag_llm import ask_with_context

from app.model_trainer import train_all_models


# ==========================================================
# Tool: AI Chat (dataset / ML reasoning)
# ==========================================================

def tool_ai_chat(df, question):

    if df is None:

        return "No dataset is loaded, so I can't answer dataset-related questions right now."

    try:

        system_instruction = build_system_instruction(df)

        reply = ai_chat_send_message(
            system_context=system_instruction,
            chat_history=[],
            new_message=question
        )

        return reply

    except Exception as e:

        return f"❌ AI Chat tool failed: {e}"


# ==========================================================
# Tool: RAG (PDF Q&A)
# ==========================================================

def tool_rag(question):

    if "rag_vectorstore" not in st.session_state or st.session_state["rag_vectorstore"].is_empty():

        return "No PDF has been uploaded yet. Please upload a PDF in the RAG tab first."

    try:

        query_embedding = create_query_embedding(question)

        retrieved_chunks = st.session_state["rag_vectorstore"].search(
            query_embedding,
            top_k=3
        )

        answer = ask_with_context(
            question=question,
            retrieved_chunks=retrieved_chunks,
            chat_history=[]
        )

        return answer

    except Exception as e:

        return f"❌ RAG tool failed: {e}"


# ==========================================================
# Tool: Quick ML Summary
# ==========================================================

def tool_ml_quick(df, target_column=None):

    if df is None:

        return "No dataset is loaded, so I can't train any models right now."

    try:

        df_clean = df.dropna(axis=1, how="all")

        if target_column and target_column in df_clean.columns:

            target = target_column

        else:

            likely_names = ["target", "label", "class", "outcome", "y", "result"]

            detected = None

            for col in df_clean.columns:

                if col.strip().lower() in likely_names:

                    detected = col

                    break

            if detected:

                target = detected

            else:

                return {
                    "needs_target_selection": True,
                    "available_columns": df_clean.columns.tolist()
                }

        feature_cols = [c for c in df_clean.columns if c != target]

        if len(feature_cols) == 0:

            return "Dataset doesn't have enough columns to train a model."

        X = pd.get_dummies(df_clean[feature_cols], drop_first=True).astype(float)

        X = X.fillna(X.median(numeric_only=True))

        y = df_clean[target]

        if y.isnull().sum() > 0:

            valid_idx = y.dropna().index

            X = X.loc[valid_idx]

            y = y.loc[valid_idx]

        if y.dtype == "object" or y.nunique() <= 10:

            problem_type = "Classification"

            y = y.astype(str)

        else:

            problem_type = "Regression"

            y = pd.to_numeric(y, errors="coerce")

        if len(X) < 5:

            return "Not enough valid rows to train a quick model."

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        result_df, best_model_name, trained_model, prediction = train_all_models(
            X_train, X_test, y_train, y_test,
            problem_type, "⚡ Quick Train"
        )

        best_row = result_df.iloc[0]

        score_col = "Accuracy" if problem_type == "Classification" else "R2 Score"

        summary = (
            f"Quick ML Summary (target column: '{target}'):\n"
            f"- Problem type detected: {problem_type}\n"
            f"- Best model: {best_model_name}\n"
            f"- {score_col}: {best_row[score_col]:.4f}\n"
            f"- Training time: {best_row['Training Time (s)']:.3f} sec\n\n"
            f"Note: this is a quick estimate using default settings. "
            f"For full control over test size, random state, and evaluation, use the Machine Learning tab."
        )

        return summary

    except Exception as e:

        return f"❌ ML tool failed: {e}"


# ==========================================================
# Helper: compute raw EDA stats (deterministic, no LLM)
# ==========================================================

def _compute_eda_stats(df):

    lines = []

    lines.append(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns.")

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) > 0:

        lines.append(
            "Missing values found in: " +
            ", ".join([f"{col} ({cnt})" for col, cnt in missing.items()])
        )

    else:

        lines.append("No missing values found.")

    numeric_df = df.select_dtypes(include="number")

    top_pairs_text = []

    if numeric_df.shape[1] >= 2:

        corr = numeric_df.corr().abs()

        corr_pairs = (
            corr.where(~corr.isna())
            .unstack()
            .sort_values(ascending=False)
        )

        corr_pairs = corr_pairs[corr_pairs < 1.0]

        seen_pairs = set()

        for (col1, col2), value in corr_pairs.items():

            pair_key = tuple(sorted([col1, col2]))

            if pair_key in seen_pairs:

                continue

            seen_pairs.add(pair_key)

            top_pairs_text.append(f"{col1} & {col2}: {value:.2f}")

            if len(top_pairs_text) >= 5:

                break

        if top_pairs_text:

            lines.append("Top correlated pairs: " + "; ".join(top_pairs_text))

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if cat_cols:

        lines.append(f"Categorical columns: {', '.join(cat_cols)}")

    num_cols = numeric_df.columns.tolist()

    if num_cols:

        lines.append(f"Numeric columns: {', '.join(num_cols)}")

        desc = numeric_df.describe().round(2)

        lines.append("Numeric summary:\n" + desc.to_string())

    return "\n".join(lines)


# ==========================================================
# Tool: EDA Summary (now phrased dynamically via LLM)
# ==========================================================

def tool_eda_summary(df, question=None):

    if df is None:

        return "No dataset is loaded, so I can't generate an EDA summary."

    try:

        raw_stats = _compute_eda_stats(df)

        question_text = question or "Give me an exploratory data analysis summary of this dataset."

        prompt = f"""The user asked: "{question_text}"

Here are the actual computed statistics for the dataset:

{raw_stats}

Answer the user's specific question using these real statistics. Do not just
list the raw stats verbatim — phrase the answer naturally based on what they
asked, highlighting whichever stats are most relevant to their question.
Keep it concise.
"""

        reply = ai_chat_send_message(
            system_context="You are a data analysis assistant. Always ground your answer in the exact statistics provided, never invent numbers.",
            chat_history=[],
            new_message=prompt
        )

        return reply

    except Exception as e:

        # Fallback to raw stats if the LLM call fails, so the tool never breaks
        return _compute_eda_stats(df)


# ==========================================================
# Helper: compute raw preprocessing stats (deterministic)
# ==========================================================

def _compute_preprocessing_stats(df):

    lines = []

    total_missing = df.isnull().sum().sum()

    lines.append(f"Total missing values: {total_missing}")

    duplicate_count = df.duplicated().sum()

    lines.append(f"Duplicate rows: {duplicate_count}")

    dtype_counts = df.dtypes.value_counts()

    lines.append("Column types: " + ", ".join([f"{dt}: {cnt}" for dt, cnt in dtype_counts.items()]))

    recommendations = []

    if total_missing > 0:

        recommendations.append("handle missing values (median/mode fill or drop rows)")

    if duplicate_count > 0:

        recommendations.append("remove duplicate rows")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if cat_cols:

        recommendations.append(f"encode categorical columns ({', '.join(cat_cols)})")

    if recommendations:

        lines.append("Recommended preprocessing: " + "; ".join(recommendations))

    else:

        lines.append("Dataset looks clean — no major preprocessing needed.")

    return "\n".join(lines)


# ==========================================================
# Tool: Preprocessing Summary (now phrased dynamically via LLM)
# ==========================================================

def tool_preprocessing_summary(df, question=None):

    if df is None:

        return "No dataset is loaded, so I can't check preprocessing needs."

    try:

        raw_stats = _compute_preprocessing_stats(df)

        question_text = question or "What preprocessing does this dataset need?"

        prompt = f"""The user asked: "{question_text}"

Here are the actual computed preprocessing checks for the dataset:

{raw_stats}

Answer the user's specific question using these real checks. Do not just list
the raw stats verbatim — phrase the answer naturally based on what they asked.
Keep it concise and practical.
"""

        reply = ai_chat_send_message(
            system_context="You are a data preprocessing assistant. Always ground your answer in the exact checks provided, never invent numbers.",
            chat_history=[],
            new_message=prompt
        )

        return reply

    except Exception as e:

        return _compute_preprocessing_stats(df)


# ==========================================================
# Tool Registry
# ==========================================================

TOOL_REGISTRY = {

    "ai_chat": {
        "description": (
            "Explains the dataset, EDA, machine learning results, best model, "
            "accuracy, business insights, or answers general questions about "
            "the currently loaded dataset."
        ),
        "needs": ["df", "question"]
    },

    "rag": {
        "description": (
            "Answers questions from an uploaded PDF document — such as "
            "summarizing it, explaining a section, or finding specific values "
            "or policies within it."
        ),
        "needs": ["question"]
    },

    "ml_quick": {
        "description": (
            "Trains machine learning models on the dataset and returns the "
            "best model with its score. Use when the user asks to train, "
            "build, or find the best model."
        ),
        "needs": ["df"]
    },

    "eda_summary": {
        "description": (
            "Generates a quick exploratory data analysis summary — shape, "
            "missing values, correlations, column types. Use when the user "
            "asks for EDA, statistics, or data overview."
        ),
        "needs": ["df"]
    },

    "preprocessing_summary": {
        "description": (
            "Checks the dataset for missing values, duplicates, and suggests "
            "preprocessing steps needed. Use when the user asks about data "
            "cleaning or preprocessing recommendations."
        ),
        "needs": ["df"]
    }

}


def execute_tool(tool_name, df, question, target_column=None):
    """
    Dispatches to the correct tool function based on tool_name.
    """

    if tool_name == "ai_chat":

        return tool_ai_chat(df, question)

    elif tool_name == "rag":

        return tool_rag(question)

    elif tool_name == "ml_quick":

        return tool_ml_quick(df, target_column)

    elif tool_name == "eda_summary":

        return tool_eda_summary(df, question)

    elif tool_name == "preprocessing_summary":

        return tool_preprocessing_summary(df, question)

    else:

        return f"❌ Unknown tool: {tool_name}"