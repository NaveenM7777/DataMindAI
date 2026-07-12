import streamlit as st
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

from app.model_trainer import train_all_models

from app.model_visualization import (
    show_confusion_matrix,
    show_roc_curve,
    show_feature_importance,
    show_actual_vs_predicted,
    show_residual_plot,
    show_best_model_summary,
    show_model_dashboard,
    show_regression_metrics_cards,
    show_classification_report_cards
)


def show_ml(df):

    st.title("🤖 Machine Learning")

    st.markdown("---")

    # =====================================================
    # Target Selection
    # =====================================================

    st.subheader("🎯 Target Column")

    target = st.selectbox(
        "Select Target Column",
        df.columns
    )

    # =====================================================
    # Feature Selection
    # =====================================================

    st.markdown("---")

    st.subheader("📌 Feature Selection")

    features = st.multiselect(
        "Select Feature Columns",
        [col for col in df.columns if col != target],
        default=[col for col in df.columns if col != target]
    )

    if len(features) == 0:

        st.warning("Please select at least one feature.")

        return

    # =====================================================
    # Drop rows where target is missing (BEFORE building X/y)
    # =====================================================

    if df[target].isnull().sum() > 0:

        st.warning(
            f"⚠️ Target column has {df[target].isnull().sum()} missing "
            f"values. These rows will be dropped."
        )

        df = df.dropna(subset=[target])

    # =====================================================
    # 🛡️ Automatic Safety-Net Preprocessing
    # (runs even if user skipped the Preprocessing tab)
    # =====================================================

    st.markdown("---")

    st.subheader("🛡️ Automatic Data Safety Check")

    feature_df = df[features].copy()

    numeric_cols = feature_df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    total_missing = feature_df.isnull().sum().sum()

    if total_missing > 0:

        st.warning(
            f"⚠️ Found {total_missing} missing values across feature columns. "
            f"Auto-filling: numeric columns with median, categorical columns with mode."
        )

        # ---- Numeric columns: fill with median ----
        for col in numeric_cols:

            if feature_df[col].isnull().sum() > 0:

                median_val = feature_df[col].median()

                feature_df[col] = feature_df[col].fillna(median_val)

        # ---- Categorical columns: fill with mode ----
        for col in categorical_cols:

            if feature_df[col].isnull().sum() > 0:

                mode_series = feature_df[col].mode()

                fill_val = mode_series[0] if len(mode_series) > 0 else "Unknown"

                feature_df[col] = feature_df[col].fillna(fill_val)

        st.success("✅ Missing values handled automatically.")

    else:

        st.success("✅ No missing values found in selected features.")

    if len(categorical_cols) > 0:

        st.info(
            f"🔤 Encoding {len(categorical_cols)} categorical column(s) "
            f"using One-Hot Encoding: {', '.join(categorical_cols)}"
        )

    # =====================================================
    # Encode + Convert to numeric
    # =====================================================

    X = pd.get_dummies(
        feature_df,
        drop_first=True
    ).astype(float)

    # Final safety check — replace any leftover inf/-inf
    X = X.replace([float("inf"), float("-inf")], pd.NA)

    if X.isnull().sum().sum() > 0:

        X = X.fillna(X.median(numeric_only=True))

    with st.expander("🔍 View Processed Feature Data Info"):

        st.write("X Shape:", X.shape)
        st.write("NaN Values Remaining:", X.isnull().sum().sum())
        st.write("Infinite Values:", (X == float("inf")).sum().sum())
        st.write(X.dtypes.value_counts())

    y = df[target]

    # =====================================================
    # Problem Detection
    # =====================================================

    st.markdown("---")

    st.subheader("🧠 Problem Detection")

    if y.dtype == "object" or str(y.dtype) == "category" or y.dtype == "bool":

        auto_detected = "Classification"

    elif pd.api.types.is_numeric_dtype(y):

        is_whole_numbers = (y.dropna() % 1 == 0).all()

        if is_whole_numbers and y.nunique() <= 10:
            auto_detected = "Classification"
        else:
            auto_detected = "Regression"

    else:

        auto_detected = "Classification"

    st.info(f"Auto-Detected : {auto_detected}")

    problem_type = st.radio(

        "Confirm Problem Type",

        ["Classification", "Regression"],

        index=0 if auto_detected == "Classification" else 1

    )

    if problem_type == "Regression":

        y = pd.to_numeric(y, errors="coerce")

        if y.isnull().sum() > 0:

            st.warning(
                f"⚠️ {y.isnull().sum()} target values could not be "
                f"converted to numbers and will be dropped."
            )

            valid_idx = y.dropna().index

            X = X.loc[valid_idx]
            y = y.loc[valid_idx]

    else:

        y = y.astype(str)

    # =====================================================
    # Training Mode
    # =====================================================

    st.markdown("---")

    st.subheader("⚡ Training Mode")

    training_mode = st.radio(

        "Choose Training Mode",

        ["⚡ Quick Train", "🚀 Full AutoML"]

    )

    # =====================================================
    # Train Test Split
    # =====================================================

    st.markdown("---")

    st.subheader("✂ Train Test Split")

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)

    random_state = st.number_input("Random State", value=42)

    # =====================================================
    # Train Models
    # =====================================================

    st.markdown("---")

    if st.button("🚀 Train Models"):

        if len(X) < 5:

            st.error(
                "Not enough valid rows remain after cleaning to train "
                "a model. Please check your target column."
            )

            return

        X_train, X_test, y_train, y_test = train_test_split(

            X, y,
            test_size=test_size,
            random_state=int(random_state)

        )

        with st.spinner("Training Models..."):

            try:

                (
                    result_df,
                    best_model,
                    trained_model,
                    prediction
                ) = train_all_models(

                    X_train, X_test, y_train, y_test,
                    problem_type, training_mode

                )

            except Exception as e:

                st.error(f"❌ Training failed: {e}")
                return

        st.success("✅ Training Completed")

        st.session_state["ml_result_df"] = result_df
        st.session_state["ml_best_model_name"] = best_model
        st.session_state["ml_trained_model"] = trained_model
        st.session_state["ml_prediction"] = prediction
        st.session_state["ml_X_test"] = X_test
        st.session_state["ml_X_train"] = X_train
        st.session_state["ml_y_test"] = y_test
        st.session_state["ml_problem_type"] = problem_type

    # =====================================================
    # Render results (from session_state so they persist
    # across reruns triggered by download buttons)
    # =====================================================

    if "ml_result_df" in st.session_state:

        result_df = st.session_state["ml_result_df"]
        best_model = st.session_state["ml_best_model_name"]
        trained_model = st.session_state["ml_trained_model"]
        prediction = st.session_state["ml_prediction"]
        X_test = st.session_state["ml_X_test"]
        X_train = st.session_state["ml_X_train"]
        y_test = st.session_state["ml_y_test"]
        problem_type = st.session_state["ml_problem_type"]

        st.markdown("---")

        show_best_model_summary(best_model, result_df, problem_type)

        st.markdown("---")

        st.subheader("🏆 Model Comparison")

        st.dataframe(result_df, use_container_width=True)

        show_model_dashboard(result_df, problem_type)

        st.markdown("---")

        st.subheader("⬇ Download Best Model")

        model_bytes = pickle.dumps(trained_model)

        st.download_button(
            label="⬇ Download Best Model (.pkl)",
            data=model_bytes,
            file_name=f"{best_model.replace(' ', '_')}.pkl",
            mime="application/octet-stream"
        )

        pred_df = pd.DataFrame({
            "Actual": y_test.values if hasattr(y_test, "values") else y_test,
            "Predicted": prediction
        })

        csv_bytes = pred_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Predictions (.csv)",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv"
        )

        st.markdown("---")

        st.subheader("📊 Model Evaluation")

        if problem_type == "Classification":

            show_classification_report_cards(y_test, prediction)

            st.markdown("---")

            show_confusion_matrix(trained_model, X_test, y_test)

            show_roc_curve(trained_model, X_test, y_test)

            show_feature_importance(trained_model, X_train, top_n=10)

        else:

            show_regression_metrics_cards(y_test, prediction)

            st.markdown("---")

            show_actual_vs_predicted(y_test, prediction)

            show_residual_plot(y_test, prediction)

            show_feature_importance(trained_model, X_train, top_n=10)