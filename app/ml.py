import streamlit as st
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

from app.model_trainer import train_all_models, get_models

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

from app.hyperparameter_tuning import (
    tune_hyperparameters,
    has_tunable_params,
    PARAM_GRIDS
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
    # Drop DateTime columns — cannot be converted to float
    # =====================================================

    datetime_cols = feature_df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    if len(datetime_cols) > 0:

        st.warning(
            f"⚠️ Dropping {len(datetime_cols)} DateTime column(s) — "
            f"not supported for ML: {', '.join(datetime_cols)}"
        )

        feature_df = feature_df.drop(columns=datetime_cols)

        # Update column lists after dropping
        numeric_cols = [c for c in numeric_cols if c not in datetime_cols]
        categorical_cols = [c for c in categorical_cols if c not in datetime_cols]

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
        st.session_state["ml_y_train"] = y_train
        st.session_state["ml_problem_type"] = problem_type

        # Clear any previous tuning results since a fresh training run happened
        st.session_state.pop("tuned_model", None)
        st.session_state.pop("tuned_model_name", None)

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
        y_train = st.session_state["ml_y_train"]
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

        # =================================================
        # Hyperparameter Tuning
        # =================================================

        st.markdown("---")

        st.subheader("🎯 Hyperparameter Tuning")

        st.caption(
            "Fine-tune a specific model's hyperparameters using Grid Search or "
            "Randomized Search, and compare performance before vs after tuning."
        )

        available_models = result_df["Model"].tolist()

        tunable_models = [m for m in available_models if has_tunable_params(m)]

        if len(tunable_models) == 0:

            st.info("No tunable models available in this comparison.")

        else:

            selected_tune_models = st.multiselect(
                "Select Model(s) to Tune",
                tunable_models,
                default=tunable_models[:1],
                key="tune_model_select"
            )

            search_type = st.radio(
                "Search Method",
                ["Grid Search", "Randomized Search"],
                horizontal=True,
                key="tune_search_type"
            )

            if selected_tune_models:

                with st.expander("🔍 View Parameter Grid(s) Being Searched"):

                    for m in selected_tune_models:

                        st.markdown(f"**{m}**")
                        st.json(PARAM_GRIDS.get(m, {}))

            if st.button("🚀 Run Hyperparameter Tuning"):

                if not selected_tune_models:

                    st.warning("Please select at least one model to tune.")

                else:

                    tuning_results = {}

                    for model_name in selected_tune_models:

                        fresh_models = get_models(problem_type, "🚀 Full AutoML")

                        if model_name not in fresh_models:

                            fresh_models = get_models(problem_type, "⚡ Quick Train")

                        base_model = fresh_models.get(model_name)

                        if base_model is None:

                            tuning_results[model_name] = {
                                "error": "Could not reconstruct the base model for tuning."
                            }

                            continue

                        with st.spinner(f"Running {search_type} on {model_name}... this may take a moment."):

                            try:

                                # Fit the base model first (untuned baseline)
                                base_model.fit(X_train, y_train)

                                result = tune_hyperparameters(
                                    model=base_model,
                                    model_name=model_name,
                                    X_train=X_train,
                                    y_train=y_train,
                                    X_test=X_test,
                                    y_test=y_test,
                                    problem_type=problem_type,
                                    search_type=search_type
                                )

                            except Exception as e:

                                result = {"error": f"Tuning failed: {e}"}

                        tuning_results[model_name] = result

                    st.session_state["tuning_results"] = tuning_results

            # =================================================
            # Render tuning results (multi-model comparison)
            # =================================================

            if "tuning_results" in st.session_state:

                tuning_results = st.session_state["tuning_results"]

                valid_results = {
                    m: r for m, r in tuning_results.items() if "error" not in r
                }

                failed_results = {
                    m: r for m, r in tuning_results.items() if "error" in r
                }

                for m, r in failed_results.items():

                    st.warning(f"⚠️ {m}: {r['error']}")

                if valid_results:

                    st.success("✅ Tuning Completed")

                    score_label = list(valid_results.values())[0]["score_label"]

                    # ---- Comparison table across all tuned models ----

                    st.markdown("#### 📊 Model Comparison — Before vs After Tuning")

                    comparison_rows = []

                    for m, r in valid_results.items():

                        comparison_rows.append({
                            "Model": m,
                            f"Before ({score_label})": r["before_score"],
                            f"After ({score_label})": r["after_score"],
                            "Improvement": r["improvement"],
                            "Tuning Time (s)": r["tuning_time"]
                        })

                    comparison_df = pd.DataFrame(comparison_rows).sort_values(
                        by=f"After ({score_label})", ascending=False
                    ).reset_index(drop=True)

                    st.dataframe(comparison_df, use_container_width=True)

                    best_tuned_name = comparison_df.iloc[0]["Model"]

                    st.info(f"🏆 Best model after tuning: **{best_tuned_name}**")

                    # ---- Per-model details ----

                    for m, r in valid_results.items():

                        with st.expander(f"🔍 Details — {m}"):

                            best_params_df = pd.DataFrame(
                                list(r["best_params"].items()),
                                columns=["Parameter", "Best Value"]
                            )

                            st.dataframe(best_params_df, use_container_width=True)

                            if r["improvement"] > 0:

                                st.success(
                                    f"🎉 Improved {score_label} by {r['improvement']:.4f} "
                                    f"({r['improvement']*100:.2f}%)."
                                )

                            elif r["improvement"] == 0:

                                st.info(
                                    "No improvement — the default parameters were already "
                                    "optimal for this search space."
                                )

                            else:

                                st.warning(
                                    "The tuned model performed slightly worse on this test "
                                    "split. This can happen with small datasets or high "
                                    "variance — consider a different search space or more "
                                    "cross-validation folds."
                                )

                    # ---- Download a tuned model ----

                    st.markdown("---")

                    download_choice = st.selectbox(
                        "Select tuned model to download",
                        list(valid_results.keys()),
                        key="tuned_download_select"
                    )

                    tuned_model_bytes = pickle.dumps(
                        valid_results[download_choice]["tuned_model"]
                    )

                    st.download_button(
                        label=f"⬇ Download Tuned Model ({download_choice}.pkl)",
                        data=tuned_model_bytes,
                        file_name=f"{download_choice.replace(' ', '_')}_tuned.pkl",
                        mime="application/octet-stream"
                    )