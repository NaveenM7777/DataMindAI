import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns


# ==========================================================
# Confusion Matrix
# ==========================================================

def show_confusion_matrix(model, X_test, y_test):

    st.subheader("🧩 Confusion Matrix")

    try:

        prediction = model.predict(X_test)

        cm = confusion_matrix(y_test, prediction)

        fig, ax = plt.subplots(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)

    except Exception as e:

        st.warning(f"Confusion Matrix could not be generated.\n{e}")


# ==========================================================
# ROC Curve
# ==========================================================

def show_roc_curve(model, X_test, y_test):

    st.subheader("📈 ROC Curve")

    if len(set(y_test)) != 2:
        st.info("ROC Curve is available only for Binary Classification.")
        return

    try:

        if hasattr(model, "predict_proba"):

            y_score = model.predict_proba(X_test)[:, 1]

            pos_label = model.classes_[1]

        elif hasattr(model, "decision_function"):

            y_score = model.decision_function(X_test)

            pos_label = model.classes_[1] if hasattr(model, "classes_") else None

        else:

            st.info("ROC Curve not supported for this model.")
            return

        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=pos_label)

        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")

        st.pyplot(fig)

    except Exception as e:

        st.warning(f"ROC Curve could not be generated.\n{e}")


# ==========================================================
# Classification Report (Feature 6)
# ==========================================================

def show_classification_report_cards(y_test, prediction):

    st.subheader("📋 Classification Report")

    try:

        report = classification_report(
            y_test,
            prediction,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        # Separate per-class rows from summary rows (accuracy, avg rows)
        summary_rows = ["accuracy", "macro avg", "weighted avg"]

        class_rows = report_df.drop(
            index=[r for r in summary_rows if r in report_df.index],
            errors="ignore"
        )

        # ---- Overall accuracy metric ----
        if "accuracy" in report:

            st.metric("Overall Accuracy", f"{report['accuracy']:.4f}")

        # ---- Per-class table, styled ----
        st.dataframe(

            class_rows[["precision", "recall", "f1-score", "support"]]
                .round(4),

            use_container_width=True

        )

        # ---- Weighted avg summary as metric cards ----
        if "weighted avg" in report:

            w = report["weighted avg"]

            col1, col2, col3 = st.columns(3)

            col1.metric("Weighted Precision", f"{w['precision']:.4f}")
            col2.metric("Weighted Recall", f"{w['recall']:.4f}")
            col3.metric("Weighted F1 Score", f"{w['f1-score']:.4f}")

    except Exception as e:

        st.warning(f"Classification Report could not be generated.\n{e}")


# ==========================================================
# Feature Importance — Top 10, sorted, horizontal chart (Feature 8)
# ==========================================================

def show_feature_importance(model, X_train, top_n=10):

    st.subheader("🌳 Feature Importance")

    feature_names = X_train.columns

    try:

        if hasattr(model, "feature_importances_"):

            importance = model.feature_importances_

        elif hasattr(model, "coef_"):

            importance = np.abs(model.coef_)

            if importance.ndim > 1:
                importance = importance[0]

        else:

            st.info(
                f"⚠️ {type(model).__name__} does not provide Feature Importance."
            )

            return

        importance_df = (

            pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance
            })
            .sort_values(by="Importance", ascending=False)
            .reset_index(drop=True)

        )

        top_features = importance_df.head(top_n)

        st.dataframe(
            top_features,
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.barh(
            top_features["Feature"],
            top_features["Importance"],
            color="#4C72B0"
        )

        ax.invert_yaxis()
        ax.set_title(f"Top {top_n} Feature Importance")
        ax.set_xlabel("Importance")

        st.pyplot(fig)

    except Exception as e:

        st.warning(f"Unable to generate Feature Importance.\n{e}")


# ==========================================================
# Actual vs Predicted (Regression)
# ==========================================================

def show_actual_vs_predicted(y_test, prediction):

    st.subheader("🎯 Actual vs Predicted")

    try:

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.scatter(y_test, prediction, alpha=0.6)

        min_val = min(min(y_test), min(prediction))
        max_val = max(max(y_test), max(prediction))

        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red")

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")

        st.pyplot(fig)

    except Exception as e:

        st.warning(f"Actual vs Predicted plot could not be generated.\n{e}")


# ==========================================================
# Residual Plot (Regression)
# ==========================================================

def show_residual_plot(y_test, prediction):

    st.subheader("📉 Residual Plot")

    try:

        residuals = np.array(y_test) - np.array(prediction)

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.scatter(prediction, residuals, alpha=0.6)
        ax.axhline(y=0, linestyle="--", color="red")

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.set_title("Residual Plot")

        st.pyplot(fig)

    except Exception as e:

        st.warning(f"Residual plot could not be generated.\n{e}")


# ==========================================================
# Regression Metrics Cards (Feature 5)
# ==========================================================

def show_regression_metrics_cards(y_test, prediction):

    st.subheader("📐 Regression Metrics")

    try:

        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        r2 = r2_score(y_test, prediction)
        mae = mean_absolute_error(y_test, prediction)
        mse = mean_squared_error(y_test, prediction)
        rmse = mse ** 0.5

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("R² Score", f"{r2:.4f}")
        col2.metric("MAE", f"{mae:.4f}")
        col3.metric("MSE", f"{mse:.4f}")
        col4.metric("RMSE", f"{rmse:.4f}")

    except Exception as e:

        st.warning(f"Regression metrics could not be generated.\n{e}")


# ==========================================================
# Best Model Summary Card (Feature 1)
# ==========================================================

def show_best_model_summary(best_model_name, result_df, problem_type):

    st.subheader("🏆 Best Model Summary")

    try:

        best_row = result_df.iloc[0]

        score_col = "Accuracy" if problem_type == "Classification" else "R2 Score"

        secondary_col = "F1 Score" if problem_type == "Classification" else "RMSE"

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Model Name", best_model_name)
        col2.metric(score_col, f"{best_row[score_col]:.4f}")

        if secondary_col in best_row:
            col3.metric(secondary_col, f"{best_row[secondary_col]:.4f}")

        col4.metric("Training Time", f"{best_row['Training Time (s)']:.3f} sec")

        st.caption(f"Rank : {int(best_row['Rank'])} / {len(result_df)}")

    except Exception as e:

        st.warning(f"Best Model Summary could not be generated.\n{e}")


# ==========================================================
# Model Comparison Dashboard (Feature 2)
# ==========================================================

def show_model_dashboard(result_df, problem_type):

    st.markdown("---")

    st.subheader("📊 Model Performance Dashboard")

    if problem_type == "Classification":
        score_col = "Accuracy"
    else:
        score_col = "R2 Score"

    # ---- Score Comparison ----
    st.write("### 📈 Model Score Comparison")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(result_df["Model"], result_df[score_col], color="#4C72B0")

    ax.set_ylabel(score_col)
    ax.set_title(f"{score_col} Comparison")

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    # ---- Training Time ----
    st.write("### ⏱ Training Time Comparison")

    time_col = "Training Time (s)"

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(result_df["Model"], result_df[time_col], color="#DD8452")

    ax.set_ylabel("Seconds")
    ax.set_title("Training Time")

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    # ---- Top 5 Models ----
    st.write("### 🏆 Top 5 Models")

    top5 = result_df.head(5)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(top5["Model"], top5[score_col], color="#55A868")

    ax.invert_yaxis()
    ax.set_xlabel(score_col)
    ax.set_title("Top 5 Models")

    st.pyplot(fig)