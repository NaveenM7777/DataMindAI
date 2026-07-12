import streamlit as st
import pandas as pd
import numpy as np


def _log_action(message):
    """Add an entry to the feature engineering history log."""

    if "fe_history" not in st.session_state:
        st.session_state["fe_history"] = []

    st.session_state["fe_history"].append(message)


def show_feature_engineering(df):

    st.title("🛠️ Feature Engineering")

    st.markdown("---")

    # Keep a working copy in session_state so operations persist
    # across reruns (button clicks, widget changes) without being lost.
    if "fe_df" not in st.session_state or st.session_state.get("fe_source_shape") != df.shape:

        st.session_state["fe_df"] = df.copy()
        st.session_state["fe_source_shape"] = df.shape
        st.session_state["fe_history"] = []

    working_df = st.session_state["fe_df"]

    st.info(f"Current Dataset Shape : {working_df.shape[0]} rows × {working_df.shape[1]} columns")

    # =====================================================
    # MODULE 1 — Feature Creation
    # =====================================================

    st.markdown("---")
    st.header("➕ Feature Creation")

    numeric_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()

    creation_type = st.radio(

        "Choose Operation",

        [
            "Add Columns",
            "Subtract Columns",
            "Multiply Columns",
            "Divide Columns",
            "Log Transform",
            "Square",
            "Cube",
            "Square Root"
        ],

        horizontal=True,

        key="fe_creation_type"
    )

    if creation_type in ["Add Columns", "Subtract Columns", "Multiply Columns", "Divide Columns"]:

        col1 = st.selectbox("Select First Column", numeric_cols, key="fe_col1")
        col2 = st.selectbox("Select Second Column", numeric_cols, key="fe_col2")

        new_col_name = st.text_input(
            "New Column Name",
            value=f"{col1}_{creation_type.split()[0].lower()}_{col2}",
            key="fe_new_col_name_pair"
        )

        if st.button("Apply", key="fe_apply_pair"):

            try:

                if creation_type == "Add Columns":
                    working_df[new_col_name] = working_df[col1] + working_df[col2]

                elif creation_type == "Subtract Columns":
                    working_df[new_col_name] = working_df[col1] - working_df[col2]

                elif creation_type == "Multiply Columns":
                    working_df[new_col_name] = working_df[col1] * working_df[col2]

                elif creation_type == "Divide Columns":
                    working_df[new_col_name] = working_df[col1] / working_df[col2].replace(0, np.nan)

                st.session_state["fe_df"] = working_df

                _log_action(f"✓ Created **{new_col_name}** ({creation_type}: {col1}, {col2})")

                st.success(f"✅ Column '{new_col_name}' created.")

                st.rerun()

            except Exception as e:

                st.error(f"❌ Operation failed: {e}")

    else:

        col = st.selectbox("Select Column", numeric_cols, key="fe_col_single")

        suffix_map = {
            "Log Transform": "log",
            "Square": "sq",
            "Cube": "cube",
            "Square Root": "sqrt"
        }

        new_col_name = st.text_input(
            "New Column Name",
            value=f"{col}_{suffix_map[creation_type]}",
            key="fe_new_col_name_single"
        )

        if st.button("Apply", key="fe_apply_single"):

            try:

                if creation_type == "Log Transform":
                    # log1p handles zero values safely; negative values become NaN
                    working_df[new_col_name] = np.log1p(working_df[col].clip(lower=0))

                elif creation_type == "Square":
                    working_df[new_col_name] = working_df[col] ** 2

                elif creation_type == "Cube":
                    working_df[new_col_name] = working_df[col] ** 3

                elif creation_type == "Square Root":
                    working_df[new_col_name] = np.sqrt(working_df[col].clip(lower=0))

                st.session_state["fe_df"] = working_df

                _log_action(f"✓ Created **{new_col_name}** ({creation_type} of {col})")

                st.success(f"✅ Column '{new_col_name}' created.")

                st.rerun()

            except Exception as e:

                st.error(f"❌ Operation failed: {e}")

    # =====================================================
    # MODULE 2 — Feature Selection
    # =====================================================

    st.markdown("---")
    st.header("🎯 Feature Selection")

    selection_type = st.radio(

        "Choose Operation",

        [
            "Remove Columns",
            "Keep Selected Columns",
            "Correlation Filter",
            "Variance Threshold"
        ],

        horizontal=True,

        key="fe_selection_type"
    )

    if selection_type == "Remove Columns":

        cols_to_remove = st.multiselect(
            "Select Columns To Remove",
            working_df.columns.tolist(),
            key="fe_remove_cols"
        )

        if st.button("Apply", key="fe_apply_remove"):

            if cols_to_remove:

                working_df = working_df.drop(columns=cols_to_remove)

                st.session_state["fe_df"] = working_df

                _log_action(f"✓ Removed columns: {', '.join(cols_to_remove)}")

                st.success(f"✅ Removed {len(cols_to_remove)} column(s).")

                st.rerun()

            else:

                st.warning("Select at least one column to remove.")

    elif selection_type == "Keep Selected Columns":

        cols_to_keep = st.multiselect(
            "Select Columns To Keep",
            working_df.columns.tolist(),
            default=working_df.columns.tolist(),
            key="fe_keep_cols"
        )

        if st.button("Apply", key="fe_apply_keep"):

            if cols_to_keep:

                dropped = [c for c in working_df.columns if c not in cols_to_keep]

                working_df = working_df[cols_to_keep]

                st.session_state["fe_df"] = working_df

                _log_action(f"✓ Kept {len(cols_to_keep)} columns, dropped: {', '.join(dropped) if dropped else 'none'}")

                st.success("✅ Column selection applied.")

                st.rerun()

            else:

                st.warning("Select at least one column to keep.")

    elif selection_type == "Correlation Filter":

        threshold = st.slider(
            "Correlation Threshold",
            0.5, 1.0, 0.9, 0.01,
            key="fe_corr_threshold"
        )

        st.caption(
            "Drops one column from each pair of numeric features whose "
            "correlation exceeds this threshold."
        )

        if st.button("Apply", key="fe_apply_corr"):

            try:

                numeric_df = working_df.select_dtypes(include=[np.number])

                corr_matrix = numeric_df.corr().abs()

                upper = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )

                to_drop = [
                    column for column in upper.columns
                    if any(upper[column] > threshold)
                ]

                if to_drop:

                    working_df = working_df.drop(columns=to_drop)

                    st.session_state["fe_df"] = working_df

                    _log_action(f"✓ Correlation Filter removed: {', '.join(to_drop)} (threshold={threshold})")

                    st.success(f"✅ Removed {len(to_drop)} highly correlated column(s): {', '.join(to_drop)}")

                    st.rerun()

                else:

                    st.info("No columns exceeded the correlation threshold.")

            except Exception as e:

                st.error(f"❌ Operation failed: {e}")

    elif selection_type == "Variance Threshold":

        var_threshold = st.slider(
            "Minimum Variance",
            0.0, 1.0, 0.01, 0.01,
            key="fe_var_threshold"
        )

        st.caption("Removes numeric columns whose variance is below this threshold.")

        if st.button("Apply", key="fe_apply_var"):

            try:

                from sklearn.feature_selection import VarianceThreshold

                numeric_df = working_df.select_dtypes(include=[np.number]).dropna(axis=1, how="any")

                if numeric_df.shape[1] == 0:

                    st.warning("No numeric columns available for variance filtering.")

                else:

                    selector = VarianceThreshold(threshold=var_threshold)

                    selector.fit(numeric_df)

                    kept_mask = selector.get_support()

                    low_variance_cols = numeric_df.columns[~kept_mask].tolist()

                    if low_variance_cols:

                        working_df = working_df.drop(columns=low_variance_cols)

                        st.session_state["fe_df"] = working_df

                        _log_action(f"✓ Variance Threshold removed: {', '.join(low_variance_cols)} (threshold={var_threshold})")

                        st.success(f"✅ Removed {len(low_variance_cols)} low-variance column(s).")

                        st.rerun()

                    else:

                        st.info("No columns fell below the variance threshold.")

            except Exception as e:

                st.error(f"❌ Operation failed: {e}")

    # =====================================================
    # MODULE 3 — Outlier Handling
    # =====================================================

    st.markdown("---")
    st.header("📦 Outlier Handling")

    numeric_cols_current = working_df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols_current) == 0:

        st.info("No numeric columns available for outlier handling.")

    else:

        outlier_col = st.selectbox("Select Numeric Column", numeric_cols_current, key="fe_outlier_col")

        method = st.radio("Detection Method", ["IQR", "Z-score"], horizontal=True, key="fe_outlier_method")

        action = st.radio("Action", ["Remove", "Cap"], horizontal=True, key="fe_outlier_action")

        if st.button("Apply", key="fe_apply_outlier"):

            try:

                series = working_df[outlier_col]

                if method == "IQR":

                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                else:  # Z-score

                    mean = series.mean()
                    std = series.std()

                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std

                outlier_mask = (series < lower_bound) | (series > upper_bound)
                outlier_count = int(outlier_mask.sum())

                if outlier_count == 0:

                    st.info(f"No outliers detected in '{outlier_col}' using {method}.")

                else:

                    if action == "Remove":

                        working_df = working_df.loc[~outlier_mask].reset_index(drop=True)

                        _log_action(f"✓ Removed {outlier_count} outliers from '{outlier_col}' ({method})")

                    else:  # Cap

                        working_df[outlier_col] = series.clip(lower=lower_bound, upper=upper_bound)

                        _log_action(f"✓ Capped {outlier_count} outliers in '{outlier_col}' ({method})")

                    st.session_state["fe_df"] = working_df

                    st.success(f"✅ {action}d {outlier_count} outlier(s) in '{outlier_col}'.")

                    st.rerun()

            except Exception as e:

                st.error(f"❌ Operation failed: {e}")

    # =====================================================
    # MODULE 4 — Date Feature Engineering
    # =====================================================

    st.markdown("---")
    st.header("📅 Date Feature Engineering")

    # Auto-detect columns that look like dates
    candidate_date_cols = []

    for col in working_df.columns:

        if working_df[col].dtype == "object" or "datetime" in str(working_df[col].dtype):

            try:

                converted = pd.to_datetime(working_df[col], errors="coerce")

                # If most values converted successfully, treat as a date column
                if converted.notnull().sum() >= 0.8 * len(converted):

                    candidate_date_cols.append(col)

            except Exception:

                pass

    if len(candidate_date_cols) == 0:

        st.info("No date-like columns were detected in the dataset.")

    else:

        date_col = st.selectbox("Select Date Column", candidate_date_cols, key="fe_date_col")

        date_features = st.multiselect(
            "Select Features To Generate",
            ["Year", "Month", "Day", "Week", "Quarter", "Day Name"],
            default=["Year", "Month", "Day"],
            key="fe_date_features"
        )

        if st.button("Apply", key="fe_apply_date"):

            try:

                parsed = pd.to_datetime(working_df[date_col], errors="coerce")

                generated = []

                if "Year" in date_features:
                    working_df[f"{date_col}_year"] = parsed.dt.year
                    generated.append(f"{date_col}_year")

                if "Month" in date_features:
                    working_df[f"{date_col}_month"] = parsed.dt.month
                    generated.append(f"{date_col}_month")

                if "Day" in date_features:
                    working_df[f"{date_col}_day"] = parsed.dt.day
                    generated.append(f"{date_col}_day")

                if "Week" in date_features:
                    working_df[f"{date_col}_week"] = parsed.dt.isocalendar().week.astype(int)
                    generated.append(f"{date_col}_week")

                if "Quarter" in date_features:
                    working_df[f"{date_col}_quarter"] = parsed.dt.quarter
                    generated.append(f"{date_col}_quarter")

                if "Day Name" in date_features:
                    working_df[f"{date_col}_day_name"] = parsed.dt.day_name()
                    generated.append(f"{date_col}_day_name")

                st.session_state["fe_df"] = working_df

                _log_action(f"✓ Generated date features from '{date_col}': {', '.join(generated)}")

                st.success(f"✅ Generated {len(generated)} date feature(s) from '{date_col}'.")

                st.rerun()

            except Exception as e:

                st.error(f"❌ Operation failed: {e}")

    # =====================================================
    # MODULE 5 — Feature Engineering History
    # =====================================================

    st.markdown("---")
    st.header("📜 Feature Engineering History")

    history = st.session_state.get("fe_history", [])

    if len(history) == 0:

        st.info("No operations performed yet.")

    else:

        for entry in history:

            st.markdown(entry)

        if st.button("🔄 Reset All Feature Engineering", key="fe_reset"):

            st.session_state["fe_df"] = df.copy()
            st.session_state["fe_history"] = []

            st.success("✅ Feature engineering reset to original preprocessed data.")

            st.rerun()

    # =====================================================
    # Preview current data
    # =====================================================

    st.markdown("---")
    st.header("👁️ Preview Engineered Dataset")

    st.dataframe(working_df.head(20), use_container_width=True)

    st.write(f"Shape: {working_df.shape[0]} rows × {working_df.shape[1]} columns")

    # =====================================================
    # MODULE 6 — Download Engineered Dataset
    # =====================================================

    st.markdown("---")
    st.header("⬇ Download Engineered Dataset")

    csv_bytes = working_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Engineered Dataset (.csv)",
        data=csv_bytes,
        file_name="engineered_dataset.csv",
        mime="text/csv"
    )

    return working_df