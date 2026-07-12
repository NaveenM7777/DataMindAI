import streamlit as st
import pandas as pd


# ==========================================================
# Data Preprocessing
# ==========================================================

def show_preprocessing(df):

    # ======================================================
    # Working Copy
    # ======================================================

    processed_df = df.copy()

    st.title("🧹 Data Preprocessing")

    st.markdown("---")

    # ======================================================
    # Dataset Health
    # ======================================================

    st.subheader("📊 Dataset Health")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Missing Values",
        int(processed_df.isnull().sum().sum())
    )

    col2.metric(
        "Duplicate Rows",
        int(processed_df.duplicated().sum())
    )

    col3.metric(
        "Total Features",
        processed_df.shape[1]
    )

    st.markdown("---")

    # ======================================================
    # Missing Value Handling
    # ======================================================

    st.subheader("🩹 Missing Value Handling")

    missing = processed_df.isnull().sum()

    missing = missing[missing > 0]

    if missing.empty:

        st.success("🎉 No Missing Values Found!")

    else:

        missing_df = pd.DataFrame({

            "Column": missing.index,

            "Missing Values": missing.values

        })

        st.dataframe(
            missing_df,
            use_container_width=True
        )

        strategy = st.selectbox(

            "Missing Value Strategy",

            [

                "Do Nothing",

                "Drop Missing Rows",

                "Fill Numeric with Mean",

                "Fill Numeric with Median",

                "Fill Categorical with Mode"

            ]

        )

        if st.button("Apply Missing Value Handling"):

            if strategy == "Drop Missing Rows":

                processed_df = processed_df.dropna()

            elif strategy == "Fill Numeric with Mean":

                numeric_cols = processed_df.select_dtypes(
                    include="number"
                ).columns

                processed_df[numeric_cols] = processed_df[
                    numeric_cols
                ].fillna(

                    processed_df[numeric_cols].mean()

                )

            elif strategy == "Fill Numeric with Median":

                numeric_cols = processed_df.select_dtypes(
                    include="number"
                ).columns

                processed_df[numeric_cols] = processed_df[
                    numeric_cols
                ].fillna(

                    processed_df[numeric_cols].median()

                )

            elif strategy == "Fill Categorical with Mode":

                categorical_cols = processed_df.select_dtypes(
                    exclude="number"
                ).columns

                for col in categorical_cols:

                    processed_df[col] = processed_df[col].fillna(

                        processed_df[col].mode()[0]

                    )

            st.success("✅ Missing Value Handling Completed")

            st.dataframe(
                processed_df.head(),
                use_container_width=True
            )

    # ======================================================
    # STOP HERE
    # ======================================================
    # ======================================================
    # Duplicate Removal
    # ======================================================

    st.markdown("---")

    st.subheader("🗑️ Duplicate Removal")

    duplicate_count = processed_df.duplicated().sum()

    st.write(f"Duplicate Rows Found : {duplicate_count}")

    if st.button("Remove Duplicate Rows"):

        processed_df = processed_df.drop_duplicates()

        st.success("✅ Duplicate Rows Removed")

        st.write(
            "Remaining Rows :",
            len(processed_df)
        )

    # ======================================================
    # Data Type Conversion
    # ======================================================

    st.markdown("---")

    st.subheader("🔄 Data Type Conversion")

    selected_column = st.selectbox(

        "Select Column",

        processed_df.columns,

        key="datatype"

    )

    new_type = st.selectbox(

        "Convert To",

        [

            "int",

            "float",

            "string"

        ],

        key="datatype_type"

    )

    if st.button("Convert Data Type"):

        try:

            if new_type == "int":

                processed_df[selected_column] = processed_df[
                    selected_column
                ].astype(int)

            elif new_type == "float":

                processed_df[selected_column] = processed_df[
                    selected_column
                ].astype(float)

            else:

                processed_df[selected_column] = processed_df[
                    selected_column
                ].astype(str)

            st.success("✅ Data Type Converted Successfully")

        except Exception as e:

            st.error(e)

    # ======================================================
    # Encoding
    # ======================================================

    st.markdown("---")

    st.subheader("🏷️ Encoding")

    categorical_columns = processed_df.select_dtypes(
        include="object"
    ).columns

    if len(categorical_columns) > 0:

        encoding_method = st.radio(

            "Encoding Method",

            [

                "Label Encoding",

                "One Hot Encoding"

            ]

        )

        encoding_column = st.selectbox(

            "Select Column",

            categorical_columns,

            key="encoding"

        )

        if st.button("Apply Encoding"):

            if encoding_method == "Label Encoding":

                from sklearn.preprocessing import LabelEncoder

                encoder = LabelEncoder()

                processed_df[encoding_column] = encoder.fit_transform(

                    processed_df[encoding_column]

                )

            else:

                processed_df = pd.get_dummies(

                    processed_df,

                    columns=[encoding_column],

                    drop_first=True

                )

            st.success("✅ Encoding Applied Successfully")

            st.dataframe(

                processed_df.head(),

                use_container_width=True

            )

    else:

        st.info("No Categorical Columns Available.")

    # ======================================================
    # STOP HERE
    # ======================================================    
    # ======================================================
    # Feature Scaling
    # ======================================================

    st.markdown("---")

    st.subheader("📏 Feature Scaling")

    numeric_columns = processed_df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) > 0:

        scaling_column = st.selectbox(

            "Select Numeric Column",

            numeric_columns,

            key="scaling_column"

        )

        scaling_method = st.selectbox(

            "Scaling Method",

            [

                "StandardScaler",

                "MinMaxScaler"

            ],

            key="scaling_method"

        )

        if st.button("Apply Scaling"):

            from sklearn.preprocessing import (
                StandardScaler,
                MinMaxScaler
            )

            if scaling_method == "StandardScaler":

                scaler = StandardScaler()

            else:

                scaler = MinMaxScaler()

            processed_df[[scaling_column]] = scaler.fit_transform(

                processed_df[[scaling_column]]

            )

            st.success("✅ Scaling Applied Successfully")

            st.dataframe(

                processed_df.head(),

                use_container_width=True

            )

    else:

        st.info("No Numeric Columns Available.")

    # ======================================================
    # Preview Processed Dataset
    # ======================================================

    st.markdown("---")

    st.subheader("👀 Processed Dataset Preview")

    st.dataframe(

        processed_df.head(),

        use_container_width=True

    )

    st.write(

        "Shape :",

        processed_df.shape

    )

    # ======================================================
    # Download Clean Dataset
    # ======================================================

    st.markdown("---")

    st.subheader("📥 Download Processed Dataset")

    csv = processed_df.to_csv(

        index=False

    ).encode("utf-8")

    st.download_button(

        label="⬇️ Download Clean Dataset",

        data=csv,

        file_name="processed_dataset.csv",

        mime="text/csv"

    )

    # ======================================================
    # Return Processed DataFrame
    # ======================================================

    return processed_df