import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


# ==========================================================
# Missing Values Chart
# ==========================================================

def missing_values_chart(df):

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        st.success("🎉 No Missing Values Found!")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    missing.sort_values(ascending=False).plot(
        kind="bar",
        color="orange",
        ax=ax
    )

    ax.set_title("Missing Values by Column")
    ax.set_xlabel("")
    ax.set_ylabel("Count")

    plt.xticks(rotation=45)

    st.pyplot(fig)


# ==========================================================
# Histogram
# ==========================================================

def histogram_chart(df):

    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        st.warning("No Numeric Columns Found.")
        return

    st.subheader("📈 Numeric Feature Distributions")

    fig = numeric_df.hist(
        figsize=(15, 10),
        bins=20
    )

    plt.tight_layout()

    st.pyplot(plt.gcf())

    plt.clf()


# ==========================================================
# Boxplot
# ==========================================================

def boxplot_chart(df):

    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        st.warning("No Numeric Columns Found.")
        return

    st.subheader("📦 Outlier Detection (Boxplots)")

    selected_column = st.selectbox(
        "Select Numeric Column",
        numeric_df.columns
    )

    fig, ax = plt.subplots(figsize=(8,4))

    ax.boxplot(numeric_df[selected_column].dropna())

    ax.set_title(f"Boxplot of {selected_column}")

    st.pyplot(fig)
# ==========================================================
# Correlation Heatmap
# ==========================================================

def correlation_heatmap(df):

    import seaborn as sns

    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.shape[1] < 2:
        st.warning("Not enough numeric columns.")
        return

    st.subheader("🔥 Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10,8))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    st.pyplot(fig)


# ==========================================================
# Count Plot
# ==========================================================

def count_plot(df):

    import seaborn as sns

    categorical = df.select_dtypes(include=["object"]).columns

    if len(categorical) == 0:
        return

    column = st.selectbox(
        "Select Categorical Column",
        categorical,
        key="countplot"
    )

    st.subheader("📊 Count Plot")

    fig, ax = plt.subplots(figsize=(10,5))

    sns.countplot(
        data=df,
        x=column,
        ax=ax
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)


# ==========================================================
# Pie Chart
# ==========================================================

def pie_chart(df):

    categorical = df.select_dtypes(include=["object"]).columns

    if len(categorical) == 0:
        return

    column = st.selectbox(
        "Select Pie Chart Column",
        categorical,
        key="piechart"
    )

    st.subheader("🥧 Pie Chart")

    counts = df[column].value_counts()

    fig, ax = plt.subplots(figsize=(7,7))

    ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%"
    )

    st.pyplot(fig)


# ==========================================================
# Scatter Plot
# ==========================================================

def scatter_plot(df):

    import seaborn as sns

    numeric = df.select_dtypes(include=["number"]).columns

    if len(numeric) < 2:
        return

    st.subheader("📈 Scatter Plot")

    x = st.selectbox(
        "Select X-axis",
        numeric,
        key="scatterx"
    )

    y = st.selectbox(
        "Select Y-axis",
        numeric,
        index=1,
        key="scattery"
    )

    fig, ax = plt.subplots(figsize=(8,5))

    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        ax=ax
    )

    st.pyplot(fig)
# ==========================================================
# Pair Plot
# ==========================================================

def pair_plot(df):

    import seaborn as sns

    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.shape[1] < 2:
        return

    st.subheader("📌 Pair Plot")

    if len(numeric_df.columns) > 5:
        numeric_df = numeric_df.iloc[:, :5]

    fig = sns.pairplot(numeric_df)

    st.pyplot(fig.figure)


# ==========================================================
# KDE Plot
# ==========================================================

def kde_plot(df):

    import seaborn as sns

    numeric = df.select_dtypes(include=["number"]).columns

    if len(numeric) == 0:
        return

    st.subheader("📈 KDE Distribution")

    column = st.selectbox(
        "Select Column for KDE",
        numeric,
        key="kde"
    )

    fig, ax = plt.subplots(figsize=(8,5))

    sns.kdeplot(
        data=df,
        x=column,
        fill=True,
        ax=ax
    )

    st.pyplot(fig)


# ==========================================================
# Violin Plot
# ==========================================================

def violin_plot(df):

    import seaborn as sns

    numeric = df.select_dtypes(include=["number"]).columns

    if len(numeric) == 0:
        return

    st.subheader("🎻 Violin Plot")

    column = st.selectbox(
        "Select Column",
        numeric,
        key="violin"
    )

    fig, ax = plt.subplots(figsize=(8,5))

    sns.violinplot(
        y=df[column],
        ax=ax
    )

    st.pyplot(fig)