import streamlit as st

from utils.charts import (
    missing_values_chart,
    histogram_chart,
    boxplot_chart,
    correlation_heatmap,
    count_plot,
    pie_chart,
    scatter_plot,
    pair_plot,
    kde_plot,
    violin_plot
)


def show_visualization(df):

    st.title("📊 Exploratory Data Analysis")

    st.markdown("---")

    missing_values_chart(df)

    st.markdown("---")

    histogram_chart(df)

    st.markdown("---")

    boxplot_chart(df)

    st.markdown("---")

    correlation_heatmap(df)

    st.markdown("---")

    count_plot(df)

    st.markdown("---")

    pie_chart(df)

    st.markdown("---")

    scatter_plot(df)

    st.markdown("---")

    pair_plot(df)

    st.markdown("---")

    kde_plot(df)

    st.markdown("---")

    violin_plot(df)