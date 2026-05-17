"""
dashboard/components/module1_overview.py
=========================================
Module 1: Buyer Segmentation Overview tab.

Real cluster distribution at full data (before filters):
  ~500 buyers per cluster in each of the 4 archetypes.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

# Shared cluster color palette — consistent across all modules
CLUSTER_COLORS = {
    "Global Investors": "#2196F3",   # blue
    "First-Time Buyers": "#4CAF50",  # green
    "Corporate Buyers": "#FF9800",   # orange
    "Luxury Investors": "#9C27B0",   # purple
}


def render_module1(df: pd.DataFrame) -> None:
    """
    Renders the Buyer Segmentation Overview tab.

    Layout
    ------
    Row 1 : Pie chart (cluster distribution) + Horizontal bar (counts)
    Row 2 : Centroid summary table (full-width, styled)
    Row 3 : Algorithm toggle (Silhouette / Dendrogram images)
            Expander: K-selection elbow & silhouette curves
    """
    if df.empty:
        st.warning("No data matches current filters.")
        return

    # =========================================================================
    # Row 1: Distribution charts
    # =========================================================================
    col1, col2 = st.columns(2)

    cluster_counts = (
        df["cluster_name"]
        .value_counts()
        .reset_index()
    )
    cluster_counts.columns = ["Cluster", "Count"]

    with col1:
        fig_pie = px.pie(
            cluster_counts,
            values="Count",
            names="Cluster",
            title="Buyer Distribution by Segment",
            color="Cluster",
            color_discrete_map=CLUSTER_COLORS,
            hole=0.35,
        )
        fig_pie.update_traces(
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} buyers (%{percent})",
        )
        fig_pie.update_layout(showlegend=True, margin=dict(t=50, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        sorted_counts = cluster_counts.sort_values("Count", ascending=True)
        fig_bar = px.bar(
            sorted_counts,
            x="Count",
            y="Cluster",
            orientation="h",
            title="Buyer Count per Segment",
            color="Cluster",
            color_discrete_map=CLUSTER_COLORS,
            text="Count",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            showlegend=False,
            xaxis_title="Number of Buyers",
            margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # =========================================================================
    # Row 2: Centroid Summary Table
    # =========================================================================
    st.subheader("Cluster Centroid Profiles")

    # Build profile: only use columns that exist in df
    profile_cols_needed = [
        "cluster_name", "buyer_age", "total_spend", "transaction_count",
        "satisfaction_score", "avg_price_per_sqft", "avg_floor_area", "loan_applied",
    ]
    profile_cols = [c for c in profile_cols_needed if c in df.columns]

    profile_df = df[profile_cols].copy()

    # Normalize loan_applied to numeric (Yes->1 / No->0)
    # Use .astype(str) comparison to handle both object and ArrowDtype strings
    if "loan_applied" in profile_df.columns:
        profile_df["loan_applied"] = (profile_df["loan_applied"].astype(str) == "Yes").astype(int)

    agg_dict = {c: "mean" for c in profile_cols if c != "cluster_name"}
    centroid_table = profile_df.groupby("cluster_name").agg(agg_dict).round(2).reset_index()

    # Rename columns for display
    rename_map = {
        "cluster_name": "Segment",
        "buyer_age": "Avg Age",
        "total_spend": "Avg Total Spend ($)",
        "transaction_count": "Avg Transactions",
        "satisfaction_score": "Avg Satisfaction",
        "avg_price_per_sqft": "Avg $/sqft",
        "avg_floor_area": "Avg Floor Area (sqft)",
        "loan_applied": "Loan Rate",
    }
    centroid_table = centroid_table.rename(columns={k: v for k, v in rename_map.items() if k in centroid_table.columns})

    # Format spend column
    if "Avg Total Spend ($)" in centroid_table.columns:
        centroid_table["Avg Total Spend ($)"] = centroid_table["Avg Total Spend ($)"].map("${:,.0f}".format)

    # Gradient-styled dataframe
    gradient_cols = [c for c in ["Avg Age", "Avg Transactions", "Avg Satisfaction", "Loan Rate"] if c in centroid_table.columns]
    styled = centroid_table.style.background_gradient(
        subset=gradient_cols, cmap="RdYlGn"
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    # =========================================================================
    # Row 3: Algorithm outputs toggle
    # =========================================================================
    algo = st.radio(
        "View Clustering Output",
        ["K-Means Silhouette", "Hierarchical Dendrogram"],
        horizontal=True,
    )

    if algo == "K-Means Silhouette":
        sil_path = "outputs/figures/silhouette.png"
        try:
            st.image(
                sil_path,
                caption="K-Means Silhouette Plot — K=4, n=2,000",
                use_container_width=True,
            )
        except Exception:
            st.info("Silhouette plot not found. Run `python src/pipeline.py` first.")
    else:
        den_path = "outputs/figures/dendrogram.png"
        try:
            st.image(
                den_path,
                caption="Hierarchical Dendrogram — Ward linkage, 10% sample (n=200)",
                use_container_width=True,
            )
        except Exception:
            st.info("Dendrogram not found. Run `python src/pipeline.py` first.")

    # K-selection elbow curve in expander
    with st.expander("K Selection — Elbow & Silhouette Curves"):
        k_path = "outputs/figures/k_selection.png"
        try:
            st.image(k_path, use_container_width=True)
        except Exception:
            st.info("K-selection figure not found. Run `python src/pipeline.py` first.")
