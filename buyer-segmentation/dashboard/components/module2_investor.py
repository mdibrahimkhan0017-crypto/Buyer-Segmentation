"""
dashboard/components/module2_investor.py
=========================================
Module 2: Investor Behavior Dashboard.

Real data facts encoded here:
  loan_applied      : 'Yes' (736) / 'No' (1264)
  acquisition_purpose: 'Home' (1385) / 'Investment' (615)
  referral_channel  : 'Website' / 'Agency' / 'Client'
  satisfaction_score: integer 1-5, ~400 per value (nearly uniform)
  total_spend range : $463K - $3.65M per client
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

CLUSTER_COLORS = {
    "Global Investors": "#2196F3",
    "First-Time Buyers": "#4CAF50",
    "Corporate Buyers": "#FF9800",
    "Luxury Investors": "#9C27B0",
}


def render_module2(df: pd.DataFrame) -> None:
    """Investor Behavior Dashboard with loan rates, spend, purpose, and satisfaction."""

    if df.empty:
        st.warning("No data matches current filters.")
        return

    # =========================================================================
    # Chart 1: Loan Applied Rate — Stacked Bar
    # =========================================================================
    st.subheader("Loan Applied Rate by Segment")

    # Normalize loan_applied to string labels for groupby
    loan_df_source = df.copy()
    # Use astype(str) to handle both object and ArrowDtype (pandas 2.x)
    loan_df_source["loan_applied"] = loan_df_source["loan_applied"].astype(str).map(
        {"1": "Yes", "0": "No", "Yes": "Yes", "No": "No"}
    ).fillna(loan_df_source["loan_applied"].astype(str))

    loan_df = (
        loan_df_source.groupby(["cluster_name", "loan_applied"])
        .size()
        .reset_index(name="count")
    )
    loan_totals = loan_df.groupby("cluster_name")["count"].transform("sum")
    loan_df["pct"] = (loan_df["count"] / loan_totals * 100).round(1)

    fig_loan = px.bar(
        loan_df,
        x="cluster_name",
        y="pct",
        color="loan_applied",
        barmode="stack",
        color_discrete_map={"Yes": "#E53935", "No": "#1E88E5"},
        labels={"pct": "% of Buyers", "cluster_name": "Segment"},
        title="Loan Applied Rate per Buyer Segment",
        text="pct",
    )
    fig_loan.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
    fig_loan.update_layout(xaxis_title="Segment", yaxis_title="% of Buyers")
    st.plotly_chart(fig_loan, use_container_width=True)

    st.divider()

    # =========================================================================
    # Charts 2 & 3: Box plots — Total Spend and Price per SqFt
    # =========================================================================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Spend Distribution")
        fig_spend = px.box(
            df,
            x="cluster_name",
            y="total_spend",
            color="cluster_name",
            color_discrete_map=CLUSTER_COLORS,
            points="outliers",
            title="Total Spend per Client ($)",
            labels={"total_spend": "Total Spend ($)", "cluster_name": ""},
        )
        fig_spend.update_yaxes(tickformat="$,.0f")
        fig_spend.update_layout(showlegend=False)
        st.plotly_chart(fig_spend, use_container_width=True)

    with col2:
        st.subheader("Price per SqFt Distribution")
        fig_psf = px.box(
            df,
            x="cluster_name",
            y="avg_price_per_sqft",
            color="cluster_name",
            color_discrete_map=CLUSTER_COLORS,
            points="outliers",
            title="Avg Price per SqFt ($)",
            labels={"avg_price_per_sqft": "$/sqft", "cluster_name": ""},
        )
        fig_psf.update_yaxes(tickformat="$,.0f")
        fig_psf.update_layout(showlegend=False)
        st.plotly_chart(fig_psf, use_container_width=True)

    st.divider()

    # =========================================================================
    # Chart 4: Acquisition Purpose — 4 donuts side by side
    # =========================================================================
    st.subheader("Acquisition Purpose Split by Segment")
    clusters = ["Global Investors", "First-Time Buyers", "Corporate Buyers", "Luxury Investors"]
    cols = st.columns(4)

    for i, cluster in enumerate(clusters):
        sub = df[df["cluster_name"] == cluster]
        if sub.empty:
            with cols[i]:
                st.caption(f"{cluster}: no data")
            continue

        purpose_counts = sub["acquisition_purpose"].value_counts()
        fig_donut = px.pie(
            values=purpose_counts.values,
            names=purpose_counts.index,
            hole=0.5,
            color=purpose_counts.index,
            color_discrete_map={"Home": "#43A047", "Investment": "#1565C0"},
            title=cluster,
        )
        fig_donut.update_layout(
            margin=dict(t=40, b=0, l=0, r=0),
            showlegend=(i == 0),
            height=220,
        )
        with cols[i]:
            st.plotly_chart(fig_donut, use_container_width=True)

    st.divider()

    # =========================================================================
    # Chart 5: Satisfaction Score Heatmap (Segment x Referral Channel)
    # =========================================================================
    st.subheader("Satisfaction Score — Segment x Referral Channel")
    # Real channels: Website, Agency, Client; scores: 1-5 nearly uniform

    if "referral_channel" in df.columns and "satisfaction_score" in df.columns:
        df_heat = df.copy()
        df_heat["satisfaction_score"] = df_heat["satisfaction_score"].astype(float)
        heatmap_data = (
            df_heat.groupby(["cluster_name", "referral_channel"])["satisfaction_score"]
            .mean()
            .unstack()
        )

        fig_heat, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=1,
            vmax=5,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Mean Satisfaction (1-5)"},
        )
        ax.set_title("Mean Satisfaction Score by Segment & Referral Channel")
        ax.set_xlabel("Referral Channel")
        ax.set_ylabel("Buyer Segment")
        plt.tight_layout()
        st.pyplot(fig_heat, use_container_width=True)
        plt.close(fig_heat)
    else:
        st.info("referral_channel or satisfaction_score column not found in filtered data.")
