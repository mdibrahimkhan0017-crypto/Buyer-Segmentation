"""
dashboard/components/module3_geography.py
==========================================
Module 3: Geographic Buyer Analysis.

Real data facts:
  10 countries: USA(1538), UK(95), Canada(85), Germany(56), France(53),
                Belgium(43), Mexico(40), Australia(39), Russia(36), Denmark(15)
  57 unique regions — heavy US concentration
  ISO Alpha-3 codes used for Plotly choropleth
"""

import pandas as pd
import plotly.express as px
import streamlit as st

# ISO Alpha-3 country code mapping for Plotly choropleth
COUNTRY_ISO = {
    "USA": "USA",
    "UK": "GBR",
    "Canada": "CAN",
    "Germany": "DEU",
    "France": "FRA",
    "Belgium": "BEL",
    "Mexico": "MEX",
    "Australia": "AUS",
    "Russia": "RUS",
    "Denmark": "DNK",
}

CLUSTER_COLORS = {
    "Global Investors": "#2196F3",
    "First-Time Buyers": "#4CAF50",
    "Corporate Buyers": "#FF9800",
    "Luxury Investors": "#9C27B0",
}


def render_module3(df: pd.DataFrame) -> None:
    """Geographic Buyer Analysis: choropleth, region drill-down, top-10 table."""

    if df.empty:
        st.warning("No data matches current filters.")
        return

    # =========================================================================
    # Chart 1: Choropleth — Dominant Cluster per Country
    # =========================================================================
    st.subheader("Buyer Cluster Density by Country")

    # Aggregate: count, dominant cluster (mode), avg spend per country
    id_col = "client_id" if "client_id" in df.columns else "cluster_name"
    geo_df = (
        df.groupby("country")
        .agg(
            buyer_count=(id_col, "count"),
            dominant_cluster=("cluster_name", lambda x: x.mode().iloc[0]),
            avg_spend=("total_spend", "mean"),
        )
        .reset_index()
    )
    geo_df["iso_alpha"] = geo_df["country"].map(COUNTRY_ISO)
    geo_df["avg_spend_fmt"] = geo_df["avg_spend"].map("${:,.0f}".format)

    fig_map = px.choropleth(
        geo_df,
        locations="iso_alpha",
        color="dominant_cluster",
        color_discrete_map=CLUSTER_COLORS,
        hover_name="country",
        hover_data={
            "buyer_count": True,
            "avg_spend_fmt": True,
            "dominant_cluster": True,
            "iso_alpha": False,
        },
        title="Dominant Buyer Segment by Country",
        labels={
            "dominant_cluster": "Dominant Segment",
            "avg_spend_fmt": "Avg Spend",
            "buyer_count": "Buyer Count",
        },
    )
    fig_map.update_geos(
        showframe=False,
        showcoastlines=True,
        projection_type="natural earth",
    )
    fig_map.update_layout(height=450, margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig_map, use_container_width=True)

    st.divider()

    # =========================================================================
    # Chart 2: Region Drill-Down (grouped bar)
    # =========================================================================
    st.subheader("Region-Level Cluster Distribution")

    country_options = sorted(df["country"].unique().tolist())
    default_idx = country_options.index("USA") if "USA" in country_options else 0
    selected_country = st.selectbox(
        "Select Country to Drill Down",
        options=country_options,
        index=default_idx,
    )

    country_df = df[df["country"] == selected_country]
    if country_df.empty:
        st.info(f"No buyers found for {selected_country} with current filters.")
    else:
        # Top 15 regions by buyer count
        top_regions = country_df["region"].value_counts().head(15).index
        region_counts = (
            country_df[country_df["region"].isin(top_regions)]
            .groupby(["region", "cluster_name"])
            .size()
            .reset_index(name="count")
        )

        fig_region = px.bar(
            region_counts,
            x="region",
            y="count",
            color="cluster_name",
            color_discrete_map=CLUSTER_COLORS,
            barmode="group",
            title=f"Cluster Distribution — {selected_country} (Top 15 Regions)",
            labels={
                "count": "Buyer Count",
                "region": "Region",
                "cluster_name": "Segment",
            },
        )
        fig_region.update_xaxes(tickangle=45)
        fig_region.update_layout(margin=dict(b=80))
        st.plotly_chart(fig_region, use_container_width=True)

    st.divider()

    # =========================================================================
    # Chart 3: Top-10 Regions Sortable Table
    # =========================================================================
    st.subheader("Top 10 Regions by Buyer Activity")

    sort_by = st.selectbox(
        "Sort by",
        ["Buyer Count", "Avg Spend", "Dominant Segment"],
    )

    top_regions_df = (
        df.groupby("region")
        .agg(
            buyer_count=("cluster_name", "count"),
            avg_spend=("total_spend", "mean"),
            dominant_segment=("cluster_name", lambda x: x.mode().iloc[0]),
        )
        .reset_index()
    )

    sort_map = {
        "Buyer Count": "buyer_count",
        "Avg Spend": "avg_spend",
        "Dominant Segment": "dominant_segment",
    }
    ascending = sort_by == "Dominant Segment"
    top_regions_df = top_regions_df.sort_values(
        sort_map[sort_by], ascending=ascending
    ).head(10)

    # Format for display
    display_df = top_regions_df.copy()
    display_df["avg_spend"] = display_df["avg_spend"].map("${:,.0f}".format)
    display_df.columns = ["Region", "Buyer Count", "Avg Spend", "Dominant Segment"]

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Buyer Count": st.column_config.NumberColumn(format="%d"),
        },
    )
