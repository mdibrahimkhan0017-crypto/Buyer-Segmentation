"""
dashboard/filters.py
====================
Sidebar filter controls for the Buyer Segmentation Dashboard.

Real data facts encoded here:
  10 countries: USA, UK, Canada, Germany, France, Belgium, Mexico, Australia, Russia, Denmark
  57 unique regions — heavy US concentration (California 633, Nevada 143, ...)
  Transaction date range: Jan 2024 - Dec 2025 (24 months)
  4 cluster_names: Global Investors, First-Time Buyers, Corporate Buyers, Luxury Investors
"""

import pandas as pd
import streamlit as st


def build_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Render sidebar filter controls and return the filtered DataFrame.

    Filters (in order)
    ------------------
    1. Country multi-select (all 10 countries)
    2. Region multi-select (dynamically updates based on country selection)
    3. Acquisition Purpose radio
    4. Client Type radio
    5. Transaction Date Range date_input
    6. Buyer Segment multi-select
    7. Reset All Filters button
    """
    st.sidebar.title("Filter Buyers")
    st.sidebar.markdown("---")

    # ---- 1. Country multi-select ------------------------------------------
    all_countries = sorted(df["country"].unique().tolist())
    countries = st.sidebar.multiselect(
        "Country",
        options=all_countries,
        default=all_countries,
        key="filter_country",
    )

    # ---- 2. Region multi-select (dynamic based on country) ----------------
    # Only show regions that exist within the currently selected countries
    country_mask = df["country"].isin(countries) if countries else pd.Series([True] * len(df))
    available_regions = sorted(df[country_mask]["region"].unique().tolist())

    regions = st.sidebar.multiselect(
        "Region",
        options=available_regions,
        default=available_regions,
        key="filter_region",
    )

    # ---- 3. Acquisition Purpose radio -------------------------------------
    purpose = st.sidebar.radio(
        "Acquisition Purpose",
        options=["All", "Home", "Investment"],
        index=0,
        key="filter_purpose",
    )

    # ---- 4. Client Type radio ---------------------------------------------
    ctype = st.sidebar.radio(
        "Client Type",
        options=["All", "Individual", "Company"],
        index=0,
        key="filter_ctype",
    )

    # ---- 5. Transaction Date Range ----------------------------------------
    # Real range: Jan 2024 - Dec 2025 (24 months)
    min_date = pd.Timestamp("2024-01-01")
    max_date = pd.Timestamp("2025-12-31")

    date_range = st.sidebar.date_input(
        "Transaction Date Range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
        key="filter_dates",
    )

    # ---- 6. Buyer Segment multi-select ------------------------------------
    all_segments = [
        "Global Investors",
        "First-Time Buyers",
        "Corporate Buyers",
        "Luxury Investors",
    ]
    cluster_filter = st.sidebar.multiselect(
        "Buyer Segment",
        options=all_segments,
        default=all_segments,
        key="filter_cluster",
    )

    # ---- 7. Reset All Filters button --------------------------------------
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset All Filters"):
        for key in [
            "filter_country",
            "filter_region",
            "filter_purpose",
            "filter_ctype",
            "filter_dates",
            "filter_cluster",
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # =========================================================================
    # Apply filters sequentially
    # =========================================================================
    filtered = df.copy()

    # Country filter
    if countries:
        filtered = filtered[filtered["country"].isin(countries)]

    # Region filter
    if regions:
        filtered = filtered[filtered["region"].isin(regions)]

    # Purpose filter
    if purpose != "All":
        filtered = filtered[filtered["acquisition_purpose"] == purpose]

    # Client type filter
    if ctype != "All":
        filtered = filtered[filtered["client_type"] == ctype]

    # Segment filter
    if cluster_filter:
        filtered = filtered[filtered["cluster_name"].isin(cluster_filter)]

    # Date filter — apply on transaction_year_latest column (2024 or 2025)
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_year = pd.Timestamp(date_range[0]).year
        end_year = pd.Timestamp(date_range[1]).year
        if "transaction_year_latest" in filtered.columns:
            filtered = filtered[
                filtered["transaction_year_latest"].between(start_year, end_year)
            ]

    # Sidebar summary badge
    st.sidebar.markdown("---")
    st.sidebar.metric("Buyers shown", f"{len(filtered):,} / {len(df):,}")

    return filtered
