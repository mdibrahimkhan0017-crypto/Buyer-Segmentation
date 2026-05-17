"""
dashboard/app.py
================
Streamlit entry point for the Buyer Segmentation Dashboard.
Parcl Co. Limited — UM-PARCL-2025-001

Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path so 'dashboard.*' imports work
project_root = str(Path(__file__).parent.parent.resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---------------------------------------------------------------------------
# Page configuration — must be the FIRST Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Buyer Segmentation | Parcl Co.",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_labeled_data() -> pd.DataFrame:
    """
    Load data/processed/labeled_data.csv.

    Expected: 2,000 rows, ~25 columns including:
      cluster, cluster_name, buyer_age, total_spend, transaction_count,
      avg_price_per_sqft, avg_floor_area, satisfaction_score, loan_applied,
      acquisition_purpose, client_type, country, region, referral_channel,
      gender, spend_tier, dominant_month, transaction_year_latest

    Raises a clear user-visible error if the pipeline has not been run yet.
    """
    path = Path("data/processed/labeled_data.csv")
    if not path.exists():
        st.error(
            """
            **Labeled data not found.** Run the ML pipeline first:

                python src/pipeline.py

            Then relaunch the dashboard:

                streamlit run dashboard/app.py
            """
        )
        st.stop()

    df = pd.read_csv(path)
    # pandas 2.x may infer ArrowDtype for string columns which breaks .mean() calls.
    # Force all columns back to standard numpy-backed dtypes.
    df = df.convert_dtypes(convert_string=False)
    # Restore spend_tier as Categorical if column exists
    if "spend_tier" in df.columns:
        df["spend_tier"] = pd.Categorical(
            df["spend_tier"],
            categories=["Budget", "Mid", "Premium", "Luxury"],
            ordered=True,
        )
    return df


@st.cache_data
def load_cluster_profiles() -> pd.DataFrame:
    """Load pre-computed cluster profiles from pipeline output."""
    path = Path("outputs/profiles/cluster_profiles.csv")
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <h1 style='margin-bottom:0'>🏠 Buyer Segmentation Dashboard</h1>
    <p style='color:gray;margin-top:4px'>
        Parcl Co. Limited &nbsp;·&nbsp;
        ML-Based Buyer Segmentation & Investment Profiling &nbsp;·&nbsp;
        UM-PARCL-2025-001
    </p>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Data load
# ---------------------------------------------------------------------------
df = load_labeled_data()
profiles = load_cluster_profiles()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
from dashboard.filters import build_sidebar_filters  # noqa: E402

filtered_df = build_sidebar_filters(df)

# ---------------------------------------------------------------------------
# Filter summary bar
# ---------------------------------------------------------------------------
n_buyers = len(filtered_df)
n_countries = filtered_df["country"].nunique() if "country" in filtered_df.columns else 0
n_clusters = filtered_df["cluster_name"].nunique() if "cluster_name" in filtered_df.columns else 0

st.info(
    f"📊 Showing **{n_buyers:,}** buyers across **{n_countries}** "
    f"{'country' if n_countries == 1 else 'countries'} | "
    f"**{n_clusters}** {'segment' if n_clusters == 1 else 'segments'} active"
)

# ---------------------------------------------------------------------------
# Tab navigation
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Segmentation Overview",
        "💰 Investor Behavior",
        "🌍 Geographic Analysis",
        "🔍 Segment Insights",
    ]
)

from dashboard.components.module1_overview import render_module1   # noqa: E402
from dashboard.components.module2_investor import render_module2   # noqa: E402
from dashboard.components.module3_geography import render_module3  # noqa: E402
from dashboard.components.module4_insights import render_module4   # noqa: E402

with tab1:
    render_module1(filtered_df)

with tab2:
    render_module2(filtered_df)

with tab3:
    render_module3(filtered_df)

with tab4:
    render_module4(filtered_df)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "© 2025 Unified Mentor · Parcl Co. Limited · CONFIDENTIAL · UM-PARCL-2025-001"
)
