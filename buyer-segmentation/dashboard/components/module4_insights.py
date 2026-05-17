"""
dashboard/components/module4_insights.py
=========================================
Module 4: Segment Insights Panel.

Deep profile for each of the 4 buyer archetypes with:
  - KPI metrics
  - Statistical profile table
  - Categorical breakdown
  - Segment-specific marketing recommendations
  - CSV and PDF export
"""

import io

import pandas as pd
import streamlit as st

# Segment-specific recommendation content grounded in real cluster characteristics
RECOMMENDATIONS = {
    "Global Investors": {
        "icon": "🌐",
        "tagline": "High-value international buyers with strong investment focus",
        "actions": [
            "🎯 Run cross-border digital campaigns in UK, Canada, Germany, France",
            "💎 Lead with premium and Investment listings; skip starter inventory",
            "🏦 Minimize loan-related messaging — this segment predominantly pays outright",
            "🤝 Assign dedicated international account managers",
            "📊 Offer market intelligence reports for cross-border property investment",
        ],
    },
    "First-Time Buyers": {
        "icon": "🏠",
        "tagline": "Younger, price-sensitive buyers seeking primary residences",
        "actions": [
            "🏦 Partner with mortgage providers for first-home financing bundles",
            "📚 Create educational content: home-buying guides, mortgage calculators",
            "🌐 Invest heavily in Website channel (dominant referral source)",
            "🏡 Prioritize smaller-unit Apartment inventory in affordable regions",
            "💬 Use reassuring, step-by-step communication — reduce friction",
        ],
    },
    "Corporate Buyers": {
        "icon": "🏢",
        "tagline": "Company clients acquiring multiple units for portfolio or operations",
        "actions": [
            "👔 Assign B2B sales representatives with dedicated SLAs",
            "📦 Bundle Office + Apartment inventory for portfolio pitches",
            "💼 Offer volume discounts and priority access to new tower releases",
            "🤝 Negotiate framework agreements for repeat acquisition deals",
            "📈 Provide portfolio performance dashboards as a retention tool",
        ],
    },
    "Luxury Investors": {
        "icon": "✨",
        "tagline": "Top-tier investors with premium property preferences",
        "actions": [
            "🥂 Provide white-glove concierge service and private viewings",
            "🚀 Offer early access to off-plan launches before public listing",
            "🤝 Explore co-investment and joint venture opportunities",
            "📐 Curate large floor-area units (>1,200 sqft) with premium $/sqft",
            "⭐ Prioritize satisfaction at every touchpoint — this segment scores highest",
        ],
    },
}


def _loan_rate(cluster_df: pd.DataFrame) -> float:
    """Return the loan_applied rate as a float (0-1), handling both string and numeric."""
    col = cluster_df["loan_applied"]
    # Use .astype(str) to handle both object dtype and pandas 2.x ArrowDtype strings
    return (col.astype(str) == "Yes").mean()


def render_module4(df: pd.DataFrame) -> None:
    """Segment Insights Panel — deep profile and recommendations per archetype."""

    if df.empty:
        st.warning("No data matches current filters.")
        return

    # ---- Cluster selector ---------------------------------------------------
    selected = st.selectbox(
        "Select Buyer Segment to Profile",
        options=["Global Investors", "First-Time Buyers", "Corporate Buyers", "Luxury Investors"],
    )
    cluster_df = df[df["cluster_name"] == selected]
    rec = RECOMMENDATIONS[selected]

    if cluster_df.empty:
        st.info(f"No buyers in '{selected}' with current filters.")
        return

    # ---- Header -------------------------------------------------------------
    st.markdown(f"## {rec['icon']} {selected}")
    st.caption(rec["tagline"])

    metric_cols = st.columns(4)
    metric_cols[0].metric("Buyers in Segment", f"{len(cluster_df):,}")
    metric_cols[1].metric(
        "Avg Total Spend",
        f"${cluster_df['total_spend'].mean():,.0f}" if "total_spend" in cluster_df.columns else "N/A",
    )
    metric_cols[2].metric(
        "Avg Satisfaction",
        f"{cluster_df['satisfaction_score'].mean():.2f} / 5"
        if "satisfaction_score" in cluster_df.columns
        else "N/A",
    )
    metric_cols[3].metric(
        "Loan Rate",
        f"{_loan_rate(cluster_df) * 100:.1f}%"
        if "loan_applied" in cluster_df.columns
        else "N/A",
    )

    st.divider()

    # ---- Statistical Profile Table ------------------------------------------
    st.subheader("Statistical Profile")
    numeric_cols = [
        c for c in [
            "buyer_age", "total_spend", "transaction_count",
            "satisfaction_score", "avg_price_per_sqft", "avg_floor_area",
        ]
        if c in cluster_df.columns
    ]

    if numeric_cols:
        stats = cluster_df[numeric_cols].agg(["mean", "median", "std", "min", "max"]).T.round(2)
        display_index_map = {
            "buyer_age": "Buyer Age",
            "total_spend": "Total Spend ($)",
            "transaction_count": "Transaction Count",
            "satisfaction_score": "Satisfaction Score",
            "avg_price_per_sqft": "Avg $/sqft",
            "avg_floor_area": "Avg Floor Area (sqft)",
        }
        stats.index = [display_index_map.get(c, c) for c in stats.index]
        st.dataframe(stats, use_container_width=True)
    else:
        stats = pd.DataFrame()
        st.info("No numeric columns available for stats.")

    # ---- Categorical breakdown ----------------------------------------------
    cat_cols_available = [c for c in ["country", "referral_channel", "acquisition_purpose"] if c in cluster_df.columns]
    if cat_cols_available:
        cat_metrics = st.columns(len(cat_cols_available))
        labels = {
            "country": "Top Country",
            "referral_channel": "Top Referral Channel",
            "acquisition_purpose": "Acquisition Purpose",
        }
        for i, col in enumerate(cat_cols_available):
            cat_metrics[i].metric(
                labels.get(col, col),
                cluster_df[col].value_counts().index[0],
            )

    st.divider()

    # ---- Recommendations panel ----------------------------------------------
    with st.expander("Marketing & Product Recommendations", expanded=True):
        for action in rec["actions"]:
            st.markdown(f"- {action}")

    st.divider()

    # ---- Export controls ----------------------------------------------------
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv_data = cluster_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Segment Data (CSV)",
            data=csv_data,
            file_name=f"{selected.replace(' ', '_')}_profile.csv",
            mime="text/csv",
        )

    with col_dl2:
        if st.button("Generate Segment Report (PDF)"):
            _generate_pdf_report(selected, rec, stats, cluster_df)


def _generate_pdf_report(
    selected: str,
    rec: dict,
    stats: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> None:
    """Generate PDF using reportlab and offer download via st.download_button."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from reportlab.lib import colors

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Title and tagline
        story.append(Paragraph(f"Buyer Segment Report: {selected}", styles["Title"]))
        story.append(Paragraph(rec["tagline"], styles["Normal"]))
        story.append(Spacer(1, 12))

        # KPI line
        story.append(
            Paragraph(
                f"Buyers: {len(cluster_df):,}  |  "
                f"Avg Spend: ${cluster_df['total_spend'].mean():,.0f}"
                if "total_spend" in cluster_df.columns
                else f"Buyers: {len(cluster_df):,}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 12))

        # Stats table
        if not stats.empty:
            story.append(Paragraph("Statistical Profile", styles["Heading2"]))
            table_data = [["Metric", "Mean", "Median", "Std"]] + [
                [
                    str(idx),
                    f"{row['mean']:.2f}",
                    f"{row['median']:.2f}",
                    f"{row['std']:.2f}",
                ]
                for idx, row in stats.iterrows()
            ]
            tbl = Table(table_data)
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ]
                )
            )
            story.append(tbl)
            story.append(Spacer(1, 12))

        # Recommendations
        story.append(Paragraph("Marketing & Product Recommendations", styles["Heading2"]))
        for action in rec["actions"]:
            story.append(Paragraph(f"• {action}", styles["Normal"]))
            story.append(Spacer(1, 4))

        doc.build(story)
        buf.seek(0)

        st.download_button(
            "Click to Download PDF",
            data=buf.getvalue(),
            file_name=f"{selected.replace(' ', '_')}_report.pdf",
            mime="application/pdf",
        )
    except ImportError:
        st.error("reportlab is not installed. Run: pip install reportlab")
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
