# Buyer Segmentation — Parcl Co.

**Project Reference:** UM-PARCL-2025-001  
**Client:** Parcl Co. Limited  
**Objective:** ML-based buyer segmentation and investment profiling on 2,000 clients × 10,000 properties.

---

## Project Structure

```
buyer-segmentation/
├── data/
│   ├── raw/                    ← clients.csv + properties.csv (not tracked in git)
│   └── processed/              ← pipeline outputs
├── notebooks/01_eda.ipynb      ← 19-chart exploratory analysis
├── src/
│   ├── data_cleaning.py        ← DataCleaner class
│   ├── feature_engineering.py  ← FeatureEngineer class
│   ├── encoding_scaling.py     ← EncoderScaler class
│   ├── clustering.py           ← find_optimal_k + BuyerSegmentationModel
│   └── pipeline.py             ← end-to-end orchestrator
├── dashboard/
│   ├── app.py                  ← Streamlit entry point
│   ├── filters.py              ← sidebar filter controls
│   └── components/
│       ├── module1_overview.py
│       ├── module2_investor.py
│       ├── module3_geography.py
│       └── module4_insights.py
├── outputs/
│   ├── models/                 ← joblib model files
│   ├── profiles/               ← cluster CSV/PDF reports
│   └── figures/                ← matplotlib/plotly PNG exports
├── tests/
│   ├── test_cleaning.py
│   ├── test_features.py
│   └── test_clustering.py
├── requirements.txt
├── Makefile
└── README.md
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Raw Data
Place `clients.csv` and `properties.csv` in `data/raw/`.

### 3. Run the ML Pipeline
```bash
make pipeline
# or: python src/pipeline.py
```

### 4. Launch the Dashboard
```bash
make dashboard
# or: streamlit run dashboard/app.py
```

### 5. Run Tests
```bash
make test
# or: pytest tests/ -v
```

### 6. Clean Outputs
```bash
make clean
```

---

## Dataset Summary

| File | Rows | Cols | Key Notes |
|------|------|------|-----------|
| `clients.csv` | 2,000 | 12 | Zero nulls; mixed DOB formats |
| `properties.csv` | 10,000 | 9 | 7,305 Sold / 2,695 Available |

### Engineered Features
| Feature | Source | Notes |
|---------|--------|-------|
| `buyer_age` | `date_of_birth` | Capped at 90 (4 records) |
| `transaction_count` | properties join | Range 3–13 |
| `total_spend` | properties join | $463K–$3.65M |
| `avg_floor_area` | properties join | 410–1,957 sqft |
| `avg_price_per_sqft` | derived | sale_price / floor_area |
| `spend_tier` | qcut(4) | Budget/Mid/Premium/Luxury (500 each) |
| `dominant_month` | transaction_date | Modal month per client |

### Cluster Archetypes (K=4)
| Cluster | Key Traits |
|---------|-----------|
| **Global Investors** | High spend, international, Investment purpose, low loan rate |
| **First-Time Buyers** | Younger, Home purpose, high loan rate, Website channel |
| **Corporate Buyers** | Company type, high transaction count, Office units |
| **Luxury Investors** | Highest $/sqft, largest floor area, high satisfaction |

---

## Pipeline Steps

1. **Data Loading & Cleaning** — handles mixed DOB formats, parses sale_price strings
2. **Feature Engineering** — 7 derived features, age cap, spend tiers
3. **Encoding & Scaling** — one-hot (countries), frequency encode (regions), StandardScaler + MinMaxScaler
4. **Optimal K Selection** — Elbow + Silhouette across K=2–10
5. **K-Means Clustering** — K=4, random_state=42
6. **Hierarchical Validation** — Ward linkage dendrogram on 10% sample
7. **Label Assignment & Profiling** — cluster CSVs + summary
8. **Silhouette Plot** — per-sample visualization
9. **Export labeled dataset** — for dashboard consumption

---

## Dashboard Modules

| Tab | Module | Key Charts |
|-----|--------|-----------|
| 📊 Segmentation Overview | `module1_overview.py` | Pie + bar distributions, centroid table, silhouette/dendrogram |
| 💰 Investor Behavior | `module2_investor.py` | Loan rates, spend boxes, purpose donuts, satisfaction heatmap |
| 🌍 Geographic Analysis | `module3_geography.py` | Choropleth, region drill-down, top-10 table |
| 🔍 Segment Insights | `module4_insights.py` | Deep profile, stats, recommendations, CSV/PDF export |

---

*© 2025 Unified Mentor · Parcl Co. Limited · CONFIDENTIAL*
