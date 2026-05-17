"""
src/pipeline.py
===============
End-to-end ML pipeline orchestrator.

Run ONCE before launching the Streamlit dashboard so all pre-computed artifacts
(CSVs, models, figures) are ready for instant loading.

Usage:
    python src/pipeline.py
    make pipeline
    python src/pipeline.py --data-dir data/raw/ --output-dir data/processed/
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def run_full_pipeline(
    data_dir: str = "data/raw/",
    output_dir: str = "data/processed/",
) -> None:
    """
    Step-by-step pipeline orchestrator. All 9 steps log progress with timing.

    Artifacts produced
    ------------------
    data/processed/cleaned_clients.csv
    data/processed/cleaned_properties.csv
    data/processed/engineered.csv          (2,000 rows)
    data/processed/X_scaled.csv
    data/processed/labeled_data.csv        (2,000 rows, ~25 cols)
    outputs/models/standard_scaler.joblib
    outputs/models/minmax_scaler.joblib
    outputs/models/kmeans_k4.joblib
    outputs/figures/k_selection.png
    outputs/figures/dendrogram.png
    outputs/figures/silhouette.png
    outputs/profiles/cluster_profiles.csv
    outputs/profiles/cluster_summary.csv
    """
    # Ensure we can import from src/ regardless of CWD
    project_root = str(Path(__file__).parent.parent.resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.data_cleaning import DataCleaner
    from src.feature_engineering import FeatureEngineer
    from src.encoding_scaling import EncoderScaler
    from src.clustering import find_optimal_k, BuyerSegmentationModel

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/profiles", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    pipeline_start = time.time()

    # =========================================================================
    # STEP 1 — Data Loading & Cleaning
    # =========================================================================
    log.info("=" * 60)
    log.info("STEP 1 — Data Loading & Cleaning")
    log.info("=" * 60)
    t0 = time.time()

    clients_path = os.path.join(data_dir, "clients.csv")
    props_path = os.path.join(data_dir, "properties.csv")

    cleaner = DataCleaner()
    df_c_raw, df_p_raw = cleaner.load_data(clients_path, props_path)

    # Clean clients — handles mixed DOB formats
    df_c = cleaner.clean_clients(df_c_raw)

    # Clean properties — parses "$300,385.62", DD-MM-YYYY, adds is_sold
    df_p = cleaner.clean_properties(df_p_raw)

    # Referential integrity — expected: 0 orphan client_refs
    cleaner.validate_referential_integrity(df_c, df_p)

    # Save
    cleaner.save_cleaned(df_c, df_p, output_dir)
    log.info("STEP 1 done in %.1fs", time.time() - t0)

    # =========================================================================
    # STEP 2 — Feature Engineering
    # =========================================================================
    log.info("=" * 60)
    log.info("STEP 2 — Feature Engineering")
    log.info("=" * 60)
    t0 = time.time()

    engineer = FeatureEngineer()
    df_eng = engineer.engineer_all(df_c, df_p)
    # Expected: 2,000 rows with buyer_age, transaction_count, total_spend,
    #           avg_floor_area, avg_price_per_sqft, spend_tier, dominant_month,
    #           transaction_year_latest, unit_category

    eng_path = os.path.join(output_dir, "engineered.csv")
    engineer.save_engineered(df_eng, path=eng_path)
    log.info("STEP 2 done in %.1fs", time.time() - t0)

    # =========================================================================
    # STEP 3 — Encoding & Scaling
    # =========================================================================
    log.info("=" * 60)
    log.info("STEP 3 — Encoding & Scaling")
    log.info("=" * 60)
    t0 = time.time()

    enc = EncoderScaler()
    X, feature_names, scalers = enc.fit_transform(df_eng)
    # X shape expected: (2000, n_features), all numeric, scaled

    x_path = os.path.join(output_dir, "X_scaled.csv")
    enc.save_feature_matrix(X, feature_names, path=x_path)
    log.info("STEP 3 done in %.1fs", time.time() - t0)

    # =========================================================================
    # STEP 4 — Optimal K Selection
    # =========================================================================
    log.info("=" * 60)
    log.info("STEP 4 — Optimal K Selection")
    log.info("=" * 60)
    t0 = time.time()

    best_k = find_optimal_k(X, k_range=range(2, 11), random_state=42)
    log.info("Elbow/silhouette suggested K: %d", best_k)
    # Domain override: always use K=4 (4 buyer archetypes by design)
    best_k = 4
    log.info("Overriding to domain K=4 (4 buyer archetypes)")
    log.info("STEP 4 done in %.1fs", time.time() - t0)

    # =========================================================================
    # STEP 5 — K-Means Clustering
    # =========================================================================
    log.info("=" * 60)
    log.info("STEP 5 — K-Means Clustering (K=%d)", best_k)
    log.info("=" * 60)
    t0 = time.time()

    model = BuyerSegmentationModel()
    labels = model.fit_kmeans(X, k=best_k, random_state=42)
    log.info("STEP 5 done in %.1fs", time.time() - t0)

    # =========================================================================
    # STEP 6 — Hierarchical Clustering (validation)
    # =========================================================================
    log.info("=" * 60)
    log.info("STEP 6 — Hierarchical Clustering (10%% sample, Ward linkage)")
    log.info("=" * 60)
    t0 = time.time()

    model.fit_hierarchical(X, k=best_k, sample_fraction=0.10, random_state=42)
    log.info("STEP 6 done in %.1fs", time.time() - t0)

    # =========================================================================
    # STEP 7 — Label Assignment & Profiling
    # =========================================================================
    log.info("=" * 60)
    log.info("STEP 7 — Label Assignment & Cluster Profiling")
    log.info("=" * 60)
    t0 = time.time()

    df_labeled = model.assign_cluster_names(df_eng, labels)
    model.generate_cluster_profiles(df_labeled)
    log.info("STEP 7 done in %.1fs", time.time() - t0)

    # =========================================================================
    # STEP 8 — Silhouette Plot
    # =========================================================================
    log.info("=" * 60)
    log.info("STEP 8 — Silhouette Plot")
    log.info("=" * 60)
    t0 = time.time()

    mean_sil = model.plot_silhouette(X, labels)
    log.info("Mean silhouette score: %.4f", mean_sil)
    log.info("STEP 8 done in %.1fs", time.time() - t0)

    # =========================================================================
    # STEP 9 — Save Labeled Dataset for Dashboard
    # =========================================================================
    log.info("=" * 60)
    log.info("STEP 9 — Save labeled_data.csv for Dashboard")
    log.info("=" * 60)
    t0 = time.time()

    labeled_path = os.path.join(output_dir, "labeled_data.csv")
    df_labeled.to_csv(labeled_path, index=False)
    log.info(
        "Saved labeled_data.csv -> %s  (%d rows x %d cols)",
        labeled_path, *df_labeled.shape,
    )
    log.info("STEP 9 done in %.1fs", time.time() - t0)

    # =========================================================================
    # Done
    # =========================================================================
    elapsed = time.time() - pipeline_start
    print()
    print("=" * 60)
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print("  Dashboard ready: streamlit run dashboard/app.py")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Buyer Segmentation ML Pipeline — Parcl Co."
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw/",
        help="Directory containing clients.csv and properties.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/",
        help="Directory for processed/cleaned outputs",
    )
    args = parser.parse_args()
    run_full_pipeline(args.data_dir, args.output_dir)
