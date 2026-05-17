"""
src/clustering.py
=================
Clustering module:
  - find_optimal_k()         : Elbow + Silhouette analysis across K=2-10
  - BuyerSegmentationModel   : K-Means + Hierarchical + profiling + silhouette plot

Real dataset context:
  2,000 client records, expected K=4, target silhouette > 0.40, random_state=42.
"""

import logging
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# Optimal K selection
# ===========================================================================

def find_optimal_k(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> int:
    """
    Determine the optimal number of clusters using Elbow + Silhouette methods.

    Real dataset context
    --------------------
    X has ~2,000 rows (one per client) after feature engineering.
    Expected optimal K: 4.  Target silhouette score: > 0.40.

    Steps
    -----
    1. Fit KMeans for each K, record WCSS (inertia) and silhouette score.
    2. Plot Elbow curve (WCSS) and Silhouette curve side-by-side.
    3. Print formatted comparison table.
    4. Select best_k = highest silhouette where score > 0.40;
       fallback to elbow inflection point.
    5. Save figure to outputs/figures/k_selection.png.
    6. Return best_k.
    """
    os.makedirs("outputs/figures", exist_ok=True)

    k_values = list(k_range)
    wcss_values: list[float] = []
    sil_values: list[float] = []

    log.info("Evaluating K-Means for K in %s …", list(k_range))

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        wcss = float(km.inertia_)
        sil = float(silhouette_score(X, labels))

        wcss_values.append(wcss)
        sil_values.append(sil)
        log.info("K=%d: WCSS=%.2f, Silhouette=%.4f", k, wcss, sil)

    # -----------------------------------------------------------------------
    # Select best K
    # -----------------------------------------------------------------------
    # Primary: highest silhouette above threshold
    threshold = 0.40
    eligible = [(k, s) for k, s in zip(k_values, sil_values) if s > threshold]
    if eligible:
        best_k, best_sil = max(eligible, key=lambda x: x[1])
        log.info("Best K=%d chosen by silhouette (%.4f > %.2f)", best_k, best_sil, threshold)
    else:
        # Fallback: elbow (largest second-order difference in WCSS)
        wcss_arr = np.array(wcss_values)
        deltas = np.diff(wcss_arr)
        second_diff = np.diff(deltas)
        elbow_idx = int(np.argmax(second_diff)) + 1   # +1 offset for double diff
        best_k = k_values[elbow_idx]
        log.warning(
            "No K exceeded silhouette threshold %.2f — using elbow: K=%d", threshold, best_k
        )

    # -----------------------------------------------------------------------
    # Print formatted table
    # -----------------------------------------------------------------------
    header = f"{'K':>3} | {'WCSS':>14} | {'Silhouette':>10} | {'Recommended'}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    for k, wcss, sil in zip(k_values, wcss_values, sil_values):
        marker = " ✅" if k == best_k else ""
        print(f"{k:>3} | {wcss:>14.2f} | {sil:>10.4f} |{marker}")
    print(separator)

    # -----------------------------------------------------------------------
    # Plot: Elbow + Silhouette side-by-side
    # -----------------------------------------------------------------------
    fig, (ax_elbow, ax_sil) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("K-Means — Optimal K Selection", fontsize=14, fontweight="bold")

    # Left — Elbow
    ax_elbow.plot(k_values, wcss_values, "o-", color="#1565C0", linewidth=2, markersize=6)
    ax_elbow.axvline(best_k, color="red", linestyle="--", linewidth=1.5)
    ax_elbow.annotate(
        f"Recommended K={best_k}",
        xy=(best_k, wcss_values[k_values.index(best_k)]),
        xytext=(best_k + 0.5, wcss_values[k_values.index(best_k)] * 1.05),
        fontsize=9, color="red",
        arrowprops=dict(arrowstyle="->", color="red"),
    )
    ax_elbow.set_title("Elbow Method — WCSS vs Number of Clusters")
    ax_elbow.set_xlabel("Number of Clusters (K)")
    ax_elbow.set_ylabel("WCSS (Inertia)")
    ax_elbow.grid(alpha=0.3)

    # Right — Silhouette
    ax_sil.plot(k_values, sil_values, "s-", color="#2E7D32", linewidth=2, markersize=6)
    ax_sil.axhline(threshold, color="orange", linestyle="--", linewidth=1.2,
                   label=f"Target threshold ({threshold})")
    ax_sil.axvline(best_k, color="red", linestyle="--", linewidth=1.5,
                   label=f"Recommended K={best_k}")
    ax_sil.set_title("Silhouette Score vs Number of Clusters")
    ax_sil.set_xlabel("Number of Clusters (K)")
    ax_sil.set_ylabel("Silhouette Score")
    ax_sil.legend(fontsize=8)
    ax_sil.grid(alpha=0.3)

    plt.tight_layout()
    out_path = "outputs/figures/k_selection.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved K-selection figure -> %s", out_path)

    return best_k


# ===========================================================================
# Buyer Segmentation Model
# ===========================================================================

class BuyerSegmentationModel:
    """
    Encapsulates K-Means clustering, hierarchical validation,
    cluster naming, profiling, and silhouette visualization.

    All random_state=42 for reproducibility.
    """

    CLUSTER_NAME_MAP = {
        0: "Global Investors",
        1: "First-Time Buyers",
        2: "Corporate Buyers",
        3: "Luxury Investors",
    }

    def __init__(self):
        self._kmeans_model = None

    # ------------------------------------------------------------------
    # K-Means
    # ------------------------------------------------------------------
    def fit_kmeans(
        self,
        X: np.ndarray,
        k: int = 4,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Fit KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300).
        Save model to outputs/models/kmeans_k{k}.joblib.
        Returns cluster labels array, shape (2000,).
        """
        os.makedirs("outputs/models", exist_ok=True)

        log.info("Fitting KMeans with K=%d, random_state=%d …", k, random_state)
        km = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
            max_iter=300,
        )
        labels = km.fit_predict(X)

        log.info(
            "KMeans converged in %d iterations | Inertia=%.2f",
            km.n_iter_, km.inertia_,
        )

        # Cluster size distribution
        unique, counts = np.unique(labels, return_counts=True)
        for c, n in zip(unique, counts):
            log.info("  Cluster %d: %d members (%.1f%%)", c, n, 100 * n / len(labels))

        # Save model
        model_path = f"outputs/models/kmeans_k{k}.joblib"
        joblib.dump(km, model_path)
        log.info("Saved KMeans model -> %s", model_path)

        self._kmeans_model = km
        return labels

    # ------------------------------------------------------------------
    # Hierarchical clustering (validation)
    # ------------------------------------------------------------------
    def fit_hierarchical(
        self,
        X: np.ndarray,
        k: int = 4,
        sample_fraction: float = 0.10,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        AgglomerativeClustering on 10% sample (~200 rows) — O(n^2) memory constraint.
        Plots Ward linkage dendrogram and saves to outputs/figures/dendrogram.png.
        Returns cluster labels for the sample.
        """
        os.makedirs("outputs/figures", exist_ok=True)

        # Sample 10% of X deterministically
        np.random.seed(random_state)
        sample_idx = np.random.choice(
            len(X), size=int(len(X) * sample_fraction), replace=False
        )
        X_sample = X[sample_idx]
        log.info(
            "Hierarchical clustering on %d-row sample (%.0f%% of %d)",
            len(X_sample), sample_fraction * 100, len(X),
        )

        # Fit AgglomerativeClustering
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        sample_labels = agg.fit_predict(X_sample)

        # Dendrogram using scipy linkage
        Z = linkage(X_sample, method="ward")

        fig, ax = plt.subplots(figsize=(14, 6))
        dendrogram(
            Z,
            ax=ax,
            truncate_mode="lastp",   # show last 30 merges
            p=30,
            leaf_rotation=90,
            leaf_font_size=9,
            show_contracted=True,
        )

        # Draw a horizontal cut line at the distance corresponding to k=4
        # The cut is between the (k-1)th and k-th largest merge distances
        if len(Z) >= k:
            cut_distance = (Z[-(k - 1), 2] + Z[-k, 2]) / 2
            ax.axhline(
                y=cut_distance,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"K={k} cut (dist={cut_distance:.2f})",
            )
            ax.legend(fontsize=9)

        ax.set_title(
            f"Hierarchical Clustering Dendrogram (10% sample, n={len(X_sample)})",
            fontsize=13,
        )
        ax.set_xlabel("Sample Index / Cluster Size")
        ax.set_ylabel("Ward Distance")
        plt.tight_layout()

        out_path = "outputs/figures/dendrogram.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved dendrogram -> %s", out_path)

        # Log cluster distribution in sample
        unique, counts = np.unique(sample_labels, return_counts=True)
        for c, n in zip(unique, counts):
            log.info("  Hierarchical cluster %d: %d members", c, n)

        return sample_labels

    # ------------------------------------------------------------------
    # Assign cluster names
    # ------------------------------------------------------------------
    def assign_cluster_names(
        self, df: pd.DataFrame, labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Add 'cluster' (int) and 'cluster_name' (str) columns to df.

        Then log centroid comparison so the user can validate/re-map if needed.

        Expected archetypes:
          Global Investors    : high total_spend, Investment purpose, international
          First-Time Buyers   : young age (~25-40), Home purpose, high loan_applied
          Corporate Buyers    : Company client_type, high transaction_count, Office units
          Luxury Investors    : highest avg_price_per_sqft, Investment, largest floor area
        """
        df = df.copy()
        df["cluster"] = labels
        df["cluster_name"] = df["cluster"].map(self.CLUSTER_NAME_MAP)

        # Log centroid profile for validation
        numeric_profile_cols = [
            c for c in [
                "buyer_age", "total_spend", "transaction_count",
                "satisfaction_score", "avg_price_per_sqft", "avg_floor_area",
            ]
            if c in df.columns
        ]

        if numeric_profile_cols:
            centroid_table = df.groupby("cluster_name")[numeric_profile_cols].mean().round(2)
            log.info("Cluster centroid comparison (validate name mapping):\n%s", centroid_table.to_string())

        return df

    # ------------------------------------------------------------------
    # Generate cluster profiles
    # ------------------------------------------------------------------
    def generate_cluster_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute full statistical profile per cluster_name.

        Saves:
          outputs/profiles/cluster_profiles.csv
          outputs/profiles/cluster_summary.csv
        Returns full profile DataFrame.
        """
        os.makedirs("outputs/profiles", exist_ok=True)

        # ---- Numeric stats (mean, median, std, min, max) ----------------
        numeric_cols = [
            c for c in [
                "buyer_age", "total_spend", "transaction_count", "satisfaction_score",
                "avg_price_per_sqft", "avg_floor_area", "dominant_month",
            ]
            if c in df.columns
        ]

        numeric_profile = df.groupby("cluster_name")[numeric_cols].agg(
            ["mean", "median", "std", "min", "max"]
        )

        # ---- Rate features (loan_applied, acquisition_purpose) ----------
        # Always use boolean-to-int conversion to support both object and
        # ArrowDtype string columns (pandas 2.x compatibility).
        rate_cols_config = {
            "loan_applied":        "Yes",
            "acquisition_purpose": "Investment",
        }
        rate_rows = {}
        for col, pos_val in rate_cols_config.items():
            if col not in df.columns:
                continue
            # Create a plain int64 column regardless of source dtype
            numeric_col = (df[col].astype(str) == pos_val).astype(int)
            rate_rows[col + "_rate"] = (
                df.assign(_tmp=numeric_col)
                .groupby("cluster_name")["_tmp"]
                .mean()
            )

        rate_df = pd.DataFrame(rate_rows).round(3) if rate_rows else pd.DataFrame()

        # ---- Categorical modes -------------------------------------------
        cat_cols = ["client_type", "referral_channel", "country", "region"]
        cat_modes = {}
        for col in cat_cols:
            if col in df.columns:
                cat_modes[col + "_mode"] = df.groupby("cluster_name")[col].agg(
                    lambda x: x.mode().iloc[0] if len(x) > 0 else "N/A"
                )

        cat_df = pd.DataFrame(cat_modes) if cat_modes else pd.DataFrame()

        # ---- Combine all into one profile DataFrame ---------------------
        profile = numeric_profile.copy()
        if not rate_df.empty:
            for col in rate_df.columns:
                profile[col] = rate_df[col]
        if not cat_df.empty:
            for col in cat_df.columns:
                profile[col] = cat_df[col]

        profile_path = "outputs/profiles/cluster_profiles.csv"
        profile.to_csv(profile_path)
        log.info("Saved cluster profiles -> %s", profile_path)

        # ---- Summary: one row per cluster, key stats only ----------------
        summary_cols = [
            c for c in ["buyer_age", "total_spend", "transaction_count",
                         "satisfaction_score", "avg_price_per_sqft"]
            if c in df.columns
        ]
        summary = df.groupby("cluster_name")[summary_cols].mean().round(2)
        summary["n_clients"] = df.groupby("cluster_name").size()

        summary_path = "outputs/profiles/cluster_summary.csv"
        summary.to_csv(summary_path)
        log.info("Saved cluster summary -> %s", summary_path)

        return profile

    # ------------------------------------------------------------------
    # Silhouette plot
    # ------------------------------------------------------------------
    def plot_silhouette(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        output_path: str = "outputs/figures/silhouette.png",
    ) -> float:
        """
        Generate silhouette plot for K=4, n=2000 and save to output_path.

        Steps:
          1. Compute per-sample silhouette values.
          2. For each cluster, plot sorted horizontal bars.
          3. Vertical dashed line at mean silhouette.
          4. Return mean silhouette score.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Per-sample silhouette values
        sample_silhouette_values = silhouette_samples(X, labels)
        mean_score = float(silhouette_score(X, labels))
        n_clusters = len(np.unique(labels))

        # Cluster colors from a consistent qualitative palette
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
                  "#F44336", "#00BCD4", "#FF5722", "#607D8B"]

        fig, ax = plt.subplots(figsize=(10, 7))
        y_lower = 10

        for i in range(n_clusters):
            cluster_sil = np.sort(sample_silhouette_values[labels == i])[::-1]
            size_cluster_i = len(cluster_sil)

            y_upper = y_lower + size_cluster_i
            color = colors[i % len(colors)]

            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_sil,
                alpha=0.75,
                color=color,
                label=self.CLUSTER_NAME_MAP.get(i, f"Cluster {i}"),
            )
            ax.text(-0.05, y_lower + size_cluster_i / 2, f"C{i}", fontsize=8)
            y_lower = y_upper + 10

        ax.axvline(x=mean_score, color="red", linestyle="--", linewidth=1.5)
        ax.annotate(
            f"Mean Silhouette = {mean_score:.3f}",
            xy=(mean_score, y_lower * 0.95),
            xytext=(mean_score + 0.02, y_lower * 0.9),
            fontsize=9, color="red",
        )

        ax.set_title(f"Silhouette Plot — K={n_clusters}, n={len(labels)}", fontsize=13)
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_ylabel("Cluster Samples (sorted by score)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim([-0.2, 1.0])
        ax.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved silhouette plot -> %s  (mean=%.3f)", output_path, mean_score)

        return mean_score
