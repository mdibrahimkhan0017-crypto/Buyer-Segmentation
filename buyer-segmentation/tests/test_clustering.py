"""
tests/test_clustering.py
=========================
Unit tests for clustering module — grounded in real dataset characteristics.

Real invariants tested:
  - KMeans with random_state=42 produces identical results on every run
  - K=4 produces exactly 4 unique cluster labels
  - No cluster has < 3% of 2,000 buyers (< 60 clients)
  - cluster_name column has exactly the 4 expected string values
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent.parent))

LABELED_PATH = Path("data/processed/labeled_data.csv")
X_SCALED_PATH = Path("data/processed/X_scaled.csv")


def _load_labeled() -> pd.DataFrame:
    """Load labeled_data.csv, skip test if not yet generated."""
    if not LABELED_PATH.exists():
        pytest.skip(
            "data/processed/labeled_data.csv not found — run 'python src/pipeline.py' first"
        )
    return pd.read_csv(LABELED_PATH)


def _load_feature_matrix() -> np.ndarray:
    """Load X_scaled.csv as numpy array, skip test if not yet generated."""
    if not X_SCALED_PATH.exists():
        pytest.skip(
            "data/processed/X_scaled.csv not found — run 'python src/pipeline.py' first"
        )
    return pd.read_csv(X_SCALED_PATH).values


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_kmeans_reproducibility():
    """Same cluster labels every run with random_state=42"""
    X = _load_feature_matrix()
    labels1 = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
    labels2 = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
    np.testing.assert_array_equal(
        labels1, labels2,
        err_msg="KMeans with random_state=42 produced different labels on repeated runs",
    )


# ---------------------------------------------------------------------------
# Cluster count and size
# ---------------------------------------------------------------------------

def test_four_clusters_produced():
    """KMeans with K=4 must produce exactly 4 unique integer labels"""
    X = _load_feature_matrix()
    labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
    unique_labels = np.unique(labels)
    assert len(unique_labels) == 4, (
        f"Expected 4 unique clusters, got {len(unique_labels)}: {unique_labels}"
    )


def test_cluster_sizes_reasonable():
    """No cluster should contain < 3% of total buyers (< 60 of 2,000)"""
    df = _load_labeled()
    total = len(df)
    min_cluster_size = df["cluster"].value_counts().min()
    threshold = int(total * 0.03)
    assert min_cluster_size >= threshold, (
        f"Smallest cluster has {min_cluster_size} members — below 3% threshold ({threshold})"
    )


# ---------------------------------------------------------------------------
# Cluster names
# ---------------------------------------------------------------------------

def test_cluster_names_valid():
    """cluster_name column must contain exactly the 4 expected string values"""
    df = _load_labeled()
    expected = {"Global Investors", "First-Time Buyers", "Corporate Buyers", "Luxury Investors"}
    actual = set(df["cluster_name"].unique())
    assert actual == expected, (
        f"cluster_name mismatch. Extra: {actual - expected}, Missing: {expected - actual}"
    )


def test_cluster_name_no_nulls():
    """cluster_name must have no null values"""
    df = _load_labeled()
    nulls = df["cluster_name"].isnull().sum()
    assert nulls == 0, f"Found {nulls} null cluster_name values"


def test_cluster_integer_range():
    """cluster column must be integers in range 0-3"""
    df = _load_labeled()
    assert df["cluster"].min() == 0, f"Min cluster label should be 0, got {df['cluster'].min()}"
    assert df["cluster"].max() == 3, f"Max cluster label should be 3, got {df['cluster'].max()}"


# ---------------------------------------------------------------------------
# Labeled dataset shape
# ---------------------------------------------------------------------------

def test_labeled_data_row_count():
    """labeled_data.csv must have exactly 2,000 rows"""
    df = _load_labeled()
    assert len(df) == 2000, f"Expected 2,000 rows in labeled_data.csv, got {len(df)}"


def test_labeled_data_has_required_columns():
    """labeled_data.csv must contain the core columns needed by the dashboard"""
    df = _load_labeled()
    required = {
        "cluster", "cluster_name", "buyer_age", "total_spend",
        "transaction_count", "satisfaction_score", "country", "region",
        "acquisition_purpose", "loan_applied", "referral_channel",
    }
    missing = required - set(df.columns)
    assert not missing, f"labeled_data.csv is missing required columns: {missing}"
