"""
tests/test_features.py
=======================
Unit tests for FeatureEngineer — grounded in real dataset characteristics.

Real invariants tested:
  - All 2,000 clients have >= 3 transactions (min=3, max=13)
  - buyer_age is capped at 90 (exactly 4 records exceed 90)
  - spend_tier pd.qcut produces exactly 500 records per tier
  - total_spend range: $463K - $3.65M per client
  - Exactly 10 unique countries in clients.csv
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

ENGINEERED_PATH = Path("data/processed/engineered.csv")
CLIENTS_PATH = Path("data/raw/clients.csv")


def _load_engineered():
    """Load engineered.csv, skip test if not yet generated."""
    if not ENGINEERED_PATH.exists():
        pytest.skip(
            "data/processed/engineered.csv not found — run 'python src/pipeline.py' first"
        )
    return pd.read_csv(ENGINEERED_PATH)


# ---------------------------------------------------------------------------
# Transaction features
# ---------------------------------------------------------------------------

def test_all_clients_have_transactions():
    """All 2,000 clients must have >= 3 transactions (real dataset invariant)"""
    df = _load_engineered()
    assert df["transaction_count"].min() >= 3, (
        f"Expected min transaction_count >= 3, got {df['transaction_count'].min()}"
    )
    assert df["transaction_count"].max() == 13, (
        f"Expected max transaction_count == 13, got {df['transaction_count'].max()}"
    )


def test_no_null_transactions():
    """transaction_count must not have any nulls after merge"""
    df = _load_engineered()
    nulls = df["transaction_count"].isnull().sum()
    assert nulls == 0, f"Found {nulls} null transaction_count values"


def test_total_spend_range():
    """Per-client total_spend range: $463K - $3.65M"""
    df = _load_engineered()
    assert df["total_spend"].min() > 400_000, (
        f"Min total_spend ${df['total_spend'].min():,.0f} is below $400K threshold"
    )
    assert df["total_spend"].max() < 4_000_000, (
        f"Max total_spend ${df['total_spend'].max():,.0f} exceeds $4M threshold"
    )


# ---------------------------------------------------------------------------
# buyer_age
# ---------------------------------------------------------------------------

def test_buyer_age_cap():
    """buyer_age must be capped at 90 — max should equal 90"""
    df = _load_engineered()
    assert df["buyer_age"].max() == 90, (
        f"Expected buyer_age max == 90, got {df['buyer_age'].max()}"
    )


def test_buyer_age_min():
    """buyer_age minimum should be >= 25 (youngest in real dataset)"""
    df = _load_engineered()
    assert df["buyer_age"].min() >= 25, (
        f"Expected buyer_age min >= 25, got {df['buyer_age'].min()}"
    )


# ---------------------------------------------------------------------------
# spend_tier
# ---------------------------------------------------------------------------

def test_spend_tier_balanced():
    """pd.qcut should produce exactly 500 records per tier (perfectly balanced)"""
    df = _load_engineered()
    counts = df["spend_tier"].value_counts()
    for tier in ["Budget", "Mid", "Premium", "Luxury"]:
        assert tier in counts.index, f"spend_tier missing tier: {tier}"
        assert counts[tier] == 500, (
            f"Expected 500 records in '{tier}' tier, got {counts[tier]}"
        )


def test_spend_tier_values():
    """spend_tier must only contain the 4 expected labels"""
    df = _load_engineered()
    valid_tiers = {"Budget", "Mid", "Premium", "Luxury"}
    actual = set(df["spend_tier"].dropna().unique())
    assert actual == valid_tiers, f"Unexpected spend_tier values: {actual - valid_tiers}"


# ---------------------------------------------------------------------------
# Country cardinality
# ---------------------------------------------------------------------------

def test_country_count():
    """Exactly 10 unique countries in clients.csv"""
    if not CLIENTS_PATH.exists():
        pytest.skip("data/raw/clients.csv not available")
    df = pd.read_csv(CLIENTS_PATH)
    n_countries = df["country"].nunique()
    assert n_countries == 10, f"Expected 10 unique countries, found {n_countries}"


def test_expected_countries():
    """The 10 countries must be exactly the specified set"""
    if not CLIENTS_PATH.exists():
        pytest.skip("data/raw/clients.csv not available")
    df = pd.read_csv(CLIENTS_PATH)
    expected = {
        "USA", "UK", "Canada", "Germany", "France",
        "Belgium", "Mexico", "Australia", "Russia", "Denmark",
    }
    actual = set(df["country"].unique())
    assert actual == expected, f"Country mismatch. Extra: {actual - expected}, Missing: {expected - actual}"


# ---------------------------------------------------------------------------
# Region frequency
# ---------------------------------------------------------------------------

def test_region_count():
    """clients.csv should have exactly 57 unique regions"""
    if not CLIENTS_PATH.exists():
        pytest.skip("data/raw/clients.csv not available")
    df = pd.read_csv(CLIENTS_PATH)
    n_regions = df["region"].nunique()
    assert n_regions == 57, f"Expected 57 unique regions, got {n_regions}"
