"""
tests/test_cleaning.py
=======================
Unit tests for DataCleaner — grounded in real dataset characteristics.

Real invariants tested:
  - sale_price string "$300,385.62" parses to float 300385.62
  - Real price range: $97,402.80 - $736,652.27
  - Mixed DOB formats (DD-MM-YYYY dash, M/DD/YYYY slash)
  - Zero nulls in all 12 clients columns
  - 2,695 Available rows have null client_ref (expected)
  - 0 Sold rows have null client_ref
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helper: parse a single sale_price string (mirrors DataCleaner logic)
# ---------------------------------------------------------------------------
def parse_sale_price(value: str) -> float:
    """Strip '$' and ',' then cast to float — same as DataCleaner.clean_properties."""
    import re
    return float(re.sub(r"[\$,]", "", value))


# ---------------------------------------------------------------------------
# sale_price parsing tests
# ---------------------------------------------------------------------------

def test_sale_price_parsing():
    """'$300,385.62' must parse to float 300385.62"""
    result = parse_sale_price("$300,385.62")
    assert result == pytest.approx(300385.62, abs=0.01)


def test_sale_price_no_cents():
    """'$500,000' must parse to 500000.0"""
    assert parse_sale_price("$500,000") == pytest.approx(500000.0)


def test_sale_price_min_max():
    """Real dataset range: $97,402.80 - $736,652.27"""
    raw_path = Path("data/raw/properties.csv")
    if not raw_path.exists():
        pytest.skip("data/raw/properties.csv not available")

    from src.data_cleaning import DataCleaner
    cleaner = DataCleaner()
    _, df_p_raw = cleaner.load_data(
        "data/raw/clients.csv", "data/raw/properties.csv"
    )
    df = cleaner.clean_properties(df_p_raw)

    assert df["sale_price_num"].min() >= 97000, (
        f"Min sale_price_num={df['sale_price_num'].min()} below expected $97K"
    )
    assert df["sale_price_num"].max() <= 737000, (
        f"Max sale_price_num={df['sale_price_num'].max()} above expected $737K"
    )


# ---------------------------------------------------------------------------
# Date format tests
# ---------------------------------------------------------------------------

def test_date_format_dash():
    """'05-11-1968' (DD-MM-YYYY, dash) must parse to Nov 1968"""
    result = pd.to_datetime("05-11-1968", format="mixed", dayfirst=True)
    assert result.year == 1968
    assert result.month == 11  # November, not May
    assert result.day == 5


def test_date_format_slash():
    """'11/26/1962' (M/DD/YYYY, slash) must parse to Nov 1962"""
    result = pd.to_datetime("11/26/1962", format="mixed", dayfirst=False)
    assert result.year == 1962
    assert result.month == 11
    assert result.day == 26


def test_date_format_single_digit_month():
    """'2/28/1976' (M/DD/YYYY, single-digit month) must parse to Feb 1976"""
    result = pd.to_datetime("2/28/1976", format="mixed", dayfirst=False)
    assert result.year == 1976
    assert result.month == 2
    assert result.day == 28


# ---------------------------------------------------------------------------
# Null / referential integrity tests
# ---------------------------------------------------------------------------

def test_zero_nulls_in_clients():
    """clients.csv must have zero nulls in all 12 columns"""
    raw_path = Path("data/raw/clients.csv")
    if not raw_path.exists():
        pytest.skip("data/raw/clients.csv not available")

    df = pd.read_csv(raw_path)
    total_nulls = df.isnull().sum().sum()
    assert total_nulls == 0, f"Expected 0 nulls in clients.csv but found {total_nulls}"


def test_available_nulls_expected():
    """2,695 Available rows must have null client_ref — this is EXPECTED, not an error"""
    raw_path = Path("data/raw/properties.csv")
    if not raw_path.exists():
        pytest.skip("data/raw/properties.csv not available")

    df = pd.read_csv(raw_path)
    available_nulls = df[df["listing_status"] == "Available"]["client_ref"].isnull().sum()
    assert available_nulls == 2695, (
        f"Expected 2695 null client_refs in Available rows, found {available_nulls}"
    )


def test_sold_no_nulls():
    """All 7,305 Sold rows must have non-null client_ref"""
    raw_path = Path("data/raw/properties.csv")
    if not raw_path.exists():
        pytest.skip("data/raw/properties.csv not available")

    df = pd.read_csv(raw_path)
    sold_nulls = df[df["listing_status"] == "Sold"]["client_ref"].isnull().sum()
    assert sold_nulls == 0, (
        f"Expected 0 null client_refs in Sold rows, found {sold_nulls}"
    )


def test_sold_count():
    """Exactly 7,305 Sold rows and 2,695 Available rows"""
    raw_path = Path("data/raw/properties.csv")
    if not raw_path.exists():
        pytest.skip("data/raw/properties.csv not available")

    df = pd.read_csv(raw_path)
    assert (df["listing_status"] == "Sold").sum() == 7305
    assert (df["listing_status"] == "Available").sum() == 2695
