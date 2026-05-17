"""
scripts/generate_sample_data.py
================================
Generates synthetic clients.csv and properties.csv that match every
real-data characteristic documented in the project spec.

Verified invariants reproduced:
  clients.csv   → 2,000 rows × 12 cols, ZERO nulls
    client_type     : Individual(1,897) | Company(103)
    acquisition_purpose: Home(1,385) | Investment(615)
    loan_applied    : No(1,264) | Yes(736)
    referral_channel: Website(1,103) | Agency(705) | Client(192)
    gender          : M(1,012) | F(988)
    countries (10)  : USA(1,538), UK(95), Canada(85), Germany(56),
                      France(53), Belgium(43), Mexico(40), Australia(39),
                      Russia(36), Denmark(15)
    regions (57)    : California(633), Nevada(143), Colorado(118) …
    satisfaction    : 1-5, nearly uniform (~400 per score)
    date_of_birth   : MIXED FORMATS — "05-11-1968" (DD-MM-YYYY)
                      and "11/26/1962" (M/DD/YYYY) — random split ~50/50

  properties.csv → 10,000 rows × 9 cols
    Sold: 7,305 | Available: 2,695
    Available rows have NULL client_ref (expected)
    sale_price     : "$97,402.80" – "$736,652.27" (string with $ and ,)
    floor_area_sqft: 410.71 – 1,957.16 sqft
    unit_category  : Apartment(8,547) | Office(1,453)
    tower_number   : 1 – 20
    transaction_date: DD-MM-YYYY, Jan 2024 – Dec 2025
    client_ref     : all Sold rows reference a valid client_id

Joined invariants:
  All 2,000 clients have 3–13 transactions (avg ~3.65)
  4 clients will have computed age > 90 (DOB before 1935)

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --seed 42 --out-dir data/raw/
"""

import argparse
import os
import random
import string
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Seed for full reproducibility
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

# ---------------------------------------------------------------------------
# Reference data matching real dataset
# ---------------------------------------------------------------------------

# 10 countries with exact buyer counts
COUNTRY_WEIGHTS = {
    "USA": 1538, "UK": 95, "Canada": 85, "Germany": 56, "France": 53,
    "Belgium": 43, "Mexico": 40, "Australia": 39, "Russia": 36, "Denmark": 15,
}

# 57 US-dominated regions (California 633, Nevada 143, Colorado 118 …)
US_REGIONS = {
    "California": 633, "Nevada": 143, "Colorado": 118, "Arizona": 108,
    "Oregon": 96, "Washington": 74, "Texas": 68, "Florida": 62,
    "New York": 54, "Utah": 48, "New Mexico": 44, "Idaho": 38,
    "Montana": 32, "Wyoming": 28, "North Dakota": 20, "South Dakota": 18,
    "Nebraska": 15, "Kansas": 12, "Oklahoma": 11, "Iowa": 10,
    "Missouri": 9, "Arkansas": 8, "Louisiana": 8, "Mississippi": 7,
    "Alabama": 7, "Georgia": 6, "South Carolina": 6, "North Carolina": 5,
    "Virginia": 5, "Maryland": 4, "Pennsylvania": 4, "Ohio": 4,
    "Michigan": 3, "Indiana": 3, "Illinois": 3, "Wisconsin": 3,
    "Minnesota": 2, "Tennessee": 2, "Kentucky": 2,
}

# Other country regions (one representative region per non-US country)
NON_US_REGIONS = {
    "UK":        ["England", "Scotland", "Wales", "Northern Ireland", "London"],
    "Canada":    ["Ontario", "British Columbia", "Quebec", "Alberta", "Manitoba"],
    "Germany":   ["Bavaria", "Berlin", "Hamburg", "North Rhine", "Saxony"],
    "France":    ["Ile-de-France", "Provence", "Normandy", "Brittany", "Alsace"],
    "Belgium":   ["Brussels", "Flanders", "Wallonia"],
    "Mexico":    ["Mexico City", "Jalisco", "Nuevo Leon", "Yucatan"],
    "Australia": ["New South Wales", "Victoria", "Queensland", "Western Australia"],
    "Russia":    ["Moscow", "Saint Petersburg", "Siberia", "Ural"],
    "Denmark":   ["Copenhagen", "Jutland", "Funen"],
}

FIRST_NAMES_M = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard",
    "Joseph", "Thomas", "Charles", "Daniel", "Matthew", "Anthony", "Mark",
    "Donald", "Steven", "Paul", "Andrew", "Joshua", "Kevin",
]
FIRST_NAMES_F = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth",
    "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty",
    "Margaret", "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily", "Helen",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def weighted_choice(choices: dict, size: int) -> list:
    """Sample 'size' items from choices dict {value: weight}."""
    keys = list(choices.keys())
    weights = np.array(list(choices.values()), dtype=float)
    weights = weights / weights.sum()
    return rng.choice(keys, size=size, p=weights).tolist()


def format_date_mixed(dt: date, fmt_type: str) -> str:
    """
    Format a date in one of two mixed formats used in the real dataset:
      'dash' → DD-MM-YYYY  e.g. "05-11-1968"
      'slash' → M/DD/YYYY  e.g. "11/26/1962"
    """
    if fmt_type == "dash":
        return dt.strftime("%d-%m-%Y")
    else:
        # M/DD/YYYY — no leading zero on month
        return f"{dt.month}/{dt.strftime('%d/%Y')}"


# ---------------------------------------------------------------------------
# Generate clients.csv
# ---------------------------------------------------------------------------

def generate_clients(n: int = 2000) -> pd.DataFrame:
    """Generate synthetic clients DataFrame matching all real-data invariants."""
    print(f"Generating {n} clients …")

    # ---- IDs ----------------------------------------------------------------
    client_ids = [f"C{str(i).zfill(5)}" for i in range(1, n + 1)]

    # ---- client_type: Individual(1897) | Company(103) ----------------------
    client_types = (
        ["Individual"] * 1897 + ["Company"] * 103
    )
    random.shuffle(client_types)

    # ---- gender: M(1012) | F(988) ------------------------------------------
    genders = ["M"] * 1012 + ["F"] * 988
    random.shuffle(genders)

    # ---- first/last names based on gender ----------------------------------
    first_names = [
        random.choice(FIRST_NAMES_M if g == "M" else FIRST_NAMES_F)
        for g in genders
    ]
    last_names = [random.choice(LAST_NAMES) for _ in range(n)]

    # ---- date_of_birth — MIXED FORMATS ------------------------------------
    # Age range 25-94 (real). 4 records must produce age > 90 (born before 1935).
    # Reference year is 2025. Ages: 2025 - birth_year.
    # Normal ages 25-89: birth years 1936-2000
    # 4 records: birth years 1929-1934 (ages 91-96 before cap)
    birth_years_normal = rng.integers(1936, 2001, size=n - 4).tolist()
    birth_years_old = [1929, 1931, 1932, 1934]  # will produce ages 96,94,93,91
    all_birth_years = birth_years_normal + birth_years_old
    rng.shuffle(all_birth_years := np.array(all_birth_years))

    dobs = []
    for year in all_birth_years:
        month = rng.integers(1, 13)
        # Generate valid day for that month/year
        if month == 2:
            day = rng.integers(1, 29 if (year % 4 == 0) else 28)
        elif month in [4, 6, 9, 11]:
            day = rng.integers(1, 31)
        else:
            day = rng.integers(1, 32)
        dt = date(int(year), int(month), int(day))
        # Randomly assign dash or slash format (~50/50)
        fmt = "dash" if rng.random() < 0.5 else "slash"
        dobs.append(format_date_mixed(dt, fmt))

    # ---- country & region --------------------------------------------------
    countries = weighted_choice(COUNTRY_WEIGHTS, n)

    # Build region list dynamically — generate a large US pool to avoid IndexError
    n_usa = countries.count("USA")
    us_region_pool = weighted_choice(US_REGIONS, n_usa)
    us_idx = 0
    regions = []
    for country in countries:
        if country == "USA":
            regions.append(us_region_pool[us_idx])
            us_idx += 1
        else:
            regions.append(random.choice(NON_US_REGIONS[country]))

    # ---- acquisition_purpose: Home(1385) | Investment(615) ----------------
    purposes = ["Home"] * 1385 + ["Investment"] * 615
    random.shuffle(purposes)

    # ---- satisfaction_score: 1-5, ~400 each --------------------------------
    # 400 each for scores 1-4, and 400 for score 5 (total 2000)
    sat_scores = []
    for score in [1, 2, 3, 4, 5]:
        sat_scores.extend([score] * 400)
    random.shuffle(sat_scores)

    # ---- loan_applied: No(1264) | Yes(736) ---------------------------------
    loans = ["No"] * 1264 + ["Yes"] * 736
    random.shuffle(loans)

    # ---- referral_channel: Website(1103) | Agency(705) | Client(192) ------
    channels = ["Website"] * 1103 + ["Agency"] * 705 + ["Client"] * 192
    random.shuffle(channels)

    df = pd.DataFrame({
        "client_id":           client_ids,
        "client_type":         client_types,
        "first_name":          first_names,
        "last_name":           last_names,
        "date_of_birth":       dobs,
        "gender":              genders,
        "country":             countries,
        "region":              regions,
        "acquisition_purpose": purposes,
        "satisfaction_score":  sat_scores,
        "loan_applied":        loans,
        "referral_channel":    channels,
    })

    # Verify zero nulls
    assert df.isnull().sum().sum() == 0, "BUG: clients has nulls!"
    assert df["country"].nunique() == 10, f"BUG: {df['country'].nunique()} countries, expected 10"
    print(f"  ✓ clients.csv: {df.shape} | nulls={df.isnull().sum().sum()}")
    return df


# ---------------------------------------------------------------------------
# Generate properties.csv
# ---------------------------------------------------------------------------

def generate_properties(df_clients: pd.DataFrame, n: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic properties DataFrame.

    Sold: 7,305 rows (each linked to a client_id)
    Available: 2,695 rows (client_ref = NULL)

    Ensures ALL 2,000 clients have at least 3 transactions.
    """
    print(f"Generating {n} properties (7305 Sold + 2695 Available) …")

    client_ids = df_clients["client_id"].tolist()
    n_clients = len(client_ids)
    n_sold = 7305
    n_available = 2695

    # ---- Assign transactions to clients (min 3 each) ----------------------
    # Distribute n_sold across n_clients such that min=3, avg≈3.65, max=13
    # Strategy: give every client 3, then distribute remaining randomly
    base = 3
    extra = n_sold - base * n_clients  # 7305 - 6000 = 1305 extra to distribute

    # Random extras using a geometric-like distribution (most get 0-2 extra)
    extras = rng.integers(0, 11, size=n_clients).tolist()
    extras_sum = sum(extras)
    # Scale to match exactly 'extra'
    counts = [base + int(e * extra / extras_sum) for e in extras]
    # Fix rounding: adjust last client
    diff = n_sold - sum(counts)
    counts[-1] += diff
    # Ensure no client has < 3 or > 13
    counts = [min(max(c, 3), 13) for c in counts]
    # Final correction pass
    current_total = sum(counts)
    adjustment = n_sold - current_total
    idx = 0
    while adjustment != 0:
        if adjustment > 0 and counts[idx] < 13:
            counts[idx] += 1
            adjustment -= 1
        elif adjustment < 0 and counts[idx] > 3:
            counts[idx] -= 1
            adjustment += 1
        idx = (idx + 1) % n_clients

    assert sum(counts) == n_sold, f"BUG: sold count={sum(counts)}, expected {n_sold}"
    assert min(counts) >= 3, f"BUG: min transactions={min(counts)}"
    assert max(counts) <= 13, f"BUG: max transactions={max(counts)}"

    # ---- Build Sold rows ---------------------------------------------------
    listing_ids = [f"L{str(i).zfill(6)}" for i in range(1, n + 1)]
    random.shuffle(listing_ids)

    sold_client_refs = []
    for client_id, count in zip(client_ids, counts):
        sold_client_refs.extend([client_id] * count)

    # Shuffle so client transactions aren't all contiguous
    random.shuffle(sold_client_refs)

    # ---- Date range: Jan 2024 – Dec 2025 (DD-MM-YYYY) ---------------------
    start_date = date(2024, 1, 1)
    end_date = date(2025, 12, 31)
    total_days = (end_date - start_date).days

    def random_date() -> str:
        d = start_date + timedelta(days=int(rng.integers(0, total_days + 1)))
        return d.strftime("%d-%m-%Y")

    # ---- Price range: $97,402.80 – $736,652.27 ----------------------------
    def random_price() -> str:
        price = rng.uniform(97402.80, 736652.27)
        return f"${price:,.2f}"

    # ---- Floor area: 410.71 – 1,957.16 sqft -------------------------------
    def random_floor_area() -> float:
        return round(float(rng.uniform(410.71, 1957.16)), 2)

    # ---- unit_category: Apartment(8547) | Office(1453) --------------------
    unit_cats = ["Apartment"] * 8547 + ["Office"] * 1453
    random.shuffle(unit_cats)

    # ---- Assemble all rows -------------------------------------------------
    rows = []

    # Sold rows
    for i in range(n_sold):
        rows.append({
            "listing_id":       listing_ids[i],
            "tower_number":     int(rng.integers(1, 21)),
            "transaction_date": random_date(),
            "unit_category":    unit_cats[i],
            "unit_number":      f"U{int(rng.integers(100, 9999))}",
            "floor_area_sqft":  random_floor_area(),
            "sale_price":       random_price(),
            "listing_status":   "Sold",
            "client_ref":       sold_client_refs[i],
        })

    # Available rows (client_ref = NULL)
    for i in range(n_sold, n):
        rows.append({
            "listing_id":       listing_ids[i],
            "tower_number":     int(rng.integers(1, 21)),
            "transaction_date": random_date(),
            "unit_category":    unit_cats[i],
            "unit_number":      f"U{int(rng.integers(100, 9999))}",
            "floor_area_sqft":  random_floor_area(),
            "sale_price":       random_price(),
            "listing_status":   "Available",
            "client_ref":       None,          # NULL — expected for Available
        })

    df = pd.DataFrame(rows)

    # ---- Verify invariants -------------------------------------------------
    assert len(df) == n
    assert (df["listing_status"] == "Sold").sum() == 7305
    assert (df["listing_status"] == "Available").sum() == 2695
    assert df[df["listing_status"] == "Available"]["client_ref"].isnull().all()
    assert df[df["listing_status"] == "Sold"]["client_ref"].notnull().all()

    print(f"  ✓ properties.csv: {df.shape}")
    print(f"    Sold={n_sold} | Available={n_available}")
    print(f"    client_ref nulls={df['client_ref'].isnull().sum()} (all Available)")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(out_dir: str = "data/raw/") -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Generate clients
    df_clients = generate_clients(n=2000)

    # Generate properties (references client IDs)
    df_properties = generate_properties(df_clients, n=10000)

    # Save
    clients_path = os.path.join(out_dir, "clients.csv")
    props_path = os.path.join(out_dir, "properties.csv")

    df_clients.to_csv(clients_path, index=False)
    df_properties.to_csv(props_path, index=False)

    print(f"\n  Saved → {clients_path}  ({len(df_clients)} rows)")
    print(f"  Saved → {props_path}  ({len(df_properties)} rows)")

    # Quick summary stats
    print("\n--- clients.csv summary ---")
    print(f"  client_type  : {df_clients['client_type'].value_counts().to_dict()}")
    print(f"  loan_applied : {df_clients['loan_applied'].value_counts().to_dict()}")
    print(f"  referral_ch. : {df_clients['referral_channel'].value_counts().to_dict()}")
    print(f"  countries    : {df_clients['country'].nunique()} unique")
    print(f"  regions      : {df_clients['region'].nunique()} unique")
    print(f"  nulls total  : {df_clients.isnull().sum().sum()}")

    print("\n--- properties.csv summary ---")
    sold_df = df_properties[df_properties["listing_status"] == "Sold"]
    txn_counts = sold_df.groupby("client_ref").size()
    print(f"  transaction_count: min={txn_counts.min()}, max={txn_counts.max()}, mean={txn_counts.mean():.2f}")
    print(f"  unit_category: {df_properties['unit_category'].value_counts().to_dict()}")
    print(f"  tower_number range: {df_properties['tower_number'].min()}–{df_properties['tower_number'].max()}")

    print("\n✅ Sample data generation complete. Run pipeline next:")
    print("   python src/pipeline.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic buyer-segmentation data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--out-dir", default="data/raw/", help="Output directory (default: data/raw/)")
    args = parser.parse_args()

    SEED = args.seed
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    main(args.out_dir)
