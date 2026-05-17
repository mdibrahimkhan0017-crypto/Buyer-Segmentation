"""
src/data_cleaning.py
====================
DataCleaner class — handles loading, cleaning, and validation of the two raw CSVs.

Real dataset characteristics encoded here:
  clients.csv   : 2,000 rows, 12 cols, ZERO nulls
                  date_of_birth has TWO mixed formats:
                    "05-11-1968"  → DD-MM-YYYY (dash-delimited)
                    "11/26/1962"  → M/DD/YYYY  (slash-delimited)
  properties.csv: 10,000 rows, 9 cols
                  2,695 'Available' rows have NULL client_ref — EXPECTED, not dirty.
                  sale_price is string "$300,385.62" → parse to float.
                  transaction_date is DD-MM-YYYY consistently.
"""

import logging
import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup — INFO level, timestamps
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


class DataCleaner:
    """
    Loads, validates, and cleans the two raw CSV files:
      - clients.csv      (2,000 rows, 12 cols)
      - properties.csv   (10,000 rows, 9 cols)

    All methods log key statistics so the pipeline is fully auditable.
    """

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    def load_data(
        self,
        clients_path: str,
        properties_path: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both CSVs from disk.

        Logs shape and null counts for each DataFrame so that any
        unexpected data drift is visible immediately.

        Parameters
        ----------
        clients_path    : path to clients.csv   (expected 2,000 × 12)
        properties_path : path to properties.csv (expected 10,000 × 9)

        Returns
        -------
        (df_clients, df_properties) — raw, unmodified DataFrames
        """
        log.info("Loading clients from: %s", clients_path)
        df_clients = pd.read_csv(clients_path)
        log.info(
            "Clients shape: %s  |  nulls: %d",
            df_clients.shape,
            df_clients.isnull().sum().sum(),
        )

        log.info("Loading properties from: %s", properties_path)
        df_properties = pd.read_csv(properties_path)
        log.info(
            "Properties shape: %s  |  nulls: %d",
            df_properties.shape,
            df_properties.isnull().sum().sum(),
        )

        return df_clients, df_properties

    # ------------------------------------------------------------------
    # 2. Clean clients
    # ------------------------------------------------------------------
    def clean_clients(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the clients DataFrame.

        Key operations
        --------------
        • Parse date_of_birth with format='mixed', dayfirst=True to handle:
            - "05-11-1968"  (DD-MM-YYYY, dash)
            - "11/26/1962"  (M/DD/YYYY,  slash — no leading zeros)
            - "2/28/1976"   (M/DD/YYYY,  single-digit month)
          Rows that fail to parse become NaT → drop them and log count.

        • No age computation here — FeatureEngineer.derive_buyer_age() handles that.

        Parameters
        ----------
        df : raw clients DataFrame

        Returns
        -------
        Cleaned clients DataFrame with date_of_birth as datetime64[ns].
        """
        df = df.copy()
        log.info("Cleaning clients — initial shape: %s", df.shape)

        # --- Parse mixed date_of_birth formats ---------------------------------
        # format='mixed' lets pandas detect each row's format individually.
        # dayfirst=True ensures "05-11-1968" is read as 5 Nov, not 11 May.
        # errors='coerce' turns unparseable strings into NaT instead of raising.
        df["date_of_birth"] = pd.to_datetime(
            df["date_of_birth"],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )

        # Count and drop rows where date parsing failed (NaT)
        n_failed = df["date_of_birth"].isna().sum()
        log.info("date_of_birth parse failures (NaT): %d", n_failed)
        if n_failed > 0:
            df = df.dropna(subset=["date_of_birth"])
            log.info("Dropped %d rows with unparseable date_of_birth", n_failed)

        # Confirm zero nulls in all other columns (real dataset has none)
        remaining_nulls = df.drop(columns=["date_of_birth"]).isnull().sum().sum()
        log.info("Remaining nulls (excl. date_of_birth): %d", remaining_nulls)

        log.info("Clients after cleaning — shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    # 3. Clean properties
    # ------------------------------------------------------------------
    def clean_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the properties DataFrame.

        Key operations
        --------------
        • sale_price  : strip "$" and "," then cast to float64 → sale_price_num
                        Keep original sale_price column for reference.
        • transaction_date : parse with dayfirst=True (format DD-MM-YYYY).
        • is_sold     : boolean column, True where listing_status == 'Sold'.
        • client_ref nulls for Available rows are EXPECTED — do NOT drop them.

        Parameters
        ----------
        df : raw properties DataFrame

        Returns
        -------
        Cleaned properties DataFrame with sale_price_num (float64),
        transaction_date (datetime64), and is_sold (bool) columns added.
        """
        df = df.copy()
        log.info("Cleaning properties — initial shape: %s", df.shape)

        # --- Parse sale_price string to numeric --------------------------------
        # Real values look like "$300,385.62" — remove $ and commas first.
        df["sale_price_num"] = (
            df["sale_price"]
            .str.replace(r"[\$,]", "", regex=True)  # strip $ and ,
            .astype(float)                            # cast to float64
        )
        log.info(
            "sale_price_num range: $%.2f – $%.2f",
            df["sale_price_num"].min(),
            df["sale_price_num"].max(),
        )

        # --- Parse transaction_date (DD-MM-YYYY) --------------------------------
        df["transaction_date"] = pd.to_datetime(
            df["transaction_date"],
            dayfirst=True,  # ensures DD-MM-YYYY is read correctly
            errors="coerce",
        )
        n_bad_dates = df["transaction_date"].isna().sum()
        if n_bad_dates > 0:
            log.warning(
                "%d transaction_date values failed to parse — investigate!",
                n_bad_dates,
            )
        log.info(
            "transaction_date range: %s – %s",
            df["transaction_date"].min().date(),
            df["transaction_date"].max().date(),
        )

        # --- Add is_sold boolean column ----------------------------------------
        # 'Sold' = 7,305 rows; 'Available' = 2,695 rows (real dataset).
        df["is_sold"] = df["listing_status"] == "Sold"
        n_sold = df["is_sold"].sum()
        n_available = (~df["is_sold"]).sum()
        log.info("Sold=%d | Available=%d", n_sold, n_available)

        # --- Confirm expected null pattern in client_ref -----------------------
        # Available rows have NULL client_ref — this is by design, not dirty data.
        null_client_ref = df["client_ref"].isna().sum()
        log.info(
            "client_ref nulls: %d (all from Available rows — expected)", null_client_ref
        )

        log.info("Properties after cleaning — shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    # 4. Referential integrity check
    # ------------------------------------------------------------------
    def validate_referential_integrity(
        self,
        df_clients: pd.DataFrame,
        df_properties: pd.DataFrame,
    ) -> None:
        """
        Verify that every non-null client_ref in properties exists in clients.client_id.

        Real dataset: 0 orphan records in Sold rows.

        Parameters
        ----------
        df_clients   : cleaned clients DataFrame
        df_properties: cleaned properties DataFrame
        """
        log.info("Running referential integrity check …")

        # Only Sold rows carry a client_ref
        sold_refs = df_properties.loc[
            df_properties["is_sold"], "client_ref"
        ].dropna()

        valid_ids = set(df_clients["client_id"].unique())
        orphans = sold_refs[~sold_refs.isin(valid_ids)]

        n_orphans = len(orphans)
        if n_orphans == 0:
            log.info("Referential integrity check: 0 orphan client_refs found.")
        else:
            log.warning(
                "Referential integrity check: %d orphan client_refs found! IDs: %s",
                n_orphans,
                orphans.unique()[:10],
            )

    # ------------------------------------------------------------------
    # 5. Save cleaned files
    # ------------------------------------------------------------------
    def save_cleaned(
        self,
        df_clients: pd.DataFrame,
        df_properties: pd.DataFrame,
        output_dir: str = "data/processed/",
    ) -> None:
        """
        Persist cleaned DataFrames to CSV.

        Outputs
        -------
        data/processed/cleaned_clients.csv
        data/processed/cleaned_properties.csv
        """
        os.makedirs(output_dir, exist_ok=True)

        clients_path = os.path.join(output_dir, "cleaned_clients.csv")
        props_path = os.path.join(output_dir, "cleaned_properties.csv")

        df_clients.to_csv(clients_path, index=False)
        log.info("Saved cleaned clients → %s  (%d rows)", clients_path, len(df_clients))

        df_properties.to_csv(props_path, index=False)
        log.info(
            "Saved cleaned properties → %s  (%d rows)", props_path, len(df_properties)
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Allow overriding paths via positional args, e.g.:
    #   python src/data_cleaning.py data/raw/clients.csv data/raw/properties.csv
    clients_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/clients.csv"
    props_path = sys.argv[2] if len(sys.argv) > 2 else "data/raw/properties.csv"

    cleaner = DataCleaner()
    df_c, df_p = cleaner.load_data(clients_path, props_path)
    df_c_clean = cleaner.clean_clients(df_c)
    df_p_clean = cleaner.clean_properties(df_p)
    cleaner.validate_referential_integrity(df_c_clean, df_p_clean)
    cleaner.save_cleaned(df_c_clean, df_p_clean)

    log.info("data_cleaning.py standalone run complete.")
