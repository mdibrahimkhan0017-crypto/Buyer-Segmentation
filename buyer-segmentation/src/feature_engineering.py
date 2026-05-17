"""
src/feature_engineering.py
===========================
FeatureEngineer class — derives 7 client-level features from the cleaned DataFrames.

Real dataset invariants:
  - All 2,000 clients have >= 3 transactions (min=3, max=13)
  - buyer_age range after cap: 25-90 (4 records capped)
  - spend_tier qcut produces exactly 500 clients per tier
  - total_spend range: $463,611 - $3,653,385
"""

import logging
import os

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


class FeatureEngineer:
    """Derives engineered features from cleaned clients + properties DataFrames."""

    def engineer_all(
        self,
        df_clients: pd.DataFrame,
        df_properties: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Orchestrate all feature derivation and return merged DataFrame (2,000 rows).
        Steps: buyer_age -> transaction features -> price_per_sqft ->
               spend_tier -> temporal features -> dominant unit_category
        """
        log.info("=== FeatureEngineer.engineer_all() START ===")
        df = df_clients.copy()

        df = self.derive_buyer_age(df)
        df = self.derive_transaction_features(df, df_properties)
        df = self.derive_price_per_sqft(df, df_properties)
        df = self.derive_spend_tier(df)
        df = self.derive_temporal_features(df, df_properties)
        df = self.derive_dominant_unit_category(df, df_properties)

        log.info("=== FeatureEngineer.engineer_all() DONE — shape: %s ===", df.shape)
        return df

    def derive_buyer_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute buyer_age = 2025 - date_of_birth.dt.year.
        Cap at 90: exactly 4 records exceed 90 in the real dataset.
        """
        df = df.copy()
        df["buyer_age"] = 2025 - df["date_of_birth"].dt.year

        # Cap ages > 90 at 90 (4 records in real data)
        over_90_mask = df["buyer_age"] > 90
        n_capped = int(over_90_mask.sum())
        df.loc[over_90_mask, "buyer_age"] = 90
        log.info("%d records capped at age 90", n_capped)
        log.info(
            "buyer_age stats — min=%d, max=%d, mean=%.1f",
            df["buyer_age"].min(), df["buyer_age"].max(), df["buyer_age"].mean(),
        )
        return df

    def derive_transaction_features(
        self, df_clients: pd.DataFrame, df_properties: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter to Sold rows, aggregate per client_ref, merge into clients.
        Aggregates: transaction_count (3-13), total_spend ($463K-$3.65M), avg_floor_area.
        """
        df_clients = df_clients.copy()
        sold = df_properties[df_properties["is_sold"]].copy()
        log.info("Sold properties for aggregation: %d rows", len(sold))

        agg = (
            sold.groupby("client_ref")
            .agg(
                transaction_count=("listing_id", "count"),
                total_spend=("sale_price_num", "sum"),
                avg_floor_area=("floor_area_sqft", "mean"),
            )
            .reset_index()
            .rename(columns={"client_ref": "client_id"})
        )

        df_merged = df_clients.merge(agg, on="client_id", how="left")

        # Invariant: all 2,000 clients have >= 3 transactions
        assert df_merged["transaction_count"].isnull().sum() == 0, (
            "Some clients have no transaction records!"
        )
        assert df_merged["transaction_count"].min() >= 3, (
            f"Min transaction_count={df_merged['transaction_count'].min()} < 3"
        )

        log.info(
            "transaction_count: min=%d, max=%d, mean=%.2f",
            df_merged["transaction_count"].min(),
            df_merged["transaction_count"].max(),
            df_merged["transaction_count"].mean(),
        )
        log.info(
            "total_spend: min=$%.0f, max=$%.0f",
            df_merged["total_spend"].min(), df_merged["total_spend"].max(),
        )
        return df_merged

    def derive_price_per_sqft(
        self, df_clients: pd.DataFrame, df_properties: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute price_per_sqft = sale_price_num / floor_area_sqft per Sold row,
        average per client_ref, merge as avg_price_per_sqft.
        """
        df_clients = df_clients.copy()
        sold = df_properties[df_properties["is_sold"]].copy()

        sold["price_per_sqft"] = sold["sale_price_num"] / sold["floor_area_sqft"]

        psf_agg = (
            sold.groupby("client_ref")["price_per_sqft"]
            .mean()
            .reset_index()
            .rename(columns={"client_ref": "client_id", "price_per_sqft": "avg_price_per_sqft"})
        )

        df_merged = df_clients.merge(psf_agg, on="client_id", how="left")
        log.info(
            "avg_price_per_sqft: min=$%.2f, max=$%.2f, mean=$%.2f",
            df_merged["avg_price_per_sqft"].min(),
            df_merged["avg_price_per_sqft"].max(),
            df_merged["avg_price_per_sqft"].mean(),
        )
        return df_merged

    def derive_spend_tier(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Quartile-cut total_spend into 4 equal bands (500 clients each).
        Labels: Budget | Mid | Premium | Luxury
        """
        df = df.copy()
        df["spend_tier"], bins = pd.qcut(
            df["total_spend"],
            q=4,
            labels=["Budget", "Mid", "Premium", "Luxury"],
            retbins=True,
        )
        log.info(
            "spend_tier quantile breakpoints: Budget<=$%.0f | Mid<=$%.0f | Premium<=$%.0f | Luxury>$%.0f",
            bins[1], bins[2], bins[3], bins[3],
        )
        log.info("spend_tier distribution:\n%s", df["spend_tier"].value_counts().sort_index().to_string())
        return df

    def derive_temporal_features(
        self, df_clients: pd.DataFrame, df_properties: pd.DataFrame
    ) -> pd.DataFrame:
        """
        From transaction_date (Jan 2024 - Dec 2025):
          dominant_month          : modal transaction month per client (1-12)
          transaction_year_latest : most recent transaction year (2024 or 2025)
        """
        df_clients = df_clients.copy()
        sold = df_properties[df_properties["is_sold"]].copy()

        sold["transaction_month"] = sold["transaction_date"].dt.month
        sold["transaction_year"] = sold["transaction_date"].dt.year

        dominant_month = (
            sold.groupby("client_ref")["transaction_month"]
            .agg(lambda x: x.mode().iloc[0])
            .reset_index()
            .rename(columns={"client_ref": "client_id", "transaction_month": "dominant_month"})
        )

        latest_year = (
            sold.groupby("client_ref")["transaction_year"]
            .max()
            .reset_index()
            .rename(columns={"client_ref": "client_id", "transaction_year": "transaction_year_latest"})
        )

        df_merged = df_clients.merge(dominant_month, on="client_id", how="left")
        df_merged = df_merged.merge(latest_year, on="client_id", how="left")

        log.info(
            "dominant_month: min=%d, max=%d",
            df_merged["dominant_month"].min(), df_merged["dominant_month"].max(),
        )
        log.info(
            "transaction_year_latest: %s",
            df_merged["transaction_year_latest"].value_counts().to_dict(),
        )
        return df_merged

    def derive_dominant_unit_category(
        self, df_clients: pd.DataFrame, df_properties: pd.DataFrame
    ) -> pd.DataFrame:
        """
        For each client, determine the most frequent unit_category among Sold transactions.
        Merges as 'unit_category' column.
        """
        df_clients = df_clients.copy()
        sold = df_properties[df_properties["is_sold"]].copy()

        dom_unit = (
            sold.groupby("client_ref")["unit_category"]
            .agg(lambda x: x.mode().iloc[0])
            .reset_index()
            .rename(columns={"client_ref": "client_id"})
        )

        df_merged = df_clients.merge(dom_unit, on="client_id", how="left")
        log.info(
            "unit_category (dominant per client):\n%s",
            df_merged["unit_category"].value_counts().to_string(),
        )
        return df_merged

    def save_engineered(
        self, df: pd.DataFrame, path: str = "data/processed/engineered.csv"
    ) -> None:
        """Save final engineered DataFrame. Log shape and first 3 rows."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        df.to_csv(path, index=False)
        log.info("Saved engineered data -> %s  (%d rows x %d cols)", path, *df.shape)
        log.info("First 3 rows:\n%s", df.head(3).to_string())


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_cleaning import DataCleaner

    cleaner = DataCleaner()
    df_c, df_p = cleaner.load_data("data/raw/clients.csv", "data/raw/properties.csv")
    df_c = cleaner.clean_clients(df_c)
    df_p = cleaner.clean_properties(df_p)

    engineer = FeatureEngineer()
    df_eng = engineer.engineer_all(df_c, df_p)
    engineer.save_engineered(df_eng)

    log.info("feature_engineering.py standalone run complete.")
