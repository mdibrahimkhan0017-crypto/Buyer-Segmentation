"""
src/encoding_scaling.py
========================
EncoderScaler class — prepares engineered features for K-Means clustering.

Encoding strategy (real-data justified):
  ONE-HOT     : client_type(2), country(10), referral_channel(3), unit_category(2)
  LABEL ENCODE: gender, loan_applied, acquisition_purpose
  FREQ ENCODE : region (57 unique values — too high for one-hot)
  STANDARD    : buyer_age, satisfaction_score, total_spend, transaction_count
  MIN-MAX     : floor_area_sqft, avg_price_per_sqft, avg_floor_area, dominant_month
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


class EncoderScaler:
    """
    Applies all encoding and scaling transformations required for clustering.
    Saves fitted scalers to outputs/models/ via joblib.
    """

    # ---------------------------------------------------------------
    # Columns to remove before building the feature matrix
    # ---------------------------------------------------------------
    COLUMNS_TO_DROP = [
        "client_id",
        "first_name",
        "last_name",
        "client_ref",        # not present after merge, but guard it anyway
        "date_of_birth",     # replaced by buyer_age
        "spend_tier",        # replaced by total_spend (numeric)
        "sale_price",        # string col — not present in engineered but guard it
    ]

    # ---------------------------------------------------------------
    # One-hot encode (drop_first=True to avoid multicollinearity)
    # ---------------------------------------------------------------
    ONE_HOT_COLS = [
        "client_type",       # 2 values: Individual, Company
        "country",           # 10 values — safe to one-hot (no 'Other' grouping needed)
        "referral_channel",  # 3 values: Website, Agency, Client
        "unit_category",     # 2 values: Apartment, Office (dominant per client)
    ]

    # ---------------------------------------------------------------
    # Binary label encoding
    # ---------------------------------------------------------------
    LABEL_ENCODE_MAP = {
        "gender":               {"M": 1, "F": 0},
        "loan_applied":         {"Yes": 1, "No": 0},
        "acquisition_purpose":  {"Investment": 1, "Home": 0},
    }

    # ---------------------------------------------------------------
    # StandardScaler columns (z-score, mean=0, std=1)
    # ---------------------------------------------------------------
    STANDARD_SCALE_COLS = [
        "buyer_age",           # range 25-90 (after cap)
        "satisfaction_score",  # range 1-5, ~400 per value
        "total_spend",         # range $463K-$3.65M
        "transaction_count",   # range 3-13
    ]

    # ---------------------------------------------------------------
    # MinMaxScaler columns (scale to [0, 1])
    # ---------------------------------------------------------------
    MINMAX_SCALE_COLS = [
        "avg_floor_area",      # derived feature
        "avg_price_per_sqft",  # derived feature
        "dominant_month",      # 1-12
    ]

    def __init__(self):
        self._standard_scaler = StandardScaler()
        self._minmax_scaler = MinMaxScaler()
        self._feature_names: list[str] = []

    # ---------------------------------------------------------------
    # Main transformation method
    # ---------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """
        Apply all encoding and scaling transformations.

        Returns
        -------
        (X_scaled_array, feature_names_list, scaler_dict)
          X_scaled_array  : numpy array, shape (2000, n_features)
          feature_names   : list of column names matching array columns
          scaler_dict     : {'standard': StandardScaler, 'minmax': MinMaxScaler}

        Saves
        -----
        outputs/models/standard_scaler.joblib
        outputs/models/minmax_scaler.joblib
        """
        df = df.copy()
        log.info("EncoderScaler.fit_transform() — input shape: %s", df.shape)

        # ---- 1. Drop irrelevant / id columns --------------------------------
        cols_to_drop = [c for c in self.COLUMNS_TO_DROP if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        log.info("Dropped columns: %s", cols_to_drop)

        # ---- 2. Label encode binary categoricals ----------------------------
        for col, mapping in self.LABEL_ENCODE_MAP.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                log.info("Label encoded '%s': %s", col, mapping)

        # ---- 3. Frequency encode region (57 unique values) ------------------
        if "region" in df.columns:
            region_freq = df["region"].value_counts(normalize=True)  # proportion
            df["region_freq"] = df["region"].map(region_freq)
            df = df.drop(columns=["region"])
            log.info(
                "Frequency encoded 'region' -> 'region_freq' (%d unique values)",
                len(region_freq),
            )

        # ---- 4. One-hot encode low-cardinality categoricals -----------------
        ohe_cols_present = [c for c in self.ONE_HOT_COLS if c in df.columns]
        if ohe_cols_present:
            df = pd.get_dummies(df, columns=ohe_cols_present, drop_first=True)
            log.info("One-hot encoded: %s", ohe_cols_present)

        # ---- 5. StandardScaler on selected numeric columns ------------------
        std_cols_present = [c for c in self.STANDARD_SCALE_COLS if c in df.columns]
        if std_cols_present:
            df[std_cols_present] = self._standard_scaler.fit_transform(df[std_cols_present])
            log.info("StandardScaler applied to: %s", std_cols_present)

        # ---- 6. MinMaxScaler on remaining numeric columns -------------------
        mm_cols_present = [c for c in self.MINMAX_SCALE_COLS if c in df.columns]
        if mm_cols_present:
            df[mm_cols_present] = self._minmax_scaler.fit_transform(df[mm_cols_present])
            log.info("MinMaxScaler applied to: %s", mm_cols_present)

        # ---- 7. Store feature names and convert to numpy array -------------
        # Drop any residual object columns that weren't handled above
        obj_cols = df.select_dtypes(include="object").columns.tolist()
        if obj_cols:
            log.warning("Dropping remaining object columns (unhandled): %s", obj_cols)
            df = df.drop(columns=obj_cols)

        # Ensure all boolean dummies become int (sklearn requirement)
        bool_cols = df.select_dtypes(include="bool").columns.tolist()
        if bool_cols:
            df[bool_cols] = df[bool_cols].astype(int)

        self._feature_names = df.columns.tolist()
        X = df.values.astype(np.float64)
        log.info(
            "Feature matrix shape: %s | %d features", X.shape, len(self._feature_names)
        )

        # ---- 8. Save fitted scalers ----------------------------------------
        os.makedirs("outputs/models", exist_ok=True)
        joblib.dump(self._standard_scaler, "outputs/models/standard_scaler.joblib")
        joblib.dump(self._minmax_scaler, "outputs/models/minmax_scaler.joblib")
        log.info("Saved scalers -> outputs/models/")

        scaler_dict = {
            "standard": self._standard_scaler,
            "minmax": self._minmax_scaler,
        }
        return X, self._feature_names, scaler_dict

    def get_feature_names(self) -> list[str]:
        """Return list of all feature names after encoding."""
        return self._feature_names

    def save_feature_matrix(
        self,
        X: np.ndarray,
        feature_names: list,
        path: str = "data/processed/X_scaled.csv",
    ) -> None:
        """Save scaled feature matrix as CSV with column headers."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        df_out = pd.DataFrame(X, columns=feature_names)
        df_out.to_csv(path, index=False)
        log.info("Saved feature matrix -> %s  shape=%s", path, df_out.shape)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_cleaning import DataCleaner
    from src.feature_engineering import FeatureEngineer

    cleaner = DataCleaner()
    df_c, df_p = cleaner.load_data("data/raw/clients.csv", "data/raw/properties.csv")
    df_c = cleaner.clean_clients(df_c)
    df_p = cleaner.clean_properties(df_p)

    engineer = FeatureEngineer()
    df_eng = engineer.engineer_all(df_c, df_p)

    enc = EncoderScaler()
    X, feature_names, scalers = enc.fit_transform(df_eng)
    enc.save_feature_matrix(X, feature_names)

    log.info("encoding_scaling.py standalone run complete.")
