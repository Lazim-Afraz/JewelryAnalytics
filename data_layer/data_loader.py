"""
Data Loader Module
data_layer/data_loader.py

Loads and preprocesses jewelry portfolio data from JewelryPortfolioDB.
All column names match the BRANCH_PERFORMANCE_SUMMARY1 table schema.

NOTE: BRANCH_PERFORMANCE_SUMMARY1 is a snapshot table with no time dimension.
      Date filtering is not supported and has been intentionally removed.
      For time-series analysis, the table schema must be extended with
      snapshot_date / invoice_date columns first.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging

from data_layer.sql_connector import SQLServerConnector

logger = logging.getLogger(__name__)


class JewelryDataLoader:
    """
    Data loader for JewelryPortfolioDB.

    Reads from BRANCH_PERFORMANCE_SUMMARY1 and provides
    preprocessing, validation, and summary helpers.

    This is a snapshot loader — BRANCH_PERFORMANCE_SUMMARY1 has no
    time dimension, so all rows are loaded for a given branch filter.
    """

    def __init__(self, connector: SQLServerConnector):
        """
        Args:
            connector: An active SQLServerConnector instance.
        """
        self.connector      = connector
        self.raw_data       = None
        self.processed_data = None

    # ─── Data loading ────────────────────────────────────────────────────────

    def load_transaction_data(self,
                               branches: Optional[List[str]] = None
                               ) -> pd.DataFrame:
        """
        Load snapshot data from BRANCH_PERFORMANCE_SUMMARY1.

        Note: This table has no date column. All records are returned
        unless filtered by branch name.

        Args:
            branches: List of BRANCHNAME values to include (None = all).

        Returns:
            Raw DataFrame from the database.
        """
        query = """
            SELECT
                REGION,
                BRANCHNAME,
                ITEMID,
                PURITY,
                FINISH,
                THEME,
                SHAPE,
                WORKSTYLE,
                BRAND,
                SALE_COUNT,
                STOCK_COUNT

            FROM BRANCH_PERFORMANCE_SUMMARY1

            WHERE 1=1
        """

        if branches:
            branch_list = "', '".join(branches)
            query += f" AND BRANCHNAME IN ('{branch_list}')"

        logger.info("Loading snapshot data...")
        logger.info(f"  Branches: {branches or 'ALL'}")

        try:
            df = self.connector.execute_query(query)

            logger.info(f"✅ Loaded {len(df):,} rows")
            logger.info(f"   Columns         : {', '.join(df.columns.tolist())}")
            logger.info(f"   Unique branches : {df['BRANCHNAME'].nunique()}")
            logger.info(f"   Unique regions  : {df['REGION'].nunique()}")

            self.raw_data = df
            return df

        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            raise

    def load_branch_master(self) -> pd.DataFrame:
        """
        Return distinct branch + region combinations.

        Returns:
            DataFrame with BRANCHNAME, REGION columns.
        """
        query = """
            SELECT DISTINCT
                BRANCHNAME,
                REGION
            FROM BRANCH_PERFORMANCE_SUMMARY1
            ORDER BY BRANCHNAME
        """
        return self.connector.execute_query(query)

    def load_product_master(self) -> pd.DataFrame:
        """
        Return distinct product attribute combinations.

        Returns:
            DataFrame with ITEMID, PURITY, FINISH, THEME, SHAPE columns.
        """
        query = """
            SELECT DISTINCT
                ITEMID,
                PURITY,
                FINISH,
                THEME,
                SHAPE
            FROM BRANCH_PERFORMANCE_SUMMARY1
            ORDER BY ITEMID, PURITY
        """
        return self.connector.execute_query(query)

    # ─── Preprocessing ───────────────────────────────────────────────────────

    def preprocess_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Clean and standardise a raw DataFrame from load_transaction_data().

        Steps:
            1. Fill nulls in numeric and categorical columns
            2. Cast SALE_COUNT / STOCK_COUNT to int
            3. Remove rows where both counts are zero
            4. Strip and upper-case all text columns

        Args:
            df: DataFrame to preprocess. Uses self.raw_data if not supplied.

        Returns:
            Cleaned DataFrame, also stored in self.processed_data.
        """
        if df is None:
            if self.raw_data is None:
                raise ValueError("No data loaded. Call load_transaction_data() first.")
            df = self.raw_data.copy()
        else:
            df = df.copy()

        logger.info("Preprocessing data...")

        # 1. Fill nulls
        logger.info("  Filling nulls...")
        numeric_defaults = {'SALE_COUNT': 0, 'STOCK_COUNT': 0}
        df.fillna(numeric_defaults, inplace=True)

        text_defaults = {
            'ITEMID':    'Unknown',
            'PURITY':    'Unknown',
            'FINISH':    'Unknown',
            'THEME':     'Unknown',
            'SHAPE':     'Unknown',
            'WORKSTYLE': 'Unknown',
            'BRAND':     'Unknown',
            'BRANCHNAME':'Unknown',
            'REGION':    'Unknown',
        }
        for col, val in text_defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        # 2. Type conversions
        logger.info("  Converting types...")
        df['SALE_COUNT']  = df['SALE_COUNT'].astype(int)
        df['STOCK_COUNT'] = df['STOCK_COUNT'].astype(int)

        # 3. Remove rows where both counts are zero (inactive records)
        logger.info("  Removing zero-activity rows...")
        initial = len(df)
        df = df[~((df['SALE_COUNT'] == 0) & (df['STOCK_COUNT'] == 0))]
        removed = initial - len(df)
        if removed:
            logger.info(f"  Removed {removed:,} zero-activity rows")

        # 4. Standardise text (strip whitespace, uppercase)
        # Cast to str first — some columns (e.g. PURITY) may arrive as float
        logger.info("  Standardising text fields...")
        text_cols = ['BRANCHNAME', 'REGION', 'ITEMID', 'PURITY',
                     'FINISH', 'THEME', 'SHAPE', 'WORKSTYLE', 'BRAND']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()

        logger.info(f"✅ Preprocessing complete: {len(df):,} rows")

        self.processed_data = df
        return df

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def get_data_summary(self, df: pd.DataFrame = None) -> Dict:
        """
        Return summary statistics for a loaded DataFrame.

        Args:
            df: DataFrame to summarise. Falls back to processed_data,
                then raw_data if not supplied.

        Returns:
            Dictionary of summary values.
        """
        if df is None:
            df = self.processed_data if self.processed_data is not None else self.raw_data
        if df is None:
            return {'error': 'No data loaded'}

        return {
            'total_rows':     len(df),
            'total_branches': df['BRANCHNAME'].nunique(),
            'total_regions':  df['REGION'].nunique(),
            'total_sales':    int(df['SALE_COUNT'].sum()),
            'total_stock':    int(df['STOCK_COUNT'].sum()),
            'attributes': {
                'item_ids':  df['ITEMID'].nunique(),
                'purities':  df['PURITY'].nunique(),
                'finishes':  df['FINISH'].nunique(),
                'themes':    df['THEME'].nunique(),
                'shapes':    df['SHAPE'].nunique(),
                'workstyles':df['WORKSTYLE'].nunique(),
                'brands':    df['BRAND'].nunique(),
            },
            'branches': sorted(df['BRANCHNAME'].unique().tolist()),
            'regions':  sorted(df['REGION'].unique().tolist()),
        }

    def validate_data(self, df: pd.DataFrame = None) -> Dict:
        """
        Run basic data-quality checks on a DataFrame.

        Args:
            df: DataFrame to validate. Falls back to processed_data,
                then raw_data if not supplied.

        Returns:
            Dictionary with keys: valid, total_rows, missing_columns,
            null_counts, duplicate_rows, issues.
        """
        if df is None:
            df = self.processed_data if self.processed_data is not None else self.raw_data
        if df is None:
            return {'valid': False, 'message': 'No data to validate'}

        required = ['BRANCHNAME', 'REGION', 'ITEMID', 'SALE_COUNT', 'STOCK_COUNT']
        missing  = [c for c in required if c not in df.columns]

        validation = {
            'valid':          len(missing) == 0,
            'total_rows':     len(df),
            'missing_columns':missing,
            'null_counts':    df.isnull().sum().to_dict(),
            'duplicate_rows': int(df.duplicated().sum()),
            'issues':         [],
        }

        if missing:
            validation['issues'].append(f"Missing required columns: {missing}")
        if df.isnull().any().any():
            null_cols = df.columns[df.isnull().any()].tolist()
            validation['issues'].append(f"Null values found in: {null_cols}")
        if validation['duplicate_rows'] > 0:
            validation['issues'].append(
                f"{validation['duplicate_rows']:,} duplicate rows found"
            )

        return validation


# ─── Convenience function ────────────────────────────────────────────────────

def load_jewelry_data(connector: SQLServerConnector,
                      branches:   Optional[List[str]] = None,
                      preprocess: bool = True) -> pd.DataFrame:
    """
    One-call loader: connect → load → (optionally) preprocess.

    Note: Date parameters have been removed. BRANCH_PERFORMANCE_SUMMARY1
    is a snapshot table with no time dimension.

    Args:
        connector:  Active SQLServerConnector instance.
        branches:   Optional list of branch names to filter (None = all).
        preprocess: If True, run preprocess_data() before returning.

    Returns:
        Loaded and optionally preprocessed DataFrame.
    """
    loader = JewelryDataLoader(connector)
    df     = loader.load_transaction_data(branches=branches)

    if preprocess:
        df = loader.preprocess_data(df)

    return df


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Data Loader Test ===\n")

    from config.database_config import DatabaseConfig
    from data_layer.sql_connector import SQLServerConnector

    connector = SQLServerConnector(
        server           = DatabaseConfig.SERVER,
        database         = DatabaseConfig.DATABASE,
        username         = DatabaseConfig.USERNAME,
        password         = DatabaseConfig.PASSWORD,
        use_windows_auth = DatabaseConfig.USE_WINDOWS_AUTH
    )

    if connector.connect():
        loader = JewelryDataLoader(connector)

        try:
            print("Loading data...")
            df = loader.load_transaction_data()
            print(f"\n✅ Rows loaded: {len(df):,}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"\nFirst 3 rows:\n{df.head(3)}")

            print("\nPreprocessing...")
            df_clean = loader.preprocess_data(df)

            print("\nSummary:")
            for k, v in loader.get_data_summary(df_clean).items():
                print(f"  {k}: {v}")

            print("\nValidation:")
            v = loader.validate_data(df_clean)
            print(f"  Valid : {v['valid']}")
            if v['issues']:
                print(f"  Issues: {v['issues']}")
            else:
                print("  No issues found.")

        except Exception as e:
            print(f"\n❌ Error: {e}")

        connector.close()

    print("\n=== Test complete ===")
