"""
Performance Metrics Analyzer
analytics/performance_metrics.py

Computes efficiency ratios, sales contribution, relative strength,
and local hero identification from BRANCH_PERFORMANCE_SUMMARY1 data.

Schema reference:
    REGION, BRANCHNAME, ITEMID, PURITY, FINISH, THEME, SHAPE,
    WORKSTYLE, BRAND, SALE_COUNT, STOCK_COUNT
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Branch-level and attribute-level performance metrics calculator.

    All column references use the real schema:
        BRANCHNAME  (not BRANCH)
        ITEMID      (not ITEMTYPE)
    """

    # Attribute columns available for grouping
    ATTRIBUTE_COLS = ['PURITY', 'FINISH', 'THEME', 'SHAPE', 'WORKSTYLE', 'BRAND']

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: Preprocessed DataFrame from JewelryDataLoader.
        """
        self.df           = df.copy()
        self.metrics_df   = None
        self.branch_summary = None
        self.local_heroes = None

    # ─── Core metric calculation ──────────────────────────────────────────────

    def calculate_all_metrics(self) -> pd.DataFrame:
        """
        Calculate all row-level performance metrics.

        Adds columns:
            efficiency_ratio        — SALE_COUNT / STOCK_COUNT
            sell_through_rate       — SALE_COUNT / (SALE_COUNT + STOCK_COUNT)
            sales_contribution_pct  — % of branch's total sales
            stock_concentration_pct — % of branch's total stock
            relative_strength       — efficiency vs global average for that ITEMID
            stock_to_sales_ratio    — STOCK_COUNT / SALE_COUNT (dead-stock signal)
            performance_category    — Low / Medium / High / Excellent

        Returns:
            DataFrame with all metrics added.
        """
        logger.info("Calculating performance metrics...")

        df = self.df.copy()

        # 1. Efficiency ratio
        logger.info("  Calculating efficiency ratios...")
        df['efficiency_ratio'] = np.where(
            df['STOCK_COUNT'] > 0,
            df['SALE_COUNT'] / df['STOCK_COUNT'],
            np.nan
        )

        # 2. Sell-through rate  (more intuitive: 0–1 range)
        df['sell_through_rate'] = np.where(
            (df['SALE_COUNT'] + df['STOCK_COUNT']) > 0,
            df['SALE_COUNT'] / (df['SALE_COUNT'] + df['STOCK_COUNT']),
            np.nan
        )

        # 3. Sales contribution % within each branch
        logger.info("  Calculating sales contribution...")
        branch_total_sales = df.groupby('BRANCHNAME')['SALE_COUNT'].transform('sum')
        df['sales_contribution_pct'] = np.where(
            branch_total_sales > 0,
            (df['SALE_COUNT'] / branch_total_sales) * 100,
            0.0
        )

        # 4. Stock concentration % within each branch
        branch_total_stock = df.groupby('BRANCHNAME')['STOCK_COUNT'].transform('sum')
        df['stock_concentration_pct'] = np.where(
            branch_total_stock > 0,
            (df['STOCK_COUNT'] / branch_total_stock) * 100,
            0.0
        )

        # 5. Relative strength — efficiency vs global average for same ITEMID
        logger.info("  Calculating relative strength...")
        global_avg_efficiency = df.groupby('ITEMID')['efficiency_ratio'].transform('mean')
        df['relative_strength'] = np.where(
            global_avg_efficiency > 0,
            df['efficiency_ratio'] / global_avg_efficiency,
            np.nan
        )

        # 6. Stock-to-sales ratio (dead-stock signal; inf = zero sales)
        df['stock_to_sales_ratio'] = np.where(
            df['SALE_COUNT'] > 0,
            df['STOCK_COUNT'] / df['SALE_COUNT'],
            np.inf
        )

        # 7. Sales rank within branch
        df['sales_rank_in_branch'] = df.groupby('BRANCHNAME')['SALE_COUNT'].rank(
            ascending=False, method='dense'
        )

        # 8. Performance category
        df['performance_category'] = pd.cut(
            df['efficiency_ratio'],
            bins=[-np.inf, 0.02, 0.05, 0.1, np.inf],
            labels=['Low', 'Medium', 'High', 'Excellent']
        )

        logger.info(f"✅ Metrics calculated for {len(df):,} records")

        self.metrics_df = df
        return df

    # ─── Aggregations ─────────────────────────────────────────────────────────

    def aggregate_by_branch(self) -> pd.DataFrame:
        """
        Aggregate metrics at branch level.

        Returns:
            DataFrame indexed by BRANCHNAME with summary columns.
        """
        if self.metrics_df is None:
            self.calculate_all_metrics()

        logger.info("Aggregating by branch...")

        branch_agg = self.metrics_df.groupby('BRANCHNAME').agg(
            REGION         = ('REGION',           'first'),
            SALE_COUNT     = ('SALE_COUNT',        'sum'),
            STOCK_COUNT    = ('STOCK_COUNT',       'sum'),
            avg_efficiency = ('efficiency_ratio',  'mean'),
            avg_sell_through = ('sell_through_rate', 'mean'),
            product_count  = ('ITEMID',            'nunique'),
        ).reset_index()

        # Branch-level sell-through (more accurate than avg of row-level)
        branch_agg['branch_sell_through'] = np.where(
            (branch_agg['SALE_COUNT'] + branch_agg['STOCK_COUNT']) > 0,
            branch_agg['SALE_COUNT'] / (branch_agg['SALE_COUNT'] + branch_agg['STOCK_COUNT']),
            0.0
        )

        # Rankings
        branch_agg['sales_rank']       = branch_agg['SALE_COUNT'].rank(ascending=False, method='dense').astype(int)
        branch_agg['efficiency_rank']  = branch_agg['branch_sell_through'].rank(ascending=False, method='dense').astype(int)

        logger.info(f"✅ Branch aggregation complete: {len(branch_agg)} branches")

        self.branch_summary = branch_agg
        return branch_agg

    def aggregate_by_attribute(self,
                                attribute_cols: List[str] = None
                                ) -> Dict[str, pd.DataFrame]:
        """
        Aggregate metrics by each product attribute per branch.

        Args:
            attribute_cols: Columns to aggregate by.
                            Defaults to ATTRIBUTE_COLS class constant.

        Returns:
            Dict mapping attribute name → aggregated DataFrame.
        """
        if self.metrics_df is None:
            self.calculate_all_metrics()

        if attribute_cols is None:
            attribute_cols = self.ATTRIBUTE_COLS

        logger.info(f"Aggregating by attributes: {attribute_cols}")

        results = {}

        for attr in attribute_cols:
            if attr not in self.metrics_df.columns:
                logger.warning(f"  Column '{attr}' not found — skipping")
                continue

            logger.info(f"  Aggregating by {attr}...")

            agg = self.metrics_df.groupby(['BRANCHNAME', attr]).agg(
                SALE_COUNT           = ('SALE_COUNT',           'sum'),
                STOCK_COUNT          = ('STOCK_COUNT',          'sum'),
                avg_efficiency       = ('efficiency_ratio',     'mean'),
                avg_sell_through     = ('sell_through_rate',    'mean'),
                sales_contribution   = ('sales_contribution_pct', 'sum'),
                avg_relative_strength= ('relative_strength',    'mean'),
                item_count           = ('ITEMID',               'nunique'),
            ).reset_index()

            # Recompute sell-through at aggregated level
            agg['sell_through'] = np.where(
                (agg['SALE_COUNT'] + agg['STOCK_COUNT']) > 0,
                agg['SALE_COUNT'] / (agg['SALE_COUNT'] + agg['STOCK_COUNT']),
                0.0
            )

            results[attr] = agg

        logger.info("✅ Attribute aggregation complete")
        return results

    # ─── Hero / underperformer identification ─────────────────────────────────

    def identify_local_heroes(self,
                               relative_strength_threshold: float = 1.2,
                               min_contribution_pct: float = 1.0,
                               min_sales: int = 1) -> pd.DataFrame:
        """
        Identify products that outperform globally within their branch.

        Args:
            relative_strength_threshold: Minimum relative strength score.
            min_contribution_pct: Minimum sales contribution % in branch.
            min_sales: Minimum absolute sale count.

        Returns:
            DataFrame of local hero records, sorted by relative_strength desc.
        """
        if self.metrics_df is None:
            self.calculate_all_metrics()

        logger.info("Identifying local heroes...")

        heroes = self.metrics_df[
            (self.metrics_df['relative_strength']      > relative_strength_threshold) &
            (self.metrics_df['sales_contribution_pct'] > min_contribution_pct) &
            (self.metrics_df['SALE_COUNT']             >= min_sales)
        ].copy()

        heroes = heroes.sort_values('relative_strength', ascending=False)
        heroes['hero_rank'] = range(1, len(heroes) + 1)

        logger.info(f"✅ Found {len(heroes)} local heroes")

        self.local_heroes = heroes
        return heroes

    def identify_underperformers(self,
                                  efficiency_threshold: float = 0.02,
                                  high_stock_threshold: float = None) -> pd.DataFrame:
        """
        Identify high-stock, low-sales records (dead stock candidates).

        Args:
            efficiency_threshold: Max efficiency_ratio to qualify.
            high_stock_threshold: Min STOCK_COUNT (default: median).

        Returns:
            DataFrame of underperformers sorted by efficiency_ratio asc.
        """
        if self.metrics_df is None:
            self.calculate_all_metrics()

        if high_stock_threshold is None:
            high_stock_threshold = self.metrics_df['STOCK_COUNT'].median()

        logger.info("Identifying underperformers...")

        under = self.metrics_df[
            (self.metrics_df['efficiency_ratio'] < efficiency_threshold) &
            (self.metrics_df['STOCK_COUNT']      > high_stock_threshold)
        ].copy().sort_values('efficiency_ratio')

        logger.info(f"✅ Found {len(under)} underperformers")
        return under

    # ─── Comparison & ranking ─────────────────────────────────────────────────

    def compare_branches(self, branches: List[str] = None) -> pd.DataFrame:
        """
        Return ranked branch comparison table.

        Args:
            branches: Subset of branch names (None = all).

        Returns:
            Sorted DataFrame with sales and efficiency ranks.
        """
        if self.branch_summary is None:
            self.aggregate_by_branch()

        comparison = self.branch_summary.copy()

        if branches:
            comparison = comparison[comparison['BRANCHNAME'].isin(branches)]

        comparison['sales_percentile']      = comparison['SALE_COUNT'].rank(pct=True) * 100
        comparison['efficiency_percentile'] = comparison['branch_sell_through'].rank(pct=True) * 100

        return comparison.sort_values('sales_rank')

    def get_top_performers(self, by: str = 'SALE_COUNT', n: int = 10) -> pd.DataFrame:
        """
        Return top N records by a given metric.

        Args:
            by: Column to rank by.
            n:  Number of records.

        Returns:
            Top-N DataFrame.
        """
        if self.metrics_df is None:
            self.calculate_all_metrics()

        return self.metrics_df.nlargest(n, by)

    # ─── Summary stats ────────────────────────────────────────────────────────

    def get_summary_stats(self) -> Dict:
        """
        Return high-level summary statistics dictionary.
        """
        if self.metrics_df is None:
            self.calculate_all_metrics()

        total_sales = int(self.metrics_df['SALE_COUNT'].sum())
        total_stock = int(self.metrics_df['STOCK_COUNT'].sum())

        return {
            'total_branches':        self.metrics_df['BRANCHNAME'].nunique(),
            'total_sales':           total_sales,
            'total_stock':           total_stock,
            'overall_sell_through':  round(total_sales / (total_sales + total_stock), 4) if (total_sales + total_stock) > 0 else 0,
            'overall_efficiency':    round(total_sales / total_stock, 4) if total_stock > 0 else 0,
            'avg_efficiency_ratio':  round(float(self.metrics_df['efficiency_ratio'].mean()), 4),
            'total_local_heroes':    len(self.local_heroes) if self.local_heroes is not None
                                     else len(self.identify_local_heroes()),
            'attributes': {
                'item_ids':   self.metrics_df['ITEMID'].nunique(),
                'purities':   self.metrics_df['PURITY'].nunique(),
                'finishes':   self.metrics_df['FINISH'].nunique(),
                'themes':     self.metrics_df['THEME'].nunique(),
                'shapes':     self.metrics_df['SHAPE'].nunique(),
                'workstyles': self.metrics_df['WORKSTYLE'].nunique(),
                'brands':     self.metrics_df['BRAND'].nunique(),
            }
        }


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Performance Analyzer Test ===\n")

    rng = np.random.default_rng(42)
    n   = 120

    sample = pd.DataFrame({
        'BRANCHNAME': rng.choice(['BR-CHENNAI-01', 'BR-MUMBAI-01', 'BR-DELHI-01'], n),
        'REGION':     rng.choice(['South', 'West', 'North'], n),
        'ITEMID':     [f'PRD{i:05d}' for i in rng.integers(1, 40, n)],
        'PURITY':     rng.choice(['14.0', '18.0'], n),
        'FINISH':     rng.choice(['Textured', 'Polished', 'Matte', 'Hammered'], n),
        'THEME':      rng.choice(['Minimalist', 'Floral', 'Geometric', 'Classic', 'Boho'], n),
        'SHAPE':      rng.choice(['Round', 'Square', 'Oval', 'Mixed'], n),
        'WORKSTYLE':  rng.choice(['Filigree', 'Plain', 'Studded', 'Engraved'], n),
        'BRAND':      rng.choice(['Signature', 'Heritage', 'Luxe'], n),
        'SALE_COUNT': rng.integers(0, 30, n),
        'STOCK_COUNT':rng.integers(5, 60, n),
    })

    analyzer = PerformanceAnalyzer(sample)

    print("Calculating metrics...")
    metrics = analyzer.calculate_all_metrics()
    new_cols = [c for c in metrics.columns if c not in sample.columns]
    print(f"✅ {len(metrics)} records, new columns: {new_cols}\n")

    print("Branch summary:")
    print(analyzer.aggregate_by_branch()[['BRANCHNAME', 'SALE_COUNT', 'STOCK_COUNT', 'branch_sell_through', 'sales_rank']].to_string(index=False))

    print(f"\nLocal heroes: {len(analyzer.identify_local_heroes())}")
    print(f"Underperformers: {len(analyzer.identify_underperformers())}")

    print("\nSummary stats:")
    for k, v in analyzer.get_summary_stats().items():
        print(f"  {k}: {v}")

    print("\n=== Test complete ===")
