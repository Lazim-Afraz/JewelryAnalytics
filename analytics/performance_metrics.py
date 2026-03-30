"""
Performance Metrics Analyzer
analytics/performance_metrics.py

Computes efficiency ratios, sales contribution, relative strength,
and local hero identification from BRANCH_PERFORMANCE_SUMMARY1 data.

Schema reference:
    REGION, BRANCHNAME, ITEMID, PURITY, FINISH, THEME, SHAPE,
    WORKSTYLE, BRAND, SALE_COUNT, STOCK_COUNT

EXTENDED (v2):
    + get_top_attributes_per_branch()   — ranks each attribute value per branch
    + get_high_performing_combos()      — best multi-attribute combos per branch
    + get_product_performance_by_region() — regional attribute breakdown
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
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
        self.df             = df.copy()
        self.metrics_df     = None
        self.branch_summary = None
        self.local_heroes   = None

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
            REGION           = ('REGION',            'first'),
            SALE_COUNT       = ('SALE_COUNT',         'sum'),
            STOCK_COUNT      = ('STOCK_COUNT',        'sum'),
            avg_efficiency   = ('efficiency_ratio',   'mean'),
            avg_sell_through = ('sell_through_rate',  'mean'),
            product_count    = ('ITEMID',             'nunique'),
        ).reset_index()

        # Branch-level sell-through (more accurate than avg of row-level)
        branch_agg['branch_sell_through'] = np.where(
            (branch_agg['SALE_COUNT'] + branch_agg['STOCK_COUNT']) > 0,
            branch_agg['SALE_COUNT'] / (branch_agg['SALE_COUNT'] + branch_agg['STOCK_COUNT']),
            0.0
        )

        # Rankings
        branch_agg['sales_rank']      = branch_agg['SALE_COUNT'].rank(ascending=False, method='dense').astype(int)
        branch_agg['efficiency_rank'] = branch_agg['branch_sell_through'].rank(ascending=False, method='dense').astype(int)

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
                SALE_COUNT            = ('SALE_COUNT',            'sum'),
                STOCK_COUNT           = ('STOCK_COUNT',           'sum'),
                avg_efficiency        = ('efficiency_ratio',      'mean'),
                avg_sell_through      = ('sell_through_rate',     'mean'),
                sales_contribution    = ('sales_contribution_pct','sum'),
                avg_relative_strength = ('relative_strength',     'mean'),
                item_count            = ('ITEMID',                'nunique'),
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
            'total_branches':       self.metrics_df['BRANCHNAME'].nunique(),
            'total_sales':          total_sales,
            'total_stock':          total_stock,
            'overall_sell_through': round(total_sales / (total_sales + total_stock), 4)
                                    if (total_sales + total_stock) > 0 else 0,
            'overall_efficiency':   round(total_sales / total_stock, 4)
                                    if total_stock > 0 else 0,
            'avg_efficiency_ratio': round(float(self.metrics_df['efficiency_ratio'].mean()), 4),
            'total_local_heroes':   len(self.local_heroes) if self.local_heroes is not None
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

    # =========================================================================
    # EXTENDED — Attribute-level intelligence (v2 additions)
    # =========================================================================

    def get_top_attributes_per_branch(self,
                                       attribute: str,
                                       metric: str = 'SALE_COUNT',
                                       top_n: int = 3,
                                       branch: Optional[str] = None
                                       ) -> pd.DataFrame:
        """
        Rank attribute values (e.g. PURITY='18K', FINISH='Polished') by a
        chosen metric within each branch.

        Args:
            attribute : One of ATTRIBUTE_COLS — e.g. 'PURITY', 'FINISH', 'THEME'.
            metric    : Aggregation target — 'SALE_COUNT', 'efficiency_ratio',
                        'sell_through_rate', or 'avg_relative_strength'.
            top_n     : How many top values to return per branch.
            branch    : Filter to a single branch (None = all branches).

        Returns:
            DataFrame with columns:
                BRANCHNAME, <attribute>, total_sales, avg_efficiency,
                sell_through, avg_relative_strength, rank_in_branch
        """
        if self.metrics_df is None:
            self.calculate_all_metrics()

        if attribute not in self.metrics_df.columns:
            raise ValueError(f"Attribute '{attribute}' not in data. "
                             f"Choose from: {self.ATTRIBUTE_COLS}")

        logger.info(f"Computing top {attribute} per branch (metric={metric})...")

        df = self.metrics_df.copy()
        if branch:
            df = df[df['BRANCHNAME'] == branch]

        agg = df.groupby(['BRANCHNAME', attribute]).agg(
            total_sales           = ('SALE_COUNT',          'sum'),
            total_stock           = ('STOCK_COUNT',         'sum'),
            avg_efficiency        = ('efficiency_ratio',    'mean'),
            avg_relative_strength = ('relative_strength',   'mean'),
            item_count            = ('ITEMID',              'nunique'),
        ).reset_index()

        # Recompute sell-through at this aggregation level
        agg['sell_through'] = np.where(
            (agg['total_sales'] + agg['total_stock']) > 0,
            agg['total_sales'] / (agg['total_sales'] + agg['total_stock']),
            0.0
        )

        # Rank within each branch by chosen metric
        sort_col = {
            'SALE_COUNT':          'total_sales',
            'efficiency_ratio':    'avg_efficiency',
            'sell_through_rate':   'sell_through',
            'avg_relative_strength': 'avg_relative_strength',
        }.get(metric, 'total_sales')

        agg['rank_in_branch'] = agg.groupby('BRANCHNAME')[sort_col].rank(
            ascending=False, method='dense'
        ).astype(int)

        result = agg[agg['rank_in_branch'] <= top_n].sort_values(
            ['BRANCHNAME', 'rank_in_branch']
        ).reset_index(drop=True)

        logger.info(f"✅ Top {attribute} complete: {len(result)} rows")
        return result

    def get_high_performing_combos(self,
                                    combo_cols: List[str] = None,
                                    metric: str = 'sell_through',
                                    top_n: int = 5,
                                    min_sales: int = 1
                                    ) -> pd.DataFrame:
        """
        Find the best multi-attribute combinations per branch.

        Groups rows by BRANCHNAME + combo_cols and ranks by the chosen metric.
        Useful for spotting which PURITY × FINISH × THEME bundles drive sales.

        Args:
            combo_cols : Attribute columns to combine.
                         Default: ['PURITY', 'FINISH', 'THEME'].
            metric     : Ranking metric — 'sell_through', 'total_sales',
                         'avg_efficiency', 'avg_relative_strength'.
            top_n      : Top N combos to return per branch.
            min_sales  : Minimum total_sales to include a combo.

        Returns:
            DataFrame with one row per (branch, combo) combination,
            ranked within each branch.
        """
        if self.metrics_df is None:
            self.calculate_all_metrics()

        if combo_cols is None:
            combo_cols = ['PURITY', 'FINISH', 'THEME']

        # Keep only columns that exist
        combo_cols = [c for c in combo_cols if c in self.metrics_df.columns]
        if not combo_cols:
            logger.warning("No valid combo columns found — returning empty DataFrame")
            return pd.DataFrame()

        logger.info(f"Computing high-performing combos: {combo_cols} by {metric}...")

        group_keys = ['BRANCHNAME'] + combo_cols

        agg = self.metrics_df.groupby(group_keys).agg(
            total_sales           = ('SALE_COUNT',         'sum'),
            total_stock           = ('STOCK_COUNT',        'sum'),
            avg_efficiency        = ('efficiency_ratio',   'mean'),
            avg_relative_strength = ('relative_strength',  'mean'),
            item_count            = ('ITEMID',             'nunique'),
        ).reset_index()

        # Sell-through at combo level
        agg['sell_through'] = np.where(
            (agg['total_sales'] + agg['total_stock']) > 0,
            agg['total_sales'] / (agg['total_sales'] + agg['total_stock']),
            0.0
        )

        # Filter minimum sales
        agg = agg[agg['total_sales'] >= min_sales]

        # Rank within branch
        valid_metrics = ['sell_through', 'total_sales', 'avg_efficiency', 'avg_relative_strength']
        sort_col = metric if metric in valid_metrics else 'sell_through'

        agg['rank_in_branch'] = agg.groupby('BRANCHNAME')[sort_col].rank(
            ascending=False, method='dense'
        ).astype(int)

        result = agg[agg['rank_in_branch'] <= top_n].sort_values(
            ['BRANCHNAME', 'rank_in_branch']
        ).reset_index(drop=True)

        # Build a human-readable combo label
        result['combo_label'] = result[combo_cols].apply(
            lambda row: ' | '.join(str(v) for v in row), axis=1
        )

        logger.info(f"✅ High-performing combos: {len(result)} rows")
        return result

    def get_product_performance_by_region(self,
                                           attribute: str = 'THEME',
                                           metric: str = 'total_sales'
                                           ) -> pd.DataFrame:
        """
        Aggregate product attribute performance at the REGION level.

        Useful for identifying which themes/purities/finishes perform best
        in each geographic region — feeds directly into the service layer
        get_product_performance_by_region() call.

        Args:
            attribute : Attribute to group by (default 'THEME').
            metric    : Sort metric — 'total_sales', 'sell_through',
                        'avg_efficiency', 'avg_relative_strength'.

        Returns:
            DataFrame with columns:
                REGION, <attribute>, total_sales, total_stock,
                avg_efficiency, sell_through, avg_relative_strength,
                branch_count, rank_in_region
        """
        if self.metrics_df is None:
            self.calculate_all_metrics()

        if attribute not in self.metrics_df.columns:
            raise ValueError(f"Attribute '{attribute}' not found. "
                             f"Choose from: {self.ATTRIBUTE_COLS}")

        logger.info(f"Computing regional performance by {attribute}...")

        agg = self.metrics_df.groupby(['REGION', attribute]).agg(
            total_sales           = ('SALE_COUNT',         'sum'),
            total_stock           = ('STOCK_COUNT',        'sum'),
            avg_efficiency        = ('efficiency_ratio',   'mean'),
            avg_relative_strength = ('relative_strength',  'mean'),
            branch_count          = ('BRANCHNAME',         'nunique'),
            item_count            = ('ITEMID',             'nunique'),
        ).reset_index()

        agg['sell_through'] = np.where(
            (agg['total_sales'] + agg['total_stock']) > 0,
            agg['total_sales'] / (agg['total_sales'] + agg['total_stock']),
            0.0
        )

        valid_metrics = ['total_sales', 'sell_through', 'avg_efficiency', 'avg_relative_strength']
        sort_col = metric if metric in valid_metrics else 'total_sales'

        agg['rank_in_region'] = agg.groupby('REGION')[sort_col].rank(
            ascending=False, method='dense'
        ).astype(int)

        result = agg.sort_values(['REGION', 'rank_in_region']).reset_index(drop=True)

        logger.info(f"✅ Regional performance by {attribute}: {len(result)} rows")
        return result

    def get_recommendations(self,
                             branch: Optional[str] = None,
                             region: Optional[str] = None,
                             top_n: int = 5
                             ) -> Dict:
        """
        Generate attribute-based recommendations for a branch or region.

        Returns a structured dict ready for the service layer and UI.

        Args:
            branch : BRANCHNAME to scope recommendations (optional).
            region : REGION to scope recommendations (optional).
            top_n  : Number of top items per attribute to return.

        Returns:
            Dict with keys:
                scope        — 'branch' | 'region' | 'global'
                scope_value  — the branch/region name
                by_purity    — top PURITY values
                by_finish    — top FINISH values
                by_theme     — top THEME values
                top_combos   — best PURITY × FINISH × THEME combos
                summary      — human-readable summary string
        """
        if self.metrics_df is None:
            self.calculate_all_metrics()

        df = self.metrics_df.copy()

        # Determine scope
        if branch:
            df = df[df['BRANCHNAME'] == branch]
            scope = 'branch'
            scope_value = branch
        elif region:
            df = df[df['REGION'] == region]
            scope = 'region'
            scope_value = region
        else:
            scope = 'global'
            scope_value = 'All'

        if df.empty:
            return {
                'scope': scope,
                'scope_value': scope_value,
                'error': f"No data found for {scope}='{scope_value}'"
            }

        def _top_attr(attribute, n):
            if attribute not in df.columns:
                return []
            agg = df.groupby(attribute).agg(
                total_sales  = ('SALE_COUNT',       'sum'),
                sell_through = ('sell_through_rate', 'mean'),
            ).reset_index()
            agg['sell_through'] = agg['sell_through'].round(4)
            return agg.nlargest(n, 'total_sales')[[attribute, 'total_sales', 'sell_through']].to_dict('records')

        # Top combos
        combo_df = self.get_high_performing_combos(
            combo_cols=['PURITY', 'FINISH', 'THEME'],
            metric='sell_through',
            top_n=top_n,
            min_sales=1
        )
        if branch:
            combo_df = combo_df[combo_df['BRANCHNAME'] == branch]
        elif region and 'REGION' in self.metrics_df.columns:
            branches_in_region = self.metrics_df[
                self.metrics_df['REGION'] == region
            ]['BRANCHNAME'].unique()
            combo_df = combo_df[combo_df['BRANCHNAME'].isin(branches_in_region)]

        top_combos = combo_df.head(top_n)[
            ['combo_label', 'total_sales', 'sell_through', 'avg_efficiency']
        ].to_dict('records') if not combo_df.empty else []

        recommendations = {
            'scope':       scope,
            'scope_value': scope_value,
            'by_purity':   _top_attr('PURITY', top_n),
            'by_finish':   _top_attr('FINISH', top_n),
            'by_theme':    _top_attr('THEME',  top_n),
            'by_shape':    _top_attr('SHAPE',  top_n),
            'top_combos':  top_combos,
            'summary': (
                f"For {scope} '{scope_value}': "
                f"Top purity is {_top_attr('PURITY', 1)[0].get('PURITY', 'N/A') if _top_attr('PURITY', 1) else 'N/A'}, "
                f"top theme is {_top_attr('THEME', 1)[0].get('THEME', 'N/A') if _top_attr('THEME', 1) else 'N/A'}, "
                f"top finish is {_top_attr('FINISH', 1)[0].get('FINISH', 'N/A') if _top_attr('FINISH', 1) else 'N/A'}."
            )
        }

        logger.info(f"✅ Recommendations generated for {scope}='{scope_value}'")
        return recommendations


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Performance Analyzer Test (v2 — with attribute intelligence) ===\n")

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

    # ── Original tests (unchanged) ────────────────────────────────────────────
    print("1. Calculating metrics...")
    metrics = analyzer.calculate_all_metrics()
    new_cols = [c for c in metrics.columns if c not in sample.columns]
    print(f"   ✅ {len(metrics)} records, new columns: {new_cols}\n")

    print("2. Branch summary:")
    print(analyzer.aggregate_by_branch()[
        ['BRANCHNAME', 'SALE_COUNT', 'STOCK_COUNT', 'branch_sell_through', 'sales_rank']
    ].to_string(index=False))

    print(f"\n3. Local heroes: {len(analyzer.identify_local_heroes())}")
    print(f"   Underperformers: {len(analyzer.identify_underperformers())}")

    print("\n4. Summary stats:")
    for k, v in analyzer.get_summary_stats().items():
        print(f"   {k}: {v}")

    # ── New v2 tests ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("V2 ATTRIBUTE INTELLIGENCE TESTS")
    print("="*60)

    print("\n5. Top PURITY per branch (by SALE_COUNT):")
    print(analyzer.get_top_attributes_per_branch('PURITY', metric='SALE_COUNT', top_n=2).to_string(index=False))

    print("\n6. Top THEME per branch (by sell_through_rate):")
    print(analyzer.get_top_attributes_per_branch('THEME', metric='sell_through_rate', top_n=2).to_string(index=False))

    print("\n7. High-performing combos (PURITY × FINISH × THEME):")
    combos = analyzer.get_high_performing_combos(top_n=3)
    print(combos[['BRANCHNAME', 'combo_label', 'total_sales', 'sell_through', 'rank_in_branch']].to_string(index=False))

    print("\n8. Product performance by REGION (attribute=THEME):")
    print(analyzer.get_product_performance_by_region(attribute='THEME').head(9).to_string(index=False))

    print("\n9. Recommendations for branch 'BR-MUMBAI-01':")
    recs = analyzer.get_recommendations(branch='BR-MUMBAI-01', top_n=3)
    print(f"   Scope     : {recs['scope']} — {recs['scope_value']}")
    print(f"   Summary   : {recs['summary']}")
    print(f"   By purity : {recs['by_purity']}")
    print(f"   Top combos: {recs['top_combos']}")

    print("\n10. Recommendations for region 'South':")
    recs_r = analyzer.get_recommendations(region='South', top_n=3)
    print(f"   Summary: {recs_r['summary']}")

    print("\n=== All tests passed ✅ ===")
