"""
Analytics Service Layer
services/analytics_service.py

Central orchestrator that wires together:
    SQLServerConnector  →  JewelryDataLoader  →  PerformanceAnalyzer
    →  BranchClusterer  →  SalesPredictionModel

All app.py pages call this service exclusively — no direct imports of
lower-level modules are needed in the UI layer.

Usage:
    from services.analytics_service import AnalyticsService
    svc = AnalyticsService()
    svc.load_data()
    dash = svc.get_dashboard_data()
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# ── Path fix: ensure project root is importable ───────────────────────────────
# Walk up from this file until we find app.py (= project root)
def _find_project_root() -> Path:
    candidate = Path(__file__).resolve().parent
    for _ in range(5):
        if (candidate / "app.py").exists():
            return candidate
        candidate = candidate.parent
    return Path(__file__).resolve().parent.parent   # fallback

_ROOT = _find_project_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Single entry-point for all analytics operations.

    Lifecycle:
        1. Instantiate  → __init__()
        2. Load data    → load_data()           (connects DB, runs pipeline)
        3. Query UI     → get_dashboard_data(), get_cluster_analysis(), …
        4. Predict      → predict_sales()
        5. Optional     → retrain_model()
    """

    def __init__(self):
        self._df           = None          # raw / preprocessed DataFrame
        self._metrics_df   = None          # metrics from PerformanceAnalyzer
        self._branch_sum   = None          # branch-level aggregation
        self._analyzer     = None          # PerformanceAnalyzer instance
        self._clusterer    = None          # BranchClusterer instance
        self._predictor    = None          # SalesPredictionModel instance
        self._connector    = None          # SQLServerConnector instance
        self._cluster_result = None        # cached cluster fit result
        self._data_loaded  = False
        logger.info("AnalyticsService initialised")

    # =========================================================================
    # Data loading — public entry point
    # =========================================================================

    def load_data(self) -> Dict:
        """
        Full pipeline: connect → load → preprocess → metrics → cluster → model.

        Returns:
            dict with keys: success (bool), message (str),
                            rows_loaded (int), branches (int)
        """
        try:
            # ── 1. Database connection ────────────────────────────────────────
            # Force project root onto sys.path every time — Streamlit resets it
            for _p in [str(_ROOT), str(Path(__file__).resolve().parent.parent)]:
                if _p not in sys.path:
                    sys.path.insert(0, _p)

            from config.database_config import DatabaseConfig
            from data_layer.sql_connector import SQLServerConnector

            logger.info("Connecting to SQL Server…")
            self._connector = SQLServerConnector(
                server           = DatabaseConfig.SERVER,
                database         = DatabaseConfig.DATABASE,
                username         = DatabaseConfig.USERNAME,
                password         = DatabaseConfig.PASSWORD,
                use_windows_auth = DatabaseConfig.USE_WINDOWS_AUTH,
            )
            test = self._connector.test_connection()
            if not test["success"]:
                return {"success": False, "message": test["message"],
                        "rows_loaded": 0, "branches": 0}

            # ── 2. Load & preprocess ──────────────────────────────────────────
            from data_layer.data_loader import JewelryDataLoader

            logger.info("Loading data…")
            loader   = JewelryDataLoader(self._connector)
            raw_df   = loader.load_transaction_data()
            self._df = loader.preprocess_data(raw_df)
            rows     = len(self._df)
            logger.info(f"Loaded {rows:,} rows")

            # ── 3. Performance metrics ────────────────────────────────────────
            from analytics.performance_metrics import PerformanceAnalyzer

            logger.info("Calculating performance metrics…")
            self._analyzer   = PerformanceAnalyzer(self._df)
            self._metrics_df = self._analyzer.calculate_all_metrics()
            self._branch_sum = self._analyzer.aggregate_by_branch()
            self._analyzer.identify_local_heroes()

            # ── 4. Clustering ─────────────────────────────────────────────────
            from analytics.clustering_engine import BranchClusterer
            from config.app_settings import AppSettings

            logger.info("Running clustering…")
            self._clusterer = BranchClusterer(self._metrics_df)
            X, _            = self._clusterer.prepare_features()

            if len(X) >= 3:
                kv, ins, ss = self._clusterer.find_optimal_clusters(
                    X, max_clusters=min(AppSettings.MAX_CLUSTERS_TO_TEST,
                                       max(2, len(X) - 1))
                )
                k = self._clusterer.suggest_optimal_k(kv, ins, ss)
                self._clusterer.fit_kmeans(X, n_clusters=k)
                # Cache quality scores after fit
                from sklearn.metrics import silhouette_score, calinski_harabasz_score
                self._cluster_result = {
                    "silhouette": float(silhouette_score(X, self._clusterer.cluster_labels)),
                    "calinski":   float(calinski_harabasz_score(X, self._clusterer.cluster_labels)),
                    "inertia":    float(self._clusterer.kmeans_model.inertia_),
                }
            else:
                logger.warning("Too few branches for clustering — skipping")

            # ── 5. Prediction model ───────────────────────────────────────────
            from analytics.prediction_model import SalesPredictionModel

            logger.info("Loading / training prediction model…")
            self._predictor = SalesPredictionModel()
            self._predictor.load_or_train(self._df)

            self._data_loaded = True
            branches = self._branch_sum["BRANCHNAME"].nunique() \
                       if self._branch_sum is not None else 0

            logger.info("✅ AnalyticsService.load_data() complete")
            return {
                "success":     True,
                "message":     "Data loaded successfully",
                "rows_loaded": rows,
                "branches":    branches,
            }

        except Exception as e:
            logger.error(f"load_data() failed: {e}", exc_info=True)
            self._data_loaded = False
            return {
                "success":     False,
                "message":     str(e),
                "rows_loaded": 0,
                "branches":    0,
            }

    def is_data_loaded(self) -> bool:
        return self._data_loaded

    # =========================================================================
    # Dashboard data
    # =========================================================================

    def get_dashboard_data(self) -> Dict:
        """
        Return KPI dict for the Dashboard page.

        Keys: total_sales, total_stock, total_branches, total_regions,
              overall_efficiency, overall_sell_through, total_local_heroes,
              cluster_count, top_branch, top_region, attributes
        """
        if not self._data_loaded:
            return {}

        try:
            stats = self._analyzer.get_summary_stats()

            cluster_count = 0
            if self._clusterer and self._clusterer.kmeans_model:
                cluster_count = self._clusterer.kmeans_model.n_clusters

            attrs = {}
            for col in ["PURITY", "FINISH", "THEME", "SHAPE", "WORKSTYLE", "BRAND"]:
                if col in self._df.columns:
                    attrs[col] = int(self._df[col].nunique())

            top_branch = ""
            top_region = ""
            if self._branch_sum is not None and not self._branch_sum.empty:
                top_row    = self._branch_sum.nlargest(1, "SALE_COUNT").iloc[0]
                top_branch = top_row["BRANCHNAME"]
                reg_sum    = self._branch_sum.groupby("REGION")["SALE_COUNT"].sum()
                top_region = reg_sum.idxmax() if not reg_sum.empty else ""

            return {
                "total_sales":         int(stats.get("total_sales", 0)),
                "total_stock":         int(stats.get("total_stock", 0)),
                "total_branches":      int(stats.get("total_branches", 0)),
                "total_regions":       int(self._df["REGION"].nunique())
                                       if "REGION" in self._df.columns else 0,
                "overall_efficiency":  float(stats.get("overall_efficiency", 0)),
                "overall_sell_through":float(stats.get("overall_sell_through", 0)),
                "total_local_heroes":  int(stats.get("total_local_heroes", 0)),
                "cluster_count":       cluster_count,
                "top_branch":          top_branch,
                "top_region":          top_region,
                "attributes":          attrs,
            }

        except Exception as e:
            logger.error(f"get_dashboard_data() error: {e}", exc_info=True)
            return {}

    # =========================================================================
    # Top branches
    # =========================================================================

    def get_top_branches(self, n: int = 10, metric: str = "SALE_COUNT") -> pd.DataFrame:
        """
        Return the top-N branches ranked by a given metric.

        Args:
            n      : Number of branches to return.
            metric : Column to sort by (must exist in branch_summary).

        Returns:
            DataFrame with rank column added.
        """
        if self._branch_sum is None or self._branch_sum.empty:
            return pd.DataFrame()

        valid = self._branch_sum.copy()
        sort_col = metric if metric in valid.columns else "SALE_COUNT"
        top      = valid.nlargest(n, sort_col).copy()
        top["rank"] = range(1, len(top) + 1)
        return top.reset_index(drop=True)

    # =========================================================================
    # Cluster analysis
    # =========================================================================

    def get_cluster_analysis(self) -> Dict:
        """
        Return cluster analysis dict for the Branch & Clusters page.

        Keys: n_clusters, quality_scores, pca_data (DataFrame),
              branch_map (DataFrame), summary (dict)
        """
        if not self._data_loaded or self._clusterer is None:
            return {
                "n_clusters":     0,
                "quality_scores": {},
                "pca_data":       pd.DataFrame(),
                "branch_map":     pd.DataFrame(),
                "summary":        {"clusters": []},
            }

        try:
            n_clusters = 0
            qs         = {}
            pca_df     = pd.DataFrame()
            branch_map = pd.DataFrame()
            summary    = {"clusters": []}

            if self._clusterer.kmeans_model:
                n_clusters = self._clusterer.kmeans_model.n_clusters

                # Quality scores from the cached fit result
                if self._cluster_result:
                    qs = {
                        "silhouette": self._cluster_result.get("silhouette", 0),
                        "calinski":   self._cluster_result.get("calinski", 0),
                        "inertia":    self._cluster_result.get("inertia", 0),
                    }

                # PCA 2-D scatter data
                try:
                    pca_df = self._clusterer.get_pca_data()
                except Exception:
                    pass

                # Branch → cluster map
                try:
                    branch_map = self._clusterer.get_branch_cluster_map()
                except Exception:
                    pass

                # Cluster summary
                try:
                    summary = self._clusterer.get_cluster_summary()
                except Exception:
                    pass

            return {
                "n_clusters":     n_clusters,
                "quality_scores": qs,
                "pca_data":       pca_df,
                "branch_map":     branch_map,
                "summary":        summary,
            }

        except Exception as e:
            logger.error(f"get_cluster_analysis() error: {e}", exc_info=True)
            return {
                "n_clusters":     0,
                "quality_scores": {},
                "pca_data":       pd.DataFrame(),
                "branch_map":     pd.DataFrame(),
                "summary":        {"clusters": []},
            }

    # =========================================================================
    # Filters (for dropdowns)
    # =========================================================================

    def get_available_filters(self) -> Dict[str, List[str]]:
        """
        Return unique values for every filter dropdown in the UI.

        Returns:
            dict with keys: regions, branches, purities, finishes,
                            themes, shapes, workstyles, brands
        """
        if not self._data_loaded or self._df is None:
            return {}

        def _uniq(col):
            if col in self._df.columns:
                vals = sorted(self._df[col].dropna().astype(str).unique().tolist())
                return [v for v in vals if v.strip()]
            return []

        return {
            "regions":    _uniq("REGION"),
            "branches":   _uniq("BRANCHNAME"),
            "purities":   _uniq("PURITY"),
            "finishes":   _uniq("FINISH"),
            "themes":     _uniq("THEME"),
            "shapes":     _uniq("SHAPE"),
            "workstyles": _uniq("WORKSTYLE"),
            "brands":     _uniq("BRAND"),
        }

    # =========================================================================
    # Sales prediction
    # =========================================================================

    def predict_sales(self, input_data: Dict) -> Dict:
        """
        Predict SALE_COUNT for a given input combination.

        Args:
            input_data: dict with keys matching model features:
                        REGION, BRANCHNAME, PURITY, FINISH, THEME,
                        SHAPE, WORKSTYLE, BRAND, STOCK_COUNT

        Returns:
            dict with keys: predicted_sales (int),
                            confidence_range (tuple[int,int]),
                            or 'error' (str) on failure.
        """
        if self._predictor is None or not self._predictor.is_trained:
            return {"error": "Prediction model is not trained. Load data first."}

        try:
            result = self._predictor.predict(input_data)
            return result
        except Exception as e:
            logger.error(f"predict_sales() error: {e}", exc_info=True)
            return {"error": str(e)}

    def get_model_info(self) -> Dict:
        """Return training metadata for display in the Prediction page."""
        if self._predictor is None:
            return {"is_trained": False}
        return {
            "is_trained":   self._predictor.is_trained,
            "eval_metrics": self._predictor._eval_metrics,
        }

    def retrain_model(self) -> Dict:
        """
        Force-retrain the RandomForest model on current data.

        Returns:
            eval_metrics dict (mae, rmse, r2) or empty dict on failure.
        """
        if not self._data_loaded or self._df is None:
            return {}
        try:
            self._predictor.train_and_save(self._df)
            return self._predictor._eval_metrics
        except Exception as e:
            logger.error(f"retrain_model() error: {e}", exc_info=True)
            return {}

    # =========================================================================
    # Recommendations
    # =========================================================================

    def get_recommendations(self,
                             branch: Optional[str] = None,
                             region: Optional[str] = None,
                             top_n: int = 5) -> Dict:
        """
        Attribute-based recommendations for a branch or region.
        Delegates to PerformanceAnalyzer.get_recommendations().
        """
        if self._analyzer is None:
            return {"error": "Data not loaded."}
        try:
            return self._analyzer.get_recommendations(
                branch=branch, region=region, top_n=top_n
            )
        except Exception as e:
            logger.error(f"get_recommendations() error: {e}", exc_info=True)
            return {"error": str(e)}

    def get_high_performing_combos(self,
                                   branch: Optional[str] = None,
                                   top_n: int = 10) -> pd.DataFrame:
        """
        Best PURITY × FINISH × THEME combos, optionally filtered by branch.
        """
        if self._analyzer is None:
            return pd.DataFrame()
        try:
            combos = self._analyzer.get_high_performing_combos(top_n=top_n)
            if branch and not combos.empty and "BRANCHNAME" in combos.columns:
                combos = combos[combos["BRANCHNAME"] == branch]
            return combos
        except Exception as e:
            logger.error(f"get_high_performing_combos() error: {e}", exc_info=True)
            return pd.DataFrame()

    # =========================================================================
    # Regional performance
    # =========================================================================

    def get_product_performance_by_region(self,
                                           attribute: str = "PURITY",
                                           metric: str = "total_sales"
                                           ) -> Optional[pd.DataFrame]:
        """
        Attribute breakdown by region for the Regional Performance page.
        """
        if self._analyzer is None:
            return None
        try:
            return self._analyzer.get_product_performance_by_region(
                attribute=attribute, metric=metric
            )
        except Exception as e:
            logger.error(f"get_product_performance_by_region() error: {e}", exc_info=True)
            return None

    # =========================================================================
    # Report data bundle (used by ReportGenerator)
    # =========================================================================

    def get_report_data(self) -> Dict:
        """
        Assemble all data needed by ReportGenerator in one call.

        Returns:
            dict with keys: dashboard, top_branches, cluster_analysis,
                            branch_summary (DataFrame), filters
        """
        return {
            "dashboard":        self.get_dashboard_data(),
            "top_branches":     self.get_top_branches(10, "SALE_COUNT"),
            "cluster_analysis": self.get_cluster_analysis(),
            "branch_summary":   self._branch_sum,
            "filters":          self.get_available_filters(),
        }
