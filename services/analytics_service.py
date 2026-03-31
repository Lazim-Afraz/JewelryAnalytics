"""
Analytics Service Layer
services/analytics_service.py

Cloud-safe version: SQL Server load_data() is stubbed out with a clear
message. CSV upload path (used in app.py sidebar) is unchanged and fully
functional. All other methods are identical to the original.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

def _find_project_root() -> Path:
    candidate = Path(__file__).resolve().parent
    for _ in range(5):
        if (candidate / "app.py").exists():
            return candidate
        candidate = candidate.parent
    return Path(__file__).resolve().parent.parent

_ROOT = _find_project_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


class AnalyticsService:
    def __init__(self):
        self._df             = None
        self._metrics_df     = None
        self._branch_sum     = None
        self._analyzer       = None
        self._clusterer      = None
        self._predictor      = None
        self._connector      = None
        self._cluster_result = None
        self._data_loaded    = False
        self._heroes_df      = None
        self._attr_data      = None
        self._cluster_summary    = None
        self._pca_df             = None
        self._branch_cluster_map = None
        logger.info("AnalyticsService initialised (cloud mode)")

    # =========================================================================
    # Data loading
    # =========================================================================

    def load_data(self) -> Dict:
        """
        SQL Server connection is not available in cloud deployment.
        Use the CSV Upload option in the sidebar instead.
        """
        return {
            "success":     False,
            "message":     (
                "Database connection is not available in the cloud deployment. "
                "Please use the 📂 CSV Upload option in the sidebar."
            ),
            "rows_loaded": 0,
            "branches":    0,
        }

    def is_data_loaded(self) -> bool:
        return self._data_loaded

    # =========================================================================
    # Dashboard data
    # =========================================================================

    def get_dashboard_data(self) -> Dict:
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
                if "REGION" in self._branch_sum.columns:
                    reg_sum    = self._branch_sum.groupby("REGION")["SALE_COUNT"].sum()
                    top_region = reg_sum.idxmax() if not reg_sum.empty else ""

            return {
                "total_sales":          int(stats.get("total_sales", 0)),
                "total_stock":          int(stats.get("total_stock", 0)),
                "total_branches":       int(stats.get("total_branches", 0)),
                "total_regions":        int(self._df["REGION"].nunique())
                                        if "REGION" in self._df.columns else 0,
                "overall_efficiency":   float(stats.get("overall_efficiency", 0)),
                "overall_sell_through": float(stats.get("overall_sell_through", 0)),
                "total_local_heroes":   int(stats.get("total_local_heroes", 0)),
                "cluster_count":        cluster_count,
                "top_branch":           top_branch,
                "top_region":           top_region,
                "attributes":           attrs,
            }
        except Exception as e:
            logger.error(f"get_dashboard_data() error: {e}", exc_info=True)
            return {}

    # =========================================================================
    # Top branches
    # =========================================================================

    def get_top_branches(self, n: int = 10, metric: str = "SALE_COUNT") -> pd.DataFrame:
        if self._branch_sum is None or self._branch_sum.empty:
            return pd.DataFrame()
        valid    = self._branch_sum.copy()
        sort_col = metric if metric in valid.columns else "SALE_COUNT"
        top      = valid.nlargest(n, sort_col).copy()
        top["rank"] = range(1, len(top) + 1)
        return top.reset_index(drop=True)

    # =========================================================================
    # Cluster analysis
    # =========================================================================

    def get_cluster_analysis(self) -> Dict:
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

                if self._cluster_result:
                    qs = {
                        "silhouette": self._cluster_result.get("silhouette", 0),
                        "calinski":   self._cluster_result.get("calinski", 0),
                        "inertia":    self._cluster_result.get("inertia", 0),
                    }

                try:
                    pca_df = self._clusterer.get_pca_data()
                except Exception:
                    pass

                try:
                    branch_map = self._clusterer.get_branch_cluster_map()
                except Exception:
                    pass

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
    # Filters
    # =========================================================================

    def get_available_filters(self) -> Dict[str, List[str]]:
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
    # Prediction
    # =========================================================================

    def predict_sales(self, input_data: Dict) -> Dict:
        if self._predictor is None or not self._predictor.is_trained:
            return {"error": "Prediction model is not trained. Load data first."}
        try:
            return self._predictor.predict(input_data)
        except Exception as e:
            logger.error(f"predict_sales() error: {e}", exc_info=True)
            return {"error": str(e)}

    def get_model_info(self) -> Dict:
        if self._predictor is None:
            return {"is_trained": False}
        return {
            "is_trained":   self._predictor.is_trained,
            "eval_metrics": self._predictor._eval_metrics,
        }

    def retrain_model(self) -> Dict:
        if not self._data_loaded or self._df is None:
            return {}
        try:
            import os
            for p in ["models/model_v2.pkl", "models/model_v2_meta.pkl",
                      "models/model.pkl",    "models/model_meta.pkl"]:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
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
        if self._analyzer is None:
            return {"error": "Data not loaded."}
        try:
            import inspect
            sig = inspect.signature(self._analyzer.get_recommendations)
            if "predictor" in sig.parameters:
                return self._analyzer.get_recommendations(
                    branch=branch, region=region, top_n=top_n,
                    predictor=self._predictor,
                )
            else:
                return self._analyzer.get_recommendations(
                    branch=branch, region=region, top_n=top_n,
                )
        except Exception as e:
            logger.error(f"get_recommendations() error: {e}", exc_info=True)
            return {"error": str(e)}

    def get_high_performing_combos(self,
                                   branch: Optional[str] = None,
                                   top_n: int = 10) -> pd.DataFrame:
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
    # Report data bundle
    # =========================================================================

    def get_report_data(self) -> Dict:
        return {
            "dashboard":        self.get_dashboard_data(),
            "top_branches":     self.get_top_branches(10, "SALE_COUNT"),
            "cluster_analysis": self.get_cluster_analysis(),
            "branch_summary":   self._branch_sum,
            "filters":          self.get_available_filters(),
        }
