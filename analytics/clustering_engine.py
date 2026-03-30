"""
K-Means Clustering Engine
analytics/clustering_engine.py

Branch segmentation via K-Means on aggregated performance features.

Schema reference:
    REGION, BRANCHNAME, ITEMID, PURITY, FINISH, THEME, SHAPE,
    WORKSTYLE, BRAND, SALE_COUNT, STOCK_COUNT
    + computed: efficiency_ratio, sell_through_rate,
                sales_contribution_pct, relative_strength

EXTENDED (v2):
    + get_cluster_summary()      — structured dict per cluster, UI-ready
    + get_pca_data()             — PCA 2-D coords for scatter plots (no matplotlib)
    + get_branch_cluster_map()   — clean branch → cluster mapping DataFrame
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BranchClusterer:
    """
    K-Means clustering for branch segmentation.

    Aggregates row-level metrics_df to branch level, then clusters
    branches by their performance profile.

    Column reference: BRANCHNAME (not BRANCH), ITEMID (not ITEMTYPE).
    """

    def __init__(self, metrics_df: pd.DataFrame):
        """
        Args:
            metrics_df: Output of PerformanceAnalyzer.calculate_all_metrics().
        """
        self.metrics_df      = metrics_df.copy()
        self.scaler          = StandardScaler()
        self.kmeans_model    = None
        self.cluster_labels  = None
        self.branch_features = None
        self.feature_names   = None
        self._pca_model      = None          # populated by get_pca_data()

    # ─── Feature preparation ──────────────────────────────────────────────────

    def prepare_features(self,
                         feature_cols: List[str] = None,
                         aggregation_method: str = 'mean'
                         ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Aggregate metrics to branch level and standardise for clustering.

        Args:
            feature_cols: Columns to use as clustering features.
                          Defaults to efficiency_ratio, sell_through_rate,
                          SALE_COUNT, STOCK_COUNT, sales_contribution_pct.
            aggregation_method: How to aggregate ratio columns ('mean' / 'median').

        Returns:
            (X_scaled, branch_features_df)
        """
        logger.info("Preparing features for clustering...")

        if feature_cols is None:
            feature_cols = [
                'efficiency_ratio',
                'sell_through_rate',
                'sales_contribution_pct',
                'SALE_COUNT',
                'STOCK_COUNT',
            ]

        # Keep only columns that actually exist
        feature_cols = [c for c in feature_cols if c in self.metrics_df.columns]
        logger.info(f"  Features: {feature_cols}")

        # Aggregate to branch level
        agg_dict = {}
        for col in feature_cols:
            agg_dict[col] = 'sum' if col in ('SALE_COUNT', 'STOCK_COUNT') else aggregation_method

        branch_features = (
            self.metrics_df
            .groupby('BRANCHNAME')
            .agg(agg_dict)
            .reset_index()
        )

        # Preserve REGION for later labelling
        if 'REGION' in self.metrics_df.columns:
            region_map = self.metrics_df.groupby('BRANCHNAME')['REGION'].first()
            branch_features['REGION'] = branch_features['BRANCHNAME'].map(region_map)

        logger.info(f"  Aggregated to {len(branch_features)} branches")

        # Clean NaN / inf
        branch_features[feature_cols] = (
            branch_features[feature_cols]
            .fillna(0)
            .replace([np.inf, -np.inf], 0)
        )

        # Standardise
        logger.info("  Standardising features (Z-score)...")
        X = branch_features[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)

        logger.info(f"✅ Features prepared: shape {X_scaled.shape}")

        self.branch_features = branch_features
        self.feature_names   = feature_cols

        return X_scaled, branch_features

    # ─── Optimal k selection ──────────────────────────────────────────────────

    def find_optimal_clusters(self,
                               X: np.ndarray,
                               min_clusters: int = 2,
                               max_clusters: int = 10,
                               random_state: int = 42
                               ) -> Tuple[List[int], List[float], List[float]]:
        """
        Elbow method + silhouette scores across a range of k.

        Args:
            X:            Scaled feature matrix.
            min_clusters: Minimum k to test.
            max_clusters: Maximum k to test (capped at n_branches - 1).
            random_state: Random seed.

        Returns:
            (k_values, inertias, silhouette_scores)
        """
        max_possible = len(X) - 1
        max_clusters = min(max_clusters, max_possible)

        logger.info(f"Finding optimal clusters (k={min_clusters} to {max_clusters})...")

        k_values               = list(range(min_clusters, max_clusters + 1))
        inertias               = []
        silhouette_scores_list = []

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=random_state,
                        n_init=10, max_iter=300)
            km.fit(X)
            inertias.append(km.inertia_)

            sil = silhouette_score(X, km.labels_) if k > 1 else 0.0
            silhouette_scores_list.append(sil)

            logger.info(f"  k={k}: inertia={km.inertia_:.2f}, "
                        f"silhouette={sil:.3f}")

        logger.info("✅ Elbow method complete")
        return k_values, inertias, silhouette_scores_list

    def suggest_optimal_k(self,
                          k_values: List[int],
                          inertias: List[float],
                          silhouette_scores: List[float]) -> int:
        """
        Pick k with the highest silhouette score.

        Returns:
            Suggested k (int).
        """
        best_idx    = int(np.argmax(silhouette_scores))
        suggested_k = k_values[best_idx]
        logger.info(f"Suggested k={suggested_k} "
                    f"(silhouette={silhouette_scores[best_idx]:.3f})")
        return suggested_k

    # ─── K-Means fitting ──────────────────────────────────────────────────────

    def fit_kmeans(self,
                   X: np.ndarray,
                   n_clusters: int = 5,
                   random_state: int = 42) -> np.ndarray:
        """
        Fit K-Means and store labels.

        Args:
            X:            Scaled feature matrix.
            n_clusters:   Number of clusters.
            random_state: Random seed.

        Returns:
            Cluster label array.
        """
        logger.info(f"Fitting K-Means with k={n_clusters}...")

        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300,
            algorithm='lloyd'
        )

        self.cluster_labels = self.kmeans_model.fit_predict(X)

        silhouette = silhouette_score(X, self.cluster_labels)
        calinski   = calinski_harabasz_score(X, self.cluster_labels)

        logger.info(f"✅ K-Means complete")
        logger.info(f"   Silhouette Score        : {silhouette:.3f}")
        logger.info(f"   Calinski-Harabasz Score : {calinski:.2f}")
        logger.info(f"   Inertia                 : {self.kmeans_model.inertia_:.2f}")

        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        for cid, cnt in zip(unique, counts):
            logger.info(f"   Cluster {cid}: {cnt} branches")

        return self.cluster_labels

    def assign_clusters_to_branches(self) -> pd.DataFrame:
        """
        Return branch_features DataFrame with a Cluster column added.
        """
        if self.cluster_labels is None:
            raise ValueError("Run fit_kmeans() first.")
        if self.branch_features is None:
            raise ValueError("Run prepare_features() first.")

        result            = self.branch_features.copy()
        result['Cluster'] = self.cluster_labels
        return result

    # ─── Cluster characterisation ─────────────────────────────────────────────

    def characterize_clusters(self) -> pd.DataFrame:
        """
        Return cluster centroids in original (un-scaled) units.

        Returns:
            DataFrame with one row per cluster.
        """
        if self.kmeans_model is None:
            raise ValueError("Run fit_kmeans() first.")

        logger.info("Characterising clusters...")

        centroids_original = self.scaler.inverse_transform(
            self.kmeans_model.cluster_centers_
        )

        cluster_chars = pd.DataFrame(centroids_original, columns=self.feature_names)
        cluster_chars.insert(0, 'Cluster', range(len(cluster_chars)))

        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        cluster_chars['num_branches'] = counts

        logger.info("✅ Cluster characterisation complete")
        return cluster_chars

    def get_cluster_members(self, cluster_id: int) -> pd.DataFrame:
        """Return all branches assigned to cluster_id."""
        df = self.assign_clusters_to_branches()
        return df[df['Cluster'] == cluster_id]

    def describe_cluster(self, cluster_id: int) -> Dict:
        """
        Detailed description dict for a single cluster.

        Returns:
            Dict with cluster_id, num_branches, branches list, avg_metrics.
        """
        members = self.get_cluster_members(cluster_id)

        return {
            'cluster_id':   cluster_id,
            'num_branches': len(members),
            'branches':     members['BRANCHNAME'].tolist(),
            'avg_metrics':  {col: round(float(members[col].mean()), 4)
                             for col in self.feature_names},
        }

    # =========================================================================
    # EXTENDED — Structured returns for service layer / UI (v2 additions)
    # =========================================================================

    def get_cluster_summary(self) -> Dict:
        """
        Return a fully structured cluster summary — UI and service-layer ready.

        Includes per-cluster stats, branch lists, performance labels,
        and overall quality scores. No matplotlib / plotting involved.

        Returns:
            Dict with keys:
                n_clusters      — number of clusters
                quality_scores  — silhouette + calinski scores
                clusters        — list of per-cluster dicts:
                    cluster_id, label, num_branches, branches,
                    avg_metrics, performance_tier
                elbow_data      — None (populated if find_optimal_clusters run)
        """
        if self.kmeans_model is None:
            raise ValueError("Run fit_kmeans() first.")

        logger.info("Building structured cluster summary...")

        assigned_df    = self.assign_clusters_to_branches()
        cluster_chars  = self.characterize_clusters()

        # Recompute quality scores on current labels
        X_current = self.scaler.transform(
            self.branch_features[self.feature_names]
            .fillna(0)
            .replace([np.inf, -np.inf], 0)
            .values
        )
        silhouette = round(float(silhouette_score(X_current, self.cluster_labels)), 4)
        calinski   = round(float(calinski_harabasz_score(X_current, self.cluster_labels)), 2)

        clusters = []
        for _, row in cluster_chars.iterrows():
            cid      = int(row['Cluster'])
            members  = assigned_df[assigned_df['Cluster'] == cid]

            # Avg metrics dict (numeric cols only)
            avg_metrics = {}
            for col in self.feature_names:
                val = members[col].mean()
                avg_metrics[col] = round(float(val), 4) if not np.isnan(val) else 0.0

            # Simple performance tier based on avg sell_through / efficiency
            eff = avg_metrics.get('efficiency_ratio', avg_metrics.get('SALE_COUNT', 0))
            if eff >= 0.15:
                tier = 'Elite'
            elif eff >= 0.08:
                tier = 'Strong'
            elif eff >= 0.04:
                tier = 'Average'
            else:
                tier = 'Underperforming'

            # Regions represented
            regions = []
            if 'REGION' in members.columns:
                regions = sorted(members['REGION'].dropna().unique().tolist())

            clusters.append({
                'cluster_id':     cid,
                'label':          f"Cluster {cid}",
                'num_branches':   int(row['num_branches']),
                'branches':       sorted(members['BRANCHNAME'].tolist()),
                'regions':        regions,
                'avg_metrics':    avg_metrics,
                'performance_tier': tier,
            })

        # Sort clusters by avg SALE_COUNT desc (most active first)
        clusters.sort(
            key=lambda c: c['avg_metrics'].get('SALE_COUNT', 0),
            reverse=True
        )

        summary = {
            'n_clusters':    int(self.kmeans_model.n_clusters),
            'quality_scores': {
                'silhouette': silhouette,
                'calinski':   calinski,
                'inertia':    round(float(self.kmeans_model.inertia_), 2),
            },
            'clusters': clusters,
        }

        logger.info(f"✅ Cluster summary built: {len(clusters)} clusters")
        return summary

    def get_pca_data(self,
                     X_scaled: Optional[np.ndarray] = None,
                     n_components: int = 2) -> pd.DataFrame:
        """
        Project branch features into 2-D PCA space for scatter visualisation.

        No matplotlib is used — returns a plain DataFrame that the UI layer
        (Streamlit / Plotly) can consume directly.

        Args:
            X_scaled    : Scaled feature matrix. If None, re-scales
                          self.branch_features automatically.
            n_components: Number of PCA components (default 2).

        Returns:
            DataFrame with columns:
                BRANCHNAME, REGION (if available), Cluster,
                PC1, PC2, explained_var_pc1, explained_var_pc2
        """
        if self.branch_features is None:
            raise ValueError("Run prepare_features() first.")
        if self.cluster_labels is None:
            raise ValueError("Run fit_kmeans() first.")

        logger.info(f"Computing PCA ({n_components} components)...")

        if X_scaled is None:
            X_scaled = self.scaler.transform(
                self.branch_features[self.feature_names]
                .fillna(0)
                .replace([np.inf, -np.inf], 0)
                .values
            )

        self._pca_model = PCA(n_components=n_components, random_state=42)
        X_pca           = self._pca_model.fit_transform(X_scaled)

        explained = self._pca_model.explained_variance_ratio_

        pca_df = pd.DataFrame(
            X_pca,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        pca_df['BRANCHNAME'] = self.branch_features['BRANCHNAME'].values
        pca_df['Cluster']    = self.cluster_labels

        if 'REGION' in self.branch_features.columns:
            pca_df['REGION'] = self.branch_features['REGION'].values

        # Attach explained variance as metadata columns
        for i in range(n_components):
            pca_df[f'explained_var_pc{i+1}'] = round(float(explained[i]), 4)

        # Attach original feature values for hover tooltips
        for col in self.feature_names:
            if col in self.branch_features.columns:
                pca_df[col] = self.branch_features[col].values

        logger.info(f"✅ PCA complete. Variance explained: "
                    f"PC1={explained[0]:.1%}, PC2={explained[1]:.1%}")

        return pca_df

    def get_branch_cluster_map(self) -> pd.DataFrame:
        """
        Return a clean, minimal branch → cluster mapping DataFrame.

        Suitable for merging with other DataFrames, lookups, and table
        display in the UI. No scaled values — just identifiers + metrics.

        Returns:
            DataFrame with columns:
                BRANCHNAME, REGION (if available), Cluster,
                performance_tier, SALE_COUNT, STOCK_COUNT,
                efficiency_ratio (if available), sell_through_rate (if available)
        """
        if self.cluster_labels is None:
            raise ValueError("Run fit_kmeans() first.")

        logger.info("Building branch-cluster map...")

        df = self.branch_features.copy()
        df['Cluster'] = self.cluster_labels

        # Add performance tier
        def _tier(row):
            eff = row.get('efficiency_ratio', row.get('SALE_COUNT', 0))
            if eff >= 0.15: return 'Elite'
            if eff >= 0.08: return 'Strong'
            if eff >= 0.04: return 'Average'
            return 'Underperforming'

        df['performance_tier'] = df.apply(_tier, axis=1)

        # Select clean output columns
        base_cols   = ['BRANCHNAME', 'Cluster', 'performance_tier']
        region_cols = ['REGION'] if 'REGION' in df.columns else []
        metric_cols = [c for c in ['SALE_COUNT', 'STOCK_COUNT',
                                    'efficiency_ratio', 'sell_through_rate']
                       if c in df.columns]

        result = df[region_cols + base_cols + metric_cols].copy()

        # Round float columns
        for col in metric_cols:
            if df[col].dtype in [np.float64, np.float32]:
                result[col] = result[col].round(4)

        result = result.sort_values(['Cluster', 'BRANCHNAME']).reset_index(drop=True)

        logger.info(f"✅ Branch-cluster map: {len(result)} branches")
        return result


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Clustering Engine Test (v2 — with structured returns) ===\n")

    rng      = np.random.default_rng(42)
    branches = [f'BR-CITY{i:02d}-0{j}' for i in range(1, 11) for j in range(1, 6)]
    n        = len(branches) * 10   # 10 items per branch

    metrics_data = pd.DataFrame({
        'BRANCHNAME':             rng.choice(branches, n),
        'REGION':                 rng.choice(['North', 'South', 'East', 'West'], n),
        'ITEMID':                 [f'PRD{i:05d}' for i in rng.integers(1, 50, n)],
        'SALE_COUNT':             rng.integers(0, 30, n),
        'STOCK_COUNT':            rng.integers(5, 60, n),
        'efficiency_ratio':       rng.uniform(0, 0.5, n),
        'sell_through_rate':      rng.uniform(0, 0.4, n),
        'sales_contribution_pct': rng.uniform(0, 10, n),
        'relative_strength':      rng.uniform(0.5, 2.0, n),
    })

    clusterer = BranchClusterer(metrics_data)

    # ── Original tests (unchanged) ────────────────────────────────────────────
    print("1. Preparing features...")
    X_scaled, branch_features = clusterer.prepare_features()
    print(f"   ✅ Features shape: {X_scaled.shape}")

    print("\n2. Finding optimal clusters...")
    k_values, inertias, sil_scores = clusterer.find_optimal_clusters(
        X_scaled, max_clusters=8
    )
    for k, inertia, sil in zip(k_values, inertias, sil_scores):
        print(f"   k={k}: inertia={inertia:.2f}, silhouette={sil:.3f}")

    optimal_k = clusterer.suggest_optimal_k(k_values, inertias, sil_scores)
    print(f"\n   Suggested k: {optimal_k}")

    labels = clusterer.fit_kmeans(X_scaled, n_clusters=optimal_k)

    print("\n3. Cluster Characteristics (original):")
    print(clusterer.characterize_clusters().to_string(index=False))

    print("\n4. Branch Assignments sample (original):")
    print(clusterer.assign_clusters_to_branches()[
        ['BRANCHNAME', 'Cluster']
    ].head(10).to_string(index=False))

    # ── New v2 tests ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("V2 STRUCTURED RETURN TESTS")
    print("="*60)

    print("\n5. get_cluster_summary():")
    summary = clusterer.get_cluster_summary()
    print(f"   n_clusters   : {summary['n_clusters']}")
    print(f"   silhouette   : {summary['quality_scores']['silhouette']}")
    print(f"   calinski     : {summary['quality_scores']['calinski']}")
    for c in summary['clusters']:
        print(f"   Cluster {c['cluster_id']} [{c['performance_tier']}]: "
              f"{c['num_branches']} branches — {c['branches'][:3]}{'...' if len(c['branches'])>3 else ''}")

    print("\n6. get_pca_data():")
    pca_df = clusterer.get_pca_data(X_scaled)
    print(pca_df[['BRANCHNAME', 'Cluster', 'PC1', 'PC2',
                  'explained_var_pc1', 'explained_var_pc2']].head(8).to_string(index=False))

    print("\n7. get_branch_cluster_map():")
    bcmap = clusterer.get_branch_cluster_map()
    print(bcmap.head(10).to_string(index=False))

    print("\n=== All tests passed ✅ ===")
