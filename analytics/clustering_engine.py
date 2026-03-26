"""
K-Means Clustering Engine
analytics/clustering_engine.py

Branch segmentation via K-Means on aggregated performance features.

Schema reference:
    REGION, BRANCHNAME, ITEMID, PURITY, FINISH, THEME, SHAPE,
    WORKSTYLE, BRAND, SALE_COUNT, STOCK_COUNT
    + computed: efficiency_ratio, sell_through_rate,
                sales_contribution_pct, relative_strength
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Tuple, List, Dict
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
        self.metrics_df     = metrics_df.copy()
        self.scaler         = StandardScaler()
        self.kmeans_model   = None
        self.cluster_labels = None
        self.branch_features = None
        self.feature_names  = None

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

        k_values             = list(range(min_clusters, max_clusters + 1))
        inertias             = []
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
        best_idx = int(np.argmax(silhouette_scores))
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
            X:          Scaled feature matrix.
            n_clusters: Number of clusters.
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

        result = self.branch_features.copy()
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


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Clustering Engine Test ===\n")

    rng = np.random.default_rng(42)
    branches = [f'BR-CITY{i:02d}-0{j}' for i in range(1, 11) for j in range(1, 6)]
    n = len(branches) * 10   # 10 items per branch

    metrics_data = pd.DataFrame({
        'BRANCHNAME':           rng.choice(branches, n),
        'REGION':               rng.choice(['North', 'South', 'East', 'West'], n),
        'ITEMID':               [f'PRD{i:05d}' for i in rng.integers(1, 50, n)],
        'SALE_COUNT':           rng.integers(0, 30, n),
        'STOCK_COUNT':          rng.integers(5, 60, n),
        'efficiency_ratio':     rng.uniform(0, 0.5, n),
        'sell_through_rate':    rng.uniform(0, 0.4, n),
        'sales_contribution_pct': rng.uniform(0, 10, n),
        'relative_strength':    rng.uniform(0.5, 2.0, n),
    })

    clusterer = BranchClusterer(metrics_data)

    print("Preparing features...")
    X_scaled, branch_features = clusterer.prepare_features()
    print(f"✅ Features shape: {X_scaled.shape}")

    print("\nFinding optimal clusters...")
    k_values, inertias, sil_scores = clusterer.find_optimal_clusters(
        X_scaled, max_clusters=8
    )
    for k, inertia, sil in zip(k_values, inertias, sil_scores):
        print(f"  k={k}: inertia={inertia:.2f}, silhouette={sil:.3f}")

    optimal_k = clusterer.suggest_optimal_k(k_values, inertias, sil_scores)
    print(f"\nSuggested k: {optimal_k}")

    labels = clusterer.fit_kmeans(X_scaled, n_clusters=optimal_k)

    print("\nCluster Characteristics:")
    print(clusterer.characterize_clusters().to_string(index=False))

    print("\nBranch Assignments (first 10):")
    print(clusterer.assign_clusters_to_branches()[['BRANCHNAME', 'Cluster']].head(10).to_string(index=False))

    print("\n=== Test complete ===")
