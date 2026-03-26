# %% [markdown]
# # Jewelry Analytics - Complete Analysis
# Run cell-by-cell to see each step

# %% CELL 1: Setup
print("="*60)
print("JEWELRY PORTFOLIO ANALYTICS - STEP BY STEP")
print("="*60)

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config.database_config import DatabaseConfig
from config.app_settings import AppSettings
from data_layer.sql_connector import SQLServerConnector
from data_layer.data_loader import JewelryDataLoader
from analytics.performance_metrics import PerformanceAnalyzer
from analytics.clustering_engine import BranchClusterer

print("\n✅ All imports successful!")

# %% CELL 2: Connect to Database
print("\n" + "="*60)
print("STEP 1: DATABASE CONNECTION")
print("="*60)

connector = SQLServerConnector(
    server=DatabaseConfig.SERVER,
    database=DatabaseConfig.DATABASE,
    use_windows_auth=DatabaseConfig.USE_WINDOWS_AUTH
)

result = connector.test_connection()
print(f"\nServer: {result['server']}")
print(f"Database: {result['database']}")
print(f"Status: {result['message']}")

if not result['success']:
    print("\n❌ Connection failed! Check database_config.py")
    exit()

print("\n✅ Connected successfully!")

# %% CELL 3: Load Data
print("\n" + "="*60)
print("STEP 2: LOAD TRANSACTION DATA")
print("="*60)

loader = JewelryDataLoader(connector)

# Load all data (no date filter since you don't have DATE column)
df = loader.load_transaction_data()

print(f"\n✅ Loaded {len(df):,} rows")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())

# %% CELL 4: Data Summary
print("\n" + "="*60)
print("STEP 3: DATA SUMMARY")
print("="*60)

summary = loader.get_data_summary(df)

print(f"\nTotal Rows: {summary['total_rows']:,}")
print(f"Total Branches: {summary['total_branches']}")
print(f"Total Sales: {summary['total_sales']:,}")
print(f"Total Stock: {summary['total_stock']:,}")

print(f"\nProduct Attributes:")
for attr, count in summary['attributes'].items():
    print(f"  {attr}: {count} unique values")

print(f"\nBranches:")
for branch in summary['branches'][:10]:  # Show first 10
    print(f"  - {branch}")
if len(summary['branches']) > 10:
    print(f"  ... and {len(summary['branches']) - 10} more")

# %% CELL 5: Preprocess Data
print("\n" + "="*60)
print("STEP 4: PREPROCESS DATA")
print("="*60)

df = loader.preprocess_data(df)

print(f"\n✅ Data cleaned: {len(df):,} rows")

# Check for missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"\nRemaining missing values:")
    print(missing[missing > 0])
else:
    print(f"\n✅ No missing values")

# %% CELL 6: Calculate Performance Metrics
print("\n" + "="*60)
print("STEP 5: CALCULATE PERFORMANCE METRICS")
print("="*60)

analyzer = PerformanceAnalyzer(df)
metrics_df = analyzer.calculate_all_metrics()

print(f"\n✅ Metrics calculated for {len(metrics_df):,} records")

new_cols = ['efficiency_ratio', 'sales_contribution_pct', 'relative_strength']
print(f"\nSample Metrics (first 10 rows):")
display_cols = ['BRANCH', 'ITEMTYPE', 'SALE_COUNT', 'STOCK_COUNT'] + new_cols
print(metrics_df[display_cols].head(10))

print(f"\nMetric Statistics:")
print(metrics_df[new_cols].describe())

# %% CELL 7: Branch Summary
print("\n" + "="*60)
print("STEP 6: BRANCH-LEVEL SUMMARY")
print("="*60)

branch_summary = analyzer.aggregate_by_branch()

print(f"\n✅ Aggregated {len(branch_summary)} branches")
print(f"\nBranch Performance:")
print(branch_summary.sort_values('SALE_COUNT', ascending=False))

# %% CELL 8: Local Heroes
print("\n" + "="*60)
print("STEP 7: IDENTIFY LOCAL HEROES")
print("="*60)

heroes = analyzer.identify_local_heroes(
    relative_strength_threshold=1.2,
    min_contribution_pct=5.0
)

print(f"\n✅ Found {len(heroes)} local heroes")

if len(heroes) > 0:
    print(f"\nTop 10 Local Heroes:")
    hero_cols = ['BRANCH', 'ITEMTYPE', 'PURITY', 'FINISH', 
                 'SALE_COUNT', 'efficiency_ratio', 'relative_strength']
    print(heroes[hero_cols].head(10))
else:
    print("\n⚠️ No local heroes found with current thresholds")
    print("   Try lowering relative_strength_threshold")

# %% CELL 9: Underperformers
print("\n" + "="*60)
print("STEP 8: IDENTIFY UNDERPERFORMERS")
print("="*60)

underperformers = analyzer.identify_underperformers(
    efficiency_threshold=0.5
)

print(f"\n✅ Found {len(underperformers)} underperformers")

if len(underperformers) > 0:
    print(f"\nTop 10 Underperformers (need action):")
    under_cols = ['BRANCH', 'ITEMTYPE', 'STOCK_COUNT', 
                  'SALE_COUNT', 'efficiency_ratio']
    print(underperformers[under_cols].head(10))
else:
    print("\n✅ No major underperformers found")

# %% CELL 10: Prepare Features for Clustering
print("\n" + "="*60)
print("STEP 9: PREPARE FEATURES FOR CLUSTERING")
print("="*60)

clusterer = BranchClusterer(metrics_df)

X_scaled, branch_features = clusterer.prepare_features(
    feature_cols=['efficiency_ratio', 'sales_contribution_pct', 
                  'SALE_COUNT', 'STOCK_COUNT']
)

print(f"\n✅ Features prepared")
print(f"   Matrix shape: {X_scaled.shape}")
print(f"   ({X_scaled.shape[0]} branches × {X_scaled.shape[1]} features)")

print(f"\nBranch Features (first 10):")
print(branch_features.head(10))

# %% CELL 11: Find Optimal Clusters (Elbow Method)
print("\n" + "="*60)
print("STEP 10: FIND OPTIMAL NUMBER OF CLUSTERS")
print("="*60)

k_values, inertias, sil_scores = clusterer.find_optimal_clusters(
    X_scaled,
    min_clusters=2,
    max_clusters=min(10, len(branch_features)-1)  # Can't have more clusters than branches
)

print(f"\n✅ Elbow method complete")
print(f"\nResults:")
print(f"{'k':<5} {'Inertia':<15} {'Silhouette'}")
print("-" * 35)
for k, inertia, sil in zip(k_values, inertias, sil_scores):
    print(f"{k:<5} {inertia:<15.2f} {sil:.3f}")

optimal_k = clusterer.suggest_optimal_k(k_values, inertias, sil_scores)
print(f"\n✅ Suggested optimal k = {optimal_k}")

# %% CELL 12: Visualize Elbow Plot
print("\n" + "="*60)
print("STEP 11: VISUALIZE ELBOW METHOD")
print("="*60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Inertia', fontsize=11, fontweight='bold')
ax1.set_title('Elbow Method - Inertia', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
ax1.legend()

# Silhouette plot
ax2.plot(k_values, sil_scores, 'go-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
ax2.set_title('Silhouette Score by k', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
ax2.legend()

plt.tight_layout()
plt.savefig('exports/visualizations/elbow_method.png', dpi=300, bbox_inches='tight')
print("\n✅ Elbow plot saved to: exports/visualizations/elbow_method.png")
plt.show()

# %% CELL 13: Fit K-Means
print("\n" + "="*60)
print(f"STEP 12: FIT K-MEANS WITH k={optimal_k}")
print("="*60)

labels = clusterer.fit_kmeans(X_scaled, n_clusters=optimal_k)

print(f"\n✅ K-Means clustering complete")

unique, counts = np.unique(labels, return_counts=True)
print(f"\nCluster Distribution:")
for cluster_id, count in zip(unique, counts):
    print(f"  Cluster {cluster_id}: {count} branches")

# %% CELL 14: Characterize Clusters
print("\n" + "="*60)
print("STEP 13: CHARACTERIZE EACH CLUSTER")
print("="*60)

cluster_chars = clusterer.characterize_clusters()

print(f"\n✅ Cluster characteristics:")
print(cluster_chars)

# %% CELL 15: Assign Clusters to Branches
print("\n" + "="*60)
print("STEP 14: ASSIGN CLUSTERS TO BRANCHES")
print("="*60)

cluster_assignments = clusterer.assign_clusters_to_branches()

print(f"\n✅ Cluster assignments complete")
print(f"\nBranch Assignments:")
print(cluster_assignments[['BRANCH', 'Cluster', 'efficiency_ratio', 
                           'SALE_COUNT', 'STOCK_COUNT']].sort_values('Cluster'))

# %% CELL 16: Visualize Clusters
print("\n" + "="*60)
print("STEP 15: VISUALIZE CLUSTER DISTRIBUTION")
print("="*60)

fig, ax = plt.subplots(figsize=(10, 6))

cluster_counts = cluster_assignments['Cluster'].value_counts().sort_index()
bars = ax.bar(cluster_counts.index, cluster_counts.values, 
              color=plt.cm.Set3(np.linspace(0, 1, len(cluster_counts))))

ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Branches', fontsize=11, fontweight='bold')
ax.set_title('Branch Distribution by Cluster', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('exports/visualizations/cluster_distribution.png', dpi=300, bbox_inches='tight')
print("\n✅ Cluster plot saved to: exports/visualizations/cluster_distribution.png")
plt.show()

# %% CELL 17: Summary Statistics
print("\n" + "="*60)
print("STEP 16: OVERALL SUMMARY")
print("="*60)

stats = analyzer.get_summary_stats()

print(f"\n📊 OVERALL PERFORMANCE:")
print(f"   Total Branches: {stats['total_branches']}")
print(f"   Total Sales: {stats['total_sales']:,}")
print(f"   Total Stock: {stats['total_stock']:,}")
print(f"   Overall Efficiency: {stats['overall_efficiency']:.2f}")
print(f"   Avg Efficiency Ratio: {stats['avg_efficiency_ratio']:.2f}")
print(f"   Local Heroes Found: {stats['total_local_heroes']}")

print(f"\n📦 PRODUCT ATTRIBUTES:")
for attr, count in stats['attributes'].items():
    print(f"   {attr}: {count} types")

# %% CELL 18: Top Performers
print("\n" + "="*60)
print("STEP 17: TOP PERFORMING BRANCHES")
print("="*60)

top_branches = branch_summary.nlargest(10, 'SALE_COUNT')

print(f"\n🏆 TOP 10 BRANCHES BY SALES:")
for idx, (_, row) in enumerate(top_branches.iterrows(), 1):
    print(f"   {idx:2d}. {row['BRANCH']:30s}: "
          f"{row['SALE_COUNT']:>8,} sales | "
          f"{row['branch_efficiency']:>5.2f} efficiency | "
          f"Cluster {cluster_assignments[cluster_assignments['BRANCH']==row['BRANCH']]['Cluster'].iloc[0] if row['BRANCH'] in cluster_assignments['BRANCH'].values else 'N/A'}")

# %% CELL 19: Cleanup
print("\n" + "="*60)
print("STEP 18: CLEANUP")
print("="*60)

connector.close()
print("\n✅ Database connection closed")

print("\n" + "="*60)
print("✅ ANALYSIS COMPLETE!")
print("="*60)
print("\n📁 Files created:")
print("   - exports/visualizations/elbow_method.png")
print("   - exports/visualizations/cluster_distribution.png")
print("\n💡 Next steps:")
print("   1. Review the cluster characteristics")
print("   2. Identify actions for each cluster")
print("   3. Export results to Excel (coming soon)")
print("="*60)