"""
Jewelry Portfolio Analytics - Desktop Application
Main Entry Point

This is the main file that launches the desktop application.
Run this file to start the GUI.

Usage:
    python main.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jewelry_analytics.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'pyodbc',
        'matplotlib', 'seaborn', 'openpyxl'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nPlease install dependencies:")
        print("   pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Main application entry point"""
    
    logger.info("="*60)
    logger.info("Jewelry Portfolio Analytics - Desktop Application")
    logger.info("="*60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    try:
        # Import GUI
        from gui.main_window import MainWindow

        logger.info("Launching Tkinter GUI...")
        app = MainWindow()
        app.mainloop()
        logger.info("Application closed")
    
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please ensure GUI modules are created")
        
        # Fallback: Run console mode
        print("\n" + "="*60)
        print("GUI not available. Running in CONSOLE MODE...")
        print("="*60 + "\n")
        run_console_mode()
    
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


def run_console_mode():
    """
    Console mode for testing without GUI
    Demonstrates the complete analytical workflow
    """
    
    print("CONSOLE MODE - Complete Analysis Workflow")
    print("-" * 60)
    
    try:
        # Import modules
        from config.database_config import DatabaseConfig
        from config.app_settings import AppSettings
        from data_layer.sql_connector import SQLServerConnector
        from data_layer.data_loader import JewelryDataLoader
        from analytics.performance_metrics import PerformanceAnalyzer
        from analytics.clustering_engine import BranchClusterer
        
        print("\n1. Connecting to SQL Server...")
        print(f"   Server: {DatabaseConfig.SERVER}")
        print(f"   Database: {DatabaseConfig.DATABASE}")
        
        # Create connector
        connector = SQLServerConnector(
            server           = DatabaseConfig.SERVER,
            database         = DatabaseConfig.DATABASE,
            username         = DatabaseConfig.USERNAME,
            password         = DatabaseConfig.PASSWORD,
            use_windows_auth = DatabaseConfig.USE_WINDOWS_AUTH
        )
        
        # Test connection
        test_result = connector.test_connection()
        if not test_result['success']:
            print(f"   ❌ Connection failed: {test_result['message']}")
            print("\n   Please check config/database_config.py")
            return
        
        print(f"   ✅ {test_result['message']}")
        
        print("\n2. Loading Data...")
        loader = JewelryDataLoader(connector)
        
        try:
            # Load data
            df = loader.load_transaction_data()
            print(f"   ✅ Loaded {len(df)} rows")
            
            # Preprocess
            df = loader.preprocess_data(df)
            print(f"   ✅ Preprocessed data ready")
            
            # Summary
            summary = loader.get_data_summary(df)
            print(f"\n   Data Summary:")
            print(f"   - Branches: {summary['total_branches']}")
            print(f"   - Total Sales: {summary['total_sales']:,}")
            print(f"   - Total Stock: {summary['total_stock']:,}")
            
        except Exception as e:
            print(f"   ❌ Data loading failed: {e}")
            print("\n   Please modify queries in data_layer/data_loader.py")
            print("   to match your database schema!")
            return
        
        print("\n3. Calculating Performance Metrics...")
        analyzer = PerformanceAnalyzer(df)
        metrics_df = analyzer.calculate_all_metrics()
        print(f"   ✅ Metrics calculated")
        
        branch_summary = analyzer.aggregate_by_branch()
        print(f"   ✅ Branch aggregation complete")
        
        heroes = analyzer.identify_local_heroes()
        print(f"   ✅ Found {len(heroes)} local heroes")
        
        print("\n4. Running K-Means Clustering...")
        clusterer = BranchClusterer(metrics_df)
        
        # Prepare features
        X_scaled, branch_features = clusterer.prepare_features()
        print(f"   ✅ Features prepared: {X_scaled.shape}")
        
        # Find optimal clusters
        k_values, inertias, sil_scores = clusterer.find_optimal_clusters(
            X_scaled,
            max_clusters=AppSettings.MAX_CLUSTERS_TO_TEST
        )
        optimal_k = clusterer.suggest_optimal_k(k_values, inertias, sil_scores)
        print(f"   ✅ Suggested optimal k: {optimal_k}")
        
        # Fit K-Means
        labels = clusterer.fit_kmeans(X_scaled, n_clusters=optimal_k)
        print(f"   ✅ K-Means clustering complete")
        
        # Characterize clusters
        cluster_chars = clusterer.characterize_clusters()
        print(f"\n   Cluster Characteristics:")
        print(cluster_chars.to_string())
        
        print("\n5. Generating Summary...")
        stats = analyzer.get_summary_stats()
        print(f"\n   Overall Statistics:")
        print(f"   - Total Branches: {stats['total_branches']}")
        print(f"   - Total Sales: {stats['total_sales']:,}")
        print(f"   - Overall Efficiency: {stats['overall_efficiency']:.2f}")
        print(f"   - Local Heroes: {stats['total_local_heroes']}")
        
        print(f"\n   Top 5 Performing Branches:")
        top_branches = branch_summary.nlargest(5, 'SALE_COUNT')
        for i, (_, row) in enumerate(top_branches.iterrows(), 1):
            print(f"   {i}. {row['BRANCHNAME']}: "
                  f"{row['SALE_COUNT']:,} sales, "
                  f"{row['avg_efficiency']:.2f} efficiency")
        
        print("\n" + "="*60)
        print("✅ CONSOLE MODE ANALYSIS COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Modify database_config.py with your SQL Server details")
        print("2. Modify data_loader.py queries to match your schema")
        print("3. Run: python main.py (to launch GUI)")
        print("="*60)
        
        connector.close()
    
    except Exception as e:
        print(f"\n❌ Error in console mode: {e}")
        logger.error("Console mode error", exc_info=True)


if __name__ == '__main__':
    main()


