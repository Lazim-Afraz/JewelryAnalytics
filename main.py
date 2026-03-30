"""
Jewelry Portfolio Analytics — Main Entry Point
main.py

Launches the Streamlit dashboard (app.py).
The old Tkinter GUI (gui/main_window.py) is preserved but no longer
the default entry point.

Usage:
    python main.py
"""

import os
import sys
import logging
import subprocess
import webbrowser
import time
from pathlib import Path

# ── Project root on path ───────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "jewelry_analytics.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
APP_FILE   = ROOT / "app.py"
HOST       = "localhost"
PORT       = 8501
URL        = f"http://{HOST}:{PORT}"
BROWSER_DELAY = 2          # seconds to wait before opening browser


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_banner():
    """Print startup banner to terminal."""
    print()
    print("=" * 60)
    print("  💎  Jewelry Portfolio Analytics")
    print("       Streamlit Dashboard")
    print("=" * 60)
    print(f"  URL  : {URL}")
    print(f"  App  : {APP_FILE.name}")
    print("  Stop : Ctrl + C")
    print("=" * 60)
    print()


def check_dependencies() -> bool:
    """
    Verify all required packages are importable.
    Returns True if all present, False if any missing.
    """
    required = {
        "streamlit":   "streamlit",
        "pandas":      "pandas",
        "numpy":       "numpy",
        "sklearn":     "scikit-learn",
        "plotly":      "plotly",
        "pyodbc":      "pyodbc",
        "openpyxl":    "openpyxl",
    }

    missing = []
    for import_name, pip_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print("❌  Missing required packages:")
        for pkg in missing:
            print(f"    pip install {pkg}")
        print()
        print("Install all at once:")
        print("    pip install -r requirements.txt")
        print()
        return False

    return True


def find_streamlit() -> str:
    """
    Return the correct streamlit command for this system.
    Prefers 'python -m streamlit' for reliability across environments.
    """
    return f"{sys.executable} -m streamlit"


def open_browser_delayed(url: str, delay: float = BROWSER_DELAY):
    """Open the browser after a short delay (gives server time to start)."""
    time.sleep(delay)
    try:
        webbrowser.open(url)
        logger.info(f"Browser opened: {url}")
    except Exception as e:
        logger.warning(f"Could not open browser automatically: {e}")
        print(f"  Open manually: {url}")


# ══════════════════════════════════════════════════════════════════════════════
# Main launcher
# ══════════════════════════════════════════════════════════════════════════════

def launch_streamlit():
    """
    Launch the Streamlit dashboard as a subprocess.
    Blocks until the user presses Ctrl+C.
    """
    if not APP_FILE.exists():
        print(f"❌  app.py not found at: {APP_FILE}")
        print("    Make sure app.py is in the project root.")
        sys.exit(1)

    streamlit_cmd = find_streamlit()

    cmd = [
        *streamlit_cmd.split(),
        "run", str(APP_FILE),
        "--server.port",          str(PORT),
        "--server.headless",      "true",
        "--server.enableCORS",    "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false",
    ]

    logger.info(f"Launching: {' '.join(cmd)}")

    try:
        # Open browser in background thread after short delay
        import threading
        t = threading.Thread(
            target=open_browser_delayed,
            args=(URL, BROWSER_DELAY),
            daemon=True,
        )
        t.start()

        # Run Streamlit (blocking)
        process = subprocess.run(cmd, cwd=str(ROOT))
        return process.returncode

    except KeyboardInterrupt:
        print("\n\n  Shutting down... Goodbye 👋")
        logger.info("Application stopped by user.")
        return 0

    except FileNotFoundError:
        print("❌  Streamlit executable not found.")
        print("    Install it with:  pip install streamlit")
        return 1

    except Exception as e:
        logger.error(f"Launch failed: {e}", exc_info=True)
        print(f"❌  Launch failed: {e}")
        return 1


# ══════════════════════════════════════════════════════════════════════════════
# Console fallback (preserved from original main.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_console_mode():
    """
    Console mode — runs a full analytics pipeline without any GUI.
    Useful for debugging or when Streamlit is not available.
    """
    print("\n" + "=" * 60)
    print("  CONSOLE MODE — Full Analytics Workflow")
    print("=" * 60 + "\n")

    try:
        from config.database_config import DatabaseConfig
        from config.app_settings    import AppSettings
        from data_layer.sql_connector import SQLServerConnector
        from data_layer.data_loader   import JewelryDataLoader
        from analytics.performance_metrics import PerformanceAnalyzer
        from analytics.clustering_engine   import BranchClusterer

        print("1. Connecting to SQL Server...")
        connector = SQLServerConnector(
            server           = DatabaseConfig.SERVER,
            database         = DatabaseConfig.DATABASE,
            username         = DatabaseConfig.USERNAME,
            password         = DatabaseConfig.PASSWORD,
            use_windows_auth = DatabaseConfig.USE_WINDOWS_AUTH,
        )
        test = connector.test_connection()
        if not test["success"]:
            print(f"   ❌ Connection failed: {test['message']}")
            print("   Please check config/database_config.py")
            return

        print(f"   ✅ {test['message']}")

        print("\n2. Loading data...")
        loader = JewelryDataLoader(connector)
        df     = loader.load_transaction_data()
        df     = loader.preprocess_data(df)
        print(f"   ✅ {len(df):,} rows loaded")

        summary = loader.get_data_summary(df)
        print(f"   Branches : {summary['total_branches']}")
        print(f"   Sales    : {summary['total_sales']:,}")
        print(f"   Stock    : {summary['total_stock']:,}")

        print("\n3. Performance metrics...")
        analyzer     = PerformanceAnalyzer(df)
        metrics_df   = analyzer.calculate_all_metrics()
        branch_sum   = analyzer.aggregate_by_branch()
        heroes       = analyzer.identify_local_heroes()
        print(f"   ✅ {len(heroes)} local heroes found")

        print("\n4. Clustering...")
        clusterer = BranchClusterer(metrics_df)
        X, _      = clusterer.prepare_features()
        kv, ins, ss = clusterer.find_optimal_clusters(
            X, max_clusters=AppSettings.MAX_CLUSTERS_TO_TEST
        )
        k = clusterer.suggest_optimal_k(kv, ins, ss)
        clusterer.fit_kmeans(X, n_clusters=k)
        print(f"   ✅ {k} clusters")

        print("\n5. Summary:")
        stats = analyzer.get_summary_stats()
        print(f"   Total Branches  : {stats['total_branches']}")
        print(f"   Total Sales     : {stats['total_sales']:,}")
        print(f"   Efficiency      : {stats['overall_efficiency']:.2f}")
        print(f"   Local Heroes    : {stats['total_local_heroes']}")

        print(f"\n   Top 5 branches:")
        top = branch_sum.nlargest(5, "SALE_COUNT")
        for i, (_, row) in enumerate(top.iterrows(), 1):
            print(f"   {i}. {row['BRANCHNAME']}: {row['SALE_COUNT']:,} sales")

        print("\n" + "=" * 60)
        print("  ✅  CONSOLE MODE COMPLETE")
        print("=" * 60)
        connector.close()

    except Exception as e:
        print(f"\n❌ Console mode error: {e}")
        logger.error("Console mode error", exc_info=True)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("Jewelry Portfolio Analytics — Starting")
    logger.info("=" * 60)

    # Dependency check
    if not check_dependencies():
        sys.exit(1)

    print_banner()

    # Check app.py exists
    if not APP_FILE.exists():
        print(f"❌  app.py not found. Run from the project root directory.")
        sys.exit(1)

    # Launch Streamlit
    exit_code = launch_streamlit()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
