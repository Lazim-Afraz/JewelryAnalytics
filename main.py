"""
Jewelry Portfolio Analytics — Main Entry Point

Usage:
    python main.py
"""

import sys
import logging
import subprocess
import webbrowser
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "jewelry_analytics.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

APP_FILE = ROOT / "app.py"
HOST = "localhost"
PORT = 8501
URL = f"http://{HOST}:{PORT}"
BROWSER_DELAY = 2


def print_banner():
    print("\n" + "=" * 60)
    print("  Jewelry Portfolio Analytics")
    print("  Streamlit Dashboard")
    print("=" * 60)
    print(f"  URL  : {URL}")
    print(f"  App  : {APP_FILE.name}")
    print("  Stop : Ctrl + C")
    print("=" * 60 + "\n")


def check_dependencies():
    required = {
        "streamlit": "streamlit",
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "plotly": "plotly",
        "pyodbc": "pyodbc",
        "openpyxl": "openpyxl",
    }

    missing = []
    for module, pkg in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  pip install {pkg}")
        print("\nOr run:\n  pip install -r requirements.txt\n")
        return False

    return True


def find_streamlit():
    return f"{sys.executable} -m streamlit"


def open_browser(url):
    time.sleep(BROWSER_DELAY)
    try:
        webbrowser.open(url)
        logger.info(f"Opened browser at {url}")
    except Exception:
        print(f"Open manually: {url}")


def launch_streamlit():
    if not APP_FILE.exists():
        print(f"app.py not found at {APP_FILE}")
        sys.exit(1)

    cmd = [
        *find_streamlit().split(),
        "run", str(APP_FILE),
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]

    logger.info(f"Launching: {' '.join(cmd)}")

    try:
        import threading
        threading.Thread(target=open_browser, args=(URL,), daemon=True).start()
        subprocess.run(cmd, cwd=str(ROOT))

    except KeyboardInterrupt:
        print("\nShutting down...")

    except Exception as e:
        logger.error(f"Error: {e}")


def main():
    logger.info("Starting Jewelry Analytics Application")

    if not check_dependencies():
        sys.exit(1)

    print_banner()
    launch_streamlit()


if __name__ == "__main__":
    main()
