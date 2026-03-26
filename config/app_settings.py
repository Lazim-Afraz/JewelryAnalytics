"""
Application Settings and Configuration

User-configurable settings for the desktop application.
"""

import json
import os
from pathlib import Path

class AppSettings:
    """Application settings and preferences"""
    
    # ===== APPLICATION INFO =====
    APP_NAME = "Jewelry Portfolio Analytics"
    APP_VERSION = "1.0.0"
    APP_AUTHOR = "Your Company"
    
    # ===== PATHS =====
    BASE_DIR = Path(__file__).parent.parent
    CONFIG_DIR = BASE_DIR / 'config'
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'
    EXPORTS_DIR = BASE_DIR / 'exports'
    REPORTS_DIR = EXPORTS_DIR / 'reports'
    VISUALIZATIONS_DIR = EXPORTS_DIR / 'visualizations'
    
    # ===== ANALYSIS DEFAULTS =====
    
    # K-Means Clustering
    DEFAULT_N_CLUSTERS = 5
    MAX_CLUSTERS_TO_TEST = 10
    KMEANS_RANDOM_STATE = 42
    KMEANS_N_INIT = 10
    
    # PCA
    PCA_N_COMPONENTS = 2
    PCA_RANDOM_STATE = 42
    
    # Prediction Model
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10
    RF_RANDOM_STATE = 42
    RF_TEST_SIZE = 0.2
    
    # Local Heroes Threshold
    LOCAL_HERO_RELATIVE_STRENGTH = 1.2
    LOCAL_HERO_MIN_CONTRIBUTION = 5.0  # percentage
    
    # ===== UI SETTINGS =====
    
    # Window Size
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 900
    
    # Colors (matching your Jupyter plots)
    BRAND_COLOR = '#366092'
    SUCCESS_COLOR = '#28a745'
    WARNING_COLOR = '#ffc107'
    DANGER_COLOR = '#dc3545'
    INFO_COLOR = '#17a2b8'
    
    # Chart Settings
    CHART_DPI = 300
    CHART_STYLE = 'seaborn-v0_8-darkgrid'
    
    # ===== EXPORT SETTINGS =====
    
    # Excel Export
    EXCEL_ENGINE = 'openpyxl'
    
    # Report Naming
    REPORT_DATE_FORMAT = '%Y_%m_%d'
    REPORT_PREFIX = 'jewelry_analytics'
    
    # ===== PERFORMANCE =====
    
    # Data Loading
    CHUNK_SIZE = 100000  # Rows per chunk for large datasets
    MAX_ROWS_IN_MEMORY = 5000000  # 5 million rows
    
    # Progress Updates
    SHOW_PROGRESS_BARS = True
    
    # ===== FEATURE FLAGS =====
    
    ENABLE_ADVANCED_FILTERS = True
    ENABLE_PREDICTIONS = True
    ENABLE_COMPARISON_TOOL = True
    ENABLE_AUTO_SAVE = True
    
    # ===== METHODS =====
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.EXPORTS_DIR,
            cls.REPORTS_DIR,
            cls.VISUALIZATIONS_DIR
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def save_settings(cls, filepath: str = None):
        """Save current settings to JSON file"""
        if filepath is None:
            filepath = cls.CONFIG_DIR / 'app_settings.json'
        
        settings = {
            'n_clusters': cls.DEFAULT_N_CLUSTERS,
            'window_width': cls.WINDOW_WIDTH,
            'window_height': cls.WINDOW_HEIGHT,
            'chart_dpi': cls.CHART_DPI,
            'show_progress': cls.SHOW_PROGRESS_BARS
        }
        
        with open(filepath, 'w') as f:
            json.dump(settings, f, indent=4)
    
    @classmethod
    def load_settings(cls, filepath: str = None):
        """Load settings from JSON file"""
        if filepath is None:
            filepath = cls.CONFIG_DIR / 'app_settings.json'
        
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            settings = json.load(f)
        
        cls.DEFAULT_N_CLUSTERS = settings.get('n_clusters', cls.DEFAULT_N_CLUSTERS)
        cls.WINDOW_WIDTH = settings.get('window_width', cls.WINDOW_WIDTH)
        cls.WINDOW_HEIGHT = settings.get('window_height', cls.WINDOW_HEIGHT)
        cls.CHART_DPI = settings.get('chart_dpi', cls.CHART_DPI)
        cls.SHOW_PROGRESS_BARS = settings.get('show_progress', cls.SHOW_PROGRESS_BARS)


# Initialize on import
AppSettings.ensure_directories()

if __name__ == '__main__':
    print("=== Application Settings ===")
    print(f"App Name: {AppSettings.APP_NAME}")
    print(f"Version: {AppSettings.APP_VERSION}")
    print(f"\nDirectories:")
    print(f"Base: {AppSettings.BASE_DIR}")
    print(f"Data: {AppSettings.DATA_DIR}")
    print(f"Exports: {AppSettings.EXPORTS_DIR}")
    print(f"\nClustering: {AppSettings.DEFAULT_N_CLUSTERS} clusters")
    print(f"Random Forest: {AppSettings.RF_N_ESTIMATORS} estimators")
