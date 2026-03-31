# 💎 Jewelry Analytics Intelligence Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jewelryanalytics-klfl7mpuvmffuze2iozbwy.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> **🚀 Live Demo → [jewelryanalytics-klfl7mpuvmffuze2iozbwy.streamlit.app](https://jewelryanalytics-klfl7mpuvmffuze2iozbwy.streamlit.app/)**

An end-to-end business analytics and machine learning platform for jewelry retail portfolio management. Built to turn raw branch stock and sales data into actionable decisions — which products to stock, which branches to prioritise, and where demand is being missed.

---

## Features

### 📊 Portfolio Dashboard
Real-time KPIs across all branches — total sales, stock levels, sell-through rate, efficiency ratios, top branch, and top region at a glance.

### 🏪 Branch & Cluster Analysis
K-Means clustering automatically segments branches by performance profile. PCA scatter plots reveal how branches group spatially. Cluster profiles identify Elite, Strong, Average, and Underperforming tiers.

### 🌍 Regional Performance
Attribute breakdowns (purity, finish, theme, shape, workstyle, brand) by region — as grouped bar charts and interactive heatmaps. Instantly see which product types dominate in North vs South vs East vs West.

### 🔮 Sales Prediction
A Random Forest model (v2) trained on your data predicts `SALE_COUNT` for any branch + attribute combination. Includes confidence range from individual tree predictions and feature importance charts.

### 💡 Smart Recommendations
Attribute-level intelligence per branch or region — top purities, finishes, themes, and combos by sell-through rate. Forward-looking suggestions identify untried or understocked combinations predicted to perform well.

### 📄 PDF Export
Branded analytics report covering KPI summary, top branch rankings, and cluster profiles — downloadable directly from the app.

### 🥇 Live Gold Price
Real-time 24K gold price in INR with a 7-day mini chart, fetched via `yfinance` and cached hourly.

---

## Try It Instantly

A sample dataset is included in the repo: **`jewelry_demo_data.csv`**

1. Open the [live app](https://jewelryanalytics-klfl7mpuvmffuze2iozbwy.streamlit.app/)
2. In the sidebar, select **📂 CSV Upload**
3. Upload `jewelry_demo_data.csv`
4. All pages unlock immediately — no account or database needed

| Column | Description |
|---|---|
| `REGION` | North / South / East / West |
| `BRANCHNAME` | Branch identifier (e.g. `BR-Mumbai-01`) |
| `ITEMID` | Product SKU |
| `PURITY` | 14 / 18 / 22 |
| `FINISH` | Textured / HighPolish / TwoTone / Matte |
| `SHAPE` | Princess / Oval / Mixed / Round |
| `THEME` | Floral / Geometric / Minimalist / Traditional / Contemporary |
| `BRAND` | Essentials / House / Signature |
| `WORKSTYLE` | Filigree / Handcrafted / KundanInspired / Cast |
| `STOCK_COUNT` | Units in stock |
| `SALE_COUNT` | Units sold |

The demo covers **25 branches** across **4 regions** with **3,385 rows**.

---

## Run Locally

```bash
git clone https://github.com/Lazim-Afraz/JewelryAnalytics.git
cd JewelryAnalytics
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), switch to **📂 CSV Upload** in the sidebar, and upload `jewelry_demo_data.csv`.

For local SQL Server use, configure `config/database_config.py` with your connection details and use the **🗄️ Database** option.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Dashboard | Streamlit, Plotly |
| Machine Learning | scikit-learn — Random Forest, K-Means, PCA |
| Data | pandas, numpy |
| Market Data | yfinance (live gold price) |
| Reports | ReportLab (PDF export) |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
JewelryAnalytics/
├── app.py                         # Streamlit app — all pages and UI
├── requirements.txt               # Python dependencies
├── jewelry_demo_data.csv          # Sample dataset for instant demo
│
├── services/
│   └── analytics_service.py      # Central orchestrator for all analytics
│
├── analytics/
│   ├── performance_metrics.py    # Efficiency ratios, sell-through, local heroes
│   ├── clustering_engine.py      # K-Means, PCA, cluster profiling
│   └── prediction_model_v2.py    # Random Forest sales predictor (v2)
│
├── config/
│   ├── app_settings.py           # App-wide settings
│   └── database_config.py        # SQL Server config (local use only)
│
├── data_layer/
│   ├── sql_connector.py          # SQL Server connector (local use only)
│   └── data_loader.py            # Data loading and preprocessing
│
└── exports/
    └── report_generator.py       # PDF report builder
```

---

## How the ML Works

**Sales Prediction (Random Forest v2)**
- Aggregates raw rows to branch + attribute level before training — eliminates zero-sale noise
- One-hot encoding for all categorical features — no false ordinal relationships
- Engineered features: `log_stock`, `branch_avg_sales`, `attr_avg_sales` priors
- 300 trees, cross-validated, confidence range derived from individual tree percentiles
- Strict feature alignment between training and inference

**Branch Clustering (K-Means)**
- Features: efficiency ratio, sell-through rate, sales contribution %, total sales, total stock
- Optimal k selected by silhouette score across k=2 to k=10
- PCA projection to 2D for scatter visualisation
- Cluster tiers: Elite / Strong / Average / Underperforming

---

## Deployment Note

The SQL Server database option is disabled in the cloud deployment as it requires a local SSMS connection. All analytics, ML, clustering, prediction, and recommendation features work fully with CSV upload.

---

## Author

**Lazim Afraz**  
[linkedin.com/in/lazim-afraz-155878246](https://linkedin.com/in/lazim-afraz-155878246)  
IEEE Access Publication · Indian Patent Filed (2026)
