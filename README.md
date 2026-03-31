# 💎 Jewelry Analytics Intelligence Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end business analytics platform for jewelry retail — combining machine learning, interactive dashboards, and demand forecasting to drive data-informed inventory and sales decisions.

> **Live demo →** [your-app-name.streamlit.app](https://your-app-name.streamlit.app)  
> *(Replace this link after deploying to Streamlit Cloud)*

---

## Features

- **Branch Performance Analytics** — efficiency ratios, revenue contribution, and local hero identification
- **K-Means Clustering** — segment branches by sales behaviour and stock patterns
- **Random Forest Forecasting** — predict revenue and surface top demand drivers
- **Interactive Dashboards** — Plotly charts with filters, drill-downs, and export
- **CSV Upload Mode** — plug in your own data instantly; no database setup needed

---

## Screenshots

> *(Add screenshots here — drag images into the GitHub editor)*

---

## Quick Start

```bash
git clone https://github.com/Lazim-Afraz/JewelryAnalytics.git
cd JewelryAnalytics
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## CSV Format

To use your own data, upload a CSV with these columns:

| Column | Type | Description |
|---|---|---|
| `BRANCH` | string | Branch name or ID |
| `ITEMTYPE` | string | e.g. Ring, Necklace, Bracelet |
| `PURITY` | string | e.g. 18K, 22K, 24K |
| `THEME` | string | e.g. Bridal, Classic, Modern |
| `SALE_COUNT` | int | Units sold |
| `STOCK_COUNT` | int | Units in stock |
| `REVENUE` | float | Revenue for that row |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Dashboard | Streamlit, Plotly |
| ML | scikit-learn (Random Forest, K-Means) |
| Data | pandas, numpy |
| Deployment | Streamlit Community Cloud |

---

## Deploying to Streamlit Cloud (free)

1. Fork or push this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select this repo → set main file to `app.py`
4. Click **Deploy** — live in ~2 minutes

---

## Project Structure

```
JewelryAnalytics/
├── app.py                  # Streamlit web app (main entry point)
├── requirements.txt        # Python dependencies
├── analytics/
│   ├── performance_metrics.py
│   └── clustering_engine.py
├── data_layer/
│   ├── sql_connector.py    # Optional: local SQL Server mode
│   └── data_loader.py
└── config/
    └── database_config.py
```

---

## Author

**Lazim Afraz**  
[linkedin.com/in/lazim-afraz-155878246](https://linkedin.com/in/lazim-afraz-155878246)  
IEEE Access Publication · Indian Patent Filed (2026)
