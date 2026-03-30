# 💎 JewelryAnalytics

A full-stack jewelry sales analytics platform built with Python — featuring clustering, sales prediction, recommendations, an interactive Streamlit dashboard, and a local AI chatbot.

---

## Features

- **Sales Analytics** — branch performance metrics, sell-through rates, efficiency scores
- **Cluster Analysis** — K-Means clustering of branches into performance tiers
- **Sales Prediction** — ML model predicts sales count from product attributes
- **Recommendations** — best-performing attribute combos per branch or region
- **Regional Performance** — heatmaps and breakdowns by region × attribute
- **PDF Export** — one-click report generation with KPIs, clusters, top branches
- **AI Chatbot** — local LLM assistant via Ollama (Mistral) with live data context

---

## Project Structure

```
JewelryAnalytics/
├── analytics/
│   ├── clustering_engine.py       # K-Means branch clustering
│   ├── performance_metrics.py     # KPI calculations
│   └── prediction_model.py        # Sales prediction ML model
├── chatbot/
│   ├── __init__.py
│   └── assistant.py               # Ollama LLM chatbot
├── exports/
│   ├── __init__.py
│   └── report_generator.py        # PDF report generation
├── services/
│   └── analytics_service.py       # Central data + analytics orchestrator
├── app.py                         # Streamlit dashboard (main UI)
├── main.py                        # Entry point launcher
└── requirements.txt
```

---

## Installation

**1. Clone the repo**
```bash
git clone https://github.com/Lazim-Afraz/JewelryAnalytics.git
cd JewelryAnalytics
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure your database**

Create a `.env` file in the project root:
```
DB_SERVER=your_server
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password
```

**4. (Optional) Install Ollama for the AI chatbot**

Download from https://ollama.com/download, then:
```bash
ollama pull mistral
```

---

## Running the App

```bash
python main.py
```

Or directly:
```bash
python -m streamlit run app.py
```

Opens at **http://localhost:8501**

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 Dashboard | KPI summary cards, top branch & region |
| 🏪 Branch & Clusters | PCA scatter, cluster profiles, top-N branches |
| 🔮 Sales Prediction | Predict sales by product attributes |
| 💡 Recommendations | Best combos per branch or region |
| 🌍 Regional Performance | Region × attribute breakdown |
| 📄 Export Report | Generate & download PDF report |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Dashboard | Streamlit, Plotly |
| ML / Analytics | scikit-learn, pandas, numpy |
| Database | SQL Server via pyodbc |
| PDF Export | ReportLab |
| AI Chatbot | Ollama (Mistral) |
| GUI (legacy) | PySide6 |
