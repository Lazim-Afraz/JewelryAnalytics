"""
Jewelry Portfolio Analytics — Streamlit Dashboard
app.py  |  Entry point

Run:
    streamlit run app.py

Theme: Gold & Dark  (luxury jewelry aesthetic)
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path ───────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# THEME & GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════════════════

GOLD        = "#C9A84C"
GOLD_LIGHT  = "#E8C96A"
GOLD_PALE   = "#F5E6B8"
DARK_BG     = "#0D0D14"
CARD_BG     = "#13131F"
CARD_BORDER = "#2A2A3E"
TEXT_PRIMARY = "#F0E6CC"
TEXT_MUTED   = "#8A8AAA"
ACCENT_RED   = "#C94C4C"
ACCENT_TEAL  = "#4CC9A8"

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_PRIMARY, family="Georgia, serif"),
        title_font=dict(color=GOLD, family="Georgia, serif", size=16),
        xaxis=dict(gridcolor="#1E1E30", linecolor=CARD_BORDER, tickcolor=TEXT_MUTED),
        yaxis=dict(gridcolor="#1E1E30", linecolor=CARD_BORDER, tickcolor=TEXT_MUTED),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=CARD_BORDER),
        colorway=[GOLD, ACCENT_TEAL, "#9B7FD4", ACCENT_RED,
                  "#4C8EC9", "#C97A4C", "#7FC98B", "#C94C8E"],
        margin=dict(l=16, r=16, t=40, b=16),
    )
)

CLUSTER_COLORS = [GOLD, ACCENT_TEAL, "#9B7FD4", ACCENT_RED,
                  "#4C8EC9", "#C97A4C", "#7FC98B"]

st.set_page_config(
    page_title="Jewelry Analytics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  html, body, [class*="css"] {{
      background-color: {DARK_BG};
      color: {TEXT_PRIMARY};
      font-family: 'Segoe UI', sans-serif;
  }}

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {{
      background: {CARD_BG} !important;
      border-right: 1px solid {CARD_BORDER};
  }}
  section[data-testid="stSidebar"] * {{ color: {TEXT_PRIMARY} !important; }}

  /* ── Metric cards ── */
  div[data-testid="metric-container"] {{
      background: {CARD_BG};
      border: 1px solid {CARD_BORDER};
      border-radius: 8px;
      padding: 16px 20px;
  }}
  div[data-testid="metric-container"] label {{
      color: {TEXT_MUTED} !important;
      font-family: 'Segoe UI', sans-serif !important;
      font-size: 0.72rem !important;
      letter-spacing: 0.1em;
      text-transform: uppercase;
  }}
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
      color: {GOLD} !important;
      font-family: 'Georgia', serif !important;
      font-size: 1.9rem !important;
      font-weight: 600;
  }}

  /* ── Page title ── */
  .page-title {{
      font-family: 'Georgia', serif;
      font-size: 2.1rem;
      font-weight: 700;
      color: {GOLD};
      letter-spacing: 0.03em;
      margin-bottom: 4px;
  }}
  .page-subtitle {{
      font-family: 'Segoe UI', sans-serif;
      font-size: 0.82rem;
      color: {TEXT_MUTED};
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 24px;
  }}

  /* ── Section header ── */
  .section-header {{
      font-family: 'Georgia', serif;
      font-size: 1.25rem;
      font-weight: 600;
      color: {GOLD_LIGHT};
      border-bottom: 1px solid {CARD_BORDER};
      padding-bottom: 6px;
      margin: 24px 0 12px 0;
  }}

  /* ── Gold divider ── */
  .gold-divider {{
      border: none;
      border-top: 1px solid {CARD_BORDER};
      margin: 20px 0;
  }}

  /* ── Info / warning boxes ── */
  .info-box {{
      background: {CARD_BG};
      border: 1px solid {CARD_BORDER};
      border-left: 3px solid {GOLD};
      border-radius: 6px;
      padding: 14px 18px;
      font-size: 0.88rem;
      color: {TEXT_MUTED};
      margin: 12px 0;
  }}

  /* ── Prediction result card ── */
  .prediction-card {{
      background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
      border: 1px solid {GOLD};
      border-radius: 10px;
      padding: 24px 28px;
      text-align: center;
  }}
  .prediction-number {{
      font-family: 'Georgia', serif;
      font-size: 3.2rem;
      font-weight: 700;
      color: {GOLD};
      line-height: 1;
  }}
  .prediction-label {{
      font-family: 'Segoe UI', sans-serif;
      font-size: 0.78rem;
      letter-spacing: 0.12em;
      color: {TEXT_MUTED};
      text-transform: uppercase;
      margin-top: 6px;
  }}
  .confidence-range {{
      font-family: 'Segoe UI', sans-serif;
      font-size: 1rem;
      color: {GOLD_PALE};
      margin-top: 10px;
  }}

  /* ── Tier badges ── */
  .tier-elite       {{ color: {GOLD};        font-weight: 500; }}
  .tier-strong      {{ color: {ACCENT_TEAL}; font-weight: 500; }}
  .tier-average     {{ color: {TEXT_MUTED};  font-weight: 500; }}
  .tier-underperform{{ color: {ACCENT_RED};  font-weight: 500; }}

  /* ── Buttons ── */
  .stButton > button {{
      background: linear-gradient(135deg, {GOLD} 0%, {GOLD_LIGHT} 100%);
      color: {DARK_BG};
      border: none;
      font-family: 'Segoe UI', sans-serif;
      font-weight: 500;
      letter-spacing: 0.06em;
      border-radius: 4px;
  }}
  .stButton > button:hover {{
      background: {GOLD_LIGHT};
      color: {DARK_BG};
  }}

  /* ── Selectbox / inputs ── */
  .stSelectbox label, .stNumberInput label, .stSlider label {{
      color: {TEXT_MUTED} !important;
      font-size: 0.78rem !important;
      letter-spacing: 0.06em;
      text-transform: uppercase;
  }}

  /* ── Dataframe ── */
  .stDataFrame {{ border: 1px solid {CARD_BORDER}; border-radius: 6px; }}

  /* ── Sidebar logo area ── */
  .sidebar-logo {{
      font-family: 'Georgia', serif;
      font-size: 1.5rem;
      font-weight: 700;
      color: {GOLD};
      text-align: center;
      padding: 8px 0 4px 0;
      letter-spacing: 0.05em;
  }}
  .sidebar-tagline {{
      font-size: 0.68rem;
      color: {TEXT_MUTED};
      text-align: center;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      margin-bottom: 16px;
  }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _not_loaded_msg():
    st.markdown(
        '<div class="info-box">💎 &nbsp;Connect to the database using the '
        '<b>Load Data</b> button in the sidebar to unlock this view.</div>',
        unsafe_allow_html=True,
    )


def _apply_plotly_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    return fig


def fmt_number(n) -> str:
    if n is None:
        return "—"
    try:
        n = float(n)
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n/1_000:.1f}K"
        return f"{n:,.0f}"
    except Exception:
        return str(n)


def tier_html(tier: str) -> str:
    cls = {
        "Elite":          "tier-elite",
        "Strong":         "tier-strong",
        "Average":        "tier-average",
        "Underperforming":"tier-underperform",
    }.get(tier, "")
    return f'<span class="{cls}">{tier}</span>'


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE & SERVICE
# ══════════════════════════════════════════════════════════════════════════════

def _get_service():
    """Lazily initialise AnalyticsService in session state."""
    if "svc" not in st.session_state:
        from services.analytics_service import AnalyticsService
        st.session_state["svc"] = AnalyticsService()
    return st.session_state["svc"]


def _is_loaded() -> bool:
    return st.session_state.get("data_loaded", False)



# ══════════════════════════════════════════════════════════════════════════════
# GOLD PRICE WIDGET
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)   # cache 1 hour
def _fetch_gold_data():
    """
    Fetch 24K gold price in INR and 7-day history using yfinance.
    Returns dict with price, change, change_pct, history_df.
    Falls back gracefully on failure.
    """
    try:
        import yfinance as yf

        # Gold futures (USD/troy oz) + USD/INR rate
        gold  = yf.Ticker("GC=F")
        inr   = yf.Ticker("INR=X")

        gold_hist = gold.history(period="10d", interval="1d")
        inr_hist  = inr.history(period="1d",  interval="1d")

        if gold_hist.empty:
            return {"error": "No gold data"}

        # Latest USD/INR rate
        usd_inr = float(inr_hist["Close"].iloc[-1]) if not inr_hist.empty else 83.5

        # Troy oz → gram  (1 troy oz = 31.1035 g)
        oz_to_g = 31.1035

        # Build 7-day history in INR per gram (24K)
        hist = gold_hist.tail(7).copy()
        hist["price_inr_g"] = (hist["Close"] / oz_to_g) * usd_inr

        # Current & previous close
        current_price = float(hist["price_inr_g"].iloc[-1])
        prev_price    = float(hist["price_inr_g"].iloc[-2]) if len(hist) >= 2 else current_price
        change        = current_price - prev_price
        change_pct    = (change / prev_price * 100) if prev_price else 0

        return {
            "price":      round(current_price, 2),
            "change":     round(change, 2),
            "change_pct": round(change_pct, 2),
            "usd_inr":    round(usd_inr, 2),
            "history":    hist[["price_inr_g"]].reset_index(),
            "error":      None,
        }

    except ImportError:
        return {"error": "yfinance not installed. Run: pip install yfinance"}
    except Exception as e:
        return {"error": str(e)}


def render_gold_price():
    """Render gold price widget in the sidebar."""
    st.divider()
    st.markdown(
        f'<div style="font-size:0.72rem;color:{TEXT_MUTED};'
        f'letter-spacing:0.1em;text-transform:uppercase;'
        f'margin-bottom:6px;">🥇 24K Gold · India</div>',
        unsafe_allow_html=True,
    )

    data = _fetch_gold_data()

    if data.get("error"):
        st.markdown(
            f'<div style="font-size:0.78rem;color:{TEXT_MUTED};">'
            f'Price unavailable<br><span style="font-size:0.68rem;">'
            f'{data["error"]}</span></div>',
            unsafe_allow_html=True,
        )
        return

    price      = data["price"]
    change     = data["change"]
    change_pct = data["change_pct"]
    arrow      = "▲" if change >= 0 else "▼"
    color      = ACCENT_TEAL if change >= 0 else ACCENT_RED

    st.markdown(
        f'<div style="background:{CARD_BG};border:1px solid {CARD_BORDER};'
        f'border-left:3px solid {GOLD};border-radius:6px;padding:10px 12px;">'
        f'<div style="font-family:Georgia,serif;font-size:1.4rem;'
        f'font-weight:700;color:{GOLD};">'
        f'₹{price:,.0f}<span style="font-size:0.7rem;color:{TEXT_MUTED};'
        f'font-weight:400;margin-left:4px;">/gram</span></div>'
        f'<div style="font-size:0.78rem;color:{color};margin-top:3px;">'
        f'{arrow} ₹{abs(change):,.0f} ({change_pct:+.2f}%)</div>'
        f'<div style="font-size:0.65rem;color:{TEXT_MUTED};margin-top:2px;">'
        f'USD/INR: {data["usd_inr"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 7-day mini chart
    hist = data.get("history")
    if hist is not None and not hist.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x    = list(range(len(hist))),
            y    = hist["price_inr_g"].tolist(),
            mode = "lines+markers",
            line = dict(color=GOLD, width=2),
            marker = dict(size=4, color=GOLD),
            fill = "tozeroy",
            fillcolor = "rgba(201,168,76,0.08)",
            hovertemplate = "₹%{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(
            height      = 90,
            margin      = dict(l=0, r=0, t=0, b=0),
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            xaxis = dict(visible=False),
            yaxis = dict(visible=False),
            showlegend = False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        f'<div style="font-size:0.62rem;color:{TEXT_MUTED};'
        f'text-align:right;margin-top:-8px;">7-day · refreshes hourly</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">💎 Jewelry Analytics</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sidebar-tagline">Portfolio Intelligence Platform</div>',
                    unsafe_allow_html=True)
        st.divider()

        # Navigation
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = "📊 Dashboard"

        def _nav_btn(label, key):
            active = st.session_state["current_page"] == label
            style = f"background:{'#2A2A1A' if active else 'transparent'};border:{'1px solid '+GOLD if active else 'none'};color:{''+GOLD if active else TEXT_PRIMARY};border-radius:4px;padding:5px 10px;width:100%;text-align:left;cursor:pointer;font-size:0.88rem;"
            if st.button(label, key=key, use_container_width=True):
                st.session_state["current_page"] = label
                st.rerun()

        with st.expander("📊 Analytics", expanded=True):
            _nav_btn("📊 Dashboard",           "nav_dash")
            _nav_btn("🏪 Branch & Clusters",   "nav_cluster")
            _nav_btn("🌍 Regional Performance","nav_regional")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        _nav_btn("🔮 Sales Prediction",  "nav_pred")
        _nav_btn("💡 Recommendations",   "nav_recs")

        with st.expander("📁 Reports", expanded=False):
            _nav_btn("📄 Export Report", "nav_export")

        page = st.session_state["current_page"]

        st.divider()

        # Data Source
        st.markdown("**⚙️ Data Source**")

        data_src = st.radio(
            "Source",
            ["🗄️ Database", "📂 CSV Upload"],
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state["data_source"] = data_src

        if data_src == "🗄️ Database":
            if st.button("🔄 Load from DB", use_container_width=True):
                svc = _get_service()
                with st.spinner("Connecting to database…"):
                    result = svc.load_data()
                if result["success"]:
                    st.session_state["data_loaded"] = True
                    st.session_state["data_source_label"] = "DB"
                    st.session_state["filters"] = svc.get_available_filters()
                    st.success(
                        f"✅ {fmt_number(result['rows_loaded'])} rows "
                        f"· {result['branches']} branches"
                    )
                else:
                    st.session_state["data_loaded"] = False
                    st.error(f"❌ {result['message']}")

        else:
            uploaded = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                label_visibility="collapsed",
                help="Same columns as DB: REGION, BRANCHNAME, PURITY, FINISH, THEME, SHAPE, WORKSTYLE, BRAND, SALE_COUNT, STOCK_COUNT",
            )
            if uploaded is not None:
                try:
                    with st.spinner("Loading CSV…"):
                        import pandas as pd
                        csv_df = pd.read_csv(uploaded)

                        # Normalise column names
                        csv_df.columns = [c.strip().upper() for c in csv_df.columns]

                        required = ["SALE_COUNT", "STOCK_COUNT"]
                        missing  = [c for c in required if c not in csv_df.columns]
                        if missing:
                            st.error(f"❌ Missing columns: {missing}")
                        else:
                            svc = _get_service()
                            # Wire CSV directly into service
                            from analytics.performance_metrics import PerformanceAnalyzer
                            from analytics.clustering_engine   import BranchClusterer
                            from analytics.prediction_model_v2 import SalesPredictionModelV2
                            import numpy as np

                            svc._df           = csv_df
                            svc._analyzer     = PerformanceAnalyzer(csv_df)
                            svc._metrics_df   = svc._analyzer.calculate_all_metrics()
                            svc._branch_summary = svc._analyzer.aggregate_by_branch()
                            svc._heroes_df    = svc._analyzer.identify_local_heroes()
                            svc._attr_data    = svc._analyzer.aggregate_by_attribute()

                            svc._clusterer    = BranchClusterer(svc._metrics_df)
                            svc._X_scaled, _  = svc._clusterer.prepare_features()
                            from config.app_settings import AppSettings
                            kv, ins, ss = svc._clusterer.find_optimal_clusters(
                                svc._X_scaled,
                                max_clusters=AppSettings.MAX_CLUSTERS_TO_TEST
                            )
                            k = svc._clusterer.suggest_optimal_k(kv, ins, ss)
                            svc._clusterer.fit_kmeans(svc._X_scaled, n_clusters=k)
                            svc._cluster_summary    = svc._clusterer.get_cluster_summary()
                            svc._pca_df             = svc._clusterer.get_pca_data(svc._X_scaled)
                            svc._branch_cluster_map = svc._clusterer.get_branch_cluster_map()

                            # Use V2 model
                            predictor_v2 = SalesPredictionModelV2()
                            predictor_v2.load_or_train(csv_df)
                            svc._predictor    = predictor_v2
                            svc._data_loaded  = True

                            st.session_state["data_loaded"] = True
                            st.session_state["data_source_label"] = "CSV"
                            st.session_state["filters"] = svc.get_available_filters()

                            diag = predictor_v2.get_data_diagnostics()
                            st.success(
                                f"✅ {len(csv_df):,} rows loaded · "
                                f"{csv_df['BRANCHNAME'].nunique() if 'BRANCHNAME' in csv_df.columns else '?'} branches"
                            )
                            if diag.get("zero_pct", 0) > 60:
                                st.warning(
                                    f"⚠️ {diag['zero_pct']}% zero-sale rows detected. "
                                    "Predictions may be conservative."
                                )
                except Exception as e:
                    st.error(f"❌ CSV load failed: {e}")

        # Retrain
        if _is_loaded():
            if st.button("🧠 Retrain Model", use_container_width=True):
                svc = _get_service()
                with st.spinner("Retraining…"):
                    metrics = svc.retrain_model()
                st.success(
                    f"✅ Retrained  |  MAE {metrics.get('mae', '—'):.2f}  "
                    f"R² {metrics.get('r2', '—'):.3f}"
                )

        # Status indicator
        st.divider()
        if _is_loaded():
            label = st.session_state.get("data_source_label", "")
            badge = f" [{label}]" if label else ""
            st.markdown(
                f'<div style="text-align:center;font-size:0.78rem;'
                f'color:{ACCENT_TEAL};">● Data Loaded{badge}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="text-align:center;font-size:0.78rem;'
                f'color:{TEXT_MUTED};">○ No Data</div>',
                unsafe_allow_html=True,
            )


    return page


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def page_dashboard():
    st.markdown('<div class="page-title">Portfolio Overview</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Key performance indicators across all branches</div>',
                unsafe_allow_html=True)

    if not _is_loaded():
        _not_loaded_msg()
        return

    svc  = _get_service()
    dash = svc.get_dashboard_data()

    # ── Row 1 KPIs ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales",      fmt_number(dash.get("total_sales")))
    c2.metric("Total Stock",      fmt_number(dash.get("total_stock")))
    c3.metric("Branches",         fmt_number(dash.get("total_branches")))
    c4.metric("Regions",          fmt_number(dash.get("total_regions")))

    # ── Row 2 KPIs ────────────────────────────────────────────────────────────
    c5, c6, c7, c8 = st.columns(4)
    eff  = dash.get("overall_efficiency",   0)
    sell = dash.get("overall_sell_through", 0)
    c5.metric("Overall Efficiency",   f"{eff:.2f}")
    c6.metric("Sell-Through Rate",    f"{sell:.1%}")
    c7.metric("Local Heroes",         fmt_number(dash.get("total_local_heroes")))
    c8.metric("Clusters",             fmt_number(dash.get("cluster_count")))

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

    # ── Highlight cards ───────────────────────────────────────────────────────
    h1, h2 = st.columns(2)
    with h1:
        st.markdown(
            f'<div class="info-box">'
            f'<span style="color:{TEXT_MUTED};font-size:0.72rem;'
            f'letter-spacing:0.1em;text-transform:uppercase;">Top Branch</span>'
            f'<div style="font-family:Georgia,serif;'
            f'font-size:1.6rem;color:{GOLD};font-weight:600;margin-top:4px;">'
            f'{dash.get("top_branch","—")}</div></div>',
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            f'<div class="info-box">'
            f'<span style="color:{TEXT_MUTED};font-size:0.72rem;'
            f'letter-spacing:0.1em;text-transform:uppercase;">Top Region</span>'
            f'<div style="font-family:Georgia,serif;'
            f'font-size:1.6rem;color:{ACCENT_TEAL};font-weight:600;margin-top:4px;">'
            f'{dash.get("top_region","—")}</div></div>',
            unsafe_allow_html=True,
        )

    # ── Attribute unique counts ───────────────────────────────────────────────
    attrs = dash.get("attributes", {})
    if attrs:
        st.markdown('<div class="section-header">Attribute Diversity</div>',
                    unsafe_allow_html=True)
        cols = st.columns(len(attrs))
        for col, (attr, cnt) in zip(cols, attrs.items()):
            col.metric(attr.capitalize(), fmt_number(cnt))

    # ── Top branches quick chart ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Top 10 Branches by Sales</div>',
                unsafe_allow_html=True)
    top_df = svc.get_top_branches(10, "SALE_COUNT")
    if not top_df.empty:
        fig = px.bar(
            top_df,
            x="SALE_COUNT",
            y="BRANCHNAME",
            orientation="h",
            color="SALE_COUNT",
            color_continuous_scale=[[0, "#2A2A3E"], [1, GOLD]],
            labels={"SALE_COUNT": "Sales", "BRANCHNAME": ""},
        )
        fig.update_coloraxes(showscale=False)
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=360)
        fig.update_yaxes(autorange="reversed", gridcolor="#1E1E30", linecolor=CARD_BORDER)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BRANCH & CLUSTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def page_clusters():
    st.markdown('<div class="page-title">Branch & Cluster Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">K-Means segmentation · PCA visualisation</div>',
                unsafe_allow_html=True)

    if not _is_loaded():
        _not_loaded_msg()
        return

    svc = _get_service()
    ca  = svc.get_cluster_analysis()

    n_clusters = ca["n_clusters"]
    qs         = ca["quality_scores"]
    pca_df     = ca["pca_data"]
    branch_map = ca["branch_map"]
    summary    = ca["summary"]

    # ── Quality scores ────────────────────────────────────────────────────────
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Clusters",     n_clusters)
    q2.metric("Silhouette",   f"{qs.get('silhouette', 0):.3f}")
    q3.metric("Calinski",     f"{qs.get('calinski', 0):,.0f}")
    q4.metric("Inertia",      f"{qs.get('inertia', 0):,.0f}")

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

    # ── PCA scatter ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Branch Clusters — PCA View</div>',
                unsafe_allow_html=True)

    if pca_df is not None and not pca_df.empty:
        pca_plot = pca_df.copy()
        pca_plot["Cluster_str"] = pca_plot["Cluster"].astype(str)

        fig = px.scatter(
            pca_plot,
            x="PC1",
            y="PC2",
            color="Cluster_str",
            hover_name="BRANCHNAME" if "BRANCHNAME" in pca_plot.columns else None,
            color_discrete_sequence=CLUSTER_COLORS,
            labels={"Cluster_str": "Cluster", "PC1": "Principal Component 1",
                    "PC2": "Principal Component 2"},
            title="Branch Positioning by Cluster",
        )
        fig.update_traces(marker=dict(size=9, opacity=0.85,
                                      line=dict(width=0.5, color=DARK_BG)))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── Cluster summary table ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Cluster Profiles</div>',
                unsafe_allow_html=True)

    cluster_rows = []
    for c in summary.get("clusters", []):
        am = c.get("avg_metrics", {})
        cluster_rows.append({
            "Cluster":      c["label"],
            "Tier":         c["performance_tier"],
            "Branches":     c["num_branches"],
            "Avg Sales":    round(am.get("SALE_COUNT", 0), 1),
            "Avg Efficiency": round(am.get("efficiency_ratio", 0), 3),
            "Avg Sell-Through": round(am.get("sell_through_rate",
                                             am.get("branch_sell_through", 0)), 3),
            "Regions":      ", ".join(c.get("regions", [])),
        })

    if cluster_rows:
        cluster_tbl = pd.DataFrame(cluster_rows)
        st.dataframe(
            cluster_tbl,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Tier": st.column_config.TextColumn("Tier"),
                "Avg Sales": st.column_config.NumberColumn(format="%.1f"),
                "Avg Efficiency": st.column_config.NumberColumn(format="%.3f"),
                "Avg Sell-Through": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    # ── Top branches ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Top Branches</div>',
                unsafe_allow_html=True)

    left, right = st.columns([1, 3])
    with left:
        metric_opt = st.selectbox(
            "Rank by",
            ["SALE_COUNT", "branch_sell_through", "avg_efficiency", "product_count"],
        )
        top_n = st.slider("Top N", 5, 30, 10)

    top_df = svc.get_top_branches(top_n, metric_opt)

    with right:
        if not top_df.empty and metric_opt in top_df.columns:
            fig2 = px.bar(
                top_df,
                x=metric_opt,
                y="BRANCHNAME",
                orientation="h",
                color=metric_opt,
                color_continuous_scale=[[0, "#1E1E30"], [1, GOLD_LIGHT]],
                labels={metric_opt: metric_opt.replace("_", " ").title(),
                        "BRANCHNAME": ""},
            )
            fig2.update_coloraxes(showscale=False)
            fig2.update_layout(**PLOTLY_TEMPLATE["layout"], height=max(300, top_n * 30))
            fig2.update_yaxes(autorange="reversed", gridcolor="#1E1E30", linecolor=CARD_BORDER)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Branch → cluster lookup ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Branch → Cluster Map</div>',
                unsafe_allow_html=True)

    with st.expander("Show full branch-cluster mapping"):
        if branch_map is not None and not branch_map.empty:
            st.dataframe(branch_map, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SALES PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def page_prediction():
    st.markdown('<div class="page-title">Sales Prediction</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">RandomForest model · Predict SALE_COUNT for any branch + attribute combo</div>',
                unsafe_allow_html=True)

    if not _is_loaded():
        _not_loaded_msg()
        return

    svc     = _get_service()
    filters = st.session_state.get("filters", svc.get_available_filters())

    def _opts(key):
        return filters.get(key, [])

    # ── Model info ────────────────────────────────────────────────────────────
    info = svc.get_model_info()
    if info.get("is_trained"):
        em = info.get("eval_metrics", {})
        i1, i2, i3 = st.columns(3)
        i1.metric("MAE",  f"{em.get('mae', '—'):.2f}" if em.get("mae") else "—")
        i2.metric("RMSE", f"{em.get('rmse', '—'):.2f}" if em.get("rmse") else "—")
        i3.metric("R²",   f"{em.get('r2', '—'):.3f}"  if em.get("r2")  else "—")
    else:
        st.info("Model not yet trained. Load data first.")

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Configure Prediction</div>',
                unsafe_allow_html=True)

    # ── Input form ────────────────────────────────────────────────────────────
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        region   = st.selectbox("Region",     [""] + _opts("regions"))
        purity   = st.selectbox("Purity",     [""] + _opts("purities"))
        workstyle= st.selectbox("Workstyle",  [""] + _opts("workstyles"))
    with r1c2:
        branch   = st.selectbox("Branch",     [""] + _opts("branches"))
        finish   = st.selectbox("Finish",     [""] + _opts("finishes"))
        brand    = st.selectbox("Brand",      [""] + _opts("brands"))
    with r1c3:
        theme    = st.selectbox("Theme",      [""] + _opts("themes"))
        shape    = st.selectbox("Shape",      [""] + _opts("shapes"))
        stock    = st.number_input("Stock Count", min_value=0, max_value=9999, value=20)

    predict_btn = st.button("✨ Predict Sales", use_container_width=False)

    if predict_btn:
        input_data = {
            "REGION":      region   or None,
            "BRANCHNAME":  branch   or None,
            "PURITY":      purity   or None,
            "FINISH":      finish   or None,
            "THEME":       theme    or None,
            "SHAPE":       shape    or None,
            "WORKSTYLE":   workstyle or None,
            "BRAND":       brand    or None,
            "STOCK_COUNT": stock,
        }
        with st.spinner("Running prediction…"):
            result = svc.predict_sales(input_data)

        if "error" in result:
            st.error(result["error"])
        else:
            pred = result.get("predicted_sales", "—")
            rng  = result.get("confidence_range", ("—", "—"))

            st.markdown('<div class="section-header">Prediction Result</div>',
                        unsafe_allow_html=True)

            col_res, col_imp = st.columns([1, 2])
            with col_res:
                st.markdown(
                    f'<div class="prediction-card">'
                    f'<div class="prediction-label">Predicted Sale Count</div>'
                    f'<div class="prediction-number">{pred}</div>'
                    f'<div class="confidence-range">'
                    f'Range: {rng[0]} – {rng[1]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with col_imp:
                fi = svc._predictor.get_feature_importance() \
                     if hasattr(svc, "_predictor") and svc._predictor else None
                if fi is not None and not fi.empty:
                    top_fi = fi.head(10)
                    fig_fi = px.bar(
                        top_fi,
                        x="importance",
                        y="feature",
                        orientation="h",
                        color="importance",
                        color_continuous_scale=[[0, "#1E1E30"], [1, GOLD]],
                        title="Feature Importance",
                        labels={"importance": "Importance", "feature": ""},
                    )
                    fig_fi.update_coloraxes(showscale=False)
                    fig_fi.update_layout(**PLOTLY_TEMPLATE["layout"], height=320)
                    fig_fi.update_yaxes(autorange="reversed", gridcolor="#1E1E30", linecolor=CARD_BORDER)
                    st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

def page_recommendations():
    st.markdown('<div class="page-title">Product Recommendations</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Attribute intelligence · Best combos per branch or region</div>',
                unsafe_allow_html=True)

    if not _is_loaded():
        _not_loaded_msg()
        return

    svc     = _get_service()
    filters = st.session_state.get("filters", svc.get_available_filters())

    # ── Scope selector ────────────────────────────────────────────────────────
    scope = st.radio("Scope", ["By Branch", "By Region"], horizontal=True)

    if scope == "By Branch":
        branches = filters.get("branches", [])
        selected_branch = st.selectbox("Select Branch", branches) \
                          if branches else None
        selected_region = None
    else:
        regions = filters.get("regions", [])
        selected_region = st.selectbox("Select Region", regions) \
                          if regions else None
        selected_branch = None

    top_n = st.slider("Top N per attribute", 3, 10, 5)

    if st.button("💡 Get Recommendations"):
        with st.spinner("Analysing…"):
            recs = svc.get_recommendations(
                branch=selected_branch,
                region=selected_region,
                top_n=top_n,
            )

        if "error" in recs:
            st.error(recs["error"])
            return

        st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

        # ── Summary ───────────────────────────────────────────────────────────
        summ = recs.get("summary", {})
        if isinstance(summ, dict) and summ:
            s1, s2, s3 = st.columns(3)
            s1.metric("Total Sales (scope)",  fmt_number(summ.get("total_sales")))
            s2.metric("Total Stock (scope)",  fmt_number(summ.get("total_stock")))
            s3.metric("Avg Sell-Through",
                      f"{summ.get('avg_sell_through', 0):.1%}"
                      if summ.get("avg_sell_through") is not None else "—")
        elif isinstance(summ, str) and summ:
            st.markdown(f'<div class="info-box">{summ}</div>', unsafe_allow_html=True)

        # ── Per-attribute breakdown ───────────────────────────────────────────
        attr_keys = [k for k in recs.keys()
                     if k.startswith("by_") and isinstance(recs[k], list)]

        for akey in attr_keys:
            attr_label = akey.replace("by_", "").upper()
            items       = recs[akey]
            if not items:
                continue

            st.markdown(f'<div class="section-header">Top {attr_label}</div>',
                        unsafe_allow_html=True)

            attr_df = pd.DataFrame(items)
            if not attr_df.empty:
                # Chart
                val_col  = [c for c in attr_df.columns
                            if c not in ("PURITY","FINISH","THEME","SHAPE",
                                         "WORKSTYLE","BRAND","attribute_value",
                                         "rank","sell_through")]
                label_col = "attribute_value" if "attribute_value" in attr_df.columns \
                            else attr_df.columns[0]
                sales_col = "total_sales" if "total_sales" in attr_df.columns \
                            else (val_col[0] if val_col else None)

                if sales_col and label_col in attr_df.columns:
                    fig_attr = px.bar(
                        attr_df.head(top_n),
                        x=label_col,
                        y=sales_col,
                        color=sales_col,
                        color_continuous_scale=[[0, "#1E1E30"], [1, GOLD]],
                        labels={sales_col: "Sales", label_col: attr_label},
                    )
                    fig_attr.update_coloraxes(showscale=False)
                    fig_attr.update_layout(**PLOTLY_TEMPLATE["layout"], height=260)
                    st.plotly_chart(fig_attr, use_container_width=True)

        # ── Best combos ───────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Best Attribute Combos '
                    '(PURITY × FINISH × THEME)</div>',
                    unsafe_allow_html=True)
        combos = svc.get_high_performing_combos(branch=selected_branch, top_n=top_n)
        if not combos.empty:
            disp_cols = [c for c in
                         ["BRANCHNAME", "combo_label", "total_sales",
                          "sell_through", "rank"]
                         if c in combos.columns]
            st.dataframe(combos[disp_cols].head(20),
                         use_container_width=True, hide_index=True)
        else:
            st.markdown(
                '<div class="info-box">No combo data available for this scope.</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — REGIONAL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

def page_regional():
    st.markdown('<div class="page-title">Regional Performance</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Attribute breakdown by region</div>',
                unsafe_allow_html=True)

    if not _is_loaded():
        _not_loaded_msg()
        return

    svc = _get_service()

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        attribute = st.selectbox(
            "Attribute",
            ["PURITY", "FINISH", "THEME", "SHAPE", "WORKSTYLE", "BRAND"],
        )
    with c2:
        metric = st.selectbox(
            "Metric",
            ["total_sales", "sell_through", "avg_efficiency",
             "avg_relative_strength"],
        )

    with st.spinner("Loading regional data…"):
        perf_df = svc.get_product_performance_by_region(
            attribute=attribute, metric=metric
        )

    if perf_df is None or perf_df.empty:
        st.markdown(
            '<div class="info-box">No regional data available.</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Bar chart ─────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">{attribute} Performance by Region</div>',
        unsafe_allow_html=True,
    )

    # Determine correct column names from the DataFrame
    region_col = "REGION" if "REGION" in perf_df.columns else perf_df.columns[0]
    attr_col   = attribute if attribute in perf_df.columns else perf_df.columns[1]
    metric_col = metric if metric in perf_df.columns else perf_df.columns[2]

    fig_bar = px.bar(
        perf_df.head(60),
        x=attr_col,
        y=metric_col,
        color=region_col,
        barmode="group",
        color_discrete_sequence=CLUSTER_COLORS,
        labels={metric_col: metric.replace("_", " ").title(),
                attr_col: attribute,
                region_col: "Region"},
        title=f"{metric.replace('_',' ').title()} by {attribute} & Region",
    )
    fig_bar.update_layout(**PLOTLY_TEMPLATE["layout"], height=420)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Heatmap View</div>',
                unsafe_allow_html=True)

    try:
        pivot = perf_df.pivot_table(
            index=region_col,
            columns=attr_col,
            values=metric_col,
            aggfunc="mean",
        ).fillna(0)

        fig_heat = px.imshow(
            pivot,
            color_continuous_scale=[[0, DARK_BG], [0.5, "#3D2B00"], [1, GOLD]],
            aspect="auto",
            labels={"color": metric.replace("_", " ").title()},
            title=f"{metric.replace('_',' ').title()} Heatmap",
        )
        fig_heat.update_layout(**PLOTLY_TEMPLATE["layout"], height=340)
        st.plotly_chart(fig_heat, use_container_width=True)
    except Exception:
        pass  # Pivot fails if not enough data — silently skip

    # ── Raw table ─────────────────────────────────────────────────────────────
    with st.expander("Show raw data table"):
        st.dataframe(perf_df, use_container_width=True, hide_index=True)



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — EXPORT REPORT
# ══════════════════════════════════════════════════════════════════════════════

def page_export():
    st.markdown('<div class="page-title">Export Report</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Generate a branded PDF analytics report</div>',
                unsafe_allow_html=True)

    if not _is_loaded():
        _not_loaded_msg()
        return

    st.markdown('<div class="section-header">Report Contents</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="info-box"><b style="color:{GOLD};">📊 KPI Summary</b><br>'
            f'<span style="font-size:0.82rem;">Total sales, stock, branches, efficiency, sell-through.</span></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="info-box"><b style="color:{GOLD};">🏪 Top 10 Branches</b><br>'
            f'<span style="font-size:0.82rem;">Ranked table with sales, efficiency, sell-through.</span></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="info-box"><b style="color:{GOLD};">🔵 Cluster Analysis</b><br>'
            f'<span style="font-size:0.82rem;">Cluster profiles, tiers, quality scores, branch mapping.</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

    try:
        import reportlab
        rl_ok = True
        rl_ver = reportlab.Version
    except ImportError:
        rl_ok  = False
        rl_ver = None

    if not rl_ok:
        st.error("⚠️ `reportlab` is not installed. Run: `pip install reportlab` then restart.")
        return

    st.markdown(
        f'<div class="info-box">reportlab {rl_ver} ready &middot; Report saves to <code>reports/</code></div>',
        unsafe_allow_html=True,
    )

    if st.button("📄 Generate PDF Report"):
        svc = _get_service()
        try:
            from exports.report_generator import ReportGenerator
            gen = ReportGenerator(service=svc)
            with st.spinner("Building report…"):
                pdf_bytes = gen.generate_pdf_bytes()
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"jewelry_report_{ts}.pdf"
            st.success(f"✅ Report generated — {len(pdf_bytes):,} bytes")
            st.download_button(
                label     = "⬇️ Download PDF",
                data      = pdf_bytes,
                file_name = filename,
                mime      = "application/pdf",
            )
        except Exception as e:
            st.error(f"❌ Report generation failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# CHATBOT — Floating Widget
# ══════════════════════════════════════════════════════════════════════════════

def _get_assistant():
    """Lazily initialise JewelryAssistant in session state."""
    if "assistant" not in st.session_state:
        try:
            from chatbot.assistant import JewelryAssistant
            svc = st.session_state.get("svc", None)
            st.session_state["assistant"] = JewelryAssistant(
                model="mistral", service=svc
            )
        except ImportError:
            st.session_state["assistant"] = None
    return st.session_state["assistant"]


def render_chat_widget():
    """
    Floating chat bubble + expandable chat panel.
    Rendered on every page after main content.
    """
    assistant = _get_assistant()
    if assistant is None:
        return

    # ── Session state defaults ────────────────────────────────────────────────
    if "chat_open"    not in st.session_state:
        st.session_state["chat_open"]    = False
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # ── Floating button CSS ───────────────────────────────────────────────────
    st.markdown(f"""
    <style>
      .chat-fab {{
          position: fixed;
          bottom: 28px;
          right: 28px;
          width: 54px;
          height: 54px;
          background: linear-gradient(135deg, {GOLD} 0%, {GOLD_LIGHT} 100%);
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 1.5rem;
          cursor: pointer;
          z-index: 9999;
          box-shadow: 0 4px 20px rgba(201,168,76,0.4);
          border: none;
          color: {DARK_BG};
          font-weight: bold;
      }}
      .chat-panel {{
          position: fixed;
          bottom: 92px;
          right: 28px;
          width: 360px;
          max-height: 520px;
          background: {CARD_BG};
          border: 1px solid {GOLD};
          border-radius: 12px;
          z-index: 9998;
          display: flex;
          flex-direction: column;
          box-shadow: 0 8px 40px rgba(0,0,0,0.6);
          overflow: hidden;
      }}
      .chat-header {{
          background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
          border-bottom: 1px solid {CARD_BORDER};
          padding: 12px 16px;
          display: flex;
          align-items: center;
          justify-content: space-between;
      }}
      .chat-header-title {{
          font-size: 0.9rem;
          font-weight: 600;
          color: {GOLD};
          letter-spacing: 0.05em;
      }}
      .chat-header-sub {{
          font-size: 0.68rem;
          color: {TEXT_MUTED};
          margin-top: 2px;
      }}
      .chat-messages {{
          flex: 1;
          overflow-y: auto;
          padding: 12px 14px;
          display: flex;
          flex-direction: column;
          gap: 10px;
          max-height: 340px;
      }}
      .msg-user {{
          background: linear-gradient(135deg, {GOLD} 0%, {GOLD_LIGHT} 100%);
          color: {DARK_BG};
          border-radius: 12px 12px 2px 12px;
          padding: 8px 12px;
          font-size: 0.83rem;
          align-self: flex-end;
          max-width: 85%;
          font-weight: 500;
      }}
      .msg-assistant {{
          background: #1A1A2E;
          border: 1px solid {CARD_BORDER};
          color: {TEXT_PRIMARY};
          border-radius: 12px 12px 12px 2px;
          padding: 8px 12px;
          font-size: 0.83rem;
          align-self: flex-start;
          max-width: 90%;
          line-height: 1.5;
      }}
    </style>
    """, unsafe_allow_html=True)

    # ── Draggable floating chat button (pure HTML/JS, no Streamlit button) ──────
    chat_open = st.session_state["chat_open"]
    icon      = "✕" if chat_open else "💬"

    st.markdown(f"""
    <style>
      #chat-fab {{
          position: fixed;
          bottom: 32px;
          right: 32px;
          width: 54px;
          height: 54px;
          background: linear-gradient(135deg, {GOLD} 0%, {GOLD_LIGHT} 100%);
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 1.5rem;
          cursor: grab;
          z-index: 999998;
          box-shadow: 0 4px 20px rgba(201,168,76,0.5);
          user-select: none;
          transition: box-shadow 0.2s;
      }}
      #chat-fab:active {{ cursor: grabbing; }}
      #chat-fab:hover  {{ box-shadow: 0 6px 28px rgba(201,168,76,0.8); }}
    </style>

    <div id="chat-fab" title="AI Assistant">{icon}</div>

    <script>
    (function() {{
      var fab = document.getElementById('chat-fab');
      if (!fab) return;

      var isDragging = false;
      var startX, startY, origLeft, origTop;

      fab.addEventListener('mousedown', function(e) {{
        isDragging = false;
        var rect = fab.getBoundingClientRect();
        startX  = e.clientX;
        startY  = e.clientY;
        origLeft = rect.left;
        origTop  = rect.top;
        fab.style.right  = 'auto';
        fab.style.bottom = 'auto';
        fab.style.left   = origLeft + 'px';
        fab.style.top    = origTop  + 'px';

        function onMove(e) {{
          var dx = e.clientX - startX;
          var dy = e.clientY - startY;
          if (Math.abs(dx) > 5 || Math.abs(dy) > 5) isDragging = true;
          fab.style.left = (origLeft + dx) + 'px';
          fab.style.top  = (origTop  + dy) + 'px';
        }}

        function onUp(e) {{
          document.removeEventListener('mousemove', onMove);
          document.removeEventListener('mouseup', onUp);
          if (!isDragging) {{
            // Single click — find & click the hidden streamlit toggle button
            var btns = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < btns.length; i++) {{
              if (btns[i].innerText.trim() === 'CHATTOGGLE') {{
                btns[i].click();
                break;
              }}
            }}
          }}
        }}

        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
        e.preventDefault();
      }});
    }})();
    </script>
    """, unsafe_allow_html=True)

    # Invisible toggle button triggered by JS click
    st.markdown("<style>.chat-toggle-btn{display:none!important;}</style>",
                unsafe_allow_html=True)
    with st.container():
        if st.button("CHATTOGGLE", key="chat_fab_btn"):
            st.session_state["chat_open"] = not st.session_state["chat_open"]
            st.rerun()
    st.markdown("""
    <style>
      div[data-testid="stButton"]:has(button:contains("CHATTOGGLE")) {display:none!important;}
      button:has(> div > p:contains("CHATTOGGLE")) {display:none!important;}
    </style>
    """, unsafe_allow_html=True)

    # ── Chat panel ────────────────────────────────────────────────────────────
    if st.session_state["chat_open"]:
        st.markdown("---")
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<span style="color:{GOLD};font-size:1.1rem;">💎</span>'
            f'<span style="color:{GOLD};font-weight:600;font-size:1rem;">'
            f'Analytics Assistant</span>'
            f'<span style="color:{TEXT_MUTED};font-size:0.72rem;margin-left:4px;">'
            f'({assistant.get_context_summary()})</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Ollama status
        if not assistant.is_ollama_running():
            st.warning(
                "⚠️ Ollama is not running.\n\n"
                "1. Download from https://ollama.com/download\n"
                "2. Install & open it\n"
                "3. Run: `ollama pull mistral`\n"
                "4. Reload this page"
            )

        # Render history
        history = st.session_state["chat_history"]
        if not history:
            st.markdown(
                f'<div class="info-box">Ask me anything about your jewelry portfolio — '
                f'branch performance, predictions, recommendations, or cluster analysis.'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            for msg in history:
                role    = msg["role"]
                content = msg["content"]
                if role == "user":
                    st.markdown(
                        f'<div class="msg-user">{content}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="msg-assistant">{content}</div>',
                        unsafe_allow_html=True,
                    )

        # Input row
        col_input, col_send, col_clear = st.columns([6, 1, 1])
        with col_input:
            user_input = st.text_input(
                "Message",
                key="chat_input",
                label_visibility="collapsed",
                placeholder="Ask about branches, predictions, recommendations…",
            )
        with col_send:
            send = st.button("➤", key="chat_send", help="Send")
        with col_clear:
            if st.button("🗑", key="chat_clear", help="Clear history"):
                st.session_state["chat_history"] = []
                assistant.reset()
                st.rerun()

        if send and user_input.strip():
            # Update service reference in case data was loaded after init
            assistant.service = st.session_state.get("svc", None)

            with st.spinner("Thinking…"):
                reply, updated = assistant.chat(
                    user_input.strip(),
                    history=st.session_state["chat_history"],
                )
            st.session_state["chat_history"] = updated
            st.rerun()

        st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def render_gold_topbar():
    """Render gold price injected into Streamlit header via CSS hack."""
    data = _fetch_gold_data()
    if data.get("error"):
        price_html = "Price unavailable"
        badge = ""
    else:
        price      = data["price"]
        change_pct = data["change_pct"]
        arrow      = "▲" if change_pct >= 0 else "▼"
        clr        = "#4CC9A8" if change_pct >= 0 else "#C94C4C"
        price_html = f'₹{price:,.0f}/g &nbsp;<span style="color:{clr}">{arrow} {change_pct:+.2f}%</span>'
        badge      = '<span style="background:#C94C4C;color:#fff;font-size:0.5rem;font-weight:700;letter-spacing:0.1em;padding:2px 5px;border-radius:8px;margin-right:6px;vertical-align:middle;">LIVE</span>'

    st.markdown(f"""
    <style>
      /* inject gold price into the Streamlit toolbar area */
      [data-testid="stHeader"]::after {{
          content: "";
      }}
      .gold-topbar {{
          position: fixed;
          top: 0px;
          right: 160px;
          height: 48px;
          display: flex;
          align-items: center;
          gap: 6px;
          z-index: 999990;
          font-family: 'Segoe UI', sans-serif;
          font-size: 0.82rem;
          color: {TEXT_PRIMARY};
          pointer-events: none;
      }}
      .gold-topbar .pill {{
          background: {CARD_BG};
          border: 1px solid {CARD_BORDER};
          border-radius: 20px;
          padding: 4px 12px;
          display: flex;
          align-items: center;
          gap: 6px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.5);
      }}
      .gold-topbar .gold-val {{
          font-weight: 700;
          color: {GOLD};
      }}
    </style>
    <div class="gold-topbar">
      <div class="pill">
        {badge}
        <span style="color:{TEXT_MUTED};font-size:0.72rem;">🥇 24K Gold</span>
        <span class="gold-val">{price_html}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    page = render_sidebar()
    render_gold_topbar()

    if page == "📊 Dashboard":
        page_dashboard()
    elif page == "🏪 Branch & Clusters":
        page_clusters()
    elif page == "🔮 Sales Prediction":
        page_prediction()
    elif page == "💡 Recommendations":
        page_recommendations()
    elif page == "🌍 Regional Performance":
        page_regional()
    elif page == "📄 Export Report":
        page_export()

    # Floating chat widget — rendered on every page
    render_chat_widget()


if __name__ == "__main__":
    main()
