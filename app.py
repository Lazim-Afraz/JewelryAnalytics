"""
Jewelry Portfolio Analytics — Streamlit Web App
Supports: demo CSV mode (Streamlit Cloud) + SQL Server mode (local)
"""

import io
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jewelry Analytics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #c9a84c;
    }
    .hero-badge {
        background: #fff3cd;
        color: #856404;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Demo data generator ───────────────────────────────────────────────────────
@st.cache_data
def generate_demo_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    branches = [f"Branch_{i:02d}" for i in range(1, 21)]
    item_types = ["Necklace", "Ring", "Bracelet", "Earring", "Pendant"]
    purities = ["18K", "22K", "24K"]
    themes = ["Classic", "Modern", "Bridal", "Casual", "Luxury"]

    df = pd.DataFrame({
        "BRANCH":    rng.choice(branches, n),
        "ITEMTYPE":  rng.choice(item_types, n),
        "PURITY":    rng.choice(purities, n),
        "THEME":     rng.choice(themes, n),
        "SALE_COUNT": rng.integers(10, 300, n),
        "STOCK_COUNT": rng.integers(50, 500, n),
        "REVENUE":   rng.uniform(5000, 150000, n).round(2),
        "DATE":      pd.date_range("2024-01-01", periods=n, freq="6h"),
    })
    df["EFFICIENCY"] = (df["SALE_COUNT"] / df["STOCK_COUNT"]).round(4)
    return df


# ── Analytics helpers ─────────────────────────────────────────────────────────
def compute_branch_metrics(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("BRANCH").agg(
        total_sales=("SALE_COUNT", "sum"),
        total_revenue=("REVENUE", "sum"),
        avg_efficiency=("EFFICIENCY", "mean"),
        avg_stock=("STOCK_COUNT", "mean"),
        transactions=("SALE_COUNT", "count"),
    ).reset_index()
    grp["revenue_per_sale"] = (grp["total_revenue"] / grp["total_sales"]).round(2)
    grp["sales_contribution_pct"] = (
        grp["total_sales"] / grp["total_sales"].sum() * 100
    ).round(2)

    # Local hero: efficiency > 75th percentile AND sales > median
    eff_thresh = grp["avg_efficiency"].quantile(0.75)
    sales_med  = grp["total_sales"].median()
    grp["is_local_hero"] = (
        (grp["avg_efficiency"] >= eff_thresh) &
        (grp["total_sales"] >= sales_med)
    )
    return grp


def run_kmeans(metrics: pd.DataFrame, k: int) -> pd.DataFrame:
    features = ["total_sales", "avg_efficiency", "revenue_per_sale"]
    X = StandardScaler().fit_transform(metrics[features])
    labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X)
    out = metrics.copy()
    out["cluster"] = labels.astype(str)
    return out


def run_random_forest(df: pd.DataFrame):
    enc = pd.get_dummies(
        df[["ITEMTYPE", "PURITY", "THEME", "STOCK_COUNT", "REVENUE"]],
        columns=["ITEMTYPE", "PURITY", "THEME"],
    )
    X, y = enc.drop(columns=["REVENUE"]), enc["REVENUE"]
    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).nlargest(10)
    return mean_absolute_error(y_te, preds), r2_score(y_te, preds), feat_imp


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💎 Jewelry Analytics")
    st.markdown("---")

    data_source = st.radio(
        "Data source",
        ["Demo data (built-in)", "Upload CSV"],
        help="Use demo data to explore, or upload your own CSV file.",
    )

    df = None

    if data_source == "Demo data (built-in)":
        df = generate_demo_data()
        st.success(f"Loaded {len(df):,} demo records")

    else:
        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            help="Must have columns: BRANCH, ITEMTYPE, PURITY, THEME, SALE_COUNT, STOCK_COUNT, REVENUE",
        )
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                required = {"BRANCH", "ITEMTYPE", "PURITY", "SALE_COUNT", "STOCK_COUNT", "REVENUE"}
                missing = required - set(df.columns)
                if missing:
                    st.error(f"Missing columns: {missing}")
                    df = None
                else:
                    if "EFFICIENCY" not in df.columns:
                        df["EFFICIENCY"] = (df["SALE_COUNT"] / df["STOCK_COUNT"]).round(4)
                    st.success(f"Loaded {len(df):,} rows")
            except Exception as e:
                st.error(f"Could not read file: {e}")

    st.markdown("---")
    k_clusters = st.slider("K-Means clusters", 2, 8, 4)
    page = st.radio(
        "View",
        ["Overview", "Branch Performance", "Clustering", "Prediction", "Raw Data"],
    )

# ── Guard ─────────────────────────────────────────────────────────────────────
if df is None:
    st.info("Upload a CSV or switch to demo data to get started.")
    st.stop()

metrics = compute_branch_metrics(df)
clustered = run_kmeans(metrics, k_clusters)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Jewelry Portfolio Analytics")
    st.caption("Branch performance · Clustering · Demand forecasting")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Branches", metrics["BRANCH"].nunique())
    c2.metric("Total Sales", f"{metrics['total_sales'].sum():,}")
    c3.metric("Total Revenue", f"₹{metrics['total_revenue'].sum():,.0f}")
    c4.metric("Local Heroes", int(metrics["is_local_hero"].sum()))

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue by Branch (Top 10)")
        top10 = metrics.nlargest(10, "total_revenue")
        fig = px.bar(
            top10, x="BRANCH", y="total_revenue",
            color="is_local_hero",
            color_discrete_map={True: "#c9a84c", False: "#6c757d"},
            labels={"total_revenue": "Revenue (₹)", "is_local_hero": "Local Hero"},
        )
        fig.update_layout(showlegend=True, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sales by Item Type")
        item_sales = df.groupby("ITEMTYPE")["SALE_COUNT"].sum().reset_index()
        fig2 = px.pie(
            item_sales, names="ITEMTYPE", values="SALE_COUNT",
            color_discrete_sequence=px.colors.sequential.Oryel,
            hole=0.4,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Efficiency vs Revenue")
    fig3 = px.scatter(
        metrics, x="avg_efficiency", y="total_revenue",
        size="total_sales", color="is_local_hero",
        hover_name="BRANCH",
        color_discrete_map={True: "#c9a84c", False: "#6c757d"},
        labels={"avg_efficiency": "Avg Efficiency", "total_revenue": "Revenue (₹)"},
    )
    fig3.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Branch Performance
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Branch Performance":
    st.title("Branch Performance")

    heroes = metrics[metrics["is_local_hero"]]
    st.markdown(f"**{len(heroes)} local heroes** — branches with high efficiency AND strong sales volume")

    st.dataframe(
        metrics.sort_values("total_revenue", ascending=False)
               .style.format({
                   "total_revenue": "₹{:,.0f}",
                   "avg_efficiency": "{:.3f}",
                   "sales_contribution_pct": "{:.1f}%",
                   "revenue_per_sale": "₹{:,.0f}",
               })
               .background_gradient(subset=["total_revenue"], cmap="YlOrBr"),
        use_container_width=True,
        height=500,
    )

    st.subheader("Sales Contribution")
    fig = px.bar(
        metrics.sort_values("sales_contribution_pct", ascending=True).tail(15),
        x="sales_contribution_pct", y="BRANCH", orientation="h",
        color="avg_efficiency",
        color_continuous_scale="YlOrBr",
        labels={"sales_contribution_pct": "Sales Contribution (%)", "avg_efficiency": "Efficiency"},
    )
    fig.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Clustering
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Clustering":
    st.title("Branch Clustering (K-Means)")
    st.caption(f"Branches grouped into {k_clusters} clusters based on sales, efficiency, and revenue per sale")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            clustered, x="total_sales", y="avg_efficiency",
            color="cluster", size="total_revenue",
            hover_name="BRANCH",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"total_sales": "Total Sales", "avg_efficiency": "Avg Efficiency"},
        )
        fig.update_layout(plot_bgcolor="white", title="Sales vs Efficiency by Cluster")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            clustered, x="revenue_per_sale", y="total_revenue",
            color="cluster", hover_name="BRANCH",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"revenue_per_sale": "Revenue / Sale (₹)", "total_revenue": "Total Revenue (₹)"},
        )
        fig2.update_layout(plot_bgcolor="white", title="Revenue per Sale vs Total Revenue")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Cluster Summary")
    cluster_summary = clustered.groupby("cluster").agg(
        branches=("BRANCH", "count"),
        avg_sales=("total_sales", "mean"),
        avg_revenue=("total_revenue", "mean"),
        avg_efficiency=("avg_efficiency", "mean"),
    ).round(2)
    st.dataframe(cluster_summary, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Prediction
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction":
    st.title("Revenue Prediction (Random Forest)")

    with st.spinner("Training model..."):
        mae, r2, feat_imp = run_random_forest(df)

    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error", f"₹{mae:,.0f}")
    col2.metric("R² Score", f"{r2:.3f}")

    st.subheader("Top Feature Importances")
    fig = px.bar(
        feat_imp.reset_index().rename(columns={"index": "Feature", 0: "Importance"}),
        x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="YlOrBr",
    )
    fig.update_layout(plot_bgcolor="white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Stock Optimisation Recommendations")
    st.markdown("Branches where stock far exceeds sales — restocking priority is low:")
    over_stocked = metrics[metrics["avg_efficiency"] < metrics["avg_efficiency"].quantile(0.25)]
    st.dataframe(
        over_stocked[["BRANCH", "avg_efficiency", "avg_stock", "total_sales"]]
        .sort_values("avg_efficiency"),
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Raw Data
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Raw Data":
    st.title("Raw Data")
    st.dataframe(df, use_container_width=True, height=500)

    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="jewelry_analytics_data.csv",
        mime="text/csv",
    )
