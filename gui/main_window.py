"""
gui/main_window.py  —  Jewelry Portfolio Analytics
Modern dark dashboard: left nav + content area, matplotlib charts, all 4 views.
"""

import sys, logging, threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.database_config import DatabaseConfig
from data_layer.sql_connector import SQLServerConnector
from data_layer.data_loader import JewelryDataLoader
from analytics.performance_metrics import PerformanceAnalyzer
from analytics.clustering_engine import BranchClusterer

logger = logging.getLogger(__name__)

# ── colour system ─────────────────────────────────────────────────────────────
BG      = "#0d0f14"
NAV     = "#13151c"
CARD    = "#1a1d27"
CARD2   = "#20232f"
BORDER  = "#2d3044"
TEXT    = "#f0f2f8"
MUTED   = "#6c7293"
DIM     = "#3a3f5c"
GOLD    = "#f5c542"
BLUE    = "#4f8ef7"
GREEN   = "#3ecf8e"
RED     = "#f26464"
PURPLE  = "#a78bfa"
ORANGE  = "#fb923c"
CLUSTER_COLS = [GOLD, BLUE, GREEN, PURPLE, ORANGE, RED, "#22d3ee", "#e879f9"]

MPL = {
    "figure.facecolor": CARD, "axes.facecolor": CARD,
    "axes.edgecolor": BORDER, "axes.labelcolor": MUTED,
    "axes.titlecolor": TEXT, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": BORDER, "grid.linewidth": 0.5,
    "text.color": TEXT, "legend.facecolor": CARD2,
    "legend.edgecolor": BORDER, "legend.fontsize": 8,
}
plt.rcParams.update(MPL)


def sep(parent, bg=BORDER):
    return tk.Frame(parent, bg=bg, height=1)


class MainWindow(tk.Tk):

    VIEWS = [
        ("branch",    "📊", "Branch Performance"),
        ("attribute", "🔍", "Attribute Analysis"),
        ("cluster",   "🗂", "Cluster View"),
        ("heroes",    "⭐", "Local Heroes"),
    ]

    def __init__(self):
        super().__init__()
        self.title("Jewelry Analytics")
        self.geometry("1400x860")
        self.minsize(1100, 700)
        self.configure(bg=BG)

        self.df = self.metrics_df = self.branch_summary = None
        self.heroes_df = self.attr_data = self.cluster_df = None

        self._nav_btns = {}
        self._content_frames = {}

        self._build_layout()
        self._load_async()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        # top bar
        topbar = tk.Frame(self, bg=NAV, height=52)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)
        tk.Label(topbar, text="◆  JEWELRY PORTFOLIO ANALYTICS",
                 bg=NAV, fg=GOLD, font=("Helvetica", 13, "bold")).pack(side="left", padx=24, pady=14)
        self._status_lbl = tk.Label(topbar, text="⏳  Connecting…",
                                    bg=NAV, fg=MUTED, font=("Helvetica", 9))
        self._status_lbl.pack(side="right", padx=20)

        # body
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # left nav
        nav = tk.Frame(body, bg=NAV, width=210)
        nav.pack(side="left", fill="y")
        nav.pack_propagate(False)
        tk.Label(nav, text="NAVIGATION", bg=NAV, fg=DIM,
                 font=("Helvetica", 7, "bold")).pack(anchor="w", padx=20, pady=(28, 8))

        for key, icon, title in self.VIEWS:
            btn = tk.Button(nav, text=f"  {icon}  {title}",
                            bg=NAV, fg=MUTED, relief="flat", bd=0,
                            font=("Helvetica", 10), anchor="w",
                            padx=12, pady=10, cursor="hand2",
                            activebackground=CARD2, activeforeground=TEXT,
                            command=lambda k=key: self._switch(k))
            btn.pack(fill="x", padx=8)
            self._nav_btns[key] = btn

        self._nav_info = tk.Label(nav, text="", bg=NAV, fg=DIM,
                                  font=("Helvetica", 7), wraplength=170, justify="left")
        self._nav_info.pack(side="bottom", anchor="w", padx=20, pady=16)

        # content area
        self._content_area = tk.Frame(body, bg=BG)
        self._content_area.pack(side="left", fill="both", expand=True)

        for key, _, _ in self.VIEWS:
            f = tk.Frame(self._content_area, bg=BG)
            self._content_frames[key] = f
            tk.Label(f, text="⏳  Loading data…",
                     bg=BG, fg=MUTED, font=("Helvetica", 13)).place(relx=.5, rely=.5, anchor="center")

        self._switch("branch")

    def _switch(self, key):
        for f in self._content_frames.values():
            f.pack_forget()
        self._content_frames[key].pack(fill="both", expand=True)
        for k, btn in self._nav_btns.items():
            if k == key:
                btn.configure(bg=CARD2, fg=GOLD, font=("Helvetica", 10, "bold"))
            else:
                btn.configure(bg=NAV, fg=MUTED, font=("Helvetica", 10))

    def _set_status(self, msg, color=MUTED):
        self._status_lbl.configure(text=msg, fg=color)

    def _style_tree(self, tree):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Treeview",
            background=CARD, foreground=TEXT, fieldbackground=CARD,
            rowheight=27, bordercolor=BORDER, font=("Helvetica", 9))
        s.configure("Treeview.Heading",
            background=CARD2, foreground=MUTED,
            font=("Helvetica", 8, "bold"), relief="flat")
        s.map("Treeview",
            background=[("selected", GOLD)],
            foreground=[("selected", BG)])

    # ── async load ────────────────────────────────────────────────────────────

    def _load_async(self):
        threading.Thread(target=self._load_data, daemon=True).start()

    def _load_data(self):
        try:
            self.after(0, lambda: self._set_status("⏳  Connecting…", GOLD))
            conn = SQLServerConnector(
                server=DatabaseConfig.SERVER, database=DatabaseConfig.DATABASE,
                username=DatabaseConfig.USERNAME, password=DatabaseConfig.PASSWORD,
                use_windows_auth=DatabaseConfig.USE_WINDOWS_AUTH)
            r = conn.test_connection()
            if not r["success"]:
                raise ConnectionError(r["message"])

            self.after(0, lambda: self._set_status("⏳  Loading data…", GOLD))
            loader = JewelryDataLoader(conn)
            df = loader.preprocess_data(loader.load_transaction_data())
            self.df = df

            self.after(0, lambda: self._set_status("⏳  Computing metrics…", GOLD))
            ana = PerformanceAnalyzer(df)
            self.metrics_df     = ana.calculate_all_metrics()
            self.branch_summary = ana.aggregate_by_branch()
            self.heroes_df      = ana.identify_local_heroes()
            self.attr_data      = ana.aggregate_by_attribute()

            self.after(0, lambda: self._set_status("⏳  Clustering…", GOLD))
            cl = BranchClusterer(self.metrics_df)
            Xs, _ = cl.prepare_features()
            kv, ins, ss = cl.find_optimal_clusters(Xs, max_clusters=8)
            k = cl.suggest_optimal_k(kv, ins, ss)
            cl.fit_kmeans(Xs, n_clusters=k)
            self.cluster_df = cl.assign_clusters_to_branches()
            conn.close()

            self.after(0, self._build_all_views)
            self.after(0, lambda: self._set_status(
                f"✅  {len(df):,} rows · {len(self.branch_summary)} branches · k={k}", GREEN))
            self.after(0, lambda: self._nav_info.configure(
                text=f"{len(self.branch_summary)} branches\n"
                     f"{int(df['SALE_COUNT'].sum())} total sales\n"
                     f"{len(self.heroes_df)} local heroes"))

        except Exception as e:
            logger.error("Load failed", exc_info=True)
            self.after(0, lambda: self._set_status(f"❌  {e}", RED))
            self.after(0, lambda: messagebox.showerror("Error", str(e)))

    def _build_all_views(self):
        for key in self._content_frames:
            for w in self._content_frames[key].winfo_children():
                w.destroy()
        self._build_branch_view()
        self._build_attribute_view()
        self._build_cluster_view()
        self._build_heroes_view()

    # ─────────────────────────────────────────────────────────────────────────
    # VIEW 1  Branch Performance
    # ─────────────────────────────────────────────────────────────────────────

    def _build_branch_view(self):
        root = self._content_frames["branch"]

        # KPI row
        kpi_row = tk.Frame(root, bg=BG)
        kpi_row.pack(fill="x", padx=20, pady=(20, 0))

        total_sales = int(self.df["SALE_COUNT"].sum())
        total_stock = int(self.df["STOCK_COUNT"].sum())
        sth = round(total_sales / (total_sales + total_stock) * 100, 1)
        top = self.branch_summary.nlargest(1, "SALE_COUNT").iloc[0]["BRANCHNAME"].replace("BR-","")

        for val, lbl, col in [
            (str(len(self.branch_summary)), "Branches",    BLUE),
            (f"{total_sales:,}",           "Total Sales", GOLD),
            (f"{total_stock:,}",           "Total Stock", PURPLE),
            (f"{sth}%",                    "Sell-Through",GREEN),
            (str(len(self.heroes_df)),     "Local Heroes",ORANGE),
            (top,                          "Top Branch",  GOLD),
        ]:
            c = tk.Frame(kpi_row, bg=CARD, padx=18, pady=12)
            c.pack(side="left", padx=(0, 10))
            tk.Frame(c, bg=col, height=2).pack(fill="x", pady=(0, 6))
            tk.Label(c, text=val, bg=CARD, fg=col,
                     font=("Helvetica", 17, "bold")).pack(anchor="w")
            tk.Label(c, text=lbl, bg=CARD, fg=MUTED,
                     font=("Helvetica", 8)).pack(anchor="w")

        # controls
        ctrl = tk.Frame(root, bg=BG)
        ctrl.pack(fill="x", padx=20, pady=(14, 0))
        tk.Label(ctrl, text="Top", bg=BG, fg=TEXT, font=("Helvetica", 9)).pack(side="left")
        self._bn_var = tk.IntVar(value=15)
        tk.Spinbox(ctrl, from_=5, to=50, increment=5,
                   textvariable=self._bn_var, width=4,
                   bg=CARD, fg=TEXT, insertbackground=TEXT,
                   relief="flat", buttonbackground=CARD2).pack(side="left", padx=6)
        tk.Label(ctrl, text="Sort by", bg=BG, fg=TEXT, font=("Helvetica", 9)).pack(side="left", padx=(10,4))
        self._bs_var = tk.StringVar(value="SALE_COUNT")
        ttk.Combobox(ctrl, textvariable=self._bs_var, width=22, state="readonly",
                     values=["SALE_COUNT","STOCK_COUNT","branch_sell_through",
                             "avg_efficiency","product_count"]).pack(side="left")
        tk.Button(ctrl, text="  Refresh  ", bg=GOLD, fg=BG, relief="flat", bd=0,
                  font=("Helvetica", 9, "bold"), pady=4, cursor="hand2",
                  command=self._refresh_branch).pack(side="left", padx=14)

        # chart + table
        row = tk.Frame(root, bg=BG)
        row.pack(fill="both", expand=True, padx=20, pady=14)

        self._branch_chart_f = tk.Frame(row, bg=CARD)
        self._branch_chart_f.pack(side="left", fill="both", expand=True)

        tbl_wrap = tk.Frame(row, bg=CARD, width=345)
        tbl_wrap.pack(side="right", fill="y", padx=(10, 0))
        tbl_wrap.pack_propagate(False)
        self._build_branch_table(tbl_wrap)

        self._refresh_branch()

    def _refresh_branch(self):
        n    = self._bn_var.get()
        sort = self._bs_var.get()
        data = self.branch_summary.nlargest(n, sort).sort_values(sort)

        for w in self._branch_chart_f.winfo_children():
            w.destroy()

        fig = Figure(dpi=96)
        fig.patch.set_facecolor(CARD)
        ax = fig.add_axes([0.27, 0.05, 0.68, 0.90])
        ax.set_facecolor(CARD)

        max_val = data[sort].max()
        colors  = [GOLD if v == max_val else BLUE for v in data[sort]]
        bars    = ax.barh(data["BRANCHNAME"], data[sort],
                          color=colors, height=0.62, edgecolor="none")

        for bar, val in zip(bars, data[sort]):
            lbl = f"{val:.3f}" if isinstance(val, float) else f"{int(val):,}"
            ax.text(val + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                    lbl, va="center", ha="left", color=MUTED, fontsize=7)

        col_name = sort.replace("_"," ").title()
        ax.set_title(f"Top {n} Branches — {col_name}", pad=10, color=TEXT)
        ax.xaxis.grid(True, alpha=0.15, color=BORDER)
        ax.set_axisbelow(True)
        for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(BORDER)
        ax.tick_params(colors=MUTED)

        canvas = FigureCanvasTkAgg(fig, self._branch_chart_f)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def _build_branch_table(self, parent):
        tk.Label(parent, text="ALL BRANCHES", bg=CARD, fg=MUTED,
                 font=("Helvetica", 8, "bold"), pady=10).pack(anchor="w", padx=12)

        cols = ("Branch","Region","Sales","Stock","S/T%","Rank")
        tree = ttk.Treeview(parent, columns=cols, show="headings", height=30)
        self._style_tree(tree)
        for col, w in zip(cols, [145,55,55,55,50,40]):
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor="center")

        for _, r in self.branch_summary.sort_values("sales_rank").iterrows():
            sth = round(r["branch_sell_through"] * 100, 1)
            tree.insert("", "end", values=(
                r["BRANCHNAME"], r["REGION"],
                int(r["SALE_COUNT"]), int(r["STOCK_COUNT"]),
                f"{sth}%", int(r["sales_rank"])))

        sb = tk.Scrollbar(parent, orient="vertical", command=tree.yview,
                          bg=CARD2, troughcolor=BG, width=8)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True, padx=(12,0))
        sb.pack(side="right", fill="y")

    # ─────────────────────────────────────────────────────────────────────────
    # VIEW 2  Attribute Analysis
    # ─────────────────────────────────────────────────────────────────────────

    def _build_attribute_view(self):
        root = self._content_frames["attribute"]

        sidebar = tk.Frame(root, bg=NAV, width=210)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="FILTERS", bg=NAV, fg=DIM,
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=20, pady=(24,0))

        for lbl_txt, varname, vals, default in [
            ("Attribute",  "_av_attr",   list(self.attr_data.keys()), "FINISH"),
            ("Branch",     "_av_branch", ["ALL"]+sorted(self.branch_summary["BRANCHNAME"].tolist()), "ALL"),
            ("Metric",     "_av_metric", ["sell_through","SALE_COUNT","STOCK_COUNT",
                                          "avg_efficiency","sales_contribution"], "sell_through"),
        ]:
            tk.Label(sidebar, text=lbl_txt, bg=NAV, fg=MUTED,
                     font=("Helvetica", 8)).pack(anchor="w", padx=20, pady=(14,2))
            var = tk.StringVar(value=default)
            setattr(self, varname, var)
            ttk.Combobox(sidebar, textvariable=var, values=vals,
                         state="readonly", width=22).pack(padx=14, fill="x")

        tk.Button(sidebar, text="Update Chart", bg=GOLD, fg=BG,
                  relief="flat", bd=0, font=("Helvetica", 9, "bold"),
                  pady=6, cursor="hand2",
                  command=self._refresh_attr).pack(padx=14, pady=20, fill="x")

        self._attr_chart_f = tk.Frame(root, bg=CARD)
        self._attr_chart_f.pack(side="left", fill="both", expand=True,
                                padx=(0,16), pady=16)
        self._refresh_attr()

    def _refresh_attr(self):
        attr   = self._av_attr.get()
        branch = self._av_branch.get()
        metric = self._av_metric.get()

        if attr not in self.attr_data:
            return
        data = self.attr_data[attr].copy()

        if branch != "ALL":
            data = data[data["BRANCHNAME"] == branch]
            if data.empty:
                return

        if branch == "ALL":
            num_cols = [c for c in ["SALE_COUNT","STOCK_COUNT","avg_efficiency",
                                     "sell_through","sales_contribution"] if c in data.columns]
            agg = {c: ("sum" if c in ("SALE_COUNT","STOCK_COUNT","sales_contribution") else "mean")
                   for c in num_cols}
            data = data.groupby(attr).agg(agg).reset_index()
            if "SALE_COUNT" in data.columns and "STOCK_COUNT" in data.columns:
                tot = data["SALE_COUNT"] + data["STOCK_COUNT"]
                data["sell_through"] = np.where(tot > 0, data["SALE_COUNT"] / tot, 0)

        if metric not in data.columns:
            metric = "SALE_COUNT"

        data = data.sort_values(metric, ascending=False)

        for w in self._attr_chart_f.winfo_children():
            w.destroy()

        fig = Figure(dpi=96)
        fig.patch.set_facecolor(CARD)
        ax = fig.add_axes([0.08, 0.20, 0.88, 0.68])
        ax.set_facecolor(CARD)

        n = len(data)
        x = np.arange(n)
        cols = [CLUSTER_COLS[i % len(CLUSTER_COLS)] for i in range(n)]
        bars = ax.bar(x, data[metric], color=cols, width=0.6, edgecolor="none")

        for bar, val in zip(bars, data[metric]):
            lbl = f"{val:.2%}" if metric == "sell_through" else (
                  f"{val:.3f}" if isinstance(val, float) else f"{int(val):,}")
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + data[metric].max() * 0.015,
                    lbl, ha="center", va="bottom", color=MUTED, fontsize=8.5)

        ax.set_xticks(x)
        ax.set_xticklabels(data[attr], rotation=20, ha="right", fontsize=9)
        ax.yaxis.grid(True, alpha=0.15, color=BORDER)
        ax.set_axisbelow(True)
        for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(BORDER)

        metric_label = {
            "sell_through":"Sell-Through Rate","SALE_COUNT":"Total Sales",
            "STOCK_COUNT":"Total Stock","avg_efficiency":"Avg Efficiency",
            "sales_contribution":"Sales Contribution %"}.get(metric, metric)
        suffix = f" — {branch}" if branch != "ALL" else " — All Branches"
        ax.set_title(f"{attr}  ×  {metric_label}{suffix}", pad=10, color=TEXT)
        ax.set_ylabel(metric_label, color=MUTED)

        canvas = FigureCanvasTkAgg(fig, self._attr_chart_f)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # VIEW 3  Cluster View
    # ─────────────────────────────────────────────────────────────────────────

    def _build_cluster_view(self):
        root = self._content_frames["cluster"]

        # cluster cards
        cards_row = tk.Frame(root, bg=BG)
        cards_row.pack(fill="x", padx=20, pady=(20,0))

        for cid in sorted(self.cluster_df["Cluster"].unique()):
            members = self.cluster_df[self.cluster_df["Cluster"] == cid]
            col = CLUSTER_COLS[cid % len(CLUSTER_COLS)]
            card = tk.Frame(cards_row, bg=CARD, padx=20, pady=14)
            card.pack(side="left", padx=(0,12))
            tk.Frame(card, bg=col, height=3).pack(fill="x", pady=(0,8))
            tk.Label(card, text=f"Cluster {cid}", bg=CARD, fg=col,
                     font=("Helvetica", 11, "bold")).pack(anchor="w")
            tk.Label(card, text=f"{len(members)} branches", bg=CARD, fg=TEXT,
                     font=("Helvetica", 9)).pack(anchor="w")
            avg_s = members["SALE_COUNT"].mean()
            avg_k = members["STOCK_COUNT"].mean()
            st = avg_s / (avg_s + avg_k) if (avg_s + avg_k) > 0 else 0
            tk.Label(card, text=f"S/T: {st:.1%}", bg=CARD, fg=MUTED,
                     font=("Helvetica", 8)).pack(anchor="w")

        # scatter + table
        row = tk.Frame(root, bg=BG)
        row.pack(fill="both", expand=True, padx=20, pady=14)

        chart_f = tk.Frame(row, bg=CARD)
        chart_f.pack(side="left", fill="both", expand=True)

        tbl_f = tk.Frame(row, bg=CARD, width=345)
        tbl_f.pack(side="right", fill="y", padx=(10,0))
        tbl_f.pack_propagate(False)

        self._draw_scatter(chart_f)
        self._build_cluster_table(tbl_f)

    def _draw_scatter(self, parent):
        df = self.cluster_df.copy()
        x_col = "SALE_COUNT"
        y_col = "sell_through_rate" if "sell_through_rate" in df.columns else "STOCK_COUNT"

        fig = Figure(dpi=96)
        fig.patch.set_facecolor(CARD)
        ax = fig.add_axes([0.10, 0.10, 0.86, 0.82])
        ax.set_facecolor(CARD)

        for cid in sorted(df["Cluster"].unique()):
            sub = df[df["Cluster"] == cid]
            col = CLUSTER_COLS[cid % len(CLUSTER_COLS)]
            ax.scatter(sub[x_col], sub[y_col], c=col, s=100,
                       alpha=0.9, label=f"Cluster {cid}", zorder=3, edgecolors="none")
            for _, r in sub.iterrows():
                ax.annotate(r["BRANCHNAME"].replace("BR-",""),
                            (r[x_col], r[y_col]),
                            fontsize=5.5, color=MUTED,
                            xytext=(4,3), textcoords="offset points")

        ax.set_xlabel(x_col.replace("_"," ").title(), color=MUTED)
        ax.set_ylabel(y_col.replace("_"," ").title(), color=MUTED)
        ax.set_title("Branch Cluster Map", pad=10, color=TEXT)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.15, color=BORDER)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        for sp in ["bottom","left"]: ax.spines[sp].set_color(BORDER)

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def _build_cluster_table(self, parent):
        tk.Label(parent, text="BRANCH ASSIGNMENTS", bg=CARD, fg=MUTED,
                 font=("Helvetica", 8, "bold"), pady=10).pack(anchor="w", padx=12)

        cols = ("Branch","Region","Cluster","Sales","Stock")
        tree = ttk.Treeview(parent, columns=cols, show="headings", height=30)
        self._style_tree(tree)
        for col, w in zip(cols, [145,55,60,55,60]):
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor="center")

        for _, r in self.cluster_df.sort_values("Cluster").iterrows():
            tree.insert("", "end", values=(
                r["BRANCHNAME"], r.get("REGION",""),
                int(r["Cluster"]), int(r["SALE_COUNT"]), int(r["STOCK_COUNT"])))

        sb = tk.Scrollbar(parent, orient="vertical", command=tree.yview,
                          bg=CARD2, troughcolor=BG, width=8)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True, padx=(12,0))
        sb.pack(side="right", fill="y")

    # ─────────────────────────────────────────────────────────────────────────
    # VIEW 4  Local Heroes
    # ─────────────────────────────────────────────────────────────────────────

    def _build_heroes_view(self):
        root = self._content_frames["heroes"]

        ctrl = tk.Frame(root, bg=BG)
        ctrl.pack(fill="x", padx=20, pady=(20,0))

        tk.Label(ctrl, text="Branch", bg=BG, fg=TEXT, font=("Helvetica", 9)).pack(side="left")
        self._hv_branch = tk.StringVar(value="ALL")
        ttk.Combobox(ctrl, textvariable=self._hv_branch, width=24, state="readonly",
                     values=["ALL"]+sorted(self.heroes_df["BRANCHNAME"].unique().tolist())
                     ).pack(side="left", padx=6)

        tk.Label(ctrl, text="  Sort", bg=BG, fg=TEXT, font=("Helvetica", 9)).pack(side="left")
        self._hv_sort = tk.StringVar(value="relative_strength")
        ttk.Combobox(ctrl, textvariable=self._hv_sort, width=22, state="readonly",
                     values=["relative_strength","SALE_COUNT",
                             "sales_contribution_pct","sell_through_rate"]
                     ).pack(side="left", padx=6)

        tk.Button(ctrl, text="  Refresh  ", bg=GOLD, fg=BG, relief="flat", bd=0,
                  font=("Helvetica", 9, "bold"), pady=4, cursor="hand2",
                  command=self._refresh_heroes).pack(side="left", padx=12)

        self._hero_count = tk.Label(ctrl, text="", bg=BG, fg=GREEN, font=("Helvetica", 9))
        self._hero_count.pack(side="right", padx=4)

        self._hero_chart_f = tk.Frame(root, bg=CARD)
        self._hero_chart_f.pack(fill="x", padx=20, pady=(14,0))

        tbl_wrap = tk.Frame(root, bg=CARD)
        tbl_wrap.pack(fill="both", expand=True, padx=20, pady=(10,16))

        cols = ("Rank","Branch","Item","Purity","Finish","Theme","Sales","Rel.Str","S/T%")
        self._hero_tree = ttk.Treeview(tbl_wrap, columns=cols, show="headings", height=18)
        self._style_tree(self._hero_tree)
        for col, w in zip(cols, [45,165,75,55,75,85,55,70,55]):
            self._hero_tree.heading(col, text=col,
                command=lambda c=col: self._sort_tree(self._hero_tree, c))
            self._hero_tree.column(col, width=w, anchor="center")

        sb = tk.Scrollbar(tbl_wrap, orient="vertical", command=self._hero_tree.yview,
                          bg=CARD2, troughcolor=BG, width=8)
        self._hero_tree.configure(yscrollcommand=sb.set)
        self._hero_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self._refresh_heroes()

    def _refresh_heroes(self):
        branch = self._hv_branch.get()
        sort   = self._hv_sort.get()
        data   = self.heroes_df.copy()

        if branch != "ALL":
            data = data[data["BRANCHNAME"] == branch]
        if sort in data.columns:
            data = data.sort_values(sort, ascending=False)

        self._hero_count.configure(
            text=f"{len(data)} heroes{' in ' + branch if branch != 'ALL' else ''}")

        top20 = data.head(20)
        for w in self._hero_chart_f.winfo_children():
            w.destroy()

        if len(top20):
            fig = Figure(figsize=(12, 2.6), dpi=96)
            fig.patch.set_facecolor(CARD)
            ax = fig.add_axes([0.03, 0.25, 0.95, 0.62])
            ax.set_facecolor(CARD)

            vals   = top20[sort].values if sort in top20.columns else top20["relative_strength"].values
            colors = [GOLD if v == vals.max() else BLUE for v in vals]
            labels = [f"{r['BRANCHNAME'].replace('BR-','')}\n{r['ITEMID']}"
                      for _, r in top20.iterrows()]

            ax.bar(range(len(labels)), vals, color=colors, width=0.65, edgecolor="none")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=6.5)
            ax.yaxis.grid(True, alpha=0.15, color=BORDER)
            ax.set_axisbelow(True)
            ax.set_title(f"Top 20 Local Heroes — {sort.replace('_',' ').title()}",
                         pad=8, color=TEXT, fontsize=10)
            for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(BORDER)

            canvas = FigureCanvasTkAgg(fig, self._hero_chart_f)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="x")
            plt.close(fig)

        for row in self._hero_tree.get_children():
            self._hero_tree.delete(row)

        for _, r in data.head(100).iterrows():
            sth = round(r.get("sell_through_rate", 0) * 100, 1)
            rs  = round(r.get("relative_strength", 0), 3)
            self._hero_tree.insert("", "end", values=(
                int(r.get("hero_rank", 0)), r["BRANCHNAME"], r["ITEMID"],
                r.get("PURITY",""), r.get("FINISH",""), r.get("THEME",""),
                int(r["SALE_COUNT"]), rs, f"{sth}%"))

    def _sort_tree(self, tree, col):
        items = [(tree.set(k, col), k) for k in tree.get_children("")]
        try:
            items.sort(key=lambda t: float(t[0].replace("%","")), reverse=True)
        except ValueError:
            items.sort(reverse=True)
        for i, (_, k) in enumerate(items):
            tree.move(k, "", i)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    app = MainWindow()
    app.mainloop()
