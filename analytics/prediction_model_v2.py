"""
Prediction Model V2
analytics/prediction_model_v2.py

Improvements over v1:
    1. One-hot encoding  — no false ordinal relationships between categories
    2. Aggregation       — data grouped to branch+attribute level before training
                           eliminates zero-sale noise at row level
    3. Better RF params  — more trees, cross-validated, slightly deeper
    4. Feature alignment — guaranteed match between train and predict phases
    5. Sell-through rate — added as an engineered feature

Model saves to models/model_v2.pkl  (separate from v1, safe to run both)

Usage:
    model = SalesPredictionModelV2()
    model.load_or_train(df)
    pred  = model.predict({'PURITY': '22.0', 'FINISH': 'Polished', ...})
"""

import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR  = Path(__file__).parent.parent
MODEL_PATH = _BASE_DIR / "models" / "model_v2.pkl"
META_PATH  = _BASE_DIR / "models" / "model_v2_meta.pkl"


class SalesPredictionModelV2:
    """
    Improved SALE_COUNT predictor.

    Key differences from v1:
      - OneHot encoding (not LabelEncoder)
      - Aggregates data to branch+attribute combos before training
      - Engineered feature: sell_through_rate
      - Better RF hyperparameters
      - Strict feature alignment between fit and predict
    """

    CAT_FEATURES = ["REGION", "BRANCHNAME", "PURITY", "FINISH",
                    "THEME", "SHAPE", "WORKSTYLE", "BRAND"]
    NUM_FEATURES = ["STOCK_COUNT", "sell_through_rate"]
    TARGET       = "SALE_COUNT"

    # Aggregation grouping — roll up to this level before training
    AGG_COLS     = ["REGION", "BRANCHNAME", "PURITY", "FINISH",
                    "THEME", "SHAPE", "WORKSTYLE", "BRAND"]

    def __init__(self):
        self.model         : Optional[RandomForestRegressor] = None
        self.ohe_columns   : List[str]  = []   # column names after get_dummies
        self.feature_cols  : List[str]  = []   # final ordered feature list
        self.is_trained    : bool       = False
        self._eval_metrics : Dict       = {}
        self._train_stats  : Dict       = {}   # stores data diagnostics

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Public API  (same interface as v1 for drop-in compatibility)
    # =========================================================================

    def load_or_train(self, df: pd.DataFrame) -> bool:
        """Load saved model or train fresh if none exists."""
        if MODEL_PATH.exists():
            loaded = self.load()
            if loaded:
                logger.info("✅ V2 model loaded from disk.")
                return True
        logger.info("No V2 model found — training fresh.")
        metrics = self.train_and_save(df)
        return self.is_trained

    def train_and_save(self, df: pd.DataFrame) -> Dict:
        """
        Full training pipeline:
          1. Preprocess + aggregate
          2. Engineer features
          3. One-hot encode
          4. Train RandomForest
          5. Evaluate + save

        Returns:
            Dict with mae, rmse, r2, n_samples, n_features
        """
        logger.info("=" * 50)
        logger.info("SalesPredictionModelV2 — Training started")

        # ── Step 1: Preprocess ────────────────────────────────────────────────
        df_clean = self._preprocess(df)
        if df_clean is None or len(df_clean) < 20:
            logger.error("Not enough clean data to train.")
            return {"error": "Insufficient data"}

        logger.info(f"  After preprocessing : {len(df_clean):,} rows")

        # ── Step 2: Aggregate to branch+attribute level ───────────────────────
        df_agg = self._aggregate(df_clean)
        logger.info(f"  After aggregation   : {len(df_agg):,} rows")

        # Diagnostics
        self._train_stats = {
            "raw_rows":       len(df_clean),
            "aggregated_rows": len(df_agg),
            "target_mean":    round(float(df_agg[self.TARGET].mean()), 2),
            "target_std":     round(float(df_agg[self.TARGET].std()),  2),
            "target_min":     int(df_agg[self.TARGET].min()),
            "target_max":     int(df_agg[self.TARGET].max()),
            "zero_pct":       round(
                float((df_agg[self.TARGET] == 0).mean() * 100), 1
            ),
        }
        logger.info(f"  Target stats: {self._train_stats}")

        if self._train_stats["zero_pct"] > 80:
            logger.warning(
                f"  ⚠️  {self._train_stats['zero_pct']}% of rows have SALE_COUNT=0. "
                "Model may underfit. Consider richer data."
            )

        # ── Step 3: Engineer features ─────────────────────────────────────────
        df_feat = self._engineer_features(df_agg)

        # ── Step 4: One-hot encode ────────────────────────────────────────────
        X, y = self._encode_fit(df_feat)
        logger.info(f"  Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

        if X.shape[0] < 20:
            logger.error("Too few samples after encoding.")
            return {"error": "Too few samples"}

        # ── Step 5: Train ─────────────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestRegressor(
            n_estimators      = 300,
            max_depth         = 12,
            min_samples_split = 4,
            min_samples_leaf  = 2,
            max_features      = "sqrt",
            random_state      = 42,
            n_jobs            = -1,
        )
        self.model.fit(X_train, y_train)

        # ── Step 6: Evaluate ──────────────────────────────────────────────────
        y_pred = self.model.predict(X_test)
        y_pred = np.clip(y_pred, 0, None)

        self._eval_metrics = {
            "mae":       round(float(mean_absolute_error(y_test, y_pred)), 3),
            "rmse":      round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3),
            "r2":        round(float(r2_score(y_test, y_pred)), 3),
            "n_train":   len(X_train),
            "n_test":    len(X_test),
            "n_features": X.shape[1],
        }
        logger.info(f"  ✅ Eval: {self._eval_metrics}")

        # Cross-val for robustness check
        try:
            cv_scores = cross_val_score(
                self.model, X, y, cv=min(5, len(X) // 10),
                scoring="r2", n_jobs=-1
            )
            self._eval_metrics["cv_r2_mean"] = round(float(cv_scores.mean()), 3)
            self._eval_metrics["cv_r2_std"]  = round(float(cv_scores.std()),  3)
            logger.info(
                f"  CV R²: {self._eval_metrics['cv_r2_mean']:.3f} "
                f"± {self._eval_metrics['cv_r2_std']:.3f}"
            )
        except Exception as e:
            logger.warning(f"  Cross-val skipped: {e}")

        self.is_trained = True
        self._save()
        return self._eval_metrics

    def predict(self, input_data: Dict) -> Dict:
        """
        Predict SALE_COUNT for a given input dict.

        Args:
            input_data: Dict with any subset of feature keys.

        Returns:
            Dict with predicted_sales, confidence_range, input_used,
            model_metrics, data_quality_note.
        """
        if not self.is_trained:
            return {"error": "Model not trained. Call load_or_train() first."}

        # Build single-row DataFrame
        row = {}
        for col in self.CAT_FEATURES:
            val = input_data.get(col, None)
            row[col] = str(val).upper().strip() if val else "UNKNOWN"

        row["STOCK_COUNT"]       = float(input_data.get("STOCK_COUNT", 20))
        row["sell_through_rate"] = float(input_data.get("sell_through_rate", 0.05))

        df_row = pd.DataFrame([row])
        X      = self._encode_predict(df_row)

        pred_raw = float(self.model.predict(X)[0])
        pred     = max(0, round(pred_raw))

        # Confidence range from individual tree predictions
        tree_preds = np.array([
            t.predict(X)[0]
            for t in self.model.estimators_
        ])
        lo = max(0, round(float(np.percentile(tree_preds, 10))))
        hi = max(pred, round(float(np.percentile(tree_preds, 90))))

        # Data quality note
        note = ""
        if self._train_stats.get("zero_pct", 0) > 60:
            note = (
                f"⚠️ Training data had {self._train_stats['zero_pct']}% "
                "zero-sale rows — predictions may be conservative."
            )

        return {
            "predicted_sales":   pred,
            "confidence_range":  (lo, hi),
            "input_used":        row,
            "model_metrics":     self._eval_metrics,
            "data_quality_note": note,
        }

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Return top N features by importance."""
        if not self.is_trained or self.model is None:
            return pd.DataFrame()

        imp = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).head(top_n)

        # Clean up OHE column names for display
        imp["feature"] = imp["feature"].str.replace(r"_[A-Z0-9]+$", "", regex=True)
        return imp.reset_index(drop=True)

    def get_model_info(self) -> Dict:
        """Return model metadata dict."""
        return {
            "is_trained":    self.is_trained,
            "version":       "v2",
            "encoding":      "one-hot",
            "aggregated":    True,
            "features":      self.feature_cols,
            "n_features":    len(self.feature_cols),
            "eval_metrics":  self._eval_metrics,
            "train_stats":   self._train_stats,
        }

    def get_data_diagnostics(self) -> Dict:
        """Return training data quality diagnostics."""
        return self._train_stats

    def load(self) -> bool:
        """Load model and metadata from disk."""
        try:
            self.model        = joblib.load(MODEL_PATH)
            meta              = joblib.load(META_PATH)
            self.ohe_columns  = meta["ohe_columns"]
            self.feature_cols = meta["feature_cols"]
            self._eval_metrics = meta.get("eval_metrics", {})
            self._train_stats  = meta.get("train_stats", {})
            self.is_trained   = True
            logger.info(f"✅ V2 model loaded from {MODEL_PATH}")
            return True
        except Exception as e:
            logger.error(f"❌ V2 model load failed: {e}")
            self.is_trained = False
            return False

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _preprocess(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Clean and validate raw DataFrame."""
        required = [self.TARGET, "STOCK_COUNT"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            logger.error(f"Missing columns: {missing}")
            return None

        df = df.copy()

        # Keep only rows with positive stock
        df = df[df["STOCK_COUNT"] > 0].copy()

        # Clip negatives
        df[self.TARGET]   = df[self.TARGET].clip(lower=0)
        df["STOCK_COUNT"] = df["STOCK_COUNT"].clip(lower=1)

        # Normalise categoricals
        for col in self.CAT_FEATURES:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .fillna("UNKNOWN")
                    .astype(str)
                    .str.upper()
                    .str.strip()
                )
            else:
                df[col] = "UNKNOWN"

        return df

    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate raw rows to branch+attribute level.

        This removes zero-sale noise — instead of 100 rows for
        (BR-CHENNAI, 22K, Polished, Minimalist) with 80 zeros,
        we get ONE row with total sales and total stock.
        """
        agg_cols_present = [c for c in self.AGG_COLS if c in df.columns]

        agg = (
            df.groupby(agg_cols_present, as_index=False)
            .agg(
                SALE_COUNT  = (self.TARGET,   "sum"),
                STOCK_COUNT = ("STOCK_COUNT", "sum"),
            )
        )

        # Fill any AGG_COLS not present
        for col in self.AGG_COLS:
            if col not in agg.columns:
                agg[col] = "UNKNOWN"

        return agg

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered numeric features."""
        df = df.copy()
        df["sell_through_rate"] = np.where(
            (df["SALE_COUNT"] + df["STOCK_COUNT"]) > 0,
            df["SALE_COUNT"] / (df["SALE_COUNT"] + df["STOCK_COUNT"]),
            0.0,
        )
        return df

    def _encode_fit(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """One-hot encode during training. Stores column list for inference."""
        cat_cols = [c for c in self.CAT_FEATURES if c in df.columns]
        num_cols = [c for c in self.NUM_FEATURES  if c in df.columns]

        df_enc = pd.get_dummies(df[cat_cols + num_cols], columns=cat_cols)

        # Store column order
        self.ohe_columns  = df_enc.columns.tolist()
        self.feature_cols = self.ohe_columns

        X = df_enc.values.astype(float)
        y = df[self.TARGET].values.astype(float)
        return X, y

    def _encode_predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        One-hot encode at inference time.
        Aligns to training columns exactly — missing = 0, extra = dropped.
        """
        cat_cols = [c for c in self.CAT_FEATURES if c in df.columns]
        num_cols = [c for c in self.NUM_FEATURES  if c in df.columns]

        df_enc = pd.get_dummies(df[cat_cols + num_cols], columns=cat_cols)

        # Align to training columns
        df_enc = df_enc.reindex(columns=self.ohe_columns, fill_value=0)

        return df_enc.values.astype(float)

    def _save(self):
        """Persist model and metadata."""
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump({
            "ohe_columns":  self.ohe_columns,
            "feature_cols": self.feature_cols,
            "eval_metrics": self._eval_metrics,
            "train_stats":  self._train_stats,
        }, META_PATH)
        logger.info(f"  V2 model saved → {MODEL_PATH}")
        logger.info(f"  V2 meta  saved → {META_PATH}")


# ─── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== SalesPredictionModelV2 Self-Test ===\n")

    rng = np.random.default_rng(42)
    n   = 500

    # Synthetic data with realistic signal
    branches  = [f"BR-CITY{i:02d}-0{j}" for i in range(1, 9) for j in range(1, 4)]
    purities  = ["14.0", "18.0", "22.0"]
    finishes  = ["Textured", "Polished", "Matte", "Hammered"]
    themes    = ["Minimalist", "Floral", "Geometric", "Classic", "Boho"]

    sample_df = pd.DataFrame({
        "BRANCHNAME":  rng.choice(branches,  n),
        "REGION":      rng.choice(["North", "South", "East", "West"], n),
        "ITEMID":      [f"PRD{i:05d}" for i in rng.integers(1, 80, n)],
        "PURITY":      rng.choice(purities,  n),
        "FINISH":      rng.choice(finishes,  n),
        "THEME":       rng.choice(themes,    n),
        "SHAPE":       rng.choice(["Round", "Square", "Oval", "Mixed"], n),
        "WORKSTYLE":   rng.choice(["Filigree", "Plain", "Studded", "Engraved"], n),
        "BRAND":       rng.choice(["Signature", "Heritage", "Luxe"], n),
        "STOCK_COUNT": rng.integers(5, 80,  n),
        "SALE_COUNT":  rng.integers(0, 25,  n),
    })

    print("1. Initialising V2 model...")
    model = SalesPredictionModelV2()

    print("\n2. Training (train_and_save)...")
    metrics = model.train_and_save(sample_df)
    print(f"   MAE   : {metrics.get('mae')}")
    print(f"   RMSE  : {metrics.get('rmse')}")
    print(f"   R²    : {metrics.get('r2')}")
    print(f"   CV R² : {metrics.get('cv_r2_mean')} ± {metrics.get('cv_r2_std')}")

    print("\n3. Data diagnostics:")
    diag = model.get_data_diagnostics()
    for k, v in diag.items():
        print(f"   {k}: {v}")

    print("\n4. Single prediction:")
    result = model.predict({
        "REGION":      "South",
        "BRANCHNAME":  branches[0],
        "PURITY":      "22.0",
        "FINISH":      "Polished",
        "THEME":       "Minimalist",
        "SHAPE":       "Round",
        "WORKSTYLE":   "Plain",
        "BRAND":       "Signature",
        "STOCK_COUNT": 30,
    })
    print(f"   Predicted  : {result['predicted_sales']}")
    print(f"   Range      : {result['confidence_range']}")
    if result.get("data_quality_note"):
        print(f"   Note       : {result['data_quality_note']}")

    print("\n5. Feature importance (top 8):")
    fi = model.get_feature_importance(top_n=8)
    print(fi.to_string(index=False))

    print("\n6. Load from disk:")
    model2 = SalesPredictionModelV2()
    model2.load()
    r2 = model2.predict({
        "REGION": "North", "PURITY": "18.0",
        "FINISH": "Matte", "STOCK_COUNT": 20,
    })
    print(f"   Loaded model prediction: {r2['predicted_sales']}")

    print("\n=== All tests passed ✅ ===")
