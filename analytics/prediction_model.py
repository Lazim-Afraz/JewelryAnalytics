"""
Prediction Model
analytics/prediction_model.py

RandomForest regression pipeline that predicts SALE_COUNT for a given
branch + product attribute combination.

Workflow:
    1. First run  → train on live data → save to models/model.pkl
    2. Every run after → load model.pkl, predict instantly (no retraining)
    3. Manual retrain → call train_and_save() explicitly

Features engineered from:
    REGION, BRANCHNAME, PURITY, FINISH, THEME, SHAPE, WORKSTYLE, BRAND,
    STOCK_COUNT  →  predict  SALE_COUNT

Schema reference:
    REGION, BRANCHNAME, ITEMID, PURITY, FINISH, THEME, SHAPE,
    WORKSTYLE, BRAND, SALE_COUNT, STOCK_COUNT
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR  = Path(__file__).parent.parent
MODEL_PATH = _BASE_DIR / 'models' / 'model.pkl'
META_PATH  = _BASE_DIR / 'models' / 'model_meta.pkl'   # encoders + feature list


class SalesPredictionModel:
    """
    RandomForest-based SALE_COUNT predictor.

    Usage (typical):
        model = SalesPredictionModel()
        model.load_or_train(df)                  # train once, then load
        pred  = model.predict({'PURITY': '18.0', 'FINISH': 'Polished', ...})

    Usage (force retrain):
        model = SalesPredictionModel()
        model.train_and_save(df)
    """

    # Categorical features to encode
    CAT_FEATURES = ['REGION', 'BRANCHNAME', 'PURITY', 'FINISH',
                    'THEME', 'SHAPE', 'WORKSTYLE', 'BRAND']

    # Numeric features used as-is
    NUM_FEATURES = ['STOCK_COUNT']

    # Target
    TARGET = 'SALE_COUNT'

    def __init__(self):
        self.model        : Optional[RandomForestRegressor] = None
        self.encoders     : Dict[str, LabelEncoder]         = {}
        self.feature_cols : List[str]                       = []
        self.is_trained   : bool                            = False
        self._eval_metrics: Dict                            = {}

        # Ensure models/ directory exists
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ─── Public API ───────────────────────────────────────────────────────────

    def load_or_train(self, df: pd.DataFrame) -> bool:
        """
        Load existing model from disk if available, otherwise train fresh.

        Args:
            df: Preprocessed DataFrame from JewelryDataLoader.

        Returns:
            True if model is ready for inference.
        """
        if MODEL_PATH.exists() and META_PATH.exists():
            logger.info("Found existing model — loading from disk...")
            return self.load()
        else:
            logger.info("No saved model found — training fresh...")
            self.train_and_save(df)
            return self.is_trained

    def train_and_save(self, df: pd.DataFrame) -> Dict:
        """
        Train RandomForest on df and persist to models/model.pkl.

        Args:
            df: Preprocessed DataFrame (must contain all CAT_FEATURES,
                NUM_FEATURES, and TARGET columns).

        Returns:
            Dict of evaluation metrics (MAE, RMSE, R²).
        """
        logger.info("Training RandomForest prediction model...")

        df_clean = self._prepare_dataframe(df)
        if df_clean is None or len(df_clean) < 20:
            logger.warning("Not enough data to train model (need ≥ 20 rows).")
            return {}

        X, y = self._encode_features(df_clean, fit=True)

        # Train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info(f"  Training on {len(X_train):,} rows, "
                    f"testing on {len(X_test):,} rows...")

        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        metrics = self._compute_metrics(y_test, y_pred)
        self._eval_metrics = metrics

        logger.info(f"  MAE  : {metrics['mae']:.3f}")
        logger.info(f"  RMSE : {metrics['rmse']:.3f}")
        logger.info(f"  R²   : {metrics['r2']:.3f}")

        # Persist
        self._save()
        self.is_trained = True

        logger.info(f"✅ Model trained and saved → {MODEL_PATH}")
        return metrics

    def predict(self, input_data: Union[Dict, pd.DataFrame]) -> Dict:
        """
        Predict SALE_COUNT for a single input or a DataFrame of inputs.

        Args:
            input_data: Dict of feature values, e.g.:
                {
                    'REGION':    'South',
                    'BRANCHNAME':'BR-CHENNAI-01',
                    'PURITY':    '18.0',
                    'FINISH':    'Polished',
                    'THEME':     'Minimalist',
                    'SHAPE':     'Round',
                    'WORKSTYLE': 'Plain',
                    'BRAND':     'Signature',
                    'STOCK_COUNT': 20
                }
                OR a DataFrame with the same columns.

        Returns:
            Dict with keys:
                predicted_sales   — int
                confidence_range  — (low, high) based on tree variance
                input_used        — the cleaned input dict
                model_metrics     — training evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model not trained. Call load_or_train(df) first."
            )

        # Normalise input to DataFrame
        if isinstance(input_data, dict):
            df_in = pd.DataFrame([input_data])
        else:
            df_in = input_data.copy()

        df_in = self._fill_missing_features(df_in)
        X, _  = self._encode_features(df_in, fit=False)

        # Predict with all trees for confidence range
        predictions_per_tree = np.array(
            [tree.predict(X) for tree in self.model.estimators_]
        )
        mean_pred = float(self.model.predict(X)[0])
        std_pred  = float(predictions_per_tree[:, 0].std())

        predicted = max(0, round(mean_pred))
        low       = max(0, round(mean_pred - std_pred))
        high      = max(0, round(mean_pred + std_pred))

        result = {
            'predicted_sales':  predicted,
            'confidence_range': (low, high),
            'input_used':       df_in.iloc[0].to_dict(),
            'model_metrics':    self._eval_metrics,
        }

        logger.info(f"Prediction: {predicted} sales "
                    f"(range {low}–{high})")
        return result

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Return top N most important features with their importance scores.

        Returns:
            DataFrame with columns: feature, importance, rank
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        importances = self.model.feature_importances_
        fi_df = pd.DataFrame({
            'feature':    self.feature_cols,
            'importance': importances,
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        fi_df['rank'] = fi_df.index + 1
        return fi_df.head(top_n)

    def get_model_info(self) -> Dict:
        """
        Return model metadata summary dict.
        """
        return {
            'is_trained':      self.is_trained,
            'model_path':      str(MODEL_PATH),
            'model_exists':    MODEL_PATH.exists(),
            'features':        self.feature_cols,
            'n_features':      len(self.feature_cols),
            'eval_metrics':    self._eval_metrics,
            'cat_features':    self.CAT_FEATURES,
            'num_features':    self.NUM_FEATURES,
            'target':          self.TARGET,
        }

    def load(self) -> bool:
        """
        Load saved model and encoders from disk.

        Returns:
            True on success, False on failure.
        """
        try:
            self.model        = joblib.load(MODEL_PATH)
            meta              = joblib.load(META_PATH)
            self.encoders     = meta['encoders']
            self.feature_cols = meta['feature_cols']
            self._eval_metrics = meta.get('eval_metrics', {})
            self.is_trained   = True
            logger.info(f"✅ Model loaded from {MODEL_PATH}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.is_trained = False
            return False

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _prepare_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Select relevant columns and drop rows with zero stock.
        """
        required = [self.TARGET] + self.NUM_FEATURES
        present_cats = [c for c in self.CAT_FEATURES if c in df.columns]

        needed = required + present_cats
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return None

        df_out = df[needed].copy()
        df_out = df_out[df_out['STOCK_COUNT'] > 0].copy()

        # Fill missing categoricals with 'Unknown'
        for col in present_cats:
            df_out[col] = df_out[col].fillna('Unknown').astype(str).str.upper().str.strip()

        df_out[self.TARGET]   = df_out[self.TARGET].clip(lower=0)
        df_out['STOCK_COUNT'] = df_out['STOCK_COUNT'].clip(lower=0)

        return df_out

    def _encode_features(self,
                          df: pd.DataFrame,
                          fit: bool = False
                          ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Label-encode categorical features and assemble feature matrix.

        Args:
            df  : Input DataFrame.
            fit : True during training (fits encoders), False during inference.

        Returns:
            (X, y) — feature matrix and target array (y is None at inference).
        """
        df = df.copy()
        encoded_cols = []

        present_cats = [c for c in self.CAT_FEATURES if c in df.columns]

        for col in present_cats:
            df[col] = df[col].fillna('Unknown').astype(str).str.upper().str.strip()

            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.encoders[col] = le
            else:
                if col in self.encoders:
                    le = self.encoders[col]
                    # Handle unseen labels gracefully
                    df[col] = df[col].apply(
                        lambda v: le.transform([v])[0]
                        if v in le.classes_
                        else -1
                    )
                else:
                    df[col] = 0

            encoded_cols.append(col)

        num_cols = [c for c in self.NUM_FEATURES if c in df.columns]
        all_cols = encoded_cols + num_cols

        if fit:
            self.feature_cols = all_cols

        X = df[self.feature_cols if not fit else all_cols].values.astype(float)
        y = df[self.TARGET].values.astype(float) if self.TARGET in df.columns else None

        return X, y

    def _fill_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing feature columns with sensible defaults for inference.
        """
        df = df.copy()
        for col in self.CAT_FEATURES:
            if col not in df.columns:
                df[col] = 'Unknown'
        for col in self.NUM_FEATURES:
            if col not in df.columns:
                df[col] = 0
        return df

    def _save(self):
        """Persist model and metadata to disk."""
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump({
            'encoders':     self.encoders,
            'feature_cols': self.feature_cols,
            'eval_metrics': self._eval_metrics,
        }, META_PATH)
        logger.info(f"  Model saved → {MODEL_PATH}")
        logger.info(f"  Meta  saved → {META_PATH}")

    @staticmethod
    def _compute_metrics(y_true: np.ndarray,
                          y_pred: np.ndarray) -> Dict:
        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2   = float(r2_score(y_true, y_pred))
        return {
            'mae':  round(mae,  3),
            'rmse': round(rmse, 3),
            'r2':   round(r2,   3),
        }


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Prediction Model Test ===\n")

    rng = np.random.default_rng(42)
    n   = 300

    sample_df = pd.DataFrame({
        'BRANCHNAME':  rng.choice(['BR-CHENNAI-01', 'BR-MUMBAI-01',
                                   'BR-DELHI-01', 'BR-KOLKATA-01'], n),
        'REGION':      rng.choice(['South', 'West', 'North', 'East'], n),
        'ITEMID':      [f'PRD{i:05d}' for i in rng.integers(1, 80, n)],
        'PURITY':      rng.choice(['14.0', '18.0', '22.0'], n),
        'FINISH':      rng.choice(['Textured', 'Polished', 'Matte', 'Hammered'], n),
        'THEME':       rng.choice(['Minimalist', 'Floral', 'Geometric',
                                   'Classic', 'Boho'], n),
        'SHAPE':       rng.choice(['Round', 'Square', 'Oval', 'Mixed'], n),
        'WORKSTYLE':   rng.choice(['Filigree', 'Plain', 'Studded', 'Engraved'], n),
        'BRAND':       rng.choice(['Signature', 'Heritage', 'Luxe'], n),
        'SALE_COUNT':  rng.integers(0, 30, n),
        'STOCK_COUNT': rng.integers(5, 60, n),
    })

    print("1. Initialising model...")
    predictor = SalesPredictionModel()

    print("\n2. Training model (load_or_train)...")
    predictor.load_or_train(sample_df)

    print("\n3. Model info:")
    info = predictor.get_model_info()
    print(f"   Trained    : {info['is_trained']}")
    print(f"   Features   : {info['features']}")
    print(f"   Eval MAE   : {info['eval_metrics'].get('mae', 'N/A')}")
    print(f"   Eval RMSE  : {info['eval_metrics'].get('rmse', 'N/A')}")
    print(f"   Eval R²    : {info['eval_metrics'].get('r2', 'N/A')}")

    print("\n4. Single prediction:")
    result = predictor.predict({
        'REGION':      'South',
        'BRANCHNAME':  'BR-CHENNAI-01',
        'PURITY':      '18.0',
        'FINISH':      'Polished',
        'THEME':       'Minimalist',
        'SHAPE':       'Round',
        'WORKSTYLE':   'Plain',
        'BRAND':       'Signature',
        'STOCK_COUNT': 20,
    })
    print(f"   Predicted sales : {result['predicted_sales']}")
    print(f"   Confidence range: {result['confidence_range']}")

    print("\n5. Feature importance (top 5):")
    fi = predictor.get_feature_importance(top_n=5)
    print(fi.to_string(index=False))

    print("\n6. Testing load from disk...")
    predictor2 = SalesPredictionModel()
    predictor2.load()
    result2 = predictor2.predict({
        'REGION':      'North',
        'BRANCHNAME':  'BR-DELHI-01',
        'PURITY':      '22.0',
        'FINISH':      'Matte',
        'THEME':       'Floral',
        'SHAPE':       'Oval',
        'WORKSTYLE':   'Filigree',
        'BRAND':       'Heritage',
        'STOCK_COUNT': 35,
    })
    print(f"   Predicted sales (loaded model): {result2['predicted_sales']}")

    print("\n=== All tests passed ✅ ===")
