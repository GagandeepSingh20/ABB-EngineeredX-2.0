"""
ML/AI layer for the digital twin.

Three model families:
  1. AnomalyDetector  - multivariate anomaly score using IsolationForest
                        on physics residuals + sensor features
  2. DGAFaultClassifier - XGBoost classifier on dissolved gas analysis,
                          including Duval Triangle coordinates as features
  3. RULForecaster    - regression model that predicts remaining useful life
                        in days from recent degradation indicators

NOTE: Production design uses an LSTM with attention for RUL (per the
submission PDF). This demo uses scikit-learn's MLPRegressor (a small MLP)
because PyTorch wasn't available in the build environment. The interface
and pipeline are identical; swap MLPRegressor for an LSTM in production.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
import xgboost as xgb

# ============================================================
# 1. Anomaly detector
# ============================================================

class AnomalyDetector:
    """
    Multivariate anomaly detection using an Isolation Forest ensemble.

    Inputs are a vector of sensor measurements *plus the physics residual*
    (measured hotspot - predicted hotspot). Feeding the residual rather than
    the raw measurement is the key hybrid-loop trick: the model learns to
    detect 'unusual given the operating conditions', not just 'unusual values'.
    """

    FEATURE_COLS = [
        "load_pu", "ambient_temp_c", "winding_hotspot_c", "oil_temp_top_c",
        "hotspot_residual",  # measured - predicted (physics)
        "h2_ppm", "ch4_ppm", "c2h2_ppm", "c2h4_ppm",
        "vibration_rms_mm_s", "partial_discharge_pc",
    ]

    def __init__(self, contamination: float = 0.03, random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fit = False

    def fit(self, df: pd.DataFrame):
        """Fit on a DataFrame containing FEATURE_COLS."""
        X = df[self.FEATURE_COLS].values
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        self.is_fit = True
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns anomaly score in [0, 1] where higher = more anomalous.
        We rescale the IsolationForest decision_function to that range.
        """
        if not self.is_fit:
            raise RuntimeError("Call fit() first")
        X = df[self.FEATURE_COLS].values
        Xs = self.scaler.transform(X)
        # decision_function: positive = inlier, negative = outlier
        raw = -self.model.decision_function(Xs)
        # Normalize to [0, 1] using the empirical range
        raw_min, raw_max = raw.min(), raw.max()
        if raw_max - raw_min < 1e-6:
            return np.zeros_like(raw)
        normalized = (raw - raw_min) / (raw_max - raw_min)
        return normalized


# ============================================================
# 2. DGA fault classifier
# ============================================================

def compute_duval_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Duval Triangle 1 ratios from DGA gases.
    %CH4 = CH4 / (CH4 + C2H4 + C2H2) * 100   etc.
    """
    df = df.copy()
    denom = df["ch4_ppm"] + df["c2h4_ppm"] + df["c2h2_ppm"]
    denom = denom.replace(0, 1e-6)
    df["duval_ch4_pct"] = 100 * df["ch4_ppm"] / denom
    df["duval_c2h4_pct"] = 100 * df["c2h4_ppm"] / denom
    df["duval_c2h2_pct"] = 100 * df["c2h2_ppm"] / denom

    # Rogers Ratio features (capped)
    df["ratio_c2h2_c2h4"] = np.clip(df["c2h2_ppm"] / (df["c2h4_ppm"] + 1e-6), 0, 100)
    df["ratio_ch4_h2"] = np.clip(df["ch4_ppm"] / (df["h2_ppm"] + 1e-6), 0, 100)
    df["ratio_c2h4_c2h6"] = np.clip(df["c2h4_ppm"] / (df["c2h6_ppm"] + 1e-6), 0, 100)
    return df


class DGAFaultClassifier:
    """
    XGBoost classifier on DGA gas concentrations + Duval Triangle features.

    Output classes:
        0 = normal ageing
        1 = partial discharge
        2 = low-energy discharge
        3 = arcing
        4 = thermal fault <300C
        5 = thermal fault >700C
    """

    CLASS_NAMES = [
        "Normal ageing",
        "Partial discharge",
        "Low-energy discharge",
        "Arcing",
        "Thermal fault <300C",
        "Thermal fault >700C",
    ]

    FEATURE_COLS = [
        "h2_ppm", "ch4_ppm", "c2h2_ppm", "c2h4_ppm", "c2h6_ppm",
        "co_ppm", "co2_ppm",
        "duval_ch4_pct", "duval_c2h4_pct", "duval_c2h2_pct",
        "ratio_c2h2_c2h4", "ratio_ch4_h2", "ratio_c2h4_c2h6",
    ]

    def __init__(self, n_estimators: int = 200, max_depth: int = 5):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=6,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
        )

    def fit(self, df_with_labels: pd.DataFrame, label_col: str = "fault_type"):
        df = compute_duval_coordinates(df_with_labels)
        X = df[self.FEATURE_COLS].values
        y = df[label_col].values
        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = compute_duval_coordinates(df)
        return self.model.predict(df[self.FEATURE_COLS].values)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        df = compute_duval_coordinates(df)
        return self.model.predict_proba(df[self.FEATURE_COLS].values)

    def feature_importance(self) -> pd.DataFrame:
        importances = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.FEATURE_COLS,
            "importance": importances,
        }).sort_values("importance", ascending=False)


# ============================================================
# 3. RUL forecaster
# ============================================================

class RULForecaster:
    """
    Remaining-useful-life regression on a sliding window of degradation indicators.

    Production design (per submission PDF) uses an LSTM with attention.
    This demo uses an MLPRegressor on engineered window features
    (mean, std, slope of each indicator over the past N days).
    """

    INDICATOR_COLS = [
        "winding_hotspot_c", "h2_ppm", "ch4_ppm", "c2h2_ppm",
        "vibration_rms_mm_s", "partial_discharge_pc",
    ]

    def __init__(self, window_days: int = 30, hidden_layer_sizes=(64, 32)):
        self.window_days = window_days
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=0.005,
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )

    def _engineer_window_features(self, df: pd.DataFrame) -> np.ndarray:
        """For each row, compute statistics of the past window_days of indicators."""
        window = self.window_days * 24  # hours
        features = []
        for col in self.INDICATOR_COLS:
            roll = df[col].rolling(window=window, min_periods=1)
            features.append(roll.mean().values)
            features.append(roll.std().fillna(0).values)
            # Trend: simple slope using diff over window
            slope = (df[col] - df[col].shift(window)).fillna(0).values / window
            features.append(slope)
        return np.column_stack(features)

    def _compute_rul_labels(self, df: pd.DataFrame,
                            design_life_hours: float = 180_000) -> np.ndarray:
        """
        Synthetic RUL ground truth: physics-based residual life.
        Uses Arrhenius to compute equivalent ageing hours and projects forward
        at the average current ageing rate.
        """
        from src.physics import AgingModel
        am = AgingModel()
        consumed = am.life_consumed_hours(df["winding_hotspot_c"].values)
        # Current ageing rate (hours of life lost per real hour, recent average)
        rate = np.maximum(np.gradient(consumed), 1e-3)
        # Smooth rate
        rate_smoothed = pd.Series(rate).rolling(window=24 * 7, min_periods=1).mean().values
        remaining_hours = np.maximum(design_life_hours - consumed, 0)
        rul_hours = remaining_hours / np.maximum(rate_smoothed, 1e-3)
        rul_days = np.clip(rul_hours / 24, 0, 1500)
        return rul_days

    def fit(self, df: pd.DataFrame):
        X = self._engineer_window_features(df)
        y = self._compute_rul_labels(df)
        # Skip the first window (insufficient history)
        valid = self.window_days * 24
        Xv, yv = X[valid:], y[valid:]
        Xs = self.scaler.fit_transform(Xv)
        self.model.fit(Xs, yv)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._engineer_window_features(df)
        Xs = self.scaler.transform(X)
        return np.clip(self.model.predict(Xs), 0, 1500)


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_generator import generate_transformer_data, generate_dga_training_set
    from src.physics import ThermalModel

    print("Generating data...")
    df = generate_transformer_data(days=180)
    tm = ThermalModel()
    df["hotspot_residual"] = tm.residual(
        df["load_pu"], df["ambient_temp_c"], df["winding_hotspot_c"]
    )

    print("\n=== Anomaly Detector ===")
    det = AnomalyDetector(contamination=0.03).fit(df.iloc[:2000])  # fit on first 2000 rows
    scores = det.score(df)
    print(f"  Anomaly score range: {scores.min():.3f} - {scores.max():.3f}")
    print(f"  Mean during fault period (rows 2160-2289): {scores[2160:2289].mean():.3f}")
    print(f"  Mean during normal period (rows 100-1000): {scores[100:1000].mean():.3f}")

    print("\n=== DGA Fault Classifier ===")
    dga = generate_dga_training_set(2000)
    X_train, X_test = train_test_split(dga, test_size=0.2, random_state=42)
    clf = DGAFaultClassifier().fit(X_train)
    preds = clf.predict(X_test)
    print(f"  Test accuracy: {accuracy_score(X_test.fault_type, preds):.3f}")

    print("\n=== RUL Forecaster ===")
    rul = RULForecaster(window_days=14).fit(df)
    pred_rul = rul.predict(df)
    print(f"  Mean predicted RUL: {pred_rul[-100:].mean():.0f} days")
    print(f"  Recent trend (last 30d): {pred_rul[-30*24]:.0f} -> {pred_rul[-1]:.0f}")
