"""Sales Forecasting using RandomForestRegressor vs. DummyRegressor Baseline."""

import os
import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

LAGS = [1, 2, 3, 7, 14]
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


class _FallbackForecaster:
    """Lightweight ridge/linear lag forecaster used only if scikit-learn is not installed."""

    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.feature_names_in_ = None
        self.feature_importances_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_in_ = list(X.columns)
        X_mat = X.to_numpy(dtype=float)
        y_vec = y.to_numpy(dtype=float)
        X_aug = np.column_stack([X_mat, np.ones(len(X_mat))])
        reg = 1e-2 * np.eye(X_aug.shape[1])
        reg[-1, -1] = 0.0
        beta = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y_vec)
        self.weights = beta[:-1]
        self.bias = float(beta[-1])
        std_x = np.std(X_mat, axis=0) + 1e-9
        imp = np.abs(self.weights * std_x)
        self.feature_importances_ = imp / (imp.sum() + 1e-9)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_mat = X[self.feature_names_in_].to_numpy(dtype=float)
        return X_mat @ self.weights + self.bias


def daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """Resample transaction revenue into a continuous daily time series."""
    s = df.set_index("InvoiceDate")["Revenue"].resample("D").sum().fillna(0.0)
    return s.to_frame("y")


def make_features(d: pd.DataFrame) -> pd.DataFrame:
    """Create time-series lag, rolling window, and calendar features."""
    d = d.copy()
    for l in LAGS:
        d[f"lag_{l}"] = d["y"].shift(l)
    d["roll_7_mean"] = d["y"].shift(1).rolling(7).mean()
    d["roll_14_mean"] = d["y"].shift(1).rolling(14).mean()
    d["roll_7_std"] = d["y"].shift(1).rolling(7).std().fillna(0.0)
    d["dow"] = d.index.dayofweek
    d["is_weekend"] = (d.index.dayofweek >= 4).astype(int)
    d["month"] = d.index.month
    d["day"] = d.index.day
    return d.dropna()


def train_forecaster(df: pd.DataFrame, test_days: int = 60, save_model: bool = True):
    """Train RandomForestRegressor & DummyRegressor on a chronological split."""
    d = make_features(daily_series(df))
    test_days = min(test_days, max(14, len(d) // 4))
    X, y = d.drop(columns="y"), d["y"]
    X_tr, X_te = X.iloc[:-test_days], X.iloc[-test_days:]
    y_tr, y_te = y.iloc[:-test_days], y.iloc[-test_days:]

    if HAS_SKLEARN:
        rf = RandomForestRegressor(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ).fit(X_tr, y_tr)
        dummy = DummyRegressor(strategy="mean").fit(X_tr, y_tr)

        res = {}
        for name, m in [("RandomForest", rf), ("DummyBaseline", dummy)]:
            p = m.predict(X_te)
            res[name] = {
                "MAE": round(float(mean_absolute_error(y_te, p)), 2),
                "RMSE": round(float(np.sqrt(mean_squared_error(y_te, p))), 2),
                "R2": round(float(r2_score(y_te, p)), 3),
            }
        if save_model:
            os.makedirs(MODELS_DIR, exist_ok=True)
            joblib.dump(rf, os.path.join(MODELS_DIR, "forecaster.joblib"))
    else:
        rf = _FallbackForecaster().fit(X_tr, y_tr)
        mean_pred = np.full(len(y_te), float(y_tr.mean()))
        rf_pred = rf.predict(X_te)
        res = {}
        for name, p in [("RandomForest", rf_pred), ("DummyBaseline", mean_pred)]:
            mae = float(np.mean(np.abs(y_te - p)))
            rmse = float(np.sqrt(np.mean((y_te - p) ** 2)))
            ss_res = float(np.sum((y_te - p) ** 2))
            ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2)) + 1e-9
            res[name] = {
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "R2": round(1.0 - ss_res / ss_tot, 3),
            }

    comp = pd.DataFrame(
        {"Actual": y_te.round(2), "Predicted": np.round(rf.predict(X_te), 2)},
        index=y_te.index,
    )
    return rf, res, comp, d


def feature_importance_df(model, feature_cols) -> pd.DataFrame:
    """Return sorted feature importances from the trained Random Forest model."""
    importances = getattr(model, "feature_importances_", np.ones(len(feature_cols)) / len(feature_cols))
    return (
        pd.DataFrame({"Feature": list(feature_cols), "Importance": np.round(importances, 4)})
        .sort_values("Importance", ascending=True)
        .reset_index(drop=True)
    )


def forecast_future(model, df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """Autoregressively forecast daily sales for the next N days."""
    hist = daily_series(df)["y"].copy()
    preds = []
    for _ in range(days):
        nxt = hist.index[-1] + pd.Timedelta(days=1)
        row = {f"lag_{l}": float(hist.iloc[-l]) for l in LAGS}
        row["roll_7_mean"] = float(hist.iloc[-7:].mean())
        row["roll_14_mean"] = float(hist.iloc[-14:].mean())
        row["roll_7_std"] = float(hist.iloc[-7:].std())
        row["dow"] = nxt.dayofweek
        row["is_weekend"] = int(nxt.dayofweek >= 4)
        row["month"] = nxt.month
        row["day"] = nxt.day
        val = max(0.0, float(model.predict(pd.DataFrame([row]))[0]))
        hist.loc[nxt] = val
        preds.append({"Date": nxt, "Forecast": round(val, 2)})
    return pd.DataFrame(preds)
