"""ML model utilities for training and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.base import BaseEstimator

POINTS_MODEL_PATH = Path("models/points_model.joblib")
POINTS_SIGMA_PATH = Path("models/points_sigma.json")
REBOUNDS_MODEL_PATH = Path("models/rebounds_model.joblib")
REBOUNDS_SIGMA_PATH = Path("models/rebounds_sigma.json")
ASSISTS_MODEL_PATH = Path("models/assists_model.joblib")
ASSISTS_SIGMA_PATH = Path("models/assists_sigma.json")
POINTS_FEATURE_IMPORTANCE_PATH = Path("models/points_feature_importance.json")
REBOUNDS_FEATURE_IMPORTANCE_PATH = Path("models/rebounds_feature_importance.json")
ASSISTS_FEATURE_IMPORTANCE_PATH = Path("models/assists_feature_importance.json")
METADATA_PATH = Path("models/metadata.json")

FEATURE_IMPORTANCE_PATHS = {
    "points": POINTS_FEATURE_IMPORTANCE_PATH,
    "rebounds": REBOUNDS_FEATURE_IMPORTANCE_PATH,
    "assists": ASSISTS_FEATURE_IMPORTANCE_PATH,
}


def train_regression_model(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[BaseEstimator, Dict[str, float]]:
    """
    Split the dataframe, train a regressor, and compute evaluation metrics.

    The split is chronological by default (no shuffling) which is more realistic
    for time-series like data. Residual std is returned via the metrics dict.
    """

    df_sorted = df.sort_values("game_date")
    X = df_sorted[list(feature_cols)]
    y = df_sorted[target_col]

    split_index = int(len(df_sorted) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = GradientBoostingRegressor(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(root_mean_squared_error(y_test, y_pred)),
        "residual_std": float(np.std(residuals)),
    }
    return model, metrics


def save_model(
    model: BaseEstimator,
    sigma: float,
    model_path: Path,
    sigma_path: Path,
) -> None:
    """Persist the model artifact and its residual std (sigma)."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    sigma_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    with sigma_path.open("w", encoding="utf-8") as fp:
        json.dump({"sigma": sigma}, fp)


def load_model(model_path: Path, sigma_path: Path) -> Tuple[BaseEstimator, float]:
    """Load a trained model and sigma."""
    model = joblib.load(model_path)
    with sigma_path.open("r", encoding="utf-8") as fp:
        sigma_data = json.load(fp)
    return model, float(sigma_data.get("sigma", 0.0))


def load_points_model() -> Tuple[BaseEstimator, float]:
    return load_model(POINTS_MODEL_PATH, POINTS_SIGMA_PATH)


def load_rebounds_model() -> Tuple[BaseEstimator, float]:
    return load_model(REBOUNDS_MODEL_PATH, REBOUNDS_SIGMA_PATH)


def load_assists_model() -> Tuple[BaseEstimator, float]:
    return load_model(ASSISTS_MODEL_PATH, ASSISTS_SIGMA_PATH)


def save_feature_importance_scores(stat_type: str, scores: Dict[str, float]) -> None:
    """Persist feature importances for downstream explainability."""
    path = FEATURE_IMPORTANCE_PATHS[stat_type]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(scores, fp)


def load_feature_importance(stat_type: str, top_k: int | None = None) -> List[str]:
    """Return ordered feature names based on saved importance."""
    path = FEATURE_IMPORTANCE_PATHS.get(stat_type)
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fp:
        scores = json.load(fp)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    names = [name for name, _ in ordered]
    if top_k is None:
        return names
    return names[:top_k]


def load_metadata() -> Dict[str, Dict[str, object]]:
    """Load stored model metadata."""
    if not METADATA_PATH.exists():
        return {}
    with METADATA_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def update_metadata_entry(model_key: str, payload: Dict[str, object]) -> None:
    """Update (or create) registry entry for a model."""
    metadata = load_metadata()
    metadata[model_key] = payload
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
