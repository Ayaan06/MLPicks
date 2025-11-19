"""Train the rebounds regression model."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from app import models as model_store

ID_COLUMNS = {
    "player_id",
    "team_id",
    "opponent_team_id",
    "game_date",
    "points_actual",
    "rebounds_actual",
    "assists_actual",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rebounds projection model.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/player_games_rebounds.csv"),
        help="CSV file containing training rows.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0.0",
        help="Model version tag for metadata tracking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data_path, parse_dates=["game_date"])
    feature_cols = [col for col in df.columns if col not in ID_COLUMNS]

    model, metrics = model_store.train_regression_model(
        df=df,
        feature_cols=feature_cols,
        target_col="rebounds_actual",
    )
    sigma = metrics["residual_std"]
    model_store.save_model(
        model=model,
        sigma=sigma,
        model_path=model_store.REBOUNDS_MODEL_PATH,
        sigma_path=model_store.REBOUNDS_SIGMA_PATH,
    )
    _persist_feature_importance(model, feature_cols)
    _update_metadata(args.version, df, metrics)

    print("Training complete.")
    print(f"MAE: {metrics['mae']:.3f} | RMSE: {metrics['rmse']:.3f} | Sigma: {sigma:.3f}")


def _persist_feature_importance(model, feature_cols: list[str]) -> None:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        scores = {
            feature: float(importance)
            for feature, importance in zip(feature_cols, importances)
        }
    else:
        scores = {feature: 0.0 for feature in feature_cols}
    model_store.save_feature_importance_scores("rebounds", scores)


def _update_metadata(version: str, df: pd.DataFrame, metrics: dict) -> None:
    min_date = df["game_date"].min().date()
    max_date = df["game_date"].max().date()
    payload = {
        "version": version,
        "trained_on": dt.date.today().isoformat(),
        "data_range": f"{min_date.isoformat()} to {max_date.isoformat()}",
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
    }
    model_store.update_metadata_entry("rebounds_model", payload)


if __name__ == "__main__":
    main()
