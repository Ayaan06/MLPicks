"""Backtesting script evaluating historical player prop performance."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List

import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import models as model_store
from app.data_providers import get_default_stats_provider
from app.feature_engineering import build_player_features_for_game
from app.utils import make_prop_pick

STAT_TYPES = ["points", "rebounds", "assists"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest player prop picks.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/backtesting_player_props.csv"),
        help="CSV containing historical games, prop lines, and actual results.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("backtesting/results_player_props.csv"),
        help="Where to store per-prop backtest results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data_path, parse_dates=["game_date"])
    stats_provider = get_default_stats_provider()
    models = {
        stat: loader()
        for stat, loader in {
            "points": model_store.load_points_model,
            "rebounds": model_store.load_rebounds_model,
            "assists": model_store.load_assists_model,
        }.items()
    }

    results: List[Dict[str, object]] = []
    for record in df.to_dict(orient="records"):
        feature_frame = build_player_features_for_game(
            stats_provider=stats_provider,
            player_id=str(record["player_id"]),
            team_id=str(record["team_id"]),
            opponent_team_id=str(record["opponent_team_id"]),
            game_date=_coerce_date(record["game_date"]),
        )
        for stat in STAT_TYPES:
            line_value = record.get(f"{stat}_line")
            actual_value = record.get(f"{stat}_actual")
            if line_value is None or pd.isna(line_value):
                continue
            if actual_value is None or pd.isna(actual_value):
                continue
            model, sigma = models[stat]
            pred = float(model.predict(feature_frame)[0])
            pick = make_prop_pick(pred_mean=pred, line=float(line_value), sigma=sigma)
            hit = _is_hit(pick["pick_side"], float(actual_value), float(line_value))
            results.append(
                {
                    "player_id": record["player_id"],
                    "team_id": record["team_id"],
                    "opponent_team_id": record["opponent_team_id"],
                    "game_date": record["game_date"],
                    "stat_type": stat,
                    "line": float(line_value),
                    "actual": float(actual_value),
                    "model_projection": pick["model_projection"],
                    "pick_side": pick["pick_side"],
                    "confidence_score": pick["confidence_score"],
                    "confidence_label": pick["confidence_label"],
                    "edge_value": pick["edge_value"],
                    "edge_prob": pick["edge_prob"],
                    "risk_label": pick["risk_label"],
                    "hit": hit,
                    "confidence_bucket": _confidence_bucket(
                        pick["confidence_score"]
                    ),
                }
            )

    if not results:
        print("No props evaluated. Check input data.")
        return

    results_df = pd.DataFrame(results)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_path, index=False)

    print("=== Hit Rate by Stat Type ===")
    print(results_df.groupby("stat_type")["hit"].mean().round(3))

    print("\n=== Hit Rate by Confidence Bucket ===")
    print(
        results_df.groupby(["stat_type", "confidence_bucket"])["hit"]
        .mean()
        .round(3)
    )
    print(f"\nSaved detailed results to {args.output_path}")


def _coerce_date(value: object) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def _is_hit(pick_side: str, actual: float, line: float) -> bool:
    if pick_side == "over":
        return actual > line
    return actual < line


def _confidence_bucket(score: float) -> str:
    if score >= 80:
        return "80+"
    if score >= 70:
        return "70-80"
    if score >= 60:
        return "60-70"
    return "50-60"


if __name__ == "__main__":
    main()
