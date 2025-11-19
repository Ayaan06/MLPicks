"""Plot hit-rate calibration curves by confidence bucket."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

STAT_TYPES = ["points", "rebounds", "assists"]
BUCKET_ORDER = ["50-60", "60-70", "70-80", "80+"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot confidence calibration.")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("backtesting/results_player_props.csv"),
        help="CSV produced by backtest_player_props.py.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("backtesting/confidence_calibration.png"),
        help="File to store the generated plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.results_path)
    if df.empty:
        print("Results CSV is empty. Run the backtest first.")
        return

    summary = (
        df.groupby(["stat_type", "confidence_bucket"])["hit"]
        .mean()
        .reindex(
            pd.MultiIndex.from_product([STAT_TYPES, BUCKET_ORDER]),
            fill_value=float("nan"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for stat in STAT_TYPES:
        subset = summary[summary["stat_type"] == stat]
        ax.plot(
            subset["confidence_bucket"],
            subset["hit"],
            marker="o",
            label=stat.title(),
        )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Hit Rate")
    ax.set_xlabel("Confidence Bucket")
    ax.set_title("Prop Hit Rate vs Confidence Bucket")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output_path)
    print(f"Saved calibration plot to {args.output_path}")


if __name__ == "__main__":
    main()
