"""Simple local caching layer for API responses."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

CACHE_DIR = Path("cache")
TEAM_CACHE_DIR = CACHE_DIR / "teams"
PLAYER_CACHE_DIR = CACHE_DIR / "players"


def save_team_games(team_id: str, games_df: pd.DataFrame) -> None:
    """Persist team games for future reuse."""
    if games_df.empty:
        return
    TEAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = TEAM_CACHE_DIR / f"{team_id}.csv"
    games_df.to_csv(path, index=False)


def load_team_games(team_id: str) -> Optional[pd.DataFrame]:
    """Return cached team games if available."""
    path = TEAM_CACHE_DIR / f"{team_id}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def save_player_games(player_id: str, games_df: pd.DataFrame) -> None:
    """Persist player games."""
    if games_df.empty:
        return
    PLAYER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = PLAYER_CACHE_DIR / f"{player_id}.csv"
    games_df.to_csv(path, index=False)


def load_player_games(player_id: str) -> Optional[pd.DataFrame]:
    """Return cached player games if available."""
    path = PLAYER_CACHE_DIR / f"{player_id}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df
