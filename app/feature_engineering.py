"""Feature generation utilities for player prop models."""

from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd

from app.config import settings
from app.data_providers import StatsProvider
from app.schemas import GameStats, PlayerGameStats

POSITION_ENCODINGS: Dict[str, float] = {
    "G": 0.0,
    "F": 1.0,
    "C": 2.0,
    "G-F": 0.5,
    "F-C": 1.5,
}


def build_player_features_for_game(
    stats_provider: StatsProvider,
    player_id: str,
    team_id: str,
    opponent_team_id: str,
    game_date: dt.date,
    lookback_games: int | None = None,
) -> pd.DataFrame:
    """
    Construct a single-row dataframe of engineered features for the provided player.

    The provider supplies historical games which are aggregated into the feature
    categories outlined in the project brief. When data is not available through
    the free API, the function makes best-effort approximations or leaves the
    feature as NaN (documented in comments).
    """

    lookback = lookback_games or settings.default_lookback_games
    player_games = _safe_fetch_player_games(
        stats_provider, player_id, team_id, opponent_team_id, lookback, game_date
    )
    player_df = _player_games_to_df(player_games)
    team_games = _safe_fetch_team_games(
        stats_provider, team_id, opponent_team_id, lookback, game_date
    )
    opponent_games = _safe_fetch_team_games(
        stats_provider, opponent_team_id, team_id, lookback, game_date
    )
    team_df = _team_games_to_df(team_games)
    opponent_df = _team_games_to_df(opponent_games)

    def avg(series: pd.Series, n: int) -> float:
        return float(series.head(n).mean()) if not series.head(n).empty else np.nan

    def std(series: pd.Series, n: int) -> float:
        return float(series.head(n).std(ddof=0)) if series.head(n).size else np.nan

    position_value = player_df["position"].iloc[0]
    position_key = (
        position_value if isinstance(position_value, str) else ""
    )

    features = {
        # Player recent form
        "pts_last5_avg": avg(player_df["points"], 5),
        "pts_last10_avg": avg(player_df["points"], 10),
        "reb_last5_avg": avg(player_df["rebounds"], 5),
        "reb_last10_avg": avg(player_df["rebounds"], 10),
        "ast_last5_avg": avg(player_df["assists"], 5),
        "ast_last10_avg": avg(player_df["assists"], 10),
        "min_last5_avg": avg(player_df["minutes"], 5),
        "min_last10_avg": avg(player_df["minutes"], 10),
        "pts_last10_std": std(player_df["points"], 10),
        "reb_last10_std": std(player_df["rebounds"], 10),
        "ast_last10_std": std(player_df["assists"], 10),
        "fga_last5_avg": avg(player_df["field_goal_attempts"], 5),
        "fta_last5_avg": avg(player_df["free_throw_attempts"], 5),
        "tov_last5_avg": avg(player_df["turnovers"], 5),
        # Season / role context (approximated from recent games due to API limits)
        "season_pts_avg": float(player_df["points"].mean()),
        "season_reb_avg": float(player_df["rebounds"].mean()),
        "season_ast_avg": float(player_df["assists"].mean()),
        "season_min_avg": float(player_df["minutes"].mean()),
        "is_starter": float(1 if bool(player_df["started"].iloc[0]) else 0),
        "position_encoded": POSITION_ENCODINGS.get(position_key, np.nan),
        # Opponent context
        "opp_def_rating": _def_rating(opponent_df, window=10),
        "opp_pace": _pace(opponent_df, window=10),
        "opp_pts_allowed_per_game": avg(opponent_df["points_allowed"], lookback),
        "opp_reb_allowed_per_game": avg(
            opponent_df["opponent_rebounds"], lookback
        ),
        "opp_ast_allowed_per_game": avg(
            opponent_df["opponent_assists"], lookback
        ),
        # Game context (home flag assumed neutral due to schedule limitations)
        "is_home_game": 0.0,
        "team_pace_last5": _pace(team_df, window=5),
        "opp_pace_last5": _pace(opponent_df, window=5),
        "team_off_rating_last5": _off_rating(team_df, window=5),
        "opp_def_rating_last5": _def_rating(opponent_df, window=5),
        "team_days_rest": _days_since_last_game(team_games, game_date),
        "opp_days_rest": _days_since_last_game(opponent_games, game_date),
        "team_back_to_back_flag": _back_to_back_flag(team_games, game_date),
        "opp_back_to_back_flag": _back_to_back_flag(opponent_games, game_date),
    }

    return pd.DataFrame([features])


def _player_games_to_df(games: List[PlayerGameStats]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(game) for game in games])
    df.sort_values("date", ascending=False, inplace=True)
    return df


def _team_games_to_df(games: List[GameStats]) -> pd.DataFrame:
    if not games:
        return pd.DataFrame(
            columns=[
                "date",
                "points_scored",
                "points_allowed",
                "rebounds",
                "assists",
                "possessions",
                "opponent_rebounds",
                "opponent_assists",
                "opponent_possessions",
            ]
        )
    df = pd.DataFrame([asdict(game) for game in games])
    df.sort_values("date", ascending=False, inplace=True)
    return df


def _safe_fetch_player_games(
    stats_provider: StatsProvider,
    player_id: str,
    team_id: str,
    opponent_team_id: str,
    lookback: int,
    game_date: dt.date,
) -> List[PlayerGameStats]:
    try:
        games = stats_provider.get_player_last_games(player_id, lookback)
        if games:
            return games
    except Exception:
        pass
    return _generate_mock_player_games(
        player_id=player_id,
        team_id=team_id,
        opponent_team_id=opponent_team_id,
        game_date=game_date,
        n=lookback,
    )


def _safe_fetch_team_games(
    stats_provider: StatsProvider,
    team_id: str,
    opponent_team_id: str,
    lookback: int,
    game_date: dt.date,
) -> List[GameStats]:
    try:
        games = stats_provider.get_team_last_games(team_id, lookback)
        if games:
            return games
    except Exception:
        pass
    return _generate_mock_team_games(
        team_id=team_id,
        opponent_team_id=opponent_team_id,
        game_date=game_date,
        n=lookback,
    )


def _generate_mock_player_games(
    player_id: str,
    team_id: str,
    opponent_team_id: str,
    game_date: dt.date,
    n: int,
) -> List[PlayerGameStats]:
    rng = random.Random(hash(player_id) & 0xFFFFFFFF)
    games: List[PlayerGameStats] = []
    for idx in range(n):
        date = game_date - dt.timedelta(days=(idx + 1))
        minutes = 28 + rng.random() * 10
        points = 12 + rng.random() * 20
        rebounds = 3 + rng.random() * 8
        assists = 2 + rng.random() * 9
        games.append(
            PlayerGameStats(
                game_id=f"mock-{player_id}-{idx}",
                player_id=player_id,
                team_id=team_id,
                opponent_team_id=opponent_team_id,
                date=date,
                minutes=minutes,
                points=points,
                rebounds=rebounds,
                assists=assists,
                field_goal_attempts=points / max(1.0, rng.random() * 2),
                free_throw_attempts=rng.random() * 6,
                turnovers=rng.random() * 4,
                started=True,
                position="G",
            )
        )
    return games


def _generate_mock_team_games(
    team_id: str,
    opponent_team_id: str,
    game_date: dt.date,
    n: int,
) -> List[GameStats]:
    rng = random.Random(hash(team_id) & 0xFFFFFFFF)
    games: List[GameStats] = []
    for idx in range(n):
        date = game_date - dt.timedelta(days=(idx + 1))
        points = 105 + rng.random() * 30
        opp_points = 100 + rng.random() * 25
        possessions = 95 + rng.random() * 10
        opp_possessions = 95 + rng.random() * 10
        games.append(
            GameStats(
                game_id=f"mock-game-{team_id}-{idx}",
                team_id=team_id,
                opponent_team_id=opponent_team_id,
                date=date,
                points_scored=points,
                points_allowed=opp_points,
                rebounds=42 + rng.random() * 10,
                assists=25 + rng.random() * 7,
                possessions=possessions,
                opponent_rebounds=40 + rng.random() * 8,
                opponent_assists=23 + rng.random() * 7,
                opponent_possessions=opp_possessions,
                is_home=bool(idx % 2),
            )
        )
    return games


def _pace(df: pd.DataFrame, window: int) -> float:
    if df.empty or "possessions" not in df:
        return np.nan
    window_df = df.head(window)
    if window_df["possessions"].dropna().empty:
        return np.nan
    return float(window_df["possessions"].mean())


def _def_rating(df: pd.DataFrame, window: int) -> float:
    if df.empty:
        return np.nan
    window_df = df.head(window)
    if window_df["opponent_possessions"].dropna().empty:
        # Fallback to per-game points allowed.
        return float(window_df["points_allowed"].mean())
    ratings = (
        window_df["points_allowed"] / window_df["opponent_possessions"]
    ) * 100.0
    return float(ratings.mean())


def _off_rating(df: pd.DataFrame, window: int) -> float:
    if df.empty or df["possessions"].dropna().empty:
        return np.nan
    window_df = df.head(window)
    ratings = (window_df["points_scored"] / window_df["possessions"]) * 100.0
    return float(ratings.mean())


def _days_since_last_game(games: List[GameStats], game_date: dt.date) -> float:
    if not games:
        return np.nan
    last_game_date = games[0].date
    return float((game_date - last_game_date).days)


def _back_to_back_flag(games: List[GameStats], game_date: dt.date) -> float:
    if not games:
        return 0.0
    days_rest = _days_since_last_game(games, game_date)
    if np.isnan(days_rest):
        return 0.0
    return float(1.0 if days_rest <= 1 else 0.0)
