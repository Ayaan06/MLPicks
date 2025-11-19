"""
NBA betting projection pipeline combining data ingestion, ML, and Gradio UI.

This single-file script downloads NBA stats, trains models, scores players
for the next game, and surfaces everything via a browser interface.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import (
    commonteamroster,
    leaguegamelog,
    leaguedashteamstats,
    playergamelog,
    scoreboardv2,
    scoreboardv3,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import gradio as gr


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RECENT_GAMES_WINDOW = 5
PLAYER_STATS_GAMES = 10
MAX_LOOKAHEAD_DAYS = 7
HISTORICAL_GAMES_LIMIT = 8000
RANDOM_SEED = 42

MODEL_PATHS = {
    "points": MODEL_DIR / "model_points.pkl",
    "rebounds": MODEL_DIR / "model_rebounds.pkl",
    "assists": MODEL_DIR / "model_assists.pkl",
    "pra": MODEL_DIR / "model_pra.pkl",
}

FEATURE_COLUMNS = [
    "recent_points_avg",
    "recent_points_std",
    "recent_rebounds_avg",
    "recent_rebounds_std",
    "recent_assists_avg",
    "recent_assists_std",
    "recent_pra_avg",
    "recent_pra_std",
    "recent_minutes_avg",
    "recent_minutes_std",
    "recent_fga_avg",
    "recent_fg3a_avg",
    "recent_fta_avg",
    "home_indicator",
    "days_rest",
    "opponent_def_rating",
]

TARGET_COLUMNS = {
    "points": "label_points_over",
    "rebounds": "label_rebounds_over",
    "assists": "label_assists_over",
    "pra": "label_pra_over",
}

DISPLAY_COLUMN_ORDER = [
    "player",
    "team",
    "opponent",
    "line_points",
    "p_over_points",
    "p_under_points",
    "line_rebounds",
    "p_over_rebounds",
    "p_under_rebounds",
    "line_assists",
    "p_over_assists",
    "p_under_assists",
    "line_pra",
    "p_over_pra",
    "p_under_pra",
]

DISPLAY_COLUMN_RENAMES = {
    "player": "Player",
    "team": "Team",
    "opponent": "Opp",
    "line_points": "Pts line",
    "p_over_points": "Pts over %",
    "p_under_points": "Pts under %",
    "line_rebounds": "Reb line",
    "p_over_rebounds": "Reb over %",
    "p_under_rebounds": "Reb under %",
    "line_assists": "Ast line",
    "p_over_assists": "Ast over %",
    "p_under_assists": "Ast under %",
    "line_pra": "PRA line",
    "p_over_pra": "PRA over %",
    "p_under_pra": "PRA under %",
}

APP_THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
)

MINIMAL_UI_CSS = """
.gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e2e8f0;
}
.page-title h1 {
    margin-bottom: 0.25rem;
    font-size: 2rem;
}
.page-subtitle p {
    color: #94a3b8;
    margin-top: 0;
}
.card {
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 16px;
    padding: 1rem 1.25rem;
    box-shadow: 0 12px 30px rgba(2, 6, 23, 0.45);
}
.game-card {
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.status-card p {
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.pill-button button {
    width: 100%;
}
.projection-table table {
    font-variant-numeric: tabular-nums;
}
.projection-table thead th {
    position: sticky;
    top: 0;
    background: #111827;
    color: #f8fafc;
    z-index: 1;
}
.callout p {
    font-size: 1rem;
    color: #e2e8f0;
}
.custom-inputs .gradio-dropdown label,
.custom-inputs .gradio-slider label {
    color: #cbd5f5;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def get_current_season_string(today: Optional[date] = None) -> str:
    today = today or datetime.today().date()
    year = today.year
    start_year = year if today.month >= 10 else year - 1
    end_year_short = str(start_year + 1)[-2:]
    return f"{start_year}-{end_year_short}"


CURRENT_SEASON = get_current_season_string()


def safe_api_call(endpoint_cls, max_retries: int = 3, sleep_seconds: int = 2, **kwargs):
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return endpoint_cls(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = sleep_seconds * (attempt + 1)
            log(f"{endpoint_cls.__name__} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    log(f"{endpoint_cls.__name__} failed after retries: {last_exc}")
    return None


ALL_TEAMS = static_teams.get_teams()
TEAM_ID_TO_INFO = {team["id"]: team for team in ALL_TEAMS}
TEAM_ABBR_TO_ID = {team["abbreviation"]: team["id"] for team in ALL_TEAMS}


def parse_minutes_to_float(minutes_str: str) -> float:
    if not minutes_str or minutes_str == "0":
        return 0.0
    try:
        minute, second = minutes_str.split(":")
        return int(minute) + int(second) / 60.0
    except Exception:  # noqa: BLE001
        return 0.0


def parse_game_date(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%b %d, %Y")
    except ValueError:
        return None


def parse_matchup(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    parts = text.split()
    if len(parts) >= 3:
        return parts[0], parts[-1]
    return "", ""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GameInfo:
    game_id: str
    start_time_utc: datetime
    start_time_local: datetime
    home_team_id: int
    home_team_name: str
    home_team_tricode: str
    away_team_id: int
    away_team_name: str
    away_team_tricode: str

    @property
    def label(self) -> str:
        start = self.start_time_local.strftime("%Y-%m-%d %I:%M %p %Z")
        return (
            f"{self.home_team_name} ({self.home_team_tricode}) vs "
            f"{self.away_team_name} ({self.away_team_tricode}) — {start}"
        )


# ---------------------------------------------------------------------------
# nba_api data layer
# ---------------------------------------------------------------------------


def get_next_game(max_days: int = MAX_LOOKAHEAD_DAYS) -> Optional[GameInfo]:
    now_utc = datetime.now(timezone.utc)
    today = now_utc.date()

    for offset in range(max_days):
        target_date = today + timedelta(days=offset)
        date_v3 = target_date.isoformat()
        date_v2 = target_date.strftime("%m/%d/%Y")

        sb3 = safe_api_call(scoreboardv3.ScoreboardV3, game_date=date_v3)
        if sb3 is None:
            continue
        sb3_games = sb3.game_header.get_data_frame()
        if sb3_games.empty:
            continue

        sb2 = safe_api_call(scoreboardv2.ScoreboardV2, game_date=date_v2)
        sb2_games = sb2.game_header.get_data_frame() if sb2 else pd.DataFrame()

        for _, row in sb3_games.iterrows():
            game_time = (
                datetime.fromisoformat(row["gameTimeUTC"].replace("Z", "+00:00"))
                if row.get("gameTimeUTC")
                else None
            )
            if game_time is None or game_time <= now_utc:
                continue

            game_id = row["gameId"]
            match_rows = sb2_games[sb2_games["GAME_ID"] == game_id]
            if match_rows.empty:
                continue

            match_row = match_rows.iloc[0]
            home_id = int(match_row["HOME_TEAM_ID"])
            away_id = int(match_row["VISITOR_TEAM_ID"])

            def _team(team_id: int) -> Tuple[str, str]:
                info = TEAM_ID_TO_INFO.get(team_id, {})
                return info.get("full_name", f"Team {team_id}"), info.get(
                    "abbreviation", ""
                )

            home_name, home_tri = _team(home_id)
            away_name, away_tri = _team(away_id)
            local_time = game_time.astimezone()

            return GameInfo(
                game_id=game_id,
                start_time_utc=game_time,
                start_time_local=local_time,
                home_team_id=home_id,
                home_team_name=home_name,
                home_team_tricode=home_tri,
                away_team_id=away_id,
                away_team_name=away_name,
                away_team_tricode=away_tri,
            )

    log("No upcoming games found in the configured window.")
    return None


def get_team_roster(team_id: int) -> List[Dict]:
    endpoint = safe_api_call(
        commonteamroster.CommonTeamRoster, season=CURRENT_SEASON, team_id=team_id
    )
    if endpoint is None:
        return []
    df = endpoint.common_team_roster.get_data_frame()
    roster = []
    for _, row in df.iterrows():
        roster.append(
            {
                "player_id": str(row["PLAYER_ID"]),
                "player_name": row["PLAYER"],
                "team_id": team_id,
            }
        )
    return roster


PLAYER_STATS_CACHE: Dict[Tuple[str, int], Optional[Dict]] = {}
LEAGUE_LOGS_CACHE: Dict[str, Optional[pd.DataFrame]] = {}


def _build_stats_from_games(games: pd.DataFrame) -> Optional[Dict]:
    if games.empty:
        return None

    numeric_cols = ["PTS", "REB", "AST", "FGA", "FG3A", "FTA"]
    for col in numeric_cols:
        games[col] = pd.to_numeric(games[col], errors="coerce")
    games["MINUTES"] = games["MIN"].apply(parse_minutes_to_float)
    games["PRA"] = games["PTS"] + games["REB"] + games["AST"]

    def _avg(series: pd.Series) -> float:
        return float(series.mean()) if not series.empty else 0.0

    def _std(series: pd.Series) -> float:
        return float(series.std(ddof=0)) if len(series) > 1 else 0.0

    last_game_date = parse_game_date(games.iloc[0]["GAME_DATE"])

    return {
        "games_played": int(len(games)),
        "points_avg": _avg(games["PTS"]),
        "points_std": _std(games["PTS"]),
        "rebounds_avg": _avg(games["REB"]),
        "rebounds_std": _std(games["REB"]),
        "assists_avg": _avg(games["AST"]),
        "assists_std": _std(games["AST"]),
        "pra_avg": _avg(games["PRA"]),
        "pra_std": _std(games["PRA"]),
        "minutes_avg": _avg(games["MINUTES"]),
        "minutes_std": _std(games["MINUTES"]),
        "fga_avg": _avg(games["FGA"]),
        "fg3a_avg": _avg(games["FG3A"]),
        "fta_avg": _avg(games["FTA"]),
        "last_game_date": last_game_date.date() if last_game_date else None,
    }


def _load_league_logs_for_season(season: str) -> Optional[pd.DataFrame]:
    if season in LEAGUE_LOGS_CACHE:
        return LEAGUE_LOGS_CACHE[season]
    log(f"Falling back to LeagueGameLog cache for season {season}...")
    endpoint = safe_api_call(
        leaguegamelog.LeagueGameLog,
        counter=0,
        direction="DESC",
        player_or_team_abbreviation="P",
        season=season,
        season_type_all_star="Regular Season",
    )
    if endpoint is None:
        LEAGUE_LOGS_CACHE[season] = None
        return None
    df = endpoint.get_data_frames()[0]
    LEAGUE_LOGS_CACHE[season] = df
    return df


def _stats_from_league_logs(player_id: str, num_games: int, season: str) -> Optional[Dict]:
    league_logs = _load_league_logs_for_season(season)
    if league_logs is None or league_logs.empty:
        return None
    pid = int(player_id)
    player_logs = (
        league_logs[league_logs["PLAYER_ID"] == pid]
        .sort_values("GAME_DATE", ascending=False)
        .head(num_games)
        .copy()
    )
    return _build_stats_from_games(player_logs)


def get_player_recent_stats(
    player_id: str, num_games: int = PLAYER_STATS_GAMES, season: str = CURRENT_SEASON
) -> Optional[Dict]:
    cache_key = (player_id, num_games)
    if cache_key in PLAYER_STATS_CACHE:
        return PLAYER_STATS_CACHE[cache_key]

    endpoint = safe_api_call(
        playergamelog.PlayerGameLog,
        player_id=player_id,
        season=season,
        season_type_all_star="Regular Season",
    )
    stats: Optional[Dict]
    if endpoint is None:
        stats = _stats_from_league_logs(player_id, num_games, season)
        PLAYER_STATS_CACHE[cache_key] = stats
        return stats

    df = endpoint.player_game_log.get_data_frame()
    if df.empty:
        stats = _stats_from_league_logs(player_id, num_games, season)
        PLAYER_STATS_CACHE[cache_key] = stats
        return stats

    recent = df.head(num_games).copy()
    stats = _build_stats_from_games(recent)
    PLAYER_STATS_CACHE[cache_key] = stats
    return stats


def get_team_defensive_ratings(season: str = CURRENT_SEASON) -> Dict[int, float]:
    endpoint = safe_api_call(
        leaguedashteamstats.LeagueDashTeamStats,
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
    )
    if endpoint is None:
        return {}
    df = endpoint.get_data_frames()[0]
    return {int(row["TEAM_ID"]): float(row["DEF_RATING"]) for _, row in df.iterrows()}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_feature_row_for_player(
    player_info: Dict,
    player_stats: Optional[Dict],
    opponent_team_id: int,
    game_date: datetime,
    is_home: bool,
    team_def_ratings: Dict[int, float],
) -> Optional[Dict]:
    if not player_stats or player_stats["games_played"] < RECENT_GAMES_WINDOW:
        return None

    opponent_rating = team_def_ratings.get(opponent_team_id)
    if opponent_rating is None:
        opponent_rating = (
            float(np.mean(list(team_def_ratings.values())))
            if team_def_ratings
            else 0.0
        )

    last_game_date = player_stats.get("last_game_date")
    rest_days = (
        (game_date.date() - last_game_date).days
        if last_game_date
        else RECENT_GAMES_WINDOW
    )

    feature_row = {
        "player_id": player_info["player_id"],
        "player_name": player_info["player_name"],
        "team_id": player_info["team_id"],
        "opponent_id": opponent_team_id,
        "line_points": player_stats["points_avg"],
        "line_rebounds": player_stats["rebounds_avg"],
        "line_assists": player_stats["assists_avg"],
        "line_pra": player_stats["pra_avg"],
        "recent_points_avg": player_stats["points_avg"],
        "recent_points_std": player_stats["points_std"],
        "recent_rebounds_avg": player_stats["rebounds_avg"],
        "recent_rebounds_std": player_stats["rebounds_std"],
        "recent_assists_avg": player_stats["assists_avg"],
        "recent_assists_std": player_stats["assists_std"],
        "recent_pra_avg": player_stats["pra_avg"],
        "recent_pra_std": player_stats["pra_std"],
        "recent_minutes_avg": player_stats["minutes_avg"],
        "recent_minutes_std": player_stats["minutes_std"],
        "recent_fga_avg": player_stats["fga_avg"],
        "recent_fg3a_avg": player_stats["fg3a_avg"],
        "recent_fta_avg": player_stats["fta_avg"],
        "home_indicator": 1 if is_home else 0,
        "days_rest": float(rest_days),
        "opponent_def_rating": opponent_rating,
    }
    return feature_row


def build_historical_dataset(
    season: str = CURRENT_SEASON, recent_games: int = RECENT_GAMES_WINDOW
) -> pd.DataFrame:
    log("Fetching league-wide player game logs for training...")
    gamelog = safe_api_call(
        leaguegamelog.LeagueGameLog,
        counter=0,
        direction="DESC",
        player_or_team_abbreviation="P",
        season=season,
        season_type_all_star="Regular Season",
    )
    if gamelog is None:
        raise RuntimeError("Failed to download league logs.")

    df = gamelog.get_data_frames()[0]
    if HISTORICAL_GAMES_LIMIT:
        df = df.head(HISTORICAL_GAMES_LIMIT).copy()

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["MINUTES"] = df["MIN"].apply(parse_minutes_to_float)
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]

    team_def_ratings = get_team_defensive_ratings(season)
    avg_def_rating = (
        float(np.mean(list(team_def_ratings.values()))) if team_def_ratings else 0.0
    )

    records = []
    grouped = df.sort_values(["PLAYER_ID", "GAME_DATE"]).groupby("PLAYER_ID")
    for player_id, group in grouped:
        group = group.reset_index(drop=True)
        if len(group) <= recent_games:
            continue
        for idx in range(recent_games, len(group)):
            history = group.iloc[idx - recent_games : idx]
            current = group.iloc[idx]
            opp_abbr = parse_matchup(current["MATCHUP"])[1]
            opponent_id = TEAM_ABBR_TO_ID.get(opp_abbr, -1)
            opponent_rating = team_def_ratings.get(opponent_id, avg_def_rating)
            prev_game_date = history.iloc[-1]["GAME_DATE"]
            rest_days = (current["GAME_DATE"] - prev_game_date).days

            feature_row = {
                "player_id": str(player_id),
                "player_name": current["PLAYER_NAME"],
                "team_id": int(current["TEAM_ID"]),
                "opponent_id": opponent_id,
                "recent_points_avg": float(history["PTS"].mean()),
                "recent_points_std": float(history["PTS"].std(ddof=0)),
                "recent_rebounds_avg": float(history["REB"].mean()),
                "recent_rebounds_std": float(history["REB"].std(ddof=0)),
                "recent_assists_avg": float(history["AST"].mean()),
                "recent_assists_std": float(history["AST"].std(ddof=0)),
                "recent_pra_avg": float(history["PRA"].mean()),
                "recent_pra_std": float(history["PRA"].std(ddof=0)),
                "recent_minutes_avg": float(history["MINUTES"].mean()),
                "recent_minutes_std": float(history["MINUTES"].std(ddof=0)),
                "recent_fga_avg": float(history["FGA"].mean()),
                "recent_fg3a_avg": float(history["FG3A"].mean()),
                "recent_fta_avg": float(history["FTA"].mean()),
                "home_indicator": 1 if "vs." in current["MATCHUP"] else 0,
                "days_rest": float(rest_days),
                "opponent_def_rating": opponent_rating,
            }

            line_points = feature_row["recent_points_avg"]
            line_reb = feature_row["recent_rebounds_avg"]
            line_ast = feature_row["recent_assists_avg"]
            line_pra = feature_row["recent_pra_avg"]

            feature_row.update(
                {
                    "line_points": line_points,
                    "line_rebounds": line_reb,
                    "line_assists": line_ast,
                    "line_pra": line_pra,
                    "label_points_over": 1
                    if float(current["PTS"]) > line_points
                    else 0,
                    "label_rebounds_over": 1
                    if float(current["REB"]) > line_reb
                    else 0,
                    "label_assists_over": 1
                    if float(current["AST"]) > line_ast
                    else 0,
                    "label_pra_over": 1
                    if float(current["PRA"]) > line_pra
                    else 0,
                }
            )
            records.append(feature_row)

    dataset = pd.DataFrame(records)
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset.fillna(0.0, inplace=True)
    log(f"Built dataset with {len(dataset)} samples.")
    return dataset


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------


def train_models(force: bool = False) -> Dict[str, Pipeline]:
    dataset = build_historical_dataset()
    if dataset.empty:
        raise RuntimeError("Training dataset is empty.")

    trained: Dict[str, Pipeline] = {}
    for market, target_col in TARGET_COLUMNS.items():
        model_path = MODEL_PATHS[market]
        if model_path.exists() and not force:
            continue

        X = dataset[FEATURE_COLUMNS]
        y = dataset[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
            ]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            roc_auc = float("nan")

        log(
            f"{market} model — acc={accuracy:.3f}, roc_auc={roc_auc:.3f}, "
            f"cm={cm.tolist()}"
        )
        joblib.dump(pipeline, model_path)
        trained[market] = pipeline
        log(f"Saved {market} model to {model_path}")
    return trained


def load_models(ensure_trained: bool = True) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}
    missing = []
    for market, path in MODEL_PATHS.items():
        if path.exists():
            models[market] = joblib.load(path)
        else:
            missing.append(market)

    if missing and ensure_trained:
        log(f"Missing models for {missing}; training now...")
        models.update(train_models())

    if not models:
        raise RuntimeError("No models available. Run with --train to build them.")
    return models


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------


PLAYER_FEATURE_CACHE: Dict[str, Dict] = {}


def generate_predictions_for_next_game(models: Dict[str, Pipeline]) -> Dict:
    game = get_next_game()
    if game is None:
        return {
            "game_info": "No upcoming game scheduled within the search window.",
            "table": pd.DataFrame(),
            "player_options": [],
        }

    team_def_ratings = get_team_defensive_ratings()
    rows = []
    feature_cache: Dict[str, Dict] = {}

    for team_id, opponent_id, is_home in [
        (game.home_team_id, game.away_team_id, True),
        (game.away_team_id, game.home_team_id, False),
    ]:
        roster = get_team_roster(team_id)
        for player in roster:
            stats = get_player_recent_stats(player["player_id"])
            feature_row = build_feature_row_for_player(
                player_info=player,
                player_stats=stats,
                opponent_team_id=opponent_id,
                game_date=game.start_time_utc,
                is_home=is_home,
                team_def_ratings=team_def_ratings,
            )
            if feature_row is None:
                continue

            features_vector = pd.DataFrame(
                [
                    {feature: float(feature_row[feature]) for feature in FEATURE_COLUMNS}
                ]
            )

            row = {
                "player": player["player_name"],
                "team": TEAM_ID_TO_INFO.get(team_id, {}).get("abbreviation", str(team_id)),
                "opponent": TEAM_ID_TO_INFO.get(opponent_id, {}).get(
                    "abbreviation", str(opponent_id)
                ),
                "line_points": feature_row["line_points"],
                "line_rebounds": feature_row["line_rebounds"],
                "line_assists": feature_row["line_assists"],
                "line_pra": feature_row["line_pra"],
            }

            for market, model in models.items():
                proba = model.predict_proba(features_vector)[0, 1]
                row[f"p_over_{market}"] = float(proba)
                row[f"p_under_{market}"] = float(1.0 - proba)

            rows.append(row)

            display_name = f"{player['player_name']} ({row['team']})"
            feature_cache[display_name] = {
                "team": row["team"],
                "opponent": row["opponent"],
                "lines": {
                    "points": row["line_points"],
                    "rebounds": row["line_rebounds"],
                    "assists": row["line_assists"],
                    "pra": row["line_pra"],
                },
                "stats": {
                    "points_avg": feature_row["recent_points_avg"],
                    "points_std": feature_row["recent_points_std"],
                    "rebounds_avg": feature_row["recent_rebounds_avg"],
                    "rebounds_std": feature_row["recent_rebounds_std"],
                    "assists_avg": feature_row["recent_assists_avg"],
                    "assists_std": feature_row["recent_assists_std"],
                    "pra_avg": feature_row["recent_pra_avg"],
                    "pra_std": feature_row["recent_pra_std"],
                },
            }

    if not rows:
        return {
            "game_info": game.label,
            "table": pd.DataFrame(),
            "player_options": [],
        }

    df = pd.DataFrame(rows)
    df = df[DISPLAY_COLUMN_ORDER]
    df.sort_values(by="p_over_points", ascending=False, inplace=True)

    display_df = df.copy()
    prob_cols = [col for col in display_df.columns if col.startswith("p_")]
    line_cols = [col for col in display_df.columns if col.startswith("line_")]
    if prob_cols:
        display_df[prob_cols] = (display_df[prob_cols] * 100).round(1)
    if line_cols:
        display_df[line_cols] = display_df[line_cols].round(1)
    display_df.rename(columns=DISPLAY_COLUMN_RENAMES, inplace=True)

    PLAYER_FEATURE_CACHE.clear()
    PLAYER_FEATURE_CACHE.update(feature_cache)

    return {
        "game_info": game.label,
        "table": display_df,
        "player_options": list(feature_cache.keys()),
    }


def compute_custom_line_probability(player_label: str, market: str, line_value: float) -> str:
    if not player_label:
        return "Select a player first."
    player_data = PLAYER_FEATURE_CACHE.get(player_label)
    if not player_data:
        return "No cached stats for that player yet."

    stats = player_data["stats"]
    avg = stats.get(f"{market}_avg", 0.0)
    std = stats.get(f"{market}_std", 0.0) or 1e-3
    z = (line_value - avg) / std
    p_under = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    p_over = 1 - p_under
    return (
        f"{player_label} vs {player_data['opponent']} | line {line_value:.1f} "
        f"{market.upper()} — P(Over)={p_over:.2f}, P(Under)={p_under:.2f}"
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def create_interface(models: Dict[str, Pipeline]) -> gr.Blocks:
    def refresh():
        result = generate_predictions_for_next_game(models)
        if result["table"].empty:
            status = "Unable to compute projections right now."
        else:
            status = f"Generated {len(result['table'])} player projections."
        dropdown_state = gr.update(
            choices=result["player_options"],
            value=result["player_options"][0] if result["player_options"] else None,
        )
        return result["game_info"], result["table"], status, dropdown_state

    def custom_line(player_label: str, market: str, line_value: float):
        return compute_custom_line_probability(player_label, market, line_value)

    with gr.Blocks(
        title="NBA Betting Projections",
        theme=APP_THEME,
        css=MINIMAL_UI_CSS,
    ) as demo:
        gr.Markdown(
            "# NBA Betting Projection Demo",
            elem_classes=["page-title"],
        )
        gr.Markdown(
            "Recent player form drives prop win probabilities for the next scheduled matchup.",
            elem_classes=["page-subtitle"],
        )

        with gr.Row():
            game_info = gr.Markdown(
                "Locating the next game...",
                elem_classes=["card", "game-card"],
            )
            with gr.Column(scale=1, min_width=260):
                status = gr.Markdown(
                    "Loading projections...",
                    elem_classes=["card", "status-card"],
                )
                refresh_button = gr.Button(
                    "Refresh Next Game Projections",
                    elem_classes=["pill-button"],
                )

        table = gr.Dataframe(
            label="Player prop probabilities",
            interactive=False,
            wrap=True,
            elem_classes=["card", "projection-table"],
        )

        with gr.Column(elem_classes=["card"]):
            gr.Markdown("### Custom line explorer")
            with gr.Row(elem_classes=["custom-inputs"]):
                player_picker = gr.Dropdown(label="Player", choices=[])
                market_picker = gr.Dropdown(
                    label="Market",
                    choices=["points", "rebounds", "assists", "pra"],
                    value="points",
                )
                custom_line_slider = gr.Slider(
                    label="Custom Line",
                    minimum=0,
                    maximum=80,
                    value=20,
                    step=0.5,
                )
            custom_line_output = gr.Markdown(
                "Use the controls above to sanity-check a custom line via a quick normal approximation.",
                elem_classes=["callout"],
            )
            compute_button = gr.Button(
                "Compute Custom Line Probability",
                elem_classes=["pill-button"],
            )

        refresh_button.click(
            refresh,
            inputs=[],
            outputs=[game_info, table, status, player_picker],
        )
        demo.load(
            refresh,
            inputs=[],
            outputs=[game_info, table, status, player_picker],
        )

        compute_button.click(
            custom_line,
            inputs=[player_picker, market_picker, custom_line_slider],
            outputs=custom_line_output,
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA betting projections app")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Force retraining before launching the interface.",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Print projections once and exit instead of launching Gradio.",
    )
    args = parser.parse_args()

    if args.train:
        log("Force training triggered by CLI flag.")
        train_models(force=True)

    models = load_models(ensure_trained=True)

    if args.no_ui:
        result = generate_predictions_for_next_game(models)
        print(result["game_info"])
        if not result["table"].empty:
            print(result["table"].to_string(index=False))
        return

    demo = create_interface(models)
    demo.launch()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
