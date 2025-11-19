"""Wrappers around free basketball stats APIs."""

from __future__ import annotations

import datetime as dt
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

from app import storage
from app.config import settings
from app.schemas import GameStats, PlayerGameStats

logger = logging.getLogger(__name__)


class StatsProvider(ABC):
    """Abstract stats provider to allow pluggable data sources."""

    @abstractmethod
    def get_player_last_games(self, player_id: str, n: int) -> List[PlayerGameStats]:
        """Return the last `n` games for the given player."""

    @abstractmethod
    def get_team_last_games(self, team_id: str, n: int) -> List[GameStats]:
        """Return the last `n` games for the given team."""

    @abstractmethod
    def get_matchup_history(
        self, team_a_id: str, team_b_id: str, n: int
    ) -> List[GameStats]:
        """Return historical games where team A played team B."""


class FreeNBABasicStatsProvider(StatsProvider):
    """Stats provider backed by the public balldontlie API."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.stats_api_base_url).rstrip("/")
        self.timeout = timeout_seconds or settings.request_timeout_seconds
        self.max_retries = max_retries or settings.provider_max_retries
        self.session = requests.Session()
        if settings.stats_api_key:
            self.session.headers.update({"Authorization": f"Bearer {settings.stats_api_key}"})
        self._game_totals_cache: Dict[str, Dict[str, Dict[str, float]]] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_player_last_games(self, player_id: str, n: int) -> List[PlayerGameStats]:
        cached_df = storage.load_player_games(player_id)
        cached_games = self._player_df_to_models(cached_df)
        if len(cached_games) >= n:
            return cached_games[:n]

        fresh_games = self._fetch_player_games(player_id, n)
        combined_df = self._merge_cached_frames(
            cached_df, self._player_games_to_df(fresh_games), subset=["game_id"]
        )
        if combined_df is not None and not combined_df.empty:
            storage.save_player_games(player_id, combined_df)
            combined_games = self._player_df_to_models(combined_df)
        else:
            combined_games = fresh_games
        return combined_games[:n]

    def get_team_last_games(self, team_id: str, n: int) -> List[GameStats]:
        cached_df = storage.load_team_games(team_id)
        cached_games = self._team_df_to_models(cached_df)
        if len(cached_games) >= n:
            return cached_games[:n]

        fresh_games = self._fetch_team_games(team_id, n)
        combined_df = self._merge_cached_frames(
            cached_df,
            self._team_games_to_df(fresh_games),
            subset=["game_id", "team_id"],
        )
        if combined_df is not None and not combined_df.empty:
            storage.save_team_games(team_id, combined_df)
            combined_games = self._team_df_to_models(combined_df)
        else:
            combined_games = fresh_games
        return combined_games[:n]

    def get_matchup_history(
        self, team_a_id: str, team_b_id: str, n: int
    ) -> List[GameStats]:
        team_games = self.get_team_last_games(team_a_id, n * 5)
        matchup = [g for g in team_games if g.opponent_team_id == str(team_b_id)]
        return matchup[:n]

    # ------------------------------------------------------------------ #
    # Internal cache-aware fetchers
    # ------------------------------------------------------------------ #
    def _fetch_player_games(self, player_id: str, n: int) -> List[PlayerGameStats]:
        limit = max(n * 2, 20)
        records = self._paginate(
            "/stats",
            params={
                "player_ids[]": player_id,
                "per_page": 100,
                "postseason": "false",
            },
            limit=limit,
        )
        parsed = sorted(
            (self._player_stat_to_model(stat) for stat in records),
            key=lambda row: row.date,
            reverse=True,
        )
        return parsed

    def _fetch_team_games(self, team_id: str, n: int) -> List[GameStats]:
        games = self._paginate(
            "/games",
            params={
                "team_ids[]": team_id,
                "per_page": 100,
                "postseason": "false",
            },
            limit=max(n * 3, 20),
        )
        parsed_games: List[GameStats] = []
        for raw in sorted(games, key=lambda r: self._parse_date(r["date"]), reverse=True):
            parsed = self._game_to_model(raw, str(team_id))
            if parsed:
                parsed_games.append(parsed)
        return parsed_games

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _request_json(self, path: str, params: Dict[str, object]) -> Dict:
        url = f"{self.base_url}{path}"
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                response = self.session.get(
                    url, params=params, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:  # pragma: no cover - network
                logger.warning(
                    "Stats API request error (attempt %s/%s): %s",
                    attempt + 1,
                    attempts,
                    exc,
                )
                if attempt + 1 >= attempts:
                    raise
        raise RuntimeError("Unreachable")  # pragma: no cover

    def _paginate(
        self, path: str, params: Dict[str, object], limit: int
    ) -> List[Dict]:
        page = 1
        collected: List[Dict] = []
        while len(collected) < limit:
            data = self._request_json(path, {**params, "page": page})
            entries = data.get("data", [])
            if not entries:
                break
            collected.extend(entries)
            meta = data.get("meta", {})
            next_page = meta.get("next_page")
            if not next_page:
                break
            page = next_page
        return collected[:limit]

    @staticmethod
    def _parse_date(value: str) -> dt.date:
        return dt.date.fromisoformat(value.split("T", 1)[0])

    @staticmethod
    def _parse_minutes(value: str | None) -> float:
        if not value:
            return 0.0
        if ":" in value:
            minutes, seconds = value.split(":", 1)
            return float(minutes) + float(seconds) / 60.0
        return float(value)

    def _player_stat_to_model(self, stat: Dict) -> PlayerGameStats:
        game = stat["game"]
        team_id = str(stat["team"]["id"])
        opponent_id = (
            str(game["home_team"]["id"])
            if team_id != str(game["home_team"]["id"])
            else str(game["visitor_team"]["id"])
        )
        return PlayerGameStats(
            game_id=str(game["id"]),
            player_id=str(stat["player"]["id"]),
            team_id=team_id,
            opponent_team_id=opponent_id,
            date=self._parse_date(game["date"]),
            minutes=self._parse_minutes(stat.get("min")),
            points=float(stat.get("pts") or 0.0),
            rebounds=float(stat.get("reb") or 0.0),
            assists=float(stat.get("ast") or 0.0),
            field_goal_attempts=float(stat.get("fga") or 0.0),
            free_throw_attempts=float(stat.get("fta") or 0.0),
            turnovers=float(stat.get("turnover") or 0.0),
            started=bool(stat.get("starter")),
            position=stat.get("player", {}).get("position"),
        )

    def _game_to_model(self, game: Dict, team_id: str) -> GameStats | None:
        date = self._parse_date(game["date"])
        home_team_id = str(game["home_team"]["id"])
        visitor_team_id = str(game["visitor_team"]["id"])
        is_home = team_id == home_team_id
        opponent_id = visitor_team_id if is_home else home_team_id
        team_score = (
            float(game["home_team_score"])
            if is_home
            else float(game["visitor_team_score"])
        )
        opp_score = (
            float(game["visitor_team_score"])
            if is_home
            else float(game["home_team_score"])
        )
        totals = self._get_game_totals(str(game["id"]))
        team_totals = totals.get(team_id)
        opponent_totals = totals.get(opponent_id) if totals else None
        return GameStats(
            game_id=str(game["id"]),
            team_id=team_id,
            opponent_team_id=opponent_id,
            date=date,
            points_scored=team_score,
            points_allowed=opp_score,
            rebounds=team_totals.get("rebounds") if team_totals else None,
            assists=team_totals.get("assists") if team_totals else None,
            possessions=team_totals.get("possessions") if team_totals else None,
            opponent_rebounds=opponent_totals.get("rebounds") if opponent_totals else None,
            opponent_assists=opponent_totals.get("assists") if opponent_totals else None,
            opponent_possessions=opponent_totals.get("possessions")
            if opponent_totals
            else None,
            is_home=is_home,
        )

    def _get_game_totals(self, game_id: str) -> Dict[str, Dict[str, float]]:
        cached = self._game_totals_cache.get(game_id)
        if cached:
            return cached
        stats_rows = self._paginate(
            "/stats",
            params={"game_ids[]": game_id, "per_page": 100},
            limit=500,
        )
        totals: Dict[str, Dict[str, float]] = {}
        for row in stats_rows:
            team_id = str(row["team"]["id"])
            bucket = totals.setdefault(
                team_id,
                {
                    "points": 0.0,
                    "rebounds": 0.0,
                    "assists": 0.0,
                    "fga": 0.0,
                    "fta": 0.0,
                    "turnovers": 0.0,
                    "off_reb": 0.0,
                },
            )
            bucket["points"] += float(row.get("pts") or 0.0)
            bucket["rebounds"] += float(row.get("reb") or 0.0)
            bucket["assists"] += float(row.get("ast") or 0.0)
            bucket["fga"] += float(row.get("fga") or 0.0)
            bucket["fta"] += float(row.get("fta") or 0.0)
            bucket["turnovers"] += float(row.get("turnover") or 0.0)
            bucket["off_reb"] += float(row.get("oreb") or 0.0)
        for bucket in totals.values():
            bucket["possessions"] = (
                bucket["fga"]
                + 0.44 * bucket["fta"]
                - bucket["off_reb"]
                + bucket["turnovers"]
            )
        self._game_totals_cache[game_id] = totals
        return totals

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #
    def _player_games_to_df(self, games: List[PlayerGameStats]) -> pd.DataFrame:
        if not games:
            return pd.DataFrame()
        df = pd.DataFrame([asdict(game) for game in games])
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _team_games_to_df(self, games: List[GameStats]) -> pd.DataFrame:
        if not games:
            return pd.DataFrame()
        df = pd.DataFrame([asdict(game) for game in games])
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _player_df_to_models(
        self, df: Optional[pd.DataFrame]
    ) -> List[PlayerGameStats]:
        if df is None or df.empty:
            return []
        df_sorted = df.sort_values("date", ascending=False)
        models: List[PlayerGameStats] = []
        for _, row in df_sorted.iterrows():
            date_value = row["date"]
            models.append(
                PlayerGameStats(
                    game_id=str(row["game_id"]),
                    player_id=str(row["player_id"]),
                    team_id=str(row["team_id"]),
                    opponent_team_id=str(row["opponent_team_id"]),
                    date=self._coerce_date(date_value),
                    minutes=float(row["minutes"]),
                    points=float(row["points"]),
                    rebounds=float(row["rebounds"]),
                    assists=float(row["assists"]),
                    field_goal_attempts=float(row["field_goal_attempts"]),
                    free_throw_attempts=float(row["free_throw_attempts"]),
                    turnovers=float(row["turnovers"]),
                    started=self._coerce_bool(row.get("started")),
                    position=row.get("position"),
                )
            )
        return models

    def _team_df_to_models(
        self, df: Optional[pd.DataFrame]
    ) -> List[GameStats]:
        if df is None or df.empty:
            return []
        df_sorted = df.sort_values("date", ascending=False)
        models: List[GameStats] = []
        for _, row in df_sorted.iterrows():
            date_value = row["date"]
            models.append(
                GameStats(
                    game_id=str(row["game_id"]),
                    team_id=str(row["team_id"]),
                    opponent_team_id=str(row["opponent_team_id"]),
                    date=self._coerce_date(date_value),
                    points_scored=float(row["points_scored"]),
                    points_allowed=float(row["points_allowed"]),
                    rebounds=self._maybe_float(row.get("rebounds")),
                    assists=self._maybe_float(row.get("assists")),
                    possessions=self._maybe_float(row.get("possessions")),
                    opponent_rebounds=self._maybe_float(
                        row.get("opponent_rebounds")
                    ),
                    opponent_assists=self._maybe_float(
                        row.get("opponent_assists")
                    ),
                    opponent_possessions=self._maybe_float(
                        row.get("opponent_possessions")
                    ),
                    is_home=self._coerce_bool(row.get("is_home")),
                )
            )
        return models

    @staticmethod
    def _merge_cached_frames(
        cached: Optional[pd.DataFrame],
        fresh: pd.DataFrame,
        subset: List[str],
    ) -> pd.DataFrame:
        frames = []
        if cached is not None and not cached.empty:
            frames.append(cached.copy())
        if fresh is not None and not fresh.empty:
            frames.append(fresh.copy())
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        combined.drop_duplicates(subset=subset, keep="first", inplace=True)
        combined.sort_values("date", ascending=False, inplace=True)
        return combined

    @staticmethod
    def _maybe_float(value: object) -> Optional[float]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, str) and value == "":
            return None
        return float(value)

    @staticmethod
    def _coerce_date(value: object) -> dt.date:
        if isinstance(value, dt.datetime):
            return value.date()
        if isinstance(value, dt.date):
            return value
        return dt.date.fromisoformat(str(value).split("T", 1)[0])

    @staticmethod
    def _coerce_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(int(value))
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "t", "yes"}
        return False


def get_default_stats_provider() -> StatsProvider:
    """Factory used by the API to get the default stats provider."""
    return FreeNBABasicStatsProvider()
