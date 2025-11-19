"""Tests for feature engineering helpers."""

from __future__ import annotations

import datetime as dt
from typing import List

from app.data_providers import StatsProvider
from app.feature_engineering import build_player_features_for_game
from app.schemas import GameStats, PlayerGameStats


class FakeStatsProvider(StatsProvider):
    def __init__(self) -> None:
        self.player_games = self._build_player_games()
        self.team_games = self._build_team_games("T1", "T2")
        self.opp_games = self._build_team_games("T2", "T1")

    def get_player_last_games(self, player_id: str, n: int) -> List[PlayerGameStats]:
        return self.player_games[:n]

    def get_team_last_games(self, team_id: str, n: int) -> List[GameStats]:
        data = self.team_games if team_id == "T1" else self.opp_games
        return data[:n]

    def get_matchup_history(
        self, team_a_id: str, team_b_id: str, n: int
    ) -> List[GameStats]:
        return []

    @staticmethod
    def _build_player_games() -> List[PlayerGameStats]:
        base_date = dt.date(2024, 1, 20)
        games: List[PlayerGameStats] = []
        for idx in range(12):
            games.append(
                PlayerGameStats(
                    game_id=str(idx),
                    player_id="P1",
                    team_id="T1",
                    opponent_team_id="T2",
                    date=base_date - dt.timedelta(days=idx),
                    minutes=30 + idx * 0.5,
                    points=20 + idx,
                    rebounds=5 + idx * 0.2,
                    assists=6 + idx * 0.3,
                    field_goal_attempts=15 + idx * 0.4,
                    free_throw_attempts=4 + idx * 0.1,
                    turnovers=2 + idx * 0.1,
                    started=True,
                    position="G",
                )
            )
        return games

    @staticmethod
    def _build_team_games(team_id: str, opponent_team_id: str) -> List[GameStats]:
        base_date = dt.date(2024, 1, 20)
        games: List[GameStats] = []
        for idx in range(12):
            games.append(
                GameStats(
                    game_id=f"{team_id}-{idx}",
                    team_id=team_id,
                    opponent_team_id=opponent_team_id,
                    date=base_date - dt.timedelta(days=idx),
                    points_scored=110 + idx,
                    points_allowed=100 - idx * 0.5,
                    rebounds=45 + idx,
                    assists=28 + idx * 0.5,
                    possessions=100 + idx,
                    opponent_rebounds=44 + idx * 0.3,
                    opponent_assists=25 + idx * 0.4,
                    opponent_possessions=99 + idx,
                    is_home=bool(idx % 2),
                )
            )
        return games


def test_build_player_features_for_game_has_values() -> None:
    provider = FakeStatsProvider()
    df = build_player_features_for_game(
        stats_provider=provider,
        player_id="P1",
        team_id="T1",
        opponent_team_id="T2",
        game_date=dt.date(2024, 1, 25),
    )
    assert df.shape == (1, len(df.columns))
    assert not df.isna().any().any()
    expected_pts_last5 = sum(20 + i for i in range(5)) / 5
    assert abs(df["pts_last5_avg"].iloc[0] - expected_pts_last5) < 1e-6
