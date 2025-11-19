"""Pydantic schemas and internal data representations."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


@dataclass
class GameStats:
    """Simple structured representation of a single team game."""

    game_id: str
    team_id: str
    opponent_team_id: str
    date: dt.date
    points_scored: float
    points_allowed: float
    rebounds: float | None
    assists: float | None
    possessions: float | None
    opponent_rebounds: float | None = None
    opponent_assists: float | None = None
    opponent_possessions: float | None = None
    is_home: bool = False


@dataclass
class PlayerGameStats:
    """Structured representation of an individual player box score."""

    game_id: str
    player_id: str
    team_id: str
    opponent_team_id: str
    date: dt.date
    minutes: float
    points: float
    rebounds: float
    assists: float
    field_goal_attempts: float
    free_throw_attempts: float
    turnovers: float
    started: bool
    position: str | None = None


class PlayerPropLine(BaseModel):
    stat_type: Literal["points", "rebounds", "assists"]
    line: float

    @validator("line")
    def validate_line(cls, v: float) -> float:
        if v < 0:
            raise ValueError("line must be non-negative")
        return v


class PlayerPropsRequest(BaseModel):
    player_id: str
    team_id: str
    opponent_team_id: str
    game_date: dt.date
    prop_lines: List[PlayerPropLine] = Field(..., min_items=1)


class SinglePropPrediction(BaseModel):
    stat_type: str
    model_projection: float
    line: float
    prob_over: float
    prob_under: float
    pick_side: Literal["over", "under"]
    confidence_score: float
    confidence_label: str
    edge_value: float
    edge_prob: float
    risk_score: float
    risk_label: Literal["low", "medium", "high"]
    reason: Optional[str] = None


class PlayerPropsResponse(BaseModel):
    player_id: str
    team_id: str
    opponent_team_id: str
    game_date: dt.date
    props: Dict[str, SinglePropPrediction]


class ModelMetadata(BaseModel):
    version: str
    trained_on: dt.date
    data_range: str
    mae: float
    rmse: Optional[float] = None


class ModelsInfoResponse(BaseModel):
    models: Dict[str, ModelMetadata]
