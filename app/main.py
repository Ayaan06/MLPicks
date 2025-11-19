"""FastAPI app exposing betting picks endpoints."""

from __future__ import annotations

import datetime as dt
import logging
from typing import Dict

from fastapi import FastAPI, HTTPException

from app import models as model_store
from app.data_providers import StatsProvider, get_default_stats_provider
from app.feature_engineering import build_player_features_for_game
from app.schemas import (
    ModelMetadata,
    ModelsInfoResponse,
    PlayerPropsRequest,
    PlayerPropsResponse,
    SinglePropPrediction,
)
from app.utils import build_simple_explanation, make_prop_pick

logger = logging.getLogger(__name__)

app = FastAPI(title="ML Picks - Basketball Props")

STATS_PROVIDER: StatsProvider = get_default_stats_provider()
MODEL_LOADERS = {
    "points": model_store.load_points_model,
    "rebounds": model_store.load_rebounds_model,
    "assists": model_store.load_assists_model,
}


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/models/info", response_model=ModelsInfoResponse)
def models_info() -> ModelsInfoResponse:
    """Expose stored model registry metadata."""
    metadata = model_store.load_metadata()
    parsed: Dict[str, ModelMetadata] = {}
    for key, payload in metadata.items():
        trained_raw = payload.get("trained_on")
        try:
            trained_on = (
                dt.date.fromisoformat(trained_raw) if trained_raw else dt.date.today()
            )
        except ValueError:
            trained_on = dt.date.today()
        parsed[key] = ModelMetadata(
            version=str(payload.get("version", "unknown")),
            trained_on=trained_on,
            data_range=str(payload.get("data_range", "")),
            mae=float(payload.get("mae", 0.0)),
            rmse=float(payload["rmse"]) if payload.get("rmse") is not None else None,
        )
    return ModelsInfoResponse(models=parsed)


@app.post("/player_props", response_model=PlayerPropsResponse)
def player_props(payload: PlayerPropsRequest) -> PlayerPropsResponse:
    """Return over/under picks for the requested prop lines."""
    try:
        features_df = build_player_features_for_game(
            stats_provider=STATS_PROVIDER,
            player_id=payload.player_id,
            team_id=payload.team_id,
            opponent_team_id=payload.opponent_team_id,
            game_date=payload.game_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    feature_vector = features_df.iloc[0]
    props: Dict[str, SinglePropPrediction] = {}
    for prop in payload.prop_lines:
        loader = MODEL_LOADERS.get(prop.stat_type)
        if not loader:
            raise HTTPException(status_code=400, detail=f"Unsupported stat {prop.stat_type}")
        try:
            model, sigma = loader()
        except FileNotFoundError as exc:
            logger.error("Model artifact missing: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Model not trained for {prop.stat_type}",
            ) from exc

        pred = float(model.predict(features_df)[0])
        pick_dict = make_prop_pick(pred_mean=pred, line=prop.line, sigma=sigma)
        explanation = ""
        top_features = model_store.load_feature_importance(prop.stat_type, top_k=3)
        if top_features:
            explanation = build_simple_explanation(feature_vector, top_features)
        props[prop.stat_type] = SinglePropPrediction(
            stat_type=prop.stat_type,
            reason=explanation or None,
            **pick_dict,
        )

    return PlayerPropsResponse(
        player_id=payload.player_id,
        team_id=payload.team_id,
        opponent_team_id=payload.opponent_team_id,
        game_date=payload.game_date,
        props=props,
    )
