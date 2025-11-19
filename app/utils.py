"""Common utility helpers."""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd
from scipy.stats import norm

RISK_SCALE = 15.0


def over_probability(pred_mean: float, line: float, sigma: float) -> float:
    """Return probability the actual value is greater than the line."""
    if sigma <= 0:
        return 1.0 if pred_mean > line else 0.0
    z = (line - pred_mean) / sigma
    return float(1.0 - norm.cdf(z))


def under_probability(pred_mean: float, line: float, sigma: float) -> float:
    """Return probability the actual value is below the line."""
    return float(1.0 - over_probability(pred_mean, line, sigma))


def make_prop_pick(pred_mean: float, line: float, sigma: float) -> Dict[str, float | str]:
    """Generate structured pick metadata for a prop line."""
    prob_over = over_probability(pred_mean, line, sigma)
    prob_under = 1.0 - prob_over
    pick_side = "over" if prob_over >= prob_under else "under"
    confidence = max(prob_over, prob_under)
    confidence_score = round(confidence * 100.0, 1)
    confidence_label = _confidence_label(confidence_score)
    edge_prob = round(confidence - 0.5, 4)
    edge_value = pred_mean - line if pick_side == "over" else line - pred_mean
    risk_score = min(1.0, abs(sigma) / RISK_SCALE) if sigma >= 0 else 0.0
    risk_label = _risk_label(risk_score)
    return {
        "model_projection": pred_mean,
        "line": line,
        "prob_over": prob_over,
        "prob_under": prob_under,
        "pick_side": pick_side,
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "edge_value": edge_value,
        "edge_prob": edge_prob,
        "risk_score": risk_score,
        "risk_label": risk_label,
    }


def build_simple_explanation(
    feature_vector: pd.Series,
    top_feature_names: Sequence[str],
) -> str:
    """
    Build a small explanation snippet referencing the top feature values.

    This is intentionally lightweight but provides enough color to give the
    consumer intuition for why a projection leans a certain direction.
    """

    phrases: List[str] = []
    for feature in top_feature_names:
        if feature not in feature_vector:
            continue
        value = feature_vector[feature]
        if pd.isna(value):
            continue
        pretty_name = feature.replace("_", " ")
        phrases.append(f"{pretty_name} ({value:.2f})")
        if len(phrases) >= 3:
            break

    if not phrases:
        return ""
    return "Key drivers: " + ", ".join(phrases)


def _confidence_label(confidence_score: float) -> str:
    if confidence_score >= 80:
        return "very_high"
    if confidence_score >= 70:
        return "high"
    if confidence_score >= 60:
        return "medium"
    return "low"


def _risk_label(risk_score: float) -> str:
    if risk_score >= 0.66:
        return "high"
    if risk_score >= 0.33:
        return "medium"
    return "low"
