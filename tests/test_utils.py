"""Tests for probability and pick utilities."""

from __future__ import annotations

import math

from app.utils import make_prop_pick, over_probability, under_probability


def test_over_under_symmetry() -> None:
    mean, line, sigma = 25.0, 20.0, 5.0
    over = over_probability(mean, line, sigma)
    under = under_probability(mean, line, sigma)
    assert math.isclose(over + under, 1.0, abs_tol=1e-6)


def test_make_prop_pick_returns_reasonable_confidence() -> None:
    pick = make_prop_pick(pred_mean=28.0, line=22.5, sigma=4.0)
    assert pick["prob_over"] > 0.5
    assert 50 <= pick["confidence_score"] <= 100
    assert pick["edge_value"] > 0
    assert pick["edge_prob"] >= 0
    assert pick["risk_score"] >= 0


def test_make_prop_pick_handles_sigma_zero() -> None:
    pick = make_prop_pick(pred_mean=15.0, line=10.0, sigma=0.0)
    assert pick["prob_over"] == 1.0
    assert pick["risk_score"] == 0.0
