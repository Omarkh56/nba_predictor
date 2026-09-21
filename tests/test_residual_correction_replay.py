"""Whole-date chronological fitting and one exact-market live correction."""

import pandas as pd
import pytest

import playerlinepredictor as predictor
from scripts.replay.evaluate_corrections import chronological_predictions


def test_same_day_outcomes_never_enter_current_date_correction():
    rows = pd.DataFrame([
        {"date": "2026-01-01", "market": "PTS", "projection": 20.0, "actual": 10.0},
        {"date": "2026-01-01", "market": "PTS", "projection": 20.0, "actual": 30.0},
        {"date": "2026-01-02", "market": "PTS", "projection": 20.0, "actual": 20.0},
    ])
    original = chronological_predictions(rows, "additive", 2)
    assert original.iloc[0] == original.iloc[1] == original.iloc[2] == 20.0
    changed = rows.copy()
    changed.loc[0, "actual"] = 100.0
    later = chronological_predictions(changed, "additive", 2)
    assert later.iloc[0] == later.iloc[1] == 20.0
    assert later.iloc[2] == pytest.approx(20 + (2 / 4) * 45)


def test_pre_cutoff_market_history_does_not_mix_markets():
    frame = pd.DataFrame([
        {"date": "2026-01-01", "market": "PTS", "projection": 20.0, "actual": 10.0},
        {"date": "2026-01-01", "market": "AST", "projection": 5.0, "actual": 10.0},
        {"date": "2026-01-02", "market": "PTS", "projection": 20.0, "actual": 20.0},
        {"date": "2026-01-02", "market": "AST", "projection": 5.0, "actual": 5.0},
    ])
    out = chronological_predictions(frame, "additive", 1)
    assert out.iloc[2] == 15.0
    assert out.iloc[3] == 7.5
