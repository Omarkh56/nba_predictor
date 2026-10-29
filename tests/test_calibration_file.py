"""calibration.json has one schema, and calibrate.py never wipes sections it doesn't own."""

import json

import numpy as np
import pandas as pd
import pytest

import calibrate
import playerlinepredictor as predictor
from tests.test_train_model import _tm as trainer

MARKET_DIRECTION_FIELDS = {"n", "hits", "raw_rate", "rate", "conf_adj"}
PLAYER_MARKET_FIELDS = {"n", "mean_residual", "adj"}


@pytest.fixture
def calib_paths(monkeypatch, tmp_path):
    log, calib = tmp_path / "bet_log.csv", tmp_path / "calibration.json"
    monkeypatch.setattr(calibrate, "LOG_FILE", log)
    monkeypatch.setattr(calibrate, "CALIB_FILE", calib)
    return log, calib


def test_calibrate_preserves_existing_book_lean_section(calib_paths):
    log, calib = calib_paths
    book_lean = {"generated": "2026-04-01", "total_graded": 12,
                 "by_market": {"PTS": {"n": 12}}, "by_market_lean": {"PTS_OVER": {"n": 7}}}
    calib.write_text(json.dumps({"market_direction": {"STALE_OVER": {"conf_adj": 9.9}},
                                 "book_lean": book_lean}))
    log.write_text(
        "date,player,market,line,pick,projection,actual,margin,result\n"
        "2026-04-18,A,PTS,20.5,OVER,23.0,25.0,4.5,HIT\n"
        "2026-04-18,B,PTS,10.5,UNDER,9.0,12.0,-1.5,MISS\n"
        "2026-04-19,A,PTS,21.5,OVER,24.0,19.0,-2.5,MISS\n"
    )
    calibrate.calibrate()
    written = json.loads(calib.read_text())
    assert written["book_lean"] == book_lean
    # Sections calibrate.py owns are still rebuilt, not merged with stale entries.
    assert "STALE_OVER" not in written["market_direction"]
    assert set(written["market_direction"]) == {"PTS_OVER", "PTS_UNDER"}
    assert written["total_bets"] == 3


def _bootstrap_players(n_games=60):
    """Two players whose actuals beat their EWMA by 2: OVERs hit, UNDERs miss."""
    rng = np.random.default_rng(0)
    rows = []
    for name in ("Player A", "Player B"):
        pts, threes, stl = rng.uniform(5, 30, n_games), rng.uniform(0.5, 4, n_games), rng.uniform(0, 3, n_games)
        rows.append(pd.DataFrame({
            "player_name": name,
            "ewma_PTS": pts, "actual_PTS": pts + 2,
            "ewma_FG3M": threes, "actual_FG3M": threes + 2,
            "ewma_STL": stl, "actual_STL": stl + 2,
        }))
    return pd.concat(rows, ignore_index=True)


def test_bootstrap_output_loads_through_load_calibration(monkeypatch, tmp_path):
    out = trainer.bootstrap_calibration(pd.DataFrame(), _bootstrap_players(), {})
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(out))
    monkeypatch.setattr(predictor, "CALIB_FILE", str(path))
    monkeypatch.setattr(predictor, "CALIB", {})
    predictor._load_calibration()
    calib = predictor.CALIB

    md, pm = calib["market_direction"], calib["player_market"]
    assert set(md) == {"PTS_OVER", "PTS_UNDER", "3PM_OVER", "3PM_UNDER"}  # no FG3M, no STL
    assert all(set(v) == MARKET_DIRECTION_FIELDS for v in md.values())
    assert md["3PM_OVER"]["conf_adj"] > 0 > md["3PM_UNDER"]["conf_adj"]
    assert set(pm) == {f"{p}|{m}" for p in ("Player A", "Player B") for m in ("PTS", "3PM")}
    assert all(set(v) == PLAYER_MARKET_FIELDS for v in pm.values())
    assert pm["Player A|3PM"]["adj"] > 0
    assert calib["total_bets"] == 240

    # The live direction penalty actually reads the bootstrapped conf_adj.
    penalty, _ = predictor._compute_direction_penalty(
        "player_threes", "3PM", "OVER", 3.0, 2.5, 5, {"po_count": 1})
    monkeypatch.setattr(predictor, "CALIB", {})
    baseline, _ = predictor._compute_direction_penalty(
        "player_threes", "3PM", "OVER", 3.0, 2.5, 5, {"po_count": 1})
    assert penalty == pytest.approx(baseline - md["3PM_OVER"]["conf_adj"])
