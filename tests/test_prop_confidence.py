"""Probability-based prop picks and their real processing/penalty call sites."""

from unittest import mock

import pytest


@pytest.fixture
def predictor():
    with mock.patch(
        "requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")
    ):
        import playerlinepredictor

        yield playerlinepredictor


@pytest.mark.parametrize(
    "market", ["player_points", "player_rebounds", "player_assists", "player_threes"]
)
def test_skewed_distribution_uses_the_more_likely_side(predictor, market):
    # Mean=1, variance=4 gives NB r=1/3, p=1/4. At line 0.5, UNDER wins
    # precisely when X=0: P(X=0)=(1/4)**(1/3), about 63%, despite mean > line.
    result = predictor.evaluate_prop(0.5, 1.0, market, sd=2.0)
    expected_under = round((0.25 ** (1 / 3)) * 100, 1)
    assert result["edge"] == 0.5
    assert result["pick"] == "UNDER"
    assert result["under_prob"] == expected_under
    assert result["over_prob"] == 37.0
    assert result["confidence"] == expected_under


@pytest.mark.parametrize("projection,pick", [(25.0, "OVER"), (15.0, "UNDER")])
def test_normal_distribution_pick_and_confidence_agree(predictor, projection, pick):
    result = predictor.evaluate_prop(20.5, projection, "player_points_assists", sd=5.0)
    assert result["pick"] == pick
    assert result["confidence"] == result[f"{pick.lower()}_prob"]


def test_equal_probabilities_use_a_consistent_under_pick(predictor):
    result = predictor.evaluate_prop(20.5, 20.5, "player_points_assists", sd=5.0)
    assert result["pick"] == "UNDER"
    assert result["confidence"] == 50.0


@pytest.mark.parametrize("penalty,expected", [(20.0, 43.0), (200.0, 0.0), (-100.0, 100.0)])
def test_penalties_adjust_the_selected_side_without_a_fifty_percent_floor(
    predictor, penalty, expected
):
    result = predictor.evaluate_prop(
        0.5, 1.0, "player_threes", sd=2.0, conf_penalty=penalty
    )
    assert result["pick"] == "UNDER"
    assert result["confidence"] == expected
    assert result["under_prob"] == 63.0  # Raw probability stays separate from the score.


@pytest.fixture
def run_game(predictor, monkeypatch):
    """Keep real parsing, consensus, evaluation, direction penalties, and flags."""
    monkeypatch.setattr(predictor, "get_team_id", mock.Mock(return_value=1))
    monkeypatch.setattr(predictor, "MARKET_ANCHOR_WEIGHT", 0.0)
    monkeypatch.setattr(predictor, "IS_PLAYOFF_SEASON", False)
    monkeypatch.setattr(predictor, "LOW_LINE_CONF_PENALTY", 0.0)
    monkeypatch.setattr(predictor, "SUSPICIOUS_EDGE_PCT", 1000.0)
    monkeypatch.setattr(predictor, "CALIB", {})
    monkeypatch.setattr(
        predictor,
        "_build_player_context",
        mock.Mock(return_value={
            "Probe": {
                "opp_abbr": "BOS", "opp_tid": 2, "injuries": [], "pos": "PG",
                "is_home": True, "team_abbr": "NYK",
            }
        }),
    )

    def run(line=0.5, projection=1.0, sd=2.0, market="player_threes", penalty=0.0,
            trend="neutral"):
        props = {"bookmakers": [
            {"title": f"Book {i}", "markets": [{
                "key": market,
                "outcomes": [
                    {"description": "Probe", "name": side, "point": line, "price": -110}
                    for side in ("Over", "Under")
                ],
            }]}
            for i in range(3)
        ]}
        monkeypatch.setattr(predictor, "get_event_player_props", mock.Mock(return_value=props))
        meta = {
            "conf_penalty": penalty, "po_count": 5, "is_bench": False,
            "mins_vol_flag": False, "road_b2b": False, "self_inj_flag": "",
            "max_usage": 1.0,
        }
        monkeypatch.setattr(
            predictor, "project_prop",
            mock.Mock(return_value=(projection, {"stat": {"trend": trend}}, sd, meta)),
        )
        rows = predictor.process_game("New York Knicks", "Boston Celtics", "NYK", "BOS",
                                      {"id": "probe-event"}, {})
        assert len(rows) == 1
        return rows[0]

    return run


def test_process_game_uses_the_pick_for_calibration_penalties_and_flags(
    predictor, monkeypatch, run_game
):
    monkeypatch.setattr(predictor, "CALIB", {"market_direction": {
        "3PM_UNDER": {"conf_adj": -4.0}, "3PM_OVER": {"conf_adj": 7.0},
    }})
    with mock.patch.object(predictor, "_compute_direction_penalty",
                           wraps=predictor._compute_direction_penalty) as penalties, \
         mock.patch.object(predictor, "_build_prop_flags",
                           wraps=predictor._build_prop_flags) as flags:
        row = run_game(trend="declining")
    assert row["Pick"] == "UNDER"
    assert penalties.call_args.args[2] == row["Pick"]
    assert flags.call_args.args[4] == row["Pick"]
    assert row["direction_penalty"] == predictor.UNDER_CONF_PENALTY + 4.0
    assert row["Confidence"] == 63.0 - row["direction_penalty"]
    assert "TREND↓" in row["Flags"]  # Declining trend does not penalize UNDER.


def test_process_game_tie_uses_under_penalties_and_combo_variance(predictor, run_game):
    with mock.patch.object(predictor, "evaluate_prop", wraps=predictor.evaluate_prop) as evaluate:
        row = run_game(line=20.5, projection=20.5, sd=5.0, market="player_points_assists")
    assert row["Pick"] == "UNDER"
    assert row["direction_penalty"] == predictor.UNDER_CONF_PENALTY
    assert row["Confidence"] == 50.0 - predictor.UNDER_CONF_PENALTY
    assert evaluate.call_args.kwargs["sd"] == 5.0 * predictor.COMBO_UNDER_SD_MULT


def test_process_game_suspicious_penalty_can_reduce_confidence_below_fifty(
    predictor, monkeypatch, run_game
):
    monkeypatch.setattr(predictor, "SUSPICIOUS_EDGE_PCT", 40.0)
    row = run_game(penalty=5.0)
    assert row["Pick"] == "UNDER"
    assert row["Confidence"] == 63.0 - 5.0 - predictor.UNDER_CONF_PENALTY - 20.0
    assert "SUSP" in row["Flags"]
    assert row["Conf_Label"] == "Low"


def test_process_game_declining_over_penalty_can_reduce_confidence_below_fifty(
    predictor, run_game
):
    row = run_game(line=20.5, projection=20.6, sd=5.0, market="player_rebounds_assists",
                   penalty=2.0, trend="declining")
    assert row["Pick"] == "OVER"
    assert row["Confidence"] == 45.8
    assert "TREND↓" in row["Flags"]
