"""Historical/live parity, season resets, and real raw-data fetch call sites."""

import json
from unittest import mock

import pandas as pd
import pytest

from game_features import (
    GAME_FEATURE_VERSION,
    GAME_FEATURES_EXTENDED,
    _add_h2h_margin,
    build_game_dataset,
    build_live_game_features,
)


@pytest.fixture
def logs():
    teams, players = [], []
    for i in range(10):
        home, away = (1, 2) if i % 2 == 0 else (2, 1)
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i * 2)
        margin = 8.0 if home == 1 else -4.0
        for tid, other in ((home, away), (away, home)):
            pm = margin if tid == home else -margin
            teams.append({
                "season": "2023-24", "GAME_ID": f"g{i}", "GAME_DATE": day,
                "TEAM_ID": tid, "MATCHUP": f"T{tid} vs. T{other}" if tid == home
                else f"T{tid} @ T{other}", "PLUS_MINUS": pm,
                "WL": "W" if pm > 0 else "L", "FGM": 38 + tid + i,
                "FGA": 80 + i, "FG3M": 10 + tid, "FTA": 20 + tid,
                "OREB": 8 + tid, "REB": 40 + tid, "TOV": 12 + tid,
            })
            for j in range(2):
                players.append({
                    "season": "2023-24", "GAME_ID": f"g{i}", "GAME_DATE": day,
                    "TEAM_ID": tid, "PLAYER_ID": tid * 10 + j,
                    "PTS": (12 + i * 3) if tid == 1 else (30 - i), "MIN": 30,
                })
    return pd.DataFrame(teams), pd.DataFrame(players)


def test_training_and_live_inputs_match_for_both_home_directions(logs):
    team, player = logs
    training = build_game_dataset(team, player)
    for _, game in training.iterrows():
        live = build_live_game_features(team, player, game.home_tid, game.away_tid,
                                        game.season, game.game_date)
        assert set(live) == set(GAME_FEATURES_EXTENDED)
        assert live == pytest.approx(game[GAME_FEATURES_EXTENDED].to_dict())
    assert training["star_form_diff"].abs().max() > 0


def test_future_box_scores_and_participants_cannot_change_inputs(logs):
    team, player = logs
    cutoff = pd.Timestamp("2024-01-17")
    expected = build_live_game_features(team, player, 1, 2, "2023-24", cutoff)
    team.loc[team.GAME_DATE >= cutoff, ["PLUS_MINUS", "FGM", "TOV"]] = 99999
    player.loc[player.GAME_DATE >= cutoff, "PTS"] = 99999
    player.loc[player.GAME_DATE >= cutoff, "TEAM_ID"] = 99
    assert build_live_game_features(team, player, 1, 2, "2023-24", cutoff) == expected
    game = build_game_dataset(team, player).query("game_id == 'g8'").iloc[0]
    assert game[GAME_FEATURES_EXTENDED].to_dict() == pytest.approx(expected)


def test_prior_season_results_do_not_change_new_season_inputs(logs):
    team, player = logs
    old_team, old_player = team.copy(), player.copy()
    old_team["season"], old_player["season"] = "2022-23", "2022-23"
    old_team["GAME_DATE"] -= pd.Timedelta(days=365)
    old_player["GAME_DATE"] -= pd.Timedelta(days=365)
    old_team["PLUS_MINUS"], old_player["PTS"] = 500, 500
    expected = build_game_dataset(team, player).set_index("game_id")
    combined = build_game_dataset(pd.concat([old_team, team]), pd.concat([old_player, player]))
    actual = combined[combined.season == "2023-24"].set_index("game_id")
    pd.testing.assert_frame_equal(actual[GAME_FEATURES_EXTENDED], expected[GAME_FEATURES_EXTENDED])
    assert "g0" not in actual.index  # Last season cannot supply early-season history.


def test_head_to_head_uses_todays_home_team_perspective():
    games = pd.DataFrame([
        {"season": "S1", "game_date": "2024-01-01", "home_tid": 1, "away_tid": 2,
         "actual_margin": 10},
        {"season": "S1", "game_date": "2024-01-03", "home_tid": 2, "away_tid": 1,
         "actual_margin": 5},
        {"season": "S1", "game_date": "2024-01-05", "home_tid": 1, "away_tid": 2,
         "actual_margin": 0},
        {"season": "S2", "game_date": "2025-01-01", "home_tid": 1, "away_tid": 2,
         "actual_margin": 0},
    ])
    values = _add_h2h_margin(games).h2h_margin.tolist()
    assert values == pytest.approx([0, -10 / 6, 2.5 * 2 / 7, 0])


def test_altitude_uses_the_training_scale_in_live_inputs(logs):
    team, player = logs
    denver = 1610612743
    team.loc[team.TEAM_ID == 1, "TEAM_ID"] = denver
    player.loc[player.TEAM_ID == 1, "TEAM_ID"] = denver
    features = build_live_game_features(team, player, denver, 2, "2023-24", "2024-01-20")
    assert features["altitude_penalty"] == 1.0  # Last game Jan 19: back-to-back.
    assert features["venue_residual"] == pytest.approx(0.0)
    assert features["star_form_diff"] > 0


def test_insufficient_history_triggers_fallback_instead_of_guessing(logs):
    team, player = logs
    with pytest.raises(ValueError, match="fewer than three"):
        build_live_game_features(team, player, 1, 2, "2023-24", "2024-01-05")


@pytest.fixture
def predictor(monkeypatch):
    with mock.patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")):
        import newnbapredictor

        monkeypatch.setattr(newnbapredictor, "USE_LEARNED_GAME_MODEL", True)
        monkeypatch.setattr(newnbapredictor, "USE_KALMAN_GAME_MODEL", False)
        monkeypatch.setattr(newnbapredictor, "_LEARNED_GAME_PARAMS", {
            "feature_names": GAME_FEATURES_EXTENDED,
        })
        monkeypatch.setattr(newnbapredictor, "_LEARNED_LOG_CACHE", {})
        monkeypatch.setattr(newnbapredictor.time, "sleep", mock.Mock())
        yield newnbapredictor


def test_real_fetch_path_uses_shared_inputs_and_reuses_season_logs(predictor, logs):
    from nba_api.stats.endpoints import leaguegamelog

    team, player = logs
    expected_game = build_game_dataset(team, player).query("game_id == 'g8'").iloc[0]
    with mock.patch.object(predictor.leaguegamefinder, "LeagueGameFinder") as team_api, \
         mock.patch.object(leaguegamelog, "LeagueGameLog") as player_api:
        team_api.return_value.get_data_frames.return_value = [team]
        player_api.return_value.get_data_frames.return_value = [player]
        for _ in range(2):
            actual = predictor._fetch_learned_features(1, 2, "2023-24", "2024-01-17")
            assert actual == pytest.approx(expected_game[GAME_FEATURES_EXTENDED].to_dict())
        assert team_api.call_count == player_api.call_count == 1


def test_failed_history_request_selects_the_fallback(predictor, caplog):
    with mock.patch.object(predictor.leaguegamefinder, "LeagueGameFinder",
                           side_effect=predictor.requests.exceptions.RequestException("offline")):
        assert predictor._fetch_learned_features(1, 2, "2023-24", "2024-01-17") is None
    assert "could not build game inputs" in caplog.text


def test_kalman_preserves_the_input_version_and_rejects_old_weights(tmp_path):
    from kalman_filter import GameEKF

    path = tmp_path / "params.json"
    model = {
        "feature_version": GAME_FEATURE_VERSION, "feature_names": ["net_rtg_diff"],
        "log_intercept": 0.1, "log_coefficients": {"net_rtg_diff": 0.8},
        "scaler_mean": [0.0], "scaler_scale": [4.0],
    }
    path.write_text(json.dumps({"game_model": model}))
    ekf = GameEKF.from_learned_params(path)
    assert GameEKF.from_dict(ekf.to_dict()).feature_version == GAME_FEATURE_VERSION
    del model["feature_version"]
    path.write_text(json.dumps({"game_model": model}))
    with pytest.raises(ValueError, match="rerun train_model.py"):
        GameEKF.from_learned_params(path)


def test_old_kalman_state_stays_inactive(predictor, monkeypatch):
    import kalman_filter

    monkeypatch.setattr(predictor, "_KALMAN_EKF", None)
    ekf = mock.Mock(feature_version=None, n_updates=10)
    with mock.patch.object(kalman_filter, "load_game_ekf", return_value=ekf), \
         mock.patch.object(kalman_filter, "load_tvp_state", return_value={"recommendation": "use_kalman"}):
        predictor._load_kalman_state()
    assert predictor.USE_KALMAN_GAME_MODEL is False


def test_complete_extended_inputs_are_used_by_live_prediction(logs, monkeypatch, tmp_path):
    from tests.test_learned_game_model import _save_model

    # The gate/inference tests cover the engine separately. Here a real fitted
    # artifact is used with the raw-log fetch path, including nonzero star form.
    with mock.patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")), \
         mock.patch("requests.get", side_effect=AssertionError("Network forbidden")):
        from nba_api.stats.endpoints import leaguegamelog

        import newnbapredictor as predictor
        import train_model as trainer

        team, player = logs
        games = build_game_dataset(team, player)
        training = pd.concat([games] * 10, ignore_index=True)
        # Both outcomes are needed for a meaningful fitted logistic model.
        training.loc[training.index % 2 == 0, "home_win"] = 0
        training.loc[training.index % 2 == 1, "home_win"] = 1
        fitted = trainer.fit_game_models(training, games, features=GAME_FEATURES_EXTENDED)
        saved = {k: v for k, v in fitted.items() if k not in ("log_model", "ridge_model", "scaler")}
        saved["recommendation"] = "swap"  # Exercise availability, independently of model selection.
        _save_model(tmp_path, saved)
        monkeypatch.setattr(predictor, "__file__", str(tmp_path / "newnbapredictor.py"))
        monkeypatch.setattr(predictor, "_LEARNED_GAME_PARAMS", {})
        monkeypatch.setattr(predictor, "USE_LEARNED_GAME_MODEL", False)
        monkeypatch.setattr(predictor, "USE_KALMAN_GAME_MODEL", False)
        monkeypatch.setattr(predictor, "_LEARNED_LOG_CACHE", {})
        monkeypatch.setattr(predictor, "_VENUE_RESIDUALS", {})
        predictor._load_learned_params()
        assert predictor.USE_LEARNED_GAME_MODEL is True
        with mock.patch.object(predictor.leaguegamefinder, "LeagueGameFinder") as team_api, \
             mock.patch.object(leaguegamelog, "LeagueGameLog") as player_api:
            team_api.return_value.get_data_frames.return_value = [team]
            player_api.return_value.get_data_frames.return_value = [player]
            features = predictor._fetch_learned_features(1, 2, "2023-24", "2024-01-17")
        raw = pd.DataFrame([features])[GAME_FEATURES_EXTENDED].values
        expected = fitted["log_model"].predict_proba(fitted["scaler"].transform(raw))[0, 1]
        assert features["star_form_diff"] != 0
        assert predictor._sigmoid_learned(features) == pytest.approx(expected)
