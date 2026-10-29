"""v9 intentionally replaces the Sep 19 PTS multiplier test with replay-correction tests."""

import pytest

import playerlinepredictor as predictor
from residual_correction import apply_correction, fit_market, load_artifact


def test_unconditional_additive_shrinks_toward_zero_for_small_sample():
    stats = fit_market([(20, 15), (10, 8)], "additive", k=8)
    assert stats["n"] == 2
    assert stats["mean_residual"] == pytest.approx(-3.5)
    assert stats["correction"] == pytest.approx(-0.7)


def test_multiplicative_fraction_scales_with_projection():
    stats = fit_market([(20, 16), (10, 8)], "multiplicative", k=2)
    artifact = {"version": "v9-unconditional-1", "form": "multiplicative",
                "markets": {"PTS": stats}}
    assert stats["correction"] == pytest.approx(-0.1)
    assert apply_correction(30, "PTS", artifact) == pytest.approx(27)
    assert apply_correction(10, "PTS", artifact) == pytest.approx(9)


def test_exact_market_not_stat_components_or_unrelated_market():
    artifact = {"version": "v9-unconditional-1", "form": "additive",
                "markets": {"PTS": {"correction": -1.3},
                            "PTS+AST": {"correction": -1.6}}}
    assert apply_correction(20, "PTS", artifact) == pytest.approx(18.7)
    assert apply_correction(25, "PTS+AST", artifact) == pytest.approx(23.4)
    assert apply_correction(5, "AST", artifact) == 5
    assert not hasattr(predictor, "MARKET_PROJ_ADJ")
    assert not hasattr(predictor, "PLAYER_PROJ_ADJ")


def test_missing_artifact_is_identity_and_invalid_artifact_rejected(tmp_path):
    assert load_artifact(tmp_path / "missing.json") == {}
    assert apply_correction(20, "PTS", {}) == 20
    invalid = tmp_path / "bad.json"
    invalid.write_text('{"version":"bad", "form":"additive"}')
    with pytest.raises(ValueError, match="unsupported"):
        load_artifact(invalid)
