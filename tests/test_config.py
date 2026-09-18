"""
tests/test_config.py — Unit tests for config.py's shared season logic.

No network access required.
"""
import os
from datetime import date

import pytest

import config


@pytest.fixture(autouse=True)
def _clear_override():
    """NBA_SEASON must not leak between tests."""
    os.environ.pop("NBA_SEASON", None)
    yield
    os.environ.pop("NBA_SEASON", None)


class TestCurrentSeason:
    def test_october_starts_the_new_season(self):
        assert config.current_season(date(2026, 10, 1)) == "2026-27"

    def test_september_is_still_the_prior_season(self):
        assert config.current_season(date(2026, 9, 30)) == "2025-26"

    def test_september_october_boundary_is_exact(self):
        # The whole point of the >= 10 check -- Sept 30 and Oct 1 must
        # land on opposite sides.
        sept = config.current_season(date(2025, 9, 30))
        octo = config.current_season(date(2025, 10, 1))
        assert sept == "2024-25"
        assert octo == "2025-26"
        assert sept != octo

    def test_mid_season_january_belongs_to_the_prior_october(self):
        # A January game is played in the season that started the PREVIOUS
        # October, not the calendar year it falls in.
        assert config.current_season(date(2027, 1, 15)) == "2026-27"

    def test_june_finals_still_belongs_to_the_october_season(self):
        assert config.current_season(date(2027, 6, 10)) == "2026-27"

    def test_defaults_to_today_when_no_date_given(self):
        assert config.current_season() == config.current_season(date.today())

    def test_nba_season_env_override(self):
        os.environ["NBA_SEASON"] = "2019-20"
        assert config.current_season(date(2026, 11, 1)) == "2019-20"

    def test_two_digit_year_wrap(self):
        # 1999-2000 season -> "1999-00", not "1999-100"
        assert config.current_season(date(1999, 11, 1)) == "1999-00"


class TestTrainingSeasons:
    def test_five_completed_seasons_before_current(self):
        # "Today" is October 2026 -> current season is 2026-27, so the
        # last 5 COMPLETED seasons run 2021-22 .. 2025-26.
        result = config.training_seasons(5, d=date(2026, 10, 15))
        assert result == ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

    def test_never_includes_the_in_progress_season(self):
        result = config.training_seasons(5, d=date(2026, 10, 15))
        assert "2026-27" not in result

    def test_oldest_first_ordering(self):
        result = config.training_seasons(3, d=date(2026, 10, 15))
        assert result == sorted(result)

    def test_n_controls_count(self):
        assert len(config.training_seasons(3, d=date(2026, 10, 15))) == 3
        assert len(config.training_seasons(8, d=date(2026, 10, 15))) == 8

    def test_respects_september_boundary_too(self):
        # In September 2026, current season is still 2025-26, so the last
        # completed season is 2024-25, not 2025-26.
        result = config.training_seasons(1, d=date(2026, 9, 15))
        assert result == ["2024-25"]


class TestValidationSeason:
    def test_is_the_single_most_recent_completed_season(self):
        assert config.validation_season(date(2026, 10, 15)) == "2025-26"
        assert config.validation_season(date(2026, 9, 15)) == "2024-25"

    def test_matches_training_seasons_last_element(self):
        d = date(2026, 11, 1)
        assert config.validation_season(d) == config.training_seasons(5, d=d)[-1]
