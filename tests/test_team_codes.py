"""team_codes: one vocabulary for ESPN codes, NBA Stats codes and full names."""

from datetime import date, datetime, timezone

import pytest

import team_codes


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("NY", "NYK"), ("GS", "GSW"), ("SA", "SAS"), ("NO", "NOP"),
        ("UTAH", "UTA"), ("WSH", "WAS"), ("UTH", "UTA"), ("PHO", "PHX"),
        ("nyk", "NYK"), (" BOS ", "BOS"),
        ("Los Angeles Clippers", "LAC"), ("LA Clippers", "LAC"),
        ("Los Angeles Lakers", "LAL"), ("los angeles lakers", "LAL"),
        ("Utah Jazz", "UTA"), ("Philadelphia 76ers", "PHI"),
    ],
)
def test_normalize_team_maps_every_source_to_nba_codes(raw, expected):
    assert team_codes.normalize_team(raw) == expected


@pytest.mark.parametrize("raw", ["", "  ", "Los", "XYZ", None, 12])
def test_normalize_team_unknown_is_empty(raw):
    assert team_codes.normalize_team(raw) == ""


def test_every_canonical_code_maps_to_itself():
    for code in team_codes.NBA_CODES:
        assert team_codes.normalize_team(code) == code
    assert len(team_codes.NBA_CODES) == 30


def test_game_keys_no_longer_collide():
    # The old tracker keyed both of these as "NEW@LOS".
    knicks_lakers = team_codes.game_key("New York Knicks", "Los Angeles Lakers")
    pels_clippers = team_codes.game_key("New Orleans Pelicans", "Los Angeles Clippers")
    assert knicks_lakers == "NYK@LAL"
    assert pels_clippers == "NOP@LAC"


def test_slate_date_uses_us_eastern_calendar():
    # 02:00 UTC on Oct 22 is 10 pm ET on Oct 21.
    assert team_codes.slate_date_of("2026-10-22T02:00:00Z") == date(2026, 10, 21)
    assert team_codes.slate_date_of("2026-10-22T23:30:00Z") == date(2026, 10, 22)
    assert team_codes.slate_date_of("not a time") is None
    assert team_codes.slate_date_of(None) is None


def test_events_for_date_keeps_only_that_slate():
    events = [
        {"id": "tonight", "commence_time": "2026-10-22T23:30:00Z"},
        {"id": "late-tonight", "commence_time": "2026-10-23T02:30:00Z"},
        {"id": "later", "commence_time": "2026-10-24T23:30:00Z"},
        {"id": "no-time"},
    ]
    kept = team_codes.events_for_date(events, date(2026, 10, 22))
    assert [ev["id"] for ev in kept] == ["tonight", "late-tonight"]


def test_today_slate_date_rolls_at_eastern_midnight():
    late_utc = datetime(2026, 10, 22, 3, 30, tzinfo=timezone.utc)  # 11:30 pm ET Oct 21
    assert team_codes.today_slate_date(late_utc) == date(2026, 10, 21)
