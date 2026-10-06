"""Pregame-only minutes forecast and low-sample rate shrinkage (experimental).

Missing lineup and absence evidence are explicit. Calendar gaps alone do not
establish injury or a return from absence. Coefficients are fixed before replay.
"""
import math

import numpy as np
import pandas as pd


def forecast_minutes(history, cutoff, spread=0.0, is_home=False,
                     starter=None, team_game_dates=None, playoffs=False):
    if history.empty or not {"GAME_DATE", "MIN"} <= set(history.columns):
        return None, {"reason": "no_prior_minutes"}
    frame = history.copy()
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    cutoff = pd.Timestamp(cutoff).normalize()
    frame = frame[(frame.GAME_DATE < cutoff) & (frame.MIN > 0)].sort_values("GAME_DATE")
    if frame.empty:
        return None, {"reason": "no_prior_minutes"}
    frame = frame.tail(20)
    values = frame.MIN.astype(float).to_numpy()
    weights = np.exp2(-np.arange(len(values) - 1, -1, -1) / 4.0)
    ewm = float(np.average(values, weights=weights))
    # A proxy is recorded separately from an observed pregame lineup flag.
    starter_source = "lineup" if starter is not None else "minutes_proxy"
    starter = bool(starter) if starter is not None else ewm >= 28
    rest = max(0, (cutoff - frame.GAME_DATE.iloc[-1]).days - 1)
    minutes = ewm - (0.5 if rest == 0 else 0.0)
    if playoffs:
        minutes *= 1.06 if starter else (1.0 if ewm >= 18 else 0.92)
    reduced = 0.0
    if starter and abs(spread) > 8 and is_home == (spread > 0):
        reduced = min(5.0, (abs(spread) - 8) * 0.5)
        minutes -= reduced
    returned = None
    if team_game_dates is not None:
        schedule = pd.to_datetime(list(team_game_dates))
        schedule = schedule[schedule < cutoff]
        dates = frame.GAME_DATE.tolist()
        for index in range(len(dates) - 1, 0, -1):
            missed = int(((schedule > dates[index - 1]) & (schedule < dates[index])).sum())
            if missed >= 3:
                returned = len(dates) - index
                break
    if returned is not None and returned < 8:
        minutes *= 1.0 - 0.08 * (8 - returned) / 8.0
    return float(np.clip(minutes, 0.0, 48.0)), {
        "ewm_minutes": ewm, "starter": starter, "starter_source": starter_source,
        "rest_days": rest, "spread": float(spread), "blowout_reduction": reduced,
        "games_since_absence_return": returned,
        "absence_source": "team_schedule" if team_game_dates is not None else "unavailable",
    }


def shrink_rate(current, previous, games):
    """Ten-game prior strength fades linearly to zero at ten current games."""
    if previous is None or not math.isfinite(previous) or previous < 0 or games >= 10:
        return current
    if current is None or not math.isfinite(current):
        return previous
    weight = max(0, games) / 10.0
    return weight * current + (1.0 - weight) * previous
