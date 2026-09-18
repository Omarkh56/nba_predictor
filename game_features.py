"""Shared pre-game inputs for training and live NBA game predictions."""

import math

import numpy as np
import pandas as pd

GAME_FEATURE_VERSION = "season-history-v1"
GAME_FEATURES = [
    "net_rtg_diff", "efg_diff", "tov_diff", "orb_diff", "ftr_diff",
    "form_diff", "home_b2b", "away_b2b", "rest_diff", "h2h_margin",
]
GAME_FEATURES_NEW = ["altitude_penalty", "venue_residual", "star_form_diff"]
GAME_FEATURES_EXTENDED = GAME_FEATURES + GAME_FEATURES_NEW
_ALTITUDE_HOME_TIDS = frozenset({1610612743, 1610612762})
_STAR_FORM_SHRINK_K = 10

def altitude_feature(home_tid: int, away_b2b: int, away_rest_days: float) -> float:
    """
    Away-team acclimatization deficit at elevation venues (DEN, UTA only).
    Returns 0.0 for all other home venues.
    Range [0, 1]: 1 = maximally unacclimatized (B2B into altitude); 0 = well-rested.

    Acclimatization is approximated by rest-days only; a proper model would
    also track the previous city's elevation, but that data isn't in the NBA
    game log.  Conservative: teams that arrive 3+ days early are treated as
    fully acclimatized (factor → 0).
    """
    if int(home_tid) not in _ALTITUDE_HOME_TIDS:
        return 0.0
    if away_b2b:
        return 1.0
    # Linear interpolation: 0 rest_days → 1.0, 3+ rest_days → 0.0
    return float(max(0.0, (3.0 - float(away_rest_days)) / 3.0))


def venue_residual_from_record(home_win_count: int, home_game_count: int,
                                total_win_count: int, total_game_count: int) -> float:
    """
    Isolated venue/crowd effect: home win% minus overall win%.
    A positive value means the team wins MORE at home than their overall quality
    predicts; a negative value means they under-perform at home.
    Returns 0.0 when fewer than 3 home or 3 total games have been played.
    """
    if home_game_count < 3 or total_game_count < 3:
        return 0.0
    home_wp  = home_win_count  / home_game_count
    total_wp = total_win_count / total_game_count
    return float(home_wp - total_wp)


class _StarFormTracker:
    """Season-local scoring history; the current game's box score is never used."""

    def __init__(self, half_life, top_k, shrink_k):
        self.decay = 0.5 ** (1.0 / half_life)
        self.top_k = top_k
        self.shrink_k = shrink_k
        self.players = {}

    def observe(self, row):
        pid, tid = int(row.PLAYER_ID), int(row.TEAM_ID)
        state = self.players.setdefault(pid, {"team": tid, "n": 0, "sum": 0.0,
                                               "num": 0.0, "den": 0.0})
        state["team"] = tid
        if float(row.MIN) < 8:
            return
        pts = float(row.PTS)
        state["n"] += 1
        state["sum"] += pts
        state["num"] = pts + self.decay * state["num"]
        state["den"] = 1.0 + self.decay * state["den"]

    def team_form(self, tid):
        eligible = [(pid, s) for pid, s in self.players.items()
                    if s["team"] == tid and s["n"] >= 3 and s["sum"] > 0]
        eligible.sort(key=lambda item: (-item[1]["sum"] / item[1]["n"], item[0]))
        total, weighted = 0.0, 0.0
        for _, state in eligible[:self.top_k]:
            n = state["n"]
            ppg = state["sum"] / n
            deviation = state["num"] / state["den"] - ppg
            weighted += ppg * deviation * n / (n + self.shrink_k)
            total += ppg
        return weighted / total if total else 0.0


def compute_star_form_index(player_logs: pd.DataFrame, ewma_halflife: float = 5.0,
                            top_k: int = 2, shrink_k: int = _STAR_FORM_SHRINK_K,
                            fixtures: pd.DataFrame = None) -> pd.DataFrame:
    """PPG-weighted recent form of the top two scorers, using only earlier dates.

    Select stars from prior scoring history rather than today's participants.
    Observations reset each season and follow a player when they change teams.
    Optional fixtures allow the same calculation for a game that has not happened.
    """
    columns = ["team_id", "game_id", "star_form", "season"]
    if player_logs.empty:
        return pd.DataFrame(columns=columns)
    pl = player_logs.copy()
    if "season" not in pl:
        pl["season"] = ""
    pl["GAME_DATE"] = pd.to_datetime(pl["GAME_DATE"]).dt.normalize()
    pl["GAME_ID"] = pl["GAME_ID"].astype(str)
    for col in ("PTS", "MIN"):
        pl[col] = pd.to_numeric(pl[col], errors="coerce").fillna(0.0)
    if fixtures is None:
        fixtures = pl[["TEAM_ID", "GAME_ID", "GAME_DATE", "season"]].drop_duplicates()
    else:
        fixtures = fixtures.copy()
        fixtures["GAME_DATE"] = pd.to_datetime(fixtures["GAME_DATE"]).dt.normalize()
    output = []
    for season, games in fixtures.groupby("season", sort=False):
        observations = iter(pl[pl["season"] == season].sort_values(
            ["GAME_DATE", "PLAYER_ID"]).itertuples())
        next_row = next(observations, None)
        tracker = _StarFormTracker(ewma_halflife, top_k, shrink_k)
        for game in games.sort_values("GAME_DATE").itertuples():
            while next_row is not None and next_row.GAME_DATE < game.GAME_DATE:
                tracker.observe(next_row)
                next_row = next(observations, None)
            output.append({"team_id": int(game.TEAM_ID), "game_id": str(game.GAME_ID),
                           "star_form": tracker.team_form(int(game.TEAM_ID)), "season": season})
    return pd.DataFrame(output, columns=columns)


def _four_factors_from_row(row: pd.Series) -> dict:
    fga  = max(float(row.get("FGA", 1)), 1)
    fta  = float(row.get("FTA", 0))
    fgm  = float(row.get("FGM", 0))
    fg3m = float(row.get("FG3M", 0))
    tov  = float(row.get("TOV", 0))
    oreb = float(row.get("OREB", 0))
    reb  = max(float(row.get("REB", 1)), 1)
    return {
        "efg":  (fgm + 0.5 * fg3m) / fga,
        "tov":  tov / max(fga + 0.44 * fta + tov, 1),
        "orb":  oreb / reb,
        "ftr":  fta / fga,
        "pm":   float(row.get("PLUS_MINUS", 0)),
        "poss": fga + 0.44 * fta + tov - oreb,   # possessions estimate
    }


def _prep_team_logs(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Add derived per-game columns (date type, home flag, four-factor stats)."""
    team_logs = team_logs.copy()
    team_logs["GAME_DATE"] = pd.to_datetime(team_logs["GAME_DATE"]).dt.normalize()
    team_logs["GAME_ID"] = team_logs["GAME_ID"].astype(str)
    if "season" not in team_logs:
        team_logs["season"] = ""
    team_logs["is_home"]   = team_logs["MATCHUP"].str.contains(r"vs\.", na=False)

    for col, fn in [
        ("efg",  lambda r: (float(r["FGM"]) + 0.5*float(r["FG3M"])) / max(float(r["FGA"]),1)),
        ("tov_r",lambda r: float(r["TOV"]) / max(float(r["FGA"]) + 0.44*float(r["FTA"]) + float(r["TOV"]),1)),
        ("orb_r",lambda r: float(r["OREB"]) / max(float(r["REB"]),1)),
        ("ftr",  lambda r: float(r["FTA"]) / max(float(r["FGA"]),1)),
    ]:
        team_logs[col] = team_logs.apply(fn, axis=1)
    return team_logs


def _compute_team_rolling_stats(team_logs: pd.DataFrame, roll_stat_cols: list) -> dict:
    """Build per-team walk-forward rolling stats, keyed by TEAM_ID, indexed by GAME_ID."""
    team_stats = {}
    for (season, tid), grp in team_logs.groupby(["season", "TEAM_ID"]):
        g = grp.sort_values("GAME_DATE").reset_index(drop=True)
        g["prev_date"] = g["GAME_DATE"].shift(1)
        g["rest_days"] = ((g["GAME_DATE"] - g["prev_date"]).dt.days
                          .clip(upper=7).fillna(3).astype(int))
        g["b2b"] = (g["rest_days"] <= 1).astype(int)

        for col in roll_stat_cols:
            g[f"roll_{col}"] = g[col].shift(1).expanding(min_periods=3).mean()

        g["l10_pm"] = g["PLUS_MINUS"].shift(1).rolling(10, min_periods=4).mean()
        g["wpct"]   = (g["WL"] == "W").shift(1).expanding(min_periods=3).mean()

        # Walk-forward venue residual: home_win% - overall_win% before this game.
        # Positive = team wins MORE at home than their overall quality predicts.
        g["is_win"]      = (g["WL"] == "W").astype(float)
        g["cum_wins"]    = g["is_win"].shift(1).expanding().sum().fillna(0)
        g["cum_games"]   = pd.Series(np.arange(len(g)), index=g.index).values  # 0,1,2,...
        g["cum_hm_wins"] = (g["is_win"] * g["is_home"].astype(float)).shift(1).expanding().sum().fillna(0)
        g["cum_hm_games"]= g["is_home"].astype(float).shift(1).expanding().sum().fillna(0)
        g["venue_res"]   = g.apply(
            lambda r: venue_residual_from_record(
                int(r["cum_hm_wins"]), int(r["cum_hm_games"]),
                int(r["cum_wins"]),    int(r["cum_games"]),
            ),
            axis=1,
        )

        team_stats[(season, int(tid))] = g.set_index("GAME_ID")
    return team_stats


def _build_game_records(team_logs: pd.DataFrame, team_stats: dict, roll_stat_cols: list) -> pd.DataFrame:
    """Match home/away sides of each game and assemble the feature+target row."""
    records = []
    for (_, game_id), pair in team_logs.groupby(["season", "GAME_ID"]):
        home = pair[pair["is_home"]]
        away = pair[~pair["is_home"]]
        if home.empty or away.empty:
            continue
        hr = home.iloc[0]
        ar = away.iloc[0]
        htid = int(hr["TEAM_ID"])
        atid = int(ar["TEAM_ID"])

        season = hr["season"]
        hkey, akey = (season, htid), (season, atid)
        if hkey not in team_stats or akey not in team_stats:
            continue
        if game_id not in team_stats[hkey].index or game_id not in team_stats[akey].index:
            continue

        hs = team_stats[hkey].loc[game_id]
        as_ = team_stats[akey].loc[game_id]

        # Skip games with insufficient history (first few games of season)
        if any(pd.isna(hs[f"roll_{c}"]) for c in roll_stat_cols):
            continue
        if any(pd.isna(as_[f"roll_{c}"]) for c in roll_stat_cols):
            continue

        h_l10 = hs.get("l10_pm", 0) if not pd.isna(hs.get("l10_pm", np.nan)) else hs["roll_PLUS_MINUS"]
        a_l10 = as_.get("l10_pm", 0) if not pd.isna(as_.get("l10_pm", np.nan)) else as_["roll_PLUS_MINUS"]

        away_rest  = float(as_["rest_days"]) if not pd.isna(as_["rest_days"]) else 3.0
        away_b2b_v = int(as_["b2b"])
        h_venue_res = float(hs.get("venue_res", 0.0)) if not pd.isna(hs.get("venue_res", np.nan)) else 0.0
        a_venue_res = float(as_.get("venue_res", 0.0)) if not pd.isna(as_.get("venue_res", np.nan)) else 0.0

        rec = {
            # Features — original 10
            "net_rtg_diff":    hs["roll_PLUS_MINUS"] - as_["roll_PLUS_MINUS"],
            "efg_diff":        hs["roll_efg"]        - as_["roll_efg"],
            "tov_diff":        as_["roll_tov_r"]     - hs["roll_tov_r"],
            "orb_diff":        hs["roll_orb_r"]      - as_["roll_orb_r"],
            "ftr_diff":        hs["roll_ftr"]         - as_["roll_ftr"],
            "form_diff":       float(h_l10)           - float(a_l10),
            "home_b2b":        int(hs["b2b"]),
            "away_b2b":        int(as_["b2b"]),
            "rest_diff":       float(np.clip(hs["rest_days"] - as_["rest_days"], -5, 5)),
            "h2h_margin":      0.0,   # filled below for within-season H2H
            "home_wpct":       float(hs["wpct"]) if not pd.isna(hs["wpct"]) else 0.5,
            # New signal 1: altitude
            "altitude_penalty": altitude_feature(htid, away_b2b_v, away_rest),
            # New signal 2: residualized venue effect
            "venue_residual":  h_venue_res - a_venue_res,
            # New signal 3: star form (filled after player_logs merge)
            "star_form_diff":  0.0,
            # Targets
            "actual_margin":   float(hr["PLUS_MINUS"]),
            "home_win":        int(hr["WL"] == "W"),
            # Metadata
            "game_id":   game_id,
            "game_date": hr["GAME_DATE"],
            "season":    hr.get("season", ""),
            "home_tid":  htid,
            "away_tid":  atid,
        }
        records.append(rec)

    return pd.DataFrame(records)


def _add_h2h_margin(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in within-season H2H margin (shrinkage toward 0, only prior games)."""
    df = df.sort_values("game_date").reset_index(drop=True)
    h2h_seen: dict = {}
    h2h_margins = []
    for _, row in df.iterrows():
        low_tid = min(row["home_tid"], row["away_tid"])
        key = (row["season"], low_tid, max(row["home_tid"], row["away_tid"]))
        prior = h2h_seen.get(key, [])
        if prior:
            raw   = np.mean(prior)
            shrink = len(prior) / (len(prior) + 5)
            h2h_val = raw * shrink * (1 if row["home_tid"] == low_tid else -1)
        else:
            h2h_val = 0.0
        h2h_margins.append(h2h_val)
        # Store one consistent team perspective; convert back for today's home team.
        margin_for_key = (row["actual_margin"]
                          if row["home_tid"] == low_tid else -row["actual_margin"])
        h2h_seen.setdefault(key, []).append(margin_for_key)

    df["h2h_margin"] = h2h_margins
    return df


def _merge_star_form(df: pd.DataFrame, player_logs) -> pd.DataFrame:
    """Signal 3: merge star form from player logs (walk-forward, pre-computed)."""
    if player_logs is None or player_logs.empty:
        return df

    fixtures = pd.concat([
        df[["home_tid", "game_id", "game_date", "season"]].rename(columns={"home_tid": "TEAM_ID"}),
        df[["away_tid", "game_id", "game_date", "season"]].rename(columns={"away_tid": "TEAM_ID"}),
    ]).rename(columns={"game_id": "GAME_ID", "game_date": "GAME_DATE"})
    sf = compute_star_form_index(player_logs, fixtures=fixtures)
    if sf.empty:
        return df

    sf = sf.rename(columns={"star_form": "_h_star"})
    sf["game_id"] = sf["game_id"].astype(str)
    df["game_id"] = df["game_id"].astype(str)

    # Home team star form
    df = df.merge(
        sf[["team_id", "game_id", "season", "_h_star"]],
        left_on=["home_tid", "game_id", "season"],
        right_on=["team_id", "game_id", "season"],
        how="left",
    ).drop(columns="team_id", errors="ignore")

    # Away team star form
    sf2 = sf.rename(columns={"_h_star": "_a_star"})
    df = df.merge(
        sf2[["team_id", "game_id", "season", "_a_star"]],
        left_on=["away_tid", "game_id", "season"],
        right_on=["team_id", "game_id", "season"],
        how="left",
    ).drop(columns="team_id", errors="ignore")

    df["star_form_diff"] = (
        df["_h_star"].fillna(0.0) - df["_a_star"].fillna(0.0)
    )
    df = df.drop(columns=["_h_star", "_a_star"], errors="ignore")
    return df


def build_game_dataset(team_logs: pd.DataFrame,
                       player_logs: pd.DataFrame = None) -> pd.DataFrame:
    """
    Walk-forward game dataset. For each game G on date D, features are
    computed exclusively from games before D (no lookahead).

    Returns one row per game (home team perspective).
    Optional player_logs enables the star_form_diff feature.
    """
    if team_logs.empty:
        return pd.DataFrame()

    team_logs = _prep_team_logs(team_logs)
    roll_stat_cols = ["PLUS_MINUS", "efg", "tov_r", "orb_r", "ftr"]
    team_stats = _compute_team_rolling_stats(team_logs, roll_stat_cols)
    df = _build_game_records(team_logs, team_stats, roll_stat_cols)
    if df.empty:
        return df

    df = _add_h2h_margin(df)
    df = _merge_star_form(df, player_logs)
    return df


def build_live_game_features(team_logs: pd.DataFrame, player_logs: pd.DataFrame,
                             home_tid: int, away_tid: int, season: str,
                             game_date) -> dict:
    """Replay an upcoming fixture through the exact same walk-forward builder.

    The placeholder's outcomes are excluded by every shifted history calculation.
    Keep all prior games for both teams, including games against other opponents.
    """
    cutoff = pd.Timestamp(game_date).tz_localize(None).normalize()
    history = team_logs.copy()
    if history.empty:
        raise ValueError("no team game history")
    if "season" not in history:
        history["season"] = season
    dates = pd.to_datetime(history["GAME_DATE"]).dt.normalize()
    history = history[(history["season"] == season) & (dates < cutoff) &
                      history["TEAM_ID"].isin([home_tid, away_tid])].copy()
    fixture_id = "__upcoming__"
    upcoming = []
    for tid, matchup in ((home_tid, "HOME vs. AWAY"), (away_tid, "AWAY @ HOME")):
        row = {column: 0 for column in ("FGM", "FGA", "FG3M", "FTA", "OREB", "REB",
                                      "TOV", "PLUS_MINUS")}
        row.update(TEAM_ID=tid, GAME_ID=fixture_id, GAME_DATE=cutoff, season=season,
                   MATCHUP=matchup, WL="L")
        upcoming.append(row)
    replay = build_game_dataset(pd.concat([history, pd.DataFrame(upcoming)], ignore_index=True),
                                player_logs=player_logs)
    match = replay[replay["game_id"] == fixture_id] if not replay.empty else replay
    if match.empty:
        raise ValueError("fewer than three prior games for one or both teams")
    features = {name: float(match.iloc[0][name]) for name in GAME_FEATURES_EXTENDED}
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("non-finite game inputs")
    return features
