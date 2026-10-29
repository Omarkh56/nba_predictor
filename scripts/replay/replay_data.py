"""Season data for the replay: cached fetch, normalization, strict as-of filtering, as-of usage.

No final-season aggregate is ever used as a model input:
  * player usage (USG) is recomputed from box-score logs strictly before the game
    (regular season: season-to-date; playoffs: the completed regular season, which is legitimately known);
  * team defense and DvP are rebuilt by the engine from league logs strictly before each game date.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from replay_cache import NBACache
from replay_config import SEASON_TYPE_PO, SEASON_TYPE_RS


class ReplayDataError(RuntimeError):
    """Required data could not be fetched. The replay refuses to run on partial data."""


def parse_minutes(val: Any) -> float:
    """Minutes as float. 'MM:SS' strings and numerics accepted; None/NaN/garbage -> 0.0 (treated as did-not-play)."""
    try:
        if isinstance(val, str) and ":" in val:
            m, s = val.split(":")[:2]
            return int(m) + int(s) / 60.0
        f = float(val)
        return 0.0 if f != f else f
    except (TypeError, ValueError):
        return 0.0


def asof_before(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """The ONLY way replay code should restrict history: keep rows with GAME_DATE strictly before `cutoff`.

    Applied on every read, even if the cached/API response contains later games."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.iloc[0:0]
    return df[df["GAME_DATE"] < cutoff]


@dataclass
class SeasonBundle:
    season: str
    players: pd.DataFrame  # played rows (MIN > 0), RS + PO, ascending by (GAME_DATE, PLAYER_ID)
    teams: pd.DataFrame  # team-level game rows, RS + PO
    positions: Dict[str, str] = field(default_factory=dict)
    name_to_pid: Dict[str, int] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    previous_players: pd.DataFrame = field(default_factory=pd.DataFrame)


def _lgf(client: NBACache, who: str, season: str, season_type: str):
    params = {
        "player_or_team_abbreviation": who,
        "season_nullable": season,
        "season_type_nullable": season_type,
        "league_id_nullable": "00",
    }
    return client.fetch("leaguegamefinder", params, season, season_type, fetch_through="open")


def _normalize(df: pd.DataFrame, season_type: str) -> pd.DataFrame:
    out = df.copy()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], format="mixed")
    out["SEASON_TYPE"] = season_type
    if "MIN" in out.columns:
        out["MIN"] = out["MIN"].apply(parse_minutes)
    return out


def usage_asof(players: pd.DataFrame, min_games: int = 5) -> pd.Series:
    """USG% (fraction, like nba_api's USG_PCT) known BEFORE each player-game.

    USG = (FGA + 0.44*FTA + TOV) * (TeamMIN/5) / (MIN * (TmFGA + 0.44*TmFTA + TmTOV)).
    Regular-season rows: season-to-date before the game (NaN until `min_games` prior RS games).
    Playoff rows: the player's completed regular-season value (known before the playoffs begin)."""
    rs = players[players["SEASON_TYPE"] == SEASON_TYPE_RS].copy()
    rs["_pu"] = rs["FGA"].astype(float) + 0.44 * rs["FTA"].astype(float) + rs["TOV"].astype(float)
    grp = rs.groupby(["GAME_ID", "TEAM_ID"])
    rs["_tpu"] = grp["_pu"].transform("sum")
    rs["_tmin"] = grp["MIN"].transform("sum")
    rs["_num"] = rs["_pu"] * rs["_tmin"] / 5.0
    rs["_den"] = rs["MIN"] * rs["_tpu"]
    rs = rs.sort_values(["PLAYER_ID", "GAME_DATE"])
    g = rs.groupby("PLAYER_ID")
    cnum = g["_num"].cumsum() - rs["_num"]  # strictly before this game
    cden = g["_den"].cumsum() - rs["_den"]
    ngames = g.cumcount()
    before = (cnum / cden.replace(0, np.nan)).where(ngames >= min_games)
    rs["_usg_before"] = before

    final = (g["_num"].sum() / g["_den"].sum().replace(0, np.nan)).rename("_usg_final")

    out = pd.Series(np.nan, index=players.index, name="USG_ASOF")
    out.loc[rs.index] = rs["_usg_before"]
    po_idx = players.index[players["SEASON_TYPE"] == SEASON_TYPE_PO]
    out.loc[po_idx] = players.loc[po_idx, "PLAYER_ID"].map(final).values
    return out


def load_season_bundle(client: NBACache, season: str, want_rosters: bool = True) -> SeasonBundle:
    """Fetch (through the cache) and normalize everything a replay needs. Raises ReplayDataError on any FAILED
    fetch of required data; an empty playoff response for a season without playoffs is a valid success."""
    frames_p, frames_t, statuses = [], [], {}
    for st in (SEASON_TYPE_RS, SEASON_TYPE_PO):
        for who, sink in (("P", frames_p), ("T", frames_t)):
            res = _lgf(client, who, season, st)
            statuses["%s/%s" % (who, st)] = res.status
            if not res.ok:
                raise ReplayDataError("required fetch failed (%s %s %s): %s" % (who, season, st, res.error))
            if not res.frame.empty:
                sink.append(_normalize(res.frame, st))
    if not frames_p:
        raise ReplayDataError("no player logs for season %s" % season)

    players = pd.concat(frames_p, ignore_index=True)
    teams = pd.concat(frames_t, ignore_index=True) if frames_t else pd.DataFrame()
    n_raw = len(players)
    players = players[players["MIN"] > 0].copy()  # a row without minutes is a DNP, not a played game
    players = players.sort_values(["GAME_DATE", "PLAYER_ID"]).reset_index(drop=True)
    players["USG_ASOF"] = usage_asof(players)

    positions: Dict[str, str] = {}
    roster_status: Dict[str, str] = {}
    if want_rosters:
        for tid in sorted(players["TEAM_ID"].unique()):
            res = client.fetch(
                "commonteamroster", {"team_id": int(tid), "season": season}, season, SEASON_TYPE_RS,
                fetch_through="open", result_set="CommonTeamRoster",
            )
            roster_status[str(int(tid))] = res.status
            if res.ok and not res.frame.empty and {"PLAYER", "POSITION"} <= set(res.frame.columns):
                for _, r in res.frame.iterrows():
                    if r["PLAYER"] and r["POSITION"]:
                        positions[str(r["PLAYER"])] = str(r["POSITION"])

    name_to_pid = {n: int(p) for n, p in players.drop_duplicates("PLAYER_ID")[["PLAYER_NAME", "PLAYER_ID"]].itertuples(index=False)}
    meta = {
        "fetch_status": statuses,
        "roster_fetch_status_counts": pd.Series(list(roster_status.values())).value_counts().to_dict() if roster_status else {},
        "raw_player_rows": int(n_raw),
        "played_rows": int(len(players)),
        "dnp_rows_dropped": int(n_raw - len(players)),
        "date_range": {
            st: [str(g["GAME_DATE"].min().date()), str(g["GAME_DATE"].max().date())]
            for st, g in players.groupby("SEASON_TYPE")
        },
        "rows_by_season_type": players["SEASON_TYPE"].value_counts().to_dict(),
    }
    return SeasonBundle(season, players, teams, positions, name_to_pid, meta)


def load_previous_regular(client: NBACache, season: str) -> pd.DataFrame:
    """Previous regular season is a rate prior, not another evaluation population."""
    result = _lgf(client, "P", season, SEASON_TYPE_RS)
    if not result.ok:
        raise ReplayDataError("previous-season rate history unavailable: " + result.error)
    if result.frame.empty:
        return pd.DataFrame()
    return _normalize(result.frame, SEASON_TYPE_RS)
