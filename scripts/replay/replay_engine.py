"""Chronological replay engine.

The REAL, unmodified playerlinepredictor.project_prop() produces every projection. Only its data sources are
replaced, and every replacement is strictly as-of (rows with GAME_DATE < the predicted game's date):

  _fetch_logs        -> the player's prior games (RS and PO kept separate, MIN filters applied by the same rule)
  get_league_logs    -> league logs strictly before the date (playoff logs once a playoff game has happened,
                        otherwise regular-season logs: exactly the live fall-back rule)
  DvP engine         -> rebuilt per date from those same as-of league logs
  PLAYER_USG_CACHE   -> as-of usage computed from box scores (season_data.usage_asof)
  get_team_stat_avg  -> 0.0 (only feeds the injury-driven usage boost, which is inert without injuries)

Held neutral / not reproducible (documented, never silently assumed): live injury status (usage boost = 1.0),
book spread (0.0), matchup pace (1.0). Worker processes have network access disabled.
"""
from __future__ import annotations

import ast
import contextlib
import io
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from replay_config import (
    ELIG_MIN_AVG_MINUTES_LAST10,
    ELIG_MIN_PRIOR_GAMES,
    MARKETS,
    NEUTRAL_PACE_FACTOR,
    NEUTRAL_SPREAD,
    SEASON_TYPE_PO,
    SEASON_TYPE_RS,
)
from replay_data import SeasonBundle, asof_before

_W: Dict[str, Any] = {}  # per-process worker state
_CTX: Dict[str, Any] = {}  # per-row context read by the patched data sources


def _blocked(*_a, **_k):
    raise RuntimeError("network access is disabled inside replay workers (a live call would be lookahead)")


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError("override must be NAME=VALUE: %s" % p)
        k, v = p.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def eligibility(minutes_before: np.ndarray) -> Tuple[bool, str]:
    """Pre-game eligibility from ALL prior played games this season (independent of any model filter)."""
    if len(minutes_before) < ELIG_MIN_PRIOR_GAMES:
        return False, "insufficient_prior_games"
    if ELIG_MIN_AVG_MINUTES_LAST10 > 0 and float(np.mean(minutes_before[-10:])) < ELIG_MIN_AVG_MINUTES_LAST10:
        return False, "low_recent_minutes"
    return True, ""


def split_dates(dates: List[pd.Timestamp], n_chunks: int, weights: Optional[List[int]] = None) -> List[List[pd.Timestamp]]:
    """Contiguous, balanced-by-weight chunks of an ascending date list (each chunk stays chronological)."""
    if not dates:
        return []
    weights = weights or [1] * len(dates)
    total, target = float(sum(weights)), float(sum(weights)) / max(1, n_chunks)
    chunks, cur, acc = [], [], 0.0
    for d, w in zip(dates, weights):
        cur.append(d)
        acc += w
        if acc >= target and len(chunks) < n_chunks - 1:
            chunks.append(cur)
            cur, acc = [], 0.0
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------------------------- worker
def init_worker(bundle: SeasonBundle, overrides: Dict[str, Any], project_root: str) -> None:
    os.environ["ODDS_API_KEY"] = "replay-dummy-key"  # the replay must never hold or use the real key
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    with contextlib.redirect_stdout(io.StringIO()):
        import playerlinepredictor as pl
    import requests

    requests.Session.request = _blocked  # any accidental live call fails loudly

    players = bundle.players
    _W["pl"] = pl
    _W["bundle"] = bundle
    _W["rs_league"] = players[players["SEASON_TYPE"] == SEASON_TYPE_RS]
    _W["po_league"] = players[players["SEASON_TYPE"] == SEASON_TYPE_PO]
    pl_players = players.copy()
    pl_players["_i"] = pl_players.groupby("PLAYER_ID").cumcount()
    _W["pgroups"] = {pid: g.reset_index(drop=True) for pid, g in pl_players.groupby("PLAYER_ID", sort=False)}
    _W["by_date"] = {d: list(zip(g["PLAYER_ID"].tolist(), g["_i"].tolist())) for d, g in pl_players.groupby("GAME_DATE")}

    pl.SEASON = bundle.season  # the model's notion of "current season" must be the replayed season
    # Always reconstruct the raw historical model first. Applying the FINAL
    # 2025-26 artifact to 2025-26 would leak future outcomes. The correction
    # evaluator fits each date using prior dates only in a separate pass.
    pl.RESIDUAL_CORRECTIONS = {}
    for name, value in (overrides or {}).items():
        if not hasattr(pl, name):
            raise AttributeError("override target %s does not exist on playerlinepredictor" % name)
        setattr(pl, name, value)
    _W["overrides"] = dict(overrides or {})

    # ---- as-of data sources -------------------------------------------------------------
    pl.get_player_id = lambda name: bundle.name_to_pid.get(name)
    pl.get_league_logs = lambda: _CTX["league_logs"]
    pl.get_team_stat_avg = lambda _abbr, _stat: 0.0

    real_parse = pl._parse_minutes

    def _fetch_logs_asof(pid, season, season_type, min_minutes, retries=3):
        if pid != _CTX["pid"]:
            raise AssertionError("replay history requested for a different player than the one being predicted")
        df = _CTX["prior_rs"] if season_type == pl.SEASON_TYPE_RS else _CTX["prior_po"]
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        keep = out["MIN"].apply(real_parse) >= min_minutes
        return out[keep].reset_index(drop=True)

    pl._fetch_logs = _fetch_logs_asof

    for name, pos in bundle.positions.items():
        pg = pl._pos_group(pos)
        if pg:
            pl.PLAYER_POS_CACHE[name] = pg
    pl.PLAYER_POS_CACHE.update(pl.MANUAL_POSITIONS)
    pl.dvp_engine.inject_positions(pl.PLAYER_POS_CACHE)
    _W["date_key"] = None


def _setup_date(d: pd.Timestamp) -> str:
    pl = _W["pl"]
    prior_po = asof_before(_W["po_league"], d)
    if len(prior_po):
        mode, window, league = "playoffs", "playoffs", prior_po
        pl.IS_PLAYOFF_SEASON = True
    else:
        mode, window, league = "regular", "season", asof_before(_W["rs_league"], d)
        pl.IS_PLAYOFF_SEASON = False
    if not league.empty and league["GAME_DATE"].max() >= d:
        raise AssertionError("lookahead: league logs contain games on/after %s" % d.date())
    _CTX["league_logs"] = league
    pl.TEAM_DEF_CACHE.clear()
    pl.PLAYER_STD_CACHE.clear()
    pl.dvp_engine.clear_cache()
    if league.empty:
        # Nothing has been played yet (season opener): no player can be eligible, and building the DvP table
        # would make dvp.py fall back to a LIVE fetch because inject_logs() ignores empty frames.
        return mode
    pl.dvp_engine.inject_logs(league, window)
    pl.dvp_engine.build_dvp_table(window)
    return mode


def run_chunk(dates: List[pd.Timestamp]):
    t0 = time.perf_counter()
    pl, pgroups, by_date = _W["pl"], _W["pgroups"], _W["by_date"]
    rows: List[Dict[str, Any]] = []
    counts = {"played_rows": 0, "ineligible_insufficient_prior_games": 0, "ineligible_low_recent_minutes": 0,
              "eligible_player_games": 0, "projection_rows": 0}
    failures: Dict[str, int] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for d in dates:
            mode = _setup_date(d)
            for pid, i in by_date.get(d, []):
                g = pgroups[pid]
                counts["played_rows"] += 1
                ok, why = eligibility(g["MIN"].values[:i])
                if not ok:
                    counts["ineligible_" + why] += 1
                    continue
                counts["eligible_player_games"] += 1
                row = g.iloc[i]
                roll_min10 = float(np.mean(g["MIN"].values[max(0, i - 10):i]))
                prior = g.iloc[:i].iloc[::-1]
                if len(prior) and prior["GAME_DATE"].max() >= d:
                    raise AssertionError("lookahead: prior history contains a game on/after %s" % d.date())
                _CTX.update(
                    pid=pid,
                    prior_rs=prior[prior["SEASON_TYPE"] == SEASON_TYPE_RS].reset_index(drop=True),
                    prior_po=prior[prior["SEASON_TYPE"] == SEASON_TYPE_PO].reset_index(drop=True),
                )
                name = str(row["PLAYER_NAME"])
                pl.PLAYER_USG_CACHE.clear()
                if not pd.isna(row["USG_ASOF"]):
                    pl.PLAYER_USG_CACHE[name] = float(row["USG_ASOF"])
                team, opp = pl._parse_matchup_teams(str(row["MATCHUP"]))
                opp_tid = pl.get_team_id(opp) if opp else None
                if team is None or opp_tid is None:
                    failures["no_team_or_opponent_id"] = failures.get("no_team_or_opponent_id", 0) + len(MARKETS)
                    continue
                is_home = " vs. " in str(row["MATCHUP"])
                for label, mkt_key, cols in MARKETS:
                    proj, details, sd, meta = pl.project_prop(
                        name, mkt_key, opp, opp_tid, [], None, is_home, team,
                        spread=NEUTRAL_SPREAD, pace_factor=NEUTRAL_PACE_FACTOR,
                    )
                    if proj is None:
                        failures["project_prop_returned_none"] = failures.get("project_prop_returned_none", 0) + 1
                        continue
                    counts["projection_rows"] += 1
                    rows.append({
                        "date": str(d.date()),
                        "game_id": row["GAME_ID"],
                        "player_id": int(pid),
                        "player": name,
                        "team": team,
                        "opp": opp,
                        "is_home": bool(is_home),
                        "season_type": row["SEASON_TYPE"],
                        "mode": mode,
                        "market": label,
                        "actual": float(sum(float(row[c]) for c in cols)),
                        "projection": float(proj),
                        "sd": float(sd) if sd is not None else np.nan,
                        "proj_min": float(np.mean([v["min"] for v in details.values()])),
                        "actual_min": float(row["MIN"]),
                        "po_count": int(meta.get("po_count", 0)),
                        "pos_group": details[cols[0]].get("pos_group"),
                        "roll_min10": roll_min10,
                        "n_prior": int(i),
                        "usg_used": float(row["USG_ASOF"]) if not pd.isna(row["USG_ASOF"]) else np.nan,
                        "rust_discount": float(details[cols[0]].get("rust_discount", 0.0) or 0.0),
                        "games_since_return": details[cols[0]].get("games_since_return"),
                    })
    return rows, counts, failures, time.perf_counter() - t0


# ---------------------------------------------------------------------------------------------- driver
def run_replay(
    bundle: SeasonBundle,
    season_types: List[str],
    workers: int,
    overrides: Optional[Dict[str, Any]],
    project_root: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    log=lambda msg: None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    players = bundle.players
    sel = players[players["SEASON_TYPE"].isin(season_types)]
    if date_from:
        sel = sel[sel["GAME_DATE"] >= pd.Timestamp(date_from)]
    if date_to:
        sel = sel[sel["GAME_DATE"] <= pd.Timestamp(date_to)]
    per_date = sel.groupby("GAME_DATE").size()
    dates = list(per_date.index)
    chunks = split_dates(dates, max(1, workers * 4), [int(per_date[d]) for d in dates])
    log("replaying %d dates in %d chunks with %d worker(s)" % (len(dates), len(chunks), workers))

    t0 = time.perf_counter()
    all_rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    failures: Dict[str, int] = {}
    busy = 0.0
    if workers <= 1:
        init_worker(bundle, overrides or {}, project_root)
        results = (run_chunk(c) for c in chunks)
        pool = None
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(workers, initializer=init_worker, initargs=(bundle, overrides or {}, project_root))
        results = pool.imap_unordered(run_chunk, chunks)
    done = 0
    try:
        for rows, cnt, fail, secs in results:
            all_rows.extend(rows)
            for k, v in cnt.items():
                counts[k] = counts.get(k, 0) + v
            for k, v in fail.items():
                failures[k] = failures.get(k, 0) + v
            busy += secs
            done += 1
            if done % max(1, len(chunks) // 10) == 0 or done == len(chunks):
                log("  chunks %d/%d, rows so far %d" % (done, len(chunks), len(all_rows)))
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["date", "player_id", "market"]).reset_index(drop=True)
    info = {
        "dates": len(dates),
        "counts": counts,
        "failures": failures,
        "wall_s": round(time.perf_counter() - t0, 2),
        "worker_busy_s": round(busy, 2),
        "workers": workers,
        "overrides": overrides or {},
    }
    return df, info
