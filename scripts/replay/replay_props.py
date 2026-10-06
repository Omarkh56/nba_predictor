#!/usr/bin/env python3
"""Chronological backtest of the NBA player-prop projection model.

    python3 scripts/replay/replay_props.py run --label baseline --season 2025-26 --season-type both
    python3 scripts/replay/replay_props.py run --label after-x --compare baseline
    python3 scripts/replay/replay_props.py report --label baseline

See scripts/replay/README.md for the cold/warm cache commands and the full method.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["ODDS_API_KEY"] = "replay-dummy-key"  # the replay never needs (or touches) the real Odds API key

import pandas as pd  # noqa: E402

import replay_engine as engine  # noqa: E402
import replay_provenance as provenance  # noqa: E402
import replay_report as rpt  # noqa: E402
from replay_metrics import load_or_create_splits  # noqa: E402
from replay_cache import NBACache  # noqa: E402
from replay_config import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    DEFAULT_CACHE_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_SEASON,
    ELIGIBILITY_VERSION,
    ELIG_MIN_PRIOR_GAMES,
    REPLAY_CONFIG_VERSION,
    REPORTS_DIR,
    SEASON_TYPE_PO,
    SEASON_TYPE_RS,
)
from replay_data import load_season_bundle, load_previous_regular  # noqa: E402

SEASON_TYPES = {"regular": [SEASON_TYPE_RS], "playoffs": [SEASON_TYPE_PO], "both": [SEASON_TYPE_RS, SEASON_TYPE_PO]}


def _out_path(out_dir: str, label: str, season: str) -> str:
    return os.path.join(out_dir, "%s_%s.csv.gz" % (label, season))


def read_replay(path):
    """NBA game IDs are identifiers, including significant leading zeroes."""
    return pd.read_csv(path, dtype={"game_id": str})


def cmd_run(a: argparse.Namespace) -> int:
    t_start = time.perf_counter()
    log = lambda m: print("[replay] " + m, flush=True)  # noqa: E731
    cache_dir = a.cache_dir
    if a.cold:
        cache_dir = tempfile.mkdtemp(prefix="replay-cache-cold-")
        log("COLD run: using an empty temporary cache at %s" % cache_dir)
    def offline_fetch(*_args):
        raise ValueError("cache-only replay: required response is not cached")
    client = NBACache(cache_dir, min_interval_s=a.min_interval,
                      fetcher=offline_fetch if a.cache_only else None,
                      retries=1 if a.cache_only else 3)
    entries_before = client.entry_count() if os.path.isdir(cache_dir) else 0

    t0 = time.perf_counter()
    bundle = load_season_bundle(client, a.season, want_rosters=not a.skip_rosters)
    load_s = time.perf_counter() - t0
    log("data loaded in %.1fs: %s" % (load_s, bundle.meta["rows_by_season_type"]))

    overrides = engine.parse_overrides(a.set)
    if overrides.get("MINUTES_FORECAST_ENABLED"):
        year = int(a.season[:4]) - 1
        previous_season = "%d-%02d" % (year, (year + 1) % 100)
        bundle.previous_players = load_previous_regular(client, previous_season)
        bundle.meta["previous_season"] = previous_season
    df, info = engine.run_replay(bundle, SEASON_TYPES[a.season_type], a.workers, overrides, ROOT,
                                 a.date_from, a.date_to, log=log, min_prior_games=a.min_prior_games)
    if df.empty:
        log("no observations produced")
        return 1
    os.makedirs(a.out_dir, exist_ok=True)
    df.to_csv(_out_path(a.out_dir, a.label, a.season), index=False, compression="gzip")

    prev = prev_label = None
    if a.compare:
        pp = _out_path(a.out_dir, a.compare, a.season)
        if os.path.exists(pp):
            prev, prev_label = read_replay(pp), a.compare
        else:
            log("WARNING: previous per-row output %s not found; no comparison written" % pp)

    partial = bool(a.date_from or a.date_to)  # a partial window must never define the frozen split boundaries
    splits = load_or_create_splits(df, a.season, persist=not partial)
    total_s = time.perf_counter() - t_start
    prov = {
        "git": provenance.git_info(ROOT),
        "code_fingerprint": provenance.code_fingerprint(ROOT),
        "replay_config_version": REPLAY_CONFIG_VERSION,
        "eligibility_version": ELIGIBILITY_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "artifacts": provenance.artifact_info(ROOT),
    }
    cache = {
        "dir": cache_dir if not a.cold else "<temporary cold cache>",
        "state": "cold" if (a.cold or entries_before == 0) else "warm",
        "entries_before": entries_before,
        "stats": client.stats.as_dict(),
    }
    runtime = {"load_s": round(load_s, 2), "replay_wall_s": info["wall_s"], "workers": info["workers"],
               "worker_busy_s": info["worker_busy_s"], "total_s": round(total_s, 2)}
    manifest = {"info": info, "provenance": prov, "cache": cache, "runtime": runtime,
                "bundle_meta": bundle.meta, "season_start": bundle.meta["date_range"][SEASON_TYPE_RS][0]}
    with open(os.path.join(a.out_dir, a.label + "_" + a.season + "_run.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    summary = rpt.build_summary(a.label, df, info, prov, a.season, [s for s in SEASON_TYPES[a.season_type]],
                                bundle.meta, cache, runtime, splits, prev, prev_label)
    from minutes_diagnostics import write_diagnostics
    write_diagnostics(df, prev, a.label, a.season, a.reports_dir,
                      bundle.meta["date_range"][SEASON_TYPE_RS][0])
    paths = rpt.write_reports(summary, a.reports_dir)
    log("wrote %s and %s (%.1fs total)" % (paths["json"], paths["markdown"], total_s))
    if a.cold:
        shutil.rmtree(cache_dir, ignore_errors=True)
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    """Rebuild a report from a saved per-row output (no model run, no API)."""
    df = read_replay(_out_path(a.out_dir, a.label, a.season))
    prev = read_replay(_out_path(a.out_dir, a.compare, a.season)) if a.compare else None
    splits = load_or_create_splits(df, a.season)
    info = {"counts": {}, "failures": {}, "overrides": {}, "wall_s": 0, "workers": 0, "worker_busy_s": 0}
    prov = {"git": provenance.git_info(ROOT), "code_fingerprint": provenance.code_fingerprint(ROOT),
            "replay_config_version": REPLAY_CONFIG_VERSION, "eligibility_version": ELIGIBILITY_VERSION,
            "cache_schema_version": CACHE_SCHEMA_VERSION, "artifacts": provenance.artifact_info(ROOT)}
    cache = {"dir": "n/a", "state": "n/a (report only)", "entries_before": 0,
             "stats": {k: 0 for k in ("requests", "retries", "failures", "cache_hits", "cache_misses", "cache_writes",
                                      "rate_limit_delay_s", "backoff_delay_s")}}
    runtime = {"load_s": 0, "replay_wall_s": 0, "workers": 0, "worker_busy_s": 0, "total_s": 0}
    bundle_meta = {}
    manifest_path = os.path.join(a.out_dir, a.label + "_" + a.season + "_run.json")
    season_start = a.season_start
    if os.path.exists(manifest_path):
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        info, prov, cache, runtime = (manifest[k] for k in ("info", "provenance", "cache", "runtime"))
        bundle_meta = manifest["bundle_meta"]
        season_start = season_start or manifest.get("season_start")
    from minutes_diagnostics import write_diagnostics
    write_diagnostics(df, prev, a.label, a.season, a.reports_dir, season_start)
    s = rpt.build_summary(a.label, df, info, prov, a.season, sorted(df["season_type"].unique()), bundle_meta, cache, runtime,
                          splits, prev, a.compare)
    print(rpt.write_reports(s, a.reports_dir))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--label", required=True)
    common.add_argument("--season", default=DEFAULT_SEASON)
    common.add_argument("--compare", help="label of a previous run to compare against (paired, chronological)")
    common.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    common.add_argument("--reports-dir", default=REPORTS_DIR)
    r = sub.add_parser("run", parents=[common], help="fetch (cached), replay and report")
    r.add_argument("--season-type", choices=sorted(SEASON_TYPES), default="both")
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    r.add_argument("--skip-rosters", action="store_true", help="use manual positions only; record absent roster coverage")
    r.add_argument("--cache-only", action="store_true", help="fail on a required cache miss without networking")
    r.add_argument("--cold", action="store_true", help="use an empty temporary cache (measures cold-cache runtime)")
    r.add_argument("--min-interval", type=float, default=1.2, help="minimum seconds between live NBA API requests")
    r.add_argument("--min-prior-games", type=int, default=ELIG_MIN_PRIOR_GAMES,
                   help="explicit replay population floor; use 1 for early-season experiments")
    r.add_argument("--date-from")
    r.add_argument("--date-to")
    r.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                   help="override a playerlinepredictor module attribute for this run (repeatable)")
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("report", parents=[common], help="rebuild a report from saved per-row output")
    p.add_argument("--season-start", help="true opener for old outputs without a run manifest")
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
