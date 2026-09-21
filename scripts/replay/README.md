# Chronological replay / backtest engine

Replays the NBA player-prop projection model over **every eligible player-game outcome** of a season and reports
bias, MAE and RMSE for each exact market. It runs the real, unmodified `playerlinepredictor.project_prop()`; only
the model's data sources are replaced with strictly as-of versions.

## Commands (run from the repository root)

```bash
# 2025-26 regular season only
python3 scripts/replay/replay_props.py run --label my-run --season 2025-26 --season-type regular

# 2026 playoffs only (the 2025-26 postseason, played April-June 2026)
python3 scripts/replay/replay_props.py run --label my-run --season 2025-26 --season-type playoffs

# both regimes in one run, plus the combined report (default), compared with a previous run
python3 scripts/replay/replay_props.py run --label my-run --season 2025-26 --season-type both --compare baseline

# rebuild a report from saved per-row output (no model run, no API)
python3 scripts/replay/replay_props.py report --label my-run --compare baseline

# COLD-cache run: empty temporary cache, every response fetched live (measures cold runtime + API counts)
python3 scripts/replay/replay_props.py run --label cold-check --season-type both --cold

# WARM-cache run: repeat the same command without --cold; nothing is fetched, results must be byte-identical
python3 scripts/replay/replay_props.py run --label warm-check --season-type both

# try a model variant without editing code: override any playerlinepredictor module attribute
python3 scripts/replay/replay_props.py run --label variant --set SOME_FLAG=True --compare baseline
```

Outputs
* per-row observations: `scripts/replay/output/<label>_<season>.csv.gz` (**gitignored**, large)
* summary: `scripts/replay/reports/<label>.json` and readable `.md` (**committed**, small)
* frozen chronological split boundaries: `scripts/replay/splits.json` (committed; created by the first full run)
* API cache: `replay-cache/` at the repo root (**gitignored**)

## Method

**Strictly chronological.** Every projection for a game on date *D* uses only rows with `GAME_DATE < D`
(`replay_data.asof_before`). This is enforced on every read, even though the cached API response contains the whole
season, and the engine raises if any history or league frame contains a game on/after *D*. `tests/test_replay_engine.py`
proves it end-to-end through the real model: changing the predicted game's own outcome, or any later game, does not
change its projection, while changing earlier games does.

**As-of inputs.**

| model input | replay source |
|---|---|
| player game history | player's prior games this season (RS and PO kept separate); rate samples keep the live minutes floors, while expected minutes use every played appearance with reduced leverage for extreme shorts |
| league logs (team defense) | league logs strictly before *D* (playoff logs once a playoff game has happened, otherwise regular season: the live fall-back rule) |
| DvP table | rebuilt each date from the same as-of league logs |
| usage (USG) | recomputed from box scores. RS: season-to-date before *D*. PO: the completed regular season |
| positions | current team rosters (positions are static labels; not an injury/result input) |
| learned-parameter artifact | `learned_params.json`, trained on 2020-21..2022-23 (validated/tested on later seasons before 2025-26): out-of-sample for 2025-26. Its hash is recorded in every report |

**Not reproducible, held neutral and recorded as limitations:** live injury status (the usage boost is 1.0), the book
spread (0.0, so no blowout minute cut), matchup pace (1.0; live mean was 1.004), calibration state at the time
(no residual correction or pick-based multiplier is loaded).

**Population.** A pre-game rule, independent of any model filter, so it is identical across model versions and
paired comparisons stay valid: the player has at least 10 prior played games this season; low-minute roles remain
eligible and are reported as their own stratum. The replay is conditional on the player having played (a DNP has no game-log row and
sportsbooks void those props). Rows without minutes are dropped as DNPs and counted in the report.

**Markets.** PTS, REB, AST, 3PM, PTS+REB, PTS+AST, REB+AST, PRA. The internal box-score column `FG3M`
(three-point field goals made) **is** the displayed/priced market `3PM` (`player_threes`); they are the same
quantity, only the name differs.

**Metrics** (per exact market and pooled): sample count, mean actual, mean projection, mean residual
(`projection - actual`), percentage bias `mean(projection - actual) / mean(actual)`, median residual, MAE, RMSE, and
the share of observations where the actual exceeded the projection. Reported for regular season, playoffs and
combined, plus chronological train / validation / final-test splits (50% / 25% / 25% of observations by date, per
regime, frozen in `splits.json`).

**Comparing versions.** `--compare <label>` pairs observations on `(date, player_id, market)` and reports the change
in % bias, MAE and RMSE, with a paired bootstrap 95% CI that resamples whole game dates (a slate's observations are
correlated). Prefer the simpler model when a difference is within that interval.

## API cache

`replay_cache.NBACache` stores raw API payloads as gzip JSON. The key covers endpoint, full request parameters,
season, season type, the request's date bound (`fetch_through`; the replay fetches whole seasons so it is `open`)
and a schema version. Because a cached payload is raw, **as-of filtering is applied on read, not baked into the
key**. Only successful responses are written. A timeout, exception, malformed response or retry exhaustion returns
`status="failed"` and is never cached and never turned into an empty frame; a successful response with zero rows is
`ok_empty` and may be cached. Live requests are rate limited (default 1.2 s minimum interval) with 3 attempts and
exponential backoff. Every request, retry, hit, miss, write and delay is counted in the report.

## Determinism

Rows are processed in date order per worker and sorted by `(date, player_id, market)` at the end; bootstrap seeds are
fixed. Two runs on the same code and cache produce identical per-row output.

## Files

`replay_props.py` CLI · `replay_engine.py` worker/driver · `replay_data.py` loading, as-of filter, as-of usage ·
`replay_cache.py` API cache · `replay_metrics.py` metrics, splits, paired bootstrap · `replay_report.py` summaries ·
`replay_provenance.py` commit/fingerprint/artifact hashes · `replay_config.py` versioned configuration.
