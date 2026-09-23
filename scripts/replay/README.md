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

# after the model-history fixes, select additive vs multiplicative corrections
# on chronological validation (the test period is held out for reporting)
python3 scripts/replay/evaluate_corrections.py \
  --input scripts/replay/output/after-correction-raw_2025-26.csv.gz

# v9: pre-game, schedule-based opportunities including DNP/voids, then
# independent L10 (primary) and L5 rolling-average half-point proxy lines
python3 scripts/replay/backtest_bets.py opportunities --workers 8
python3 scripts/replay/backtest_bets.py build

# Regenerate logged real-line probabilities from the fixed model, fit L10
# proxy probabilities as-of, and apply the held-out real-line deployment gate
python3 scripts/replay/clean_invalid_bets.py
python3 calibrate.py
python3 scripts/replay/calibrate_probabilities.py

# Let the fixed models act as a zero-vig "house": quote both sides of every
# cached 2025-26 game and every L10-eligible prop at the model's own half line
python3 scripts/replay/model_book_backtest.py --season 2025-26
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

**Not reproducible, held neutral and recorded as limitations:** historical injury-status and team-schedule evidence
(the return-from-injury ramp now fails closed; same-day injury reports and usage boosts are unavailable), the book
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

**v9 correction.** `evaluate_corrections.py` compares per-exact-market additive residual
`n/(n+k) × mean(actual − raw projection)` against a multiplicative residual fraction
`n/(n+k) × sum(actual − raw projection)/sum(raw projection)`. Same-date outcomes
never contribute to that date's estimate. The validation period chooses the form
and k; final-test dates are used only to report performance. The full 2025-26
season then fits `residual_corrections.json` for future-season live use. Replay
workers explicitly disable this final artifact and project the raw model first:
applying it to its own training dates would leak future results. The corrected
chronological per-row output is `output/corrected-chronological_2025-26.csv.gz`.
Selected-bet projection multipliers and player residual nudges are retired.

**v9 synthetic probability check.** `backtest_bets.py opportunities` infers a
pre-game candidate roster from each player's last prior played team and the
day's schedule. A DNP is retained as `DNP_VOID` and excluded from calibration.
Historical transactions/injury eligibility are not fully observable; the
candidate rules and gaps are reported in `reports/opportunity-coverage.md`.
The L10 primary proxy line is the mean of the last 10 played games before the
game date, rounded down to the previous integer plus 0.5; L5 repeats the
same method as a sensitivity check. Per-market proxy calibration uses weekly
expanding fits from strictly earlier game dates. The final
`probability_calibration.json` is trained on the full completed season for
2026-27 use only. The real-line check regenerates model probabilities at
the original logged lines and compares identity with Platt on later real
dates using a paired date bootstrap. A failed non-inferiority gate deploys
identity. See `reports/probability-calibration.md` for metrics and the JSON
for reliability bins by exact market and both directions. Synthetic lines
are **calibration only**; no ROI, edge or profit can be inferred from them.

Large per-row `backtest_bets.csv` and `output/` files are gitignored.
Never write simulated bets into `bet_log.csv`; that file contains only real
logged picks. `clean_invalid_bets.py` backs up the local log and moves
historical DNPs misgraded as misses into an ignored void file before
rerunning the older `calibrate.py` diagnostic.

**Model-book check.** `model_book_backtest.py` converts the saved game's
home-win probability into no-vig home/away fair odds. For player props it sets
the house line to `floor(as-of corrected projection) + 0.5`, recomputes the
predictive probability at that line, applies a calibrator fitted only on prior
weeks, and quotes both OVER and UNDER. Regular season and playoffs are reported
separately. Historical results never set a price; they are used only after the
quote to score Brier, log loss, accuracy and reliability. The game artifact's
training, validation and test seasons are rejected as holdouts. The run is
cache-only and aborts on a cache miss. These are the model's fair prices, so the
report supports prediction/calibration claims only; it cannot establish ROI,
profit or an edge over a sportsbook.

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
