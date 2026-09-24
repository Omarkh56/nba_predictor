# Finance record schema

`bet_log.csv` remains backward compatible. Existing columns keep their names,
including `line`, `p_model`, `p_market`, `decimal_odds`, `stake_pct`, `result`,
`actual`, and `realized_profit`. New columns are appended by
`bet_log_utils.upsert_bet_log()`.

`decimal_odds` remains an alias of `entry_decimal_odds` for older consumers.
`p_market` is the selected side's de-vig probability. The raw selected-side
probability is `market_implied_probability`. `kelly_fraction` stores full Kelly;
`stake_pct` stores the configured fractional-Kelly recommendation after caps.

Missing historical values remain blank. In particular, no migration assumes
`-110`, reconstructs a stake after the fact, or treats a same-day odds row as a
closing price. `bet_id` is a stable hash of the existing identity:
`date + player + market + line + pick`.

SQLite `prop_predictions` uses additive `ALTER TABLE` migrations, so databases
created before Phase 3 remain readable. CSV readers retain unknown columns and
writers preserve their original order after the documented public fields.

Set `NBA_STARTING_BANKROLL` to a positive amount to enable bankroll tracking.
All bets on the same date are sized from that date's opening bankroll, so CSV
row order cannot change stake amounts. Once an opening bankroll is recorded,
the persisted value remains authoritative on repeated grading runs.

`checkresults.py` settles profit only when a row has valid recorded entry odds
and a positive stake amount. It writes `reports/finance_summary.json`,
`reports/finance_summary.md`, and `reports/bankroll_curve.csv`. Rows without
finance data remain in hit-rate reporting and are counted as excluded from ROI.

Each `oddstracker.py` run also appends immutable bookmaker observations to the
gitignored `prop_odds_snapshots.csv`. Every observation carries snapshot time,
event ID, game start, book, player, market, exact line, side, price, and paired
de-vig data when both sides exist. `odds_tracker.csv` remains the backward-
compatible latest daily consensus used by the live predictor.

Closing joins select the final snapshot strictly before game start. Directional
line CLV and same-line price CLV stay separate. De-vig price CLV requires both
sides; raw implied-probability CLV may still be recorded when the opposing side
is missing. Failed joins keep CLV blank and store a machine-readable reason in
`clv_unmatched_reason`. Reports are written to `reports/clv_summary.json` and
`reports/clv_summary.md`.
