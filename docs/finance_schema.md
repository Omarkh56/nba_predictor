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
