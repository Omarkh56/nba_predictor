# Phase 3 finance audit

Audit date: 2026-09-23. No projection or probability logic was changed.

## Existing records

- `bet_log.csv`: 956 real graded picks, 2026-04-18 through 2026-05-12.
- All 956 have projection, actual, result and model probability.
- 130 have a de-vig market probability; 118 have a decimal price.
- 0 have a stored stake or realized profit.
- Results: 494 HIT, 462 MISS.
- SQLite: 1,090 prop predictions, 2026-04-18 through 2026-06-08. Its current
  rows have 0 stored model probabilities, market probabilities, prices or stakes.

Decimal-price coverage by market is AST 39, REB 35, PTS 14, REB+AST 11,
PRA 8, 3PM 7, PTS+AST 4 and PTS+REB 0. Coverage exists on April 30, May 1–6,
May 8–12 only.

## Join quality

The player odds tracker has 2,046 daily consensus rows from 2026-04-30 through
2026-06-08. It matches 130 bets on date/player/market. Only 96 also match the
logged line exactly, and 92 of those contain a valid selected-side American
price. Twenty-six currently populated decimal prices came from a different line
and must be excluded from finance calculations.

## Current Kelly and persistence

`compute_stake_pct()` uses 35% of full Kelly, applies a risk-penalty multiplier,
and caps at 5%. It rejects missing odds and nonpositive EV. It does not enforce
the configured minimum edge, compare edge with paired-market vig, validate data
freshness, or return a machine-readable skip reason.

`nba_combined.py` passes decimal odds and `stake_pct` into SQLite;
`checkresults.py` carries them to CSV. The current stored rows predate usable
capture, so both stores have no stakes. There is no bankroll or stake-amount
ledger. Existing grading calculates profit as a bankroll fraction, skips PUSH
and VOID, and cannot produce bankroll reports.

## Idempotency

CSV identity is normalized date/player/market/line/pick. Repeated grading fills
blank fields, does not append duplicates, and aborts on conflicting populated
values. This protects row grading, but no idempotent bankroll calculation exists
yet.

## CLV evidence

`odds_tracker.csv` stores one consensus row per date/player/market. It has no
snapshot timestamp, event ID, commence time, sportsbook identity, or repeated
snapshots. Therefore it cannot identify a closing snapshot or support historical
CLV. The game-moneyline tracker has six timestamped rows for 2026-10-20, outside
the prop log's date range.

## Safe historical recovery

The 92 exact-line valid consensus prices and their paired de-vig/vig values can
be copied into a separate reviewed backfill file. They are pregame snapshots by
the tracker's fetch path, but their exact capture times and books are unknown.
Historical stakes, stake amounts, bankrolls, realized profits, closing prices,
CLV, logged timestamps and closing timestamps cannot be recovered and must stay
blank.
